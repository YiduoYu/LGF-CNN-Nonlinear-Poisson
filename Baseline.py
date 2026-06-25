"""
CNN baseline for a nonlinear Poisson equation in flux form.

Solves:  -div(a(u) grad u) = f,  a(u) = u^k,  u = g on boundary
on Omega = [-1, 1]^2 with Dirichlet boundary conditions.

The loss is the conservative finite-difference flux residual on interior
grid nodes. Boundary values are enforced exactly via output projection.

Recommended:
    python Baseline.py --N 32 --train-samples 2048 --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. Utilities
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_run_id() -> str:
    """Generate a sortable, human-readable run identifier from current time."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_run_info(path: Path, info: dict) -> None:
    """Write a human-readable run summary file.

    `info` should contain keys like model, N, epochs, k, metrics dict, etc.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Run: {info.get('run_id', 'unknown')}",
             f"Model: {info.get('model', 'unknown')}",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    # key parameters
    for k in ("N", "epochs", "train_samples", "k"):
        if k in info:
            lines.append(f"{k} = {info[k]}")
    # metrics (if any)
    metrics = info.get("metrics", {})
    if metrics:
        lines.append("")
        lines.append("Metrics:")
        for mk, mv in metrics.items():
            lines.append(f"  {mk}: {mv}")
    lines.append("")
    with open(str(path), "w") as f:
        f.write("\n".join(lines))


def _update_latest_symlink(link_dir: Path, run_id: str) -> None:
    """Create or update `link_dir/latest` to point to `run_id`.

    Example: _update_latest_symlink(Path("checkpoints"), "20260624_143022")
    creates checkpoints/latest -> 20260624_143022
    """
    link_path = link_dir / "latest"
    link_dir.mkdir(parents=True, exist_ok=True)
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(run_id, target_is_directory=True)
    except OSError:
        # symlink not available (e.g. Windows without admin) — fallback to text file
        link_path.with_suffix(".txt").write_text(run_id)


# ===========================================================================
# 1b. Config-key based storage
# ===========================================================================

def config_key(N: int, epochs: int, k: float, base_ch: int = 32) -> str:
    """Generate a human-readable config directory name.

    Example: config_key(32, 40, 2.0) -> "N32_e40_k2.0"
    """
    # Format k to be filesystem-friendly
    k_str = f"k{k:.1f}".rstrip("0").rstrip(".") if k == int(k) else f"k{k:.2f}"
    return f"N{N}_e{epochs}_{k_str}"


def resolve_paths(N: int, epochs: int, k: float, base_ch: int = 32,
                  model_type: str = "cnn") -> dict:
    """Return a dict of resolved paths for a given config.

    Keys: ckpt_dir, best_path, last_path, history_path, convergence_path,
          run_info_path, figure_dir, results_dir, config_name
    """
    ckey = config_key(N, epochs, k, base_ch)
    ckpt_dir = Path("checkpoints") / model_type / ckey
    results_dir = Path("results") / model_type / ckey

    return {
        "ckpt_dir": ckpt_dir,
        "best_path": ckpt_dir / "best.pt",
        "last_path": ckpt_dir / "last.pt",
        "history_path": ckpt_dir / "history.csv",
        "convergence_path": ckpt_dir / "convergence.png",
        "run_info_path": ckpt_dir / "run_info.txt",
        "figure_dir": results_dir / "figures",
        "results_dir": results_dir,
        "config_name": f"{model_type}/{ckey}",
    }


def find_existing(ckpt_dir: Path, N: int, epochs: int) -> Path | None:
    """Return path to best.pt if it was trained with matching config, else None."""
    best = ckpt_dir / "best.pt"
    if not best.exists():
        return None
    try:
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        if isinstance(ckpt_args, argparse.Namespace):
            ckpt_args = vars(ckpt_args)
        if (ckpt_args.get("N") == N and ckpt_args.get("epochs") == epochs):
            return best
    except Exception:
        pass
    return None


def update_registry(model_type: str, ckey: str, info: dict) -> None:
    """Update checkpoints/registry.json with run metadata."""
    import json
    reg_path = Path("checkpoints") / "registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    registry: dict = {}
    if reg_path.exists():
        try:
            registry = json.loads(reg_path.read_text())
        except (json.JSONDecodeError, ValueError):
            registry = {}

    registry.setdefault(model_type, {})[ckey] = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **info,
    }
    reg_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


def migrate_legacy_checkpoint(ckpt_dir: Path) -> None:
    """Migrate root-level demo_best.pt / demo.csv to new config dir if needed.

    Only runs once — if best.pt already exists in the target, do nothing.
    """
    best = ckpt_dir / "best.pt"
    if best.exists():
        return

    legacy_best = Path("checkpoints") / "demo_best.pt"
    legacy_csv = Path("checkpoints") / "demo.csv"
    legacy_conv = Path("checkpoints") / "demo.png"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    migrated = []

    if legacy_best.exists():
        legacy_best.rename(best)
        migrated.append("demo_best.pt → best.pt")

    if legacy_csv.exists():
        legacy_csv.rename(ckpt_dir / "history.csv")
        migrated.append("demo.csv → history.csv")

    if legacy_conv.exists():
        legacy_conv.rename(ckpt_dir / "convergence.png")
        migrated.append("demo.png → convergence.png")

    if migrated:
        print(f"Migrated legacy files: {', '.join(migrated)}")


# ===========================================================================
# 2. Grid, masks, and input channels
# ===========================================================================

def make_grid(N: int, device: torch.device = "cpu",
              dtype: torch.dtype = torch.float32):
    """Return (x, y, boundary_mask, interior_mask, h) as [1,1,H,W] tensors.

    Full grid includes boundary nodes: H = W = N + 2, h = 2/(N+1).
    """
    H = W = N + 2
    h = 2.0 / (N + 1)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x = xx[None, None, :, :]
    y = yy[None, None, :, :]

    # boundary mask: 1 on the four edges, 0 in interior
    boundary = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    boundary[:, :,  0,  :] = 1.0
    boundary[:, :, -1,  :] = 1.0
    boundary[:, :,  :,  0] = 1.0
    boundary[:, :,  :, -1] = 1.0
    interior = 1.0 - boundary

    return x, y, boundary, interior, h


def build_input(f_norm: torch.Tensor, g: torch.Tensor,
                boundary: torch.Tensor, k_norm: torch.Tensor,
                x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Build 6-channel CNN input: [f_norm, g, boundary, k_norm, x, y].

    All tensors expanded to batch size B. f_norm and k_norm should already
    be normalized by the caller.
    """
    B = f_norm.shape[0]
    if boundary.shape[0] == 1:
        boundary = boundary.expand(B, -1, -1, -1)
    if x.shape[0] == 1:
        x = x.expand(B, -1, -1, -1)
    if y.shape[0] == 1:
        y = y.expand(B, -1, -1, -1)

    k_ch = k_norm.view(B, 1, 1, 1).expand(B, 1, f_norm.shape[-2], f_norm.shape[-1])
    return torch.cat([f_norm, g, boundary, k_ch, x, y], dim=1)


def enforce_dirichlet(u_raw: torch.Tensor, g: torch.Tensor,
                      boundary: torch.Tensor) -> torch.Tensor:
    """Hard-impose Dirichlet boundary values: u = g on boundary nodes."""
    return boundary * g + (1.0 - boundary) * u_raw


# ===========================================================================
# 3. Flux-form PDE residual
# ===========================================================================

K_DEFAULT = 2.0


def _safe_pow(u: torch.Tensor, k: torch.Tensor, eps: float = 1e-6
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (u^k, k * u^{k-1}) handling integer vs fractional k.

    For integer k, no clamp — safe for negative u.
    For fractional k, clamps u > 0 to prevent NaN.
    """
    k = k.to(device=u.device, dtype=u.dtype)
    while k.ndim < 4:
        k = k.unsqueeze(-1)

    if torch.allclose(k, k.round(), atol=1e-6, rtol=0.0):
        return torch.pow(u, k), k * torch.pow(u, k - 1.0)
    return (torch.pow(torch.clamp(u, min=eps), k),
            k * torch.pow(torch.clamp(u, min=eps), k - 1.0))


def power_coefficient(u: torch.Tensor, k: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """Compute a(u) = u^k.  See _safe_pow for integer/fractional handling."""
    a, _ = _safe_pow(u, k, eps)
    return a


def flux_divergence(u: torch.Tensor, k: torch.Tensor, h: float,
                    eps: float = 1e-6) -> torch.Tensor:
    """Return -div_h(a(u) grad_h u) on interior nodes via conservative FD.

    Face diffusion coefficients are arithmetic averages of adjacent cell
    centers, giving a single conservative flux shared by neighbouring
    control volumes. Output shape: [B, 1, N, N].
    """
    a = power_coefficient(u, k, eps)

    # interior slices
    uc = u[:, :, 1:-1, 1:-1]
    ac = a[:, :, 1:-1, 1:-1]

    # face-averaged coefficients
    ae = 0.5 * (ac + a[:, :, 1:-1, 2:])
    aw = 0.5 * (ac + a[:, :, 1:-1, :-2])
    an = 0.5 * (ac + a[:, :, 2:, 1:-1])
    as_ = 0.5 * (ac + a[:, :, :-2, 1:-1])

    # fluxes on faces
    fe = ae * (u[:, :, 1:-1, 2:] - uc) / h
    fw = aw * (uc - u[:, :, 1:-1, :-2]) / h
    fn = an * (u[:, :, 2:, 1:-1] - uc) / h
    fs = as_ * (uc - u[:, :, :-2, 1:-1]) / h

    return -(fe - fw + fn - fs) / h


def pde_residual(u: torch.Tensor, f: torch.Tensor, k: torch.Tensor,
                 h: float, eps: float = 1e-6) -> torch.Tensor:
    """R = -div_h(a(u) grad_h u) - f  on interior nodes."""
    return flux_divergence(u, k, h, eps) - f[:, :, 1:-1, 1:-1]


# ===========================================================================
# 4. U-Net model
# ===========================================================================

def _group_count(ch: int, max_groups: int = 8) -> int:
    """Largest divisor of ch not exceeding max_groups."""
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return g


class ConvBlock(nn.Module):
    """Double 3x3 conv with GroupNorm + GELU."""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g = _group_count(out_ch, groups)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedBlock(nn.Module):
    """Residual block with a dilated first convolution for larger RF."""
    def __init__(self, ch: int, dilation: int = 1, groups: int = 8):
        super().__init__()
        g = _group_count(ch, groups)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation,
                               padding_mode="replicate")
        self.gn1 = nn.GroupNorm(g, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")
        self.gn2 = nn.GroupNorm(g, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.gn1(self.conv1(x)))
        return F.gelu(x + self.gn2(self.conv2(z)))


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8):
    """Pad spatial dims to next multiple of `multiple`. Returns (padded, pad)."""
    _, _, H, W = x.shape
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    pad = (0, pw, 0, ph)
    if ph or pw:
        x = F.pad(x, pad, mode="replicate")
    return x, pad


def _unpad(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    l, r, t, b = pad
    return x[:, :, t:x.shape[-2] - b, l:x.shape[-1] - r]


class UNet(nn.Module):
    """2D U-Net with GroupNorm, GELU, dilated bottleneck, skip connections.

    Args:
        in_ch:  input channels (default 6)
        base_ch: channels in first encoder block (default 32)
    """
    def __init__(self, in_ch: int = 6, base_ch: int = 32):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, 2 * base_ch)
        self.enc3 = ConvBlock(2 * base_ch, 4 * base_ch)
        # bottleneck with dilated residuals
        self.bottleneck = nn.Sequential(
            ConvBlock(4 * base_ch, 8 * base_ch),
            DilatedBlock(8 * base_ch, dilation=2),
            DilatedBlock(8 * base_ch, dilation=4),
        )
        # decoder (concat skip → 1.5× channels)
        self.up3 = nn.Conv2d(12 * base_ch, 4 * base_ch, 3, padding=1,
                             padding_mode="replicate")
        self.dec3 = ConvBlock(4 * base_ch, 4 * base_ch)
        self.up2 = nn.Conv2d(6 * base_ch, 2 * base_ch, 3, padding=1,
                             padding_mode="replicate")
        self.dec2 = ConvBlock(2 * base_ch, 2 * base_ch)
        self.up1 = nn.Conv2d(3 * base_ch, base_ch, 3, padding=1,
                             padding_mode="replicate")
        self.dec1 = ConvBlock(base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad = _pad_to_multiple(x, 8)

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear",
                           align_corners=False)
        d3 = self.dec3(self.up3(torch.cat([d3, e3], dim=1)))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear",
                           align_corners=False)
        d2 = self.dec2(self.up2(torch.cat([d2, e2], dim=1)))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear",
                           align_corners=False)
        d1 = self.dec1(self.up1(torch.cat([d1, e1], dim=1)))

        return _unpad(self.head(d1), pad)


# ===========================================================================
# 5. Manufactured solution & dataset
# ===========================================================================

def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Manufactured solution: u(x,y) = sin(pi*x) * cos(pi*y)."""
    return torch.sin(math.pi * x) * torch.cos(math.pi * y)


def forcing(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor,
            eps: float = 1e-6) -> torch.Tensor:
    """Analytic f = -div(u^k grad u) for u = sin(pi*x)*cos(pi*y).

    For integer k (e.g. k=2), u^k is well-defined for negative u, so no
    clamp is applied. For fractional k, a small positive clamp prevents
    NaN gradients from negative bases.
    """
    pi = math.pi
    u = u_exact(x, y)
    u_x = pi * torch.cos(pi * x) * torch.cos(pi * y)
    u_y = -pi * torch.sin(pi * x) * torch.sin(pi * y)
    laplacian = -2.0 * pi * pi * u
    grad_sq = u_x * u_x + u_y * u_y

    a, da = _safe_pow(u, k, eps)
    return -(da * grad_sq + a * laplacian)


def generate_dataset(N: int, n_samples: int, device: torch.device,
                     k_val: float = K_DEFAULT):
    """Return a batch of manufactured-problem instances as a dict.

    Keys: f, g, mask, interior_mask, x, y, k, u_exact, h.
    f, u_exact, g have shape [n_samples, 1, N+2, N+2].
    k has shape [n_samples, 1].
    mask, interior_mask, x, y have shape [1, 1, N+2, N+2] (shared).
    h is a float.
    """
    x, y, boundary, interior, h = make_grid(N, device)

    # u_exact and f are the same for all samples (k is fixed)
    ue = u_exact(x, y)                        # [1, 1, H, W]
    k = torch.full((1, 1), k_val, device=device)
    f = forcing(x, y, k)                      # [1, 1, H, W]

    # replicate to batch
    f = f.expand(n_samples, -1, -1, -1)
    ue = ue.expand(n_samples, -1, -1, -1)
    k = torch.full((n_samples, 1), k_val, device=device)
    g = boundary * ue

    return {
        "f": f, "g": g,
        "mask": boundary, "interior_mask": interior,
        "x": x, "y": y, "k": k,
        "u_exact": ue, "h": h,
    }


@dataclass
class InputStats:
    """Normalization statistics for input channels."""
    f_mean: float
    f_std: float
    k_mean: float
    k_std: float


def compute_stats(data: dict) -> InputStats:
    """Compute normalisation stats from interior f and k."""
    f_int = data["f"][:, :, 1:-1, 1:-1]
    k = data["k"]
    return InputStats(
        f_mean=float(f_int.mean()),
        f_std=float(f_int.std().clamp_min(1e-8)),
        k_mean=float(k.mean()),
        k_std=float(k.std(unbiased=False).clamp_min(1e-8)),
    )


# ===========================================================================
# 6. Training
# ===========================================================================

def _take_batch(data: dict, idx: torch.Tensor) -> dict:
    """Slice a mini-batch from the full dataset dict."""
    return {
        **{k: data[k][idx] for k in ("f", "g", "u_exact", "k")},
        **{k: data[k][:1] for k in ("mask", "interior_mask", "x", "y")},
        "h": data["h"],
    }


def _forward(model: nn.Module, batch: dict, stats: InputStats,
             norm_k: bool) -> torch.Tensor:
    """Normalize inputs, run CNN, enforce Dirichlet BC."""
    f_norm = (batch["f"] - stats.f_mean) / stats.f_std
    k_in = ((batch["k"] - stats.k_mean) / stats.k_std) if norm_k else batch["k"]
    inp = build_input(f_norm, batch["g"], batch["mask"], k_in,
                      batch["x"], batch["y"])
    return enforce_dirichlet(model(inp), batch["g"], batch["mask"])


def train(args: argparse.Namespace, best_path: Path, last_path: Path,
          history_path: Path, convergence_path: Path) -> tuple[nn.Module, float]:
    set_seed(args.seed)
    device = select_device(args.device)
    t0 = time.time()

    # ---- datasets ----
    print(f"generating {args.train_samples} train samples (N={args.N})", flush=True)
    train_data = generate_dataset(args.N, args.train_samples, device)
    print(f"generating {args.val_samples} val samples", flush=True)
    val_data = generate_dataset(args.N, args.val_samples, device)
    stats = compute_stats(train_data)

    # oracle floor: PDE residual of the exact solution (truncation error)
    with torch.no_grad():
        res_exact = pde_residual(val_data["u_exact"], val_data["f"],
                                 val_data["k"], val_data["h"])
        oracle_floor = float(res_exact.pow(2).mean().sqrt())

    # ---- model, optimizer, scheduler ----
    model = UNet(in_ch=6, base_ch=args.base_ch).to(device)

    # AdamW: no weight decay on bias / 1-d params
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith(".bias") or p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)

    steps_per_epoch = math.ceil(args.train_samples / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ---- io ----
    best_val_relL2 = math.inf
    eval_bs = args.eval_batch_size or args.batch_size

    print(f"device={device}  N={args.N}  "
          f"train={args.train_samples}  val={args.val_samples}  "
          f"gen_time={time.time() - t0:.1f}s")
    print(f"f stats: mean={stats.f_mean:.3f}  std={stats.f_std:.3f}")
    print(f"oracle resRMSE (FD truncation floor): {oracle_floor:.4f}")
    print(f"steps/epoch={steps_per_epoch}  total_steps={total_steps}")

    # ---- training loop ----
    history: list[dict] = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.train_samples, device=device)

        for start in range(0, args.train_samples, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch = _take_batch(train_data, idx)
            u_pred = _forward(model, batch, stats, args.normalize_k)

            # relative PDE residual loss
            res = pde_residual(u_pred, batch["f"], batch["k"], batch["h"])
            f_int = batch["f"][:, :, 1:-1, 1:-1]
            denom = f_int.pow(2).mean(dim=(1, 2, 3), keepdim=True) \
                           .sqrt().clamp_min(1e-3)
            loss = (res / denom).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

        # ---- eval ----
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            train_res_rmse = float(res.pow(2).mean().sqrt())
            train_bmax = float(
                ((u_pred - batch["g"]).abs() * batch["mask"]).max()
            )
            val_m = evaluate(model, val_data, stats, eval_bs, args.normalize_k)

            # save best
            if val_m["relL2"] < best_val_relL2:
                best_val_relL2 = val_m["relL2"]
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "input_stats": vars(stats),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val_relL2": best_val_relL2,
                }, best_path)
                print(f"saved best checkpoint: {best_path}")

            print(
                f"epoch={epoch:03d}  step={step:05d}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"loss={float(loss):.3e}  "
                f"train_resRMSE={train_res_rmse:.4f}  "
                f"bmax={train_bmax:.2e} | "
                f"val relL2={val_m['relL2']:.4e}  "
                f"linf={val_m['linf']:.4e}  "
                f"resRMSE={val_m['resRMSE']:.4f}  "
                f"resLinf={val_m['resLinf']:.4e}  "
                f"gradRelL2={val_m['gradRelL2']:.4e}"
            )
            history.append({
                "epoch": epoch, "step": step,
                "lr": float(scheduler.get_last_lr()[0]),
                "loss": float(loss),
                "train_resRMSE": train_res_rmse,
                "val_relL2": val_m["relL2"],
                "val_resRMSE": val_m["resRMSE"],
                "val_linf": val_m["linf"],
                "val_gradRelL2": val_m["gradRelL2"],
            })

    # ---- save final ----
    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_stats": vars(stats),
        "args": vars(args),
    }, last_path)
    print(f"saved final checkpoint: {last_path}")

    # training history CSV
    _write_csv(history, history_path)
    print(f"saved training history: {history_path}")

    # convergence plot
    _save_convergence_plot(history, convergence_path, oracle_floor)
    print(f"saved convergence plot: {convergence_path}")

    return model, oracle_floor


# ===========================================================================
# 7. Evaluation
# ===========================================================================

def _error_tensors(u_pred: torch.Tensor, batch: dict):
    """Compute error & residual tensors shared by evaluate and per_sample_metrics.

    Returns (err, res_pred, res_exact, bnd_err_abs, interior, u_exact, h).
    """
    ue = batch["u_exact"]
    interior = batch["interior_mask"].expand_as(u_pred)
    boundary = batch["mask"]
    if boundary.shape[0] == 1:
        boundary = boundary.expand_as(u_pred)

    err = (u_pred - ue) * interior
    res_pred = pde_residual(u_pred, batch["f"], batch["k"], batch["h"])
    res_exact = pde_residual(ue, batch["f"], batch["k"], batch["h"])
    bnd_err = (u_pred - batch["g"]).abs() * boundary

    return err, res_pred, res_exact, bnd_err, interior, ue, float(batch["h"])


@torch.no_grad()
def evaluate(model: nn.Module, data: dict, stats: InputStats,
             batch_size: int, norm_k: bool) -> dict:
    """Evaluate model over the full dataset. Returns aggregate metric dict."""
    model.eval()
    n = data["f"].shape[0]
    device = data["f"].device

    sums = {"relL2": 0.0, "linf": 0.0, "res_sq": 0.0, "res_linf": 0.0,
            "bmax": 0.0, "grad_sum": 0.0, "count": 0}

    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=device)
        batch = _take_batch(data, idx)
        u_pred = _forward(model, batch, stats, norm_k)

        err, res_pred, _, bnd_err, interior, ue, h = _error_tensors(u_pred, batch)

        m = u_pred.shape[0]
        sums["count"] += m
        sums["relL2"] += float(
            (err.flatten(1).norm(dim=1)
             / ue.mul(interior).flatten(1).norm(dim=1).clamp_min(1e-12)).sum()
        )
        sums["linf"] = max(sums["linf"], float(err.abs().max()))
        sums["res_sq"] += float(res_pred.pow(2).sum())
        sums["res_linf"] = max(sums["res_linf"], float(res_pred.abs().max()))
        sums["bmax"] = max(sums["bmax"], float(bnd_err.max()))

        # gradient relative L2 (not in _error_tensors — only needed here)
        du = lambda v: (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2.0 * h)
        dv = lambda v: (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2.0 * h)
        p_grad = torch.cat([du(u_pred), dv(u_pred)], dim=1)
        t_grad = torch.cat([du(ue), dv(ue)], dim=1)
        sums["grad_sum"] += float(
            ((p_grad - t_grad).flatten(1).norm(dim=1)
             / t_grad.flatten(1).norm(dim=1).clamp_min(1e-12)).sum()
        )

    c = sums["count"]
    # number of interior nodes across all samples
    res_n = c * (data["f"].shape[-2] - 2) * (data["f"].shape[-1] - 2)
    return {
        "relL2": sums["relL2"] / c,
        "linf": sums["linf"],
        "resRMSE": math.sqrt(sums["res_sq"] / res_n),
        "resLinf": sums["res_linf"],
        "bmax": sums["bmax"],
        "gradRelL2": sums["grad_sum"] / c,
    }


def per_sample_metrics(u_pred: torch.Tensor, batch: dict) -> list[dict]:
    """Compute per-sample metrics (for CSV export)."""
    err, res_pred, res_exact, bnd_err, _, ue, _ = _error_tensors(u_pred, batch)

    int_err = err[:, :, 1:-1, 1:-1].flatten(1)
    int_ue = ue[:, :, 1:-1, 1:-1].flatten(1)
    res_flat = res_pred.flatten(1)
    res_exact_flat = res_exact.flatten(1)
    bnd_flat = bnd_err.flatten(1)
    f_flat = batch["f"].flatten(1)

    relL2 = int_err.norm(dim=1) / int_ue.norm(dim=1).clamp_min(1e-12)
    rmse = int_err.pow(2).mean(1).sqrt()
    mae = int_err.abs().mean(1)
    linf = int_err.abs().max(1).values
    bmax = bnd_flat.max(1).values
    pred_res_rmse = res_flat.pow(2).mean(1).sqrt()
    exact_res_rmse = res_exact_flat.pow(2).mean(1).sqrt()
    f_rmse = f_flat.pow(2).mean(1).sqrt()

    rows = []
    for i in range(u_pred.shape[0]):
        rows.append({
            "sample_id": i,
            "k": float(batch["k"][i]),
            "relL2": float(relL2[i]),
            "RMSE": float(rmse[i]),
            "MAE": float(mae[i]),
            "Linf": float(linf[i]),
            "boundary_max": float(bmax[i]),
            "pred_flux_resRMSE": float(pred_res_rmse[i]),
            "exact_flux_resRMSE": float(exact_res_rmse[i]),
            "f_RMSE": float(f_rmse[i]),
        })
    return rows


# ===========================================================================
# 8. Prediction & visualization
# ===========================================================================

def load_checkpoint(path: str | Path, device: torch.device):
    """Load model, stats, and args from a training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_args = ckpt.get("args", {})
    if isinstance(raw_args, argparse.Namespace):
        raw_args = vars(raw_args)

    base_ch = raw_args.get("base_ch", 32)
    model = UNet(in_ch=6, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    s = ckpt["input_stats"]
    stats = InputStats(f_mean=s["f_mean"], f_std=s["f_std"],
                       k_mean=s["k_mean"], k_std=s["k_std"])
    return model, stats, argparse.Namespace(**raw_args)


def _get_plt():
    """Import matplotlib (non-interactive Agg backend)."""
    cache = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_panel(u_exact: torch.Tensor, u_pred: torch.Tensor,
                error: torch.Tensor, f_map: torch.Tensor,
                residual_pred: torch.Tensor, residual_exact: torch.Tensor,
                path: Path, title: str,
                full_ext: list, resid_ext: list, dpi: int = 200):
    """Save a 2×3 comparison panel.

    Row 1: true u | CNN prediction | signed error
    Row 2: forcing f | pred PDE residual | exact PDE residual (oracle floor)
    """
    plt = _get_plt()

    def _sym_lim(v):
        mx = max(float(v.abs().max().cpu()), 1e-12)
        return -mx, mx

    u_min = min(float(u_exact.min()), float(u_pred.min()))
    u_max = max(float(u_exact.max()), float(u_pred.max()))
    err_lim = _sym_lim(error)
    f_lim = _sym_lim(f_map)
    pred_res_lim = _sym_lim(residual_pred)
    exact_res_lim = _sym_lim(residual_exact)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (0, 0, u_exact,        "true u",                  "viridis",   full_ext,  u_min, u_max),
        (0, 1, u_pred,         "CNN prediction",           "viridis",   full_ext,  u_min, u_max),
        (0, 2, error,          "signed error",             "coolwarm",  full_ext,  *err_lim),
        (1, 0, f_map,          "forcing f",                "coolwarm",  full_ext,  *f_lim),
        (1, 1, residual_pred,  "pred PDE residual",        "coolwarm",  resid_ext, *pred_res_lim),
        (1, 2, residual_exact, "exact PDE residual (floor)","coolwarm",  resid_ext, *exact_res_lim),
    ]
    for r, c, vals, lbl, cmap, ext, v0, v1 in panels:
        ax = axes[r][c]
        im = ax.imshow(vals.detach().cpu().numpy(), origin="lower", extent=ext,
                       cmap=cmap, vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(lbl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_plot(history: list[dict], path: Path,
                           oracle_floor: float = 0.0):
    """Save a 2-panel convergence plot: loss + validation metrics vs epoch."""
    if len(history) < 2:
        return
    plt = _get_plt()

    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    train_res = [h["train_resRMSE"] for h in history]
    val_relL2 = [h["val_relL2"] for h in history]
    val_res = [h["val_resRMSE"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Training convergence")

    # left panel: training loss + residual
    ax1b = ax1.twinx()
    l1, = ax1.plot(epochs, loss, "b-", label="PDE loss (train)")
    l2, = ax1b.plot(epochs, train_res, "r--", label="train resRMSE")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1b.set_ylabel("resRMSE")
    ax1.set_title("Training loss")
    ax1.legend(handles=[l1, l2], loc="upper right")
    ax1.grid(True, alpha=0.3)

    # right panel: validation metrics
    ax2b = ax2.twinx()
    l3, = ax2.plot(epochs, val_relL2, "g-", label="val relL2")
    l4, = ax2b.plot(epochs, val_res, "m--", label="val resRMSE")
    if oracle_floor > 0:
        ax2b.axhline(oracle_floor, color="gray", ls=":", label=f"oracle floor ({oracle_floor:.4f})")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("relL2"); ax2b.set_ylabel("resRMSE")
    ax2.set_title("Validation metrics")
    ax2.legend(handles=[l3, l4], loc="upper right")
    ax2.grid(True, alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: list[dict], path: Path):
    """Write list of dicts as CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _panel_title(row: dict) -> str:
    return (f"sample {row['sample_id']} | k={row['k']:.4g}  "
            f"relL2={row['relL2']:.3e}  RMSE={row['RMSE']:.3e}  "
            f"Linf={row['Linf']:.3e}  "
            f"pred_resRMSE={row['pred_flux_resRMSE']:.3e}  "
            f"exact_resRMSE={row['exact_flux_resRMSE']:.3e}")


def run_prediction(args: argparse.Namespace) -> None:
    """Load checkpoint, run prediction, save metrics and figures."""
    if not args.load_path:
        raise ValueError("--load-path is required for prediction")

    set_seed(args.seed)
    device = select_device(args.device)
    model, stats, ckpt_args = load_checkpoint(args.load_path, device)

    # resolve config (runtime overrides checkpoint)
    N = args.predict_N or getattr(ckpt_args, "N", 32)
    k_val = getattr(ckpt_args, "k_min", 2.0)
    norm_k = getattr(ckpt_args, "normalize_k", True)

    print(f"prediction: N={N}  k={k_val}  samples={args.predict_samples}")
    set_seed(args.test_seed)
    batch = generate_dataset(N, args.predict_samples, device, k_val)

    # forward pass
    model.eval()
    with torch.no_grad():
        u_pred = _forward(model, batch, stats, norm_k)

    # aggregate metrics
    vm = evaluate(model, batch, stats, args.predict_samples, norm_k)
    print(f"relL2={vm['relL2']:.4e}  linf={vm['linf']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  resLinf={vm['resLinf']:.4e}  "
          f"bmax={vm['bmax']:.2e}  gradRelL2={vm['gradRelL2']:.4e}")

    rows = per_sample_metrics(u_pred, batch)

    # ---- save .pt payload ----
    if args.predict_save:
        path = Path(args.predict_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "k": batch["k"].cpu(), "u_exact": batch["u_exact"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]),
            "metrics": vm,
            "config": {"N": N, "k_val": k_val},
        }, path)
        print(f"saved prediction: {path}")

    # ---- prediction panels ----
    if args.plot:
        out = Path(args.plot_dir)
        h = float(batch["h"])
        full_ext = [-1.0, 1.0, -1.0, 1.0]
        resid_ext = [-1.0 + h, 1.0 - h, -1.0 + h, 1.0 - h]

        for i, row in enumerate(rows):
            sid = row["sample_id"]
            ui = batch["u_exact"][i, 0]
            up = u_pred[i, 0]
            err = up - ui
            res_pred = pde_residual(u_pred[i:i + 1], batch["f"][i:i + 1],
                                    batch["k"][i:i + 1], h)[0, 0]
            res_exact = pde_residual(batch["u_exact"][i:i + 1],
                                     batch["f"][i:i + 1],
                                     batch["k"][i:i + 1], h)[0, 0]

            _save_panel(ui, up, err, batch["f"][i, 0], res_pred, res_exact,
                        out / f"sample_{sid:03d}_panel.png",
                        _panel_title(row), full_ext, resid_ext)
        _write_csv(rows, out / "metrics.csv")
        print(f"saved panels + metrics: {out}")



# ===========================================================================
# 9. CLI & main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # data
    p.add_argument("--N", type=int, default=32, help="Interior grid size")
    p.add_argument("--train-samples", type=int, default=2048)
    p.add_argument("--val-samples", type=int, default=128)

    # training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=0,
                   help="Defaults to --batch-size")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--base-ch", type=int, default=32,
                   help="U-Net base channel count")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--normalize-k", action=argparse.BooleanOptionalAction,
                   default=True)

    # system
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)

    # io
    p.add_argument("--save-path", default="",
                   help="Deprecated: paths are auto-generated from config")
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining even if a matching checkpoint exists")
    p.add_argument("--load-path", default="")

    # prediction
    p.add_argument("--predict", action="store_true",
                   help="Run prediction from checkpoint")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="",
                   help="Save predictions as .pt file")
    p.add_argument("--predict-N", type=int, default=None,
                   help="Override grid size")
    p.add_argument("--test-seed", type=int, default=1234)

    # visualization
    p.add_argument("--plot", action="store_true",
                   help="Save prediction panel figures")
    p.add_argument("--plot-dir", default="results/figures")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # default with no arguments: demo mode
    if len(sys.argv) == 1:
        args.demo = True

    # ── Resolve config and paths ──
    model_type = "cnn"

    if args.demo:
        args.N = 32
        args.epochs = 40
        args.train_samples = 2048
        args.val_samples = 128
        args.batch_size = 16
        args.eval_batch_size = 0
        args.eval_every = 1

    paths = resolve_paths(args.N, args.epochs, K_DEFAULT,
                          args.base_ch, model_type)

    # ── Migrate legacy files on first run ──
    migrate_legacy_checkpoint(paths["ckpt_dir"])

    # ── Check for existing trained model ──
    existing = find_existing(paths["ckpt_dir"], args.N, args.epochs)

    ckey = config_key(args.N, args.epochs, K_DEFAULT, args.base_ch)

    if existing and not args.retrain:
        print("=" * 60)
        print(f"  Config:  {paths['config_name']}")
        print(f"  ✓ Found trained model: {existing}")
        print(f"  Skipping training. Use --retrain to force retraining.")
        print("=" * 60)
        # Still update registry with existing run info
        update_registry(model_type, ckey, {
            "N": args.N, "epochs": args.epochs, "k": K_DEFAULT,
        })
    else:
        if args.retrain and existing:
            print(f"⟳ Retraining requested for {paths['config_name']}")

        print("=" * 60)
        print(f"  Config:  {paths['config_name']}")
        print(f"  Output:  {paths['ckpt_dir']}")
        print("=" * 60)

        train_args = argparse.Namespace(**vars(args))
        model, oracle_floor = train(
            train_args,
            paths["best_path"],
            paths["last_path"],
            paths["history_path"],
            paths["convergence_path"],
        )
        existing = paths["best_path"]

        # ── Update registry ──
        update_registry(model_type, ckey, {
            "N": args.N,
            "epochs": args.epochs,
            "k": K_DEFAULT,
            "oracle_floor": oracle_floor,
        })

    # ── Predict ──
    if args.demo or args.plot or args.predict:
        pred_ckpt = existing if existing else paths["best_path"]
        print(f"\nPredicting from: {pred_ckpt}")

        pred_args = argparse.Namespace(**vars(args))
        pred_args.load_path = str(pred_ckpt)
        pred_args.predict_samples = 1 if args.demo else args.predict_samples
        pred_args.predict_save = ""
        pred_args.plot = True
        pred_args.plot_dir = str(paths["figure_dir"])
        pred_args.predict_N = None
        pred_args.test_seed = args.seed
        run_prediction(pred_args)

    # ── Run summary ──
    run_params = {
        "run_id": get_run_id(), "model": "U-Net (Baseline)",
        "N": args.N, "epochs": args.epochs,
        "train_samples": args.train_samples,
        "k": K_DEFAULT,
    }
    _write_run_info(paths["run_info_path"], run_params)
    _update_latest_symlink(Path("checkpoints") / model_type, ckey)
    _update_latest_symlink(Path("results") / model_type, ckey)

    print(f"\nDone. Config: {paths['config_name']}")
    print(f"Checkpoints: {paths['ckpt_dir']}")
    print(f"Figures:     {paths['figure_dir']}")
    print(f"Latest:      checkpoints/{model_type}/latest -> {ckey}")
    return


if __name__ == "__main__":
    main()
