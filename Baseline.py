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

def power_coefficient(u: torch.Tensor, k: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """Compute a(u) = u^k.

    For integer k, u^k is well-defined for all real u (including negative),
    so no clamp is applied. For non-integer k, a small positive clamp guards
    against NaN gradients from negative bases.
    """
    if not torch.is_tensor(k):
        k = torch.tensor(k, device=u.device, dtype=u.dtype)
    k = k.to(device=u.device, dtype=u.dtype)
    while k.ndim < 4:
        k = k.unsqueeze(-1)

    if torch.allclose(k, k.round(), atol=1e-6, rtol=0.0):
        return torch.pow(u, k)                     # integer exponent: safe
    return torch.pow(torch.clamp(u, min=eps), k)   # fractional exponent: guard


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

    k = k.to(device=u.device, dtype=u.dtype)
    while k.ndim < 4:
        k = k.unsqueeze(-1)

    if torch.allclose(k, k.round(), atol=1e-6, rtol=0.0):
        a = torch.pow(u, k)                    # integer k: safe for all u
        da = k * torch.pow(u, k - 1.0)
    else:
        a = torch.pow(torch.clamp(u, min=eps), k)      # fractional k: guard
        da = k * torch.pow(torch.clamp(u, min=eps), k - 1.0)

    return -(da * grad_sq + a * laplacian)


def generate_dataset(N: int, n_samples: int, device: torch.device,
                     k_val: float = 2.0):
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


def train(args: argparse.Namespace) -> nn.Module:
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
    save_path = Path(args.save_path) if args.save_path else None
    best_path = (save_path.with_stem(save_path.stem + "_best")
                 if save_path else None)
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
            if best_path and val_m["relL2"] < best_val_relL2:
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
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_stats": vars(stats),
            "args": vars(args),
        }, sp)
        print(f"saved checkpoint: {sp}")

        # training history CSV
        hist_path = sp.with_suffix(".csv")
        _write_csv(history, hist_path)
        print(f"saved training history: {hist_path}")

        # convergence plot
        plot_path = sp.with_suffix(".png")
        _save_convergence_plot(history, plot_path, oracle_floor)
        print(f"saved convergence plot: {plot_path}")

    return model


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
                error: torch.Tensor, abs_error: torch.Tensor,
                residual_pred: torch.Tensor, residual_exact: torch.Tensor,
                path: Path, title: str,
                full_ext: list, resid_ext: list, dpi: int = 200):
    """Save a 2×3 comparison panel.

    Row 1: true u | CNN prediction | signed error
    Row 2: |error| | pred PDE residual | exact PDE residual (oracle floor)
    """
    plt = _get_plt()

    def _sym_lim(v):
        mx = max(float(v.abs().max().cpu()), 1e-12)
        return -mx, mx

    u_min = min(float(u_exact.min()), float(u_pred.min()))
    u_max = max(float(u_exact.max()), float(u_pred.max()))
    err_lim = _sym_lim(error)
    abs_max = max(float(abs_error.max().cpu()), 1e-12)
    pred_res_lim = _sym_lim(residual_pred)
    exact_res_lim = _sym_lim(residual_exact)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (0, 0, u_exact,        "true u",                  "viridis",   full_ext,  u_min, u_max),
        (0, 1, u_pred,         "CNN prediction",           "viridis",   full_ext,  u_min, u_max),
        (0, 2, error,          "signed error",             "coolwarm",  full_ext,  *err_lim),
        (1, 0, abs_error,      "absolute error",           "magma",     full_ext,  0.0, abs_max),
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

            _save_panel(ui, up, err, err.abs(), res_pred, res_exact,
                        out / f"sample_{sid:03d}_panel.png",
                        _panel_title(row), full_ext, resid_ext)
        _write_csv(rows, out / "metrics.csv")
        print(f"saved panels + metrics: {out}")



# ===========================================================================
# 9. CLI & main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # modes
    p.add_argument("--demo", action="store_true",
                   help="Run a complete train→predict demo")

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
    p.add_argument("--save-path", default="")
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

    if args.demo:
        checkpoint = Path("checkpoints/demo.pt")
        best_checkpoint = Path("checkpoints/demo_best.pt")
        figure_dir = Path("results/demo_figures")

        # --- train (skip if checkpoint exists) ---
        if checkpoint.exists():
            print("=" * 60)
            print("  Checkpoint found — skipping training.")
            print(f"  Delete {checkpoint} to force retraining.")
            print("=" * 60)
        else:
            print("=" * 60)
            print("  Demo: train → predict pipeline")
            print("=" * 60)
            train_args = argparse.Namespace(**vars(args))
            train_args.N = 32
            train_args.train_samples = 2048
            train_args.val_samples = 128
            train_args.epochs = 40
            train_args.batch_size = 16
            train_args.eval_batch_size = 0
            train_args.eval_every = 1
            train_args.save_path = str(checkpoint)
            train(train_args)

        # --- predict ---
        pred_ckpt = best_checkpoint if best_checkpoint.exists() else checkpoint
        print(f"\nPredicting from: {pred_ckpt}")

        pred_args = argparse.Namespace(**vars(args))
        pred_args.load_path = str(pred_ckpt)
        pred_args.predict_samples = 4
        pred_args.predict_save = ""
        pred_args.plot = True
        pred_args.plot_dir = str(figure_dir)
        pred_args.predict_N = None
        pred_args.test_seed = 1234
        run_prediction(pred_args)

        print(f"\nDone. Checkpoint: {checkpoint}")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Figures: {figure_dir}")
        return

    # predict mode
    if args.predict or args.load_path:
        return run_prediction(args)

    # train mode
    train(args)


if __name__ == "__main__":
    main()
