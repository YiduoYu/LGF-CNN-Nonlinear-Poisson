"""
TransUNet solver for a nonlinear Poisson equation in flux form.

Solves:  -div(a(u) grad u) = f,  a(u) = u^k,  u = g on boundary
on Omega = [-1, 1]^2 with Dirichlet boundary conditions.

Architecture:
  CNN encoder → Transformer bottleneck → CNN decoder
  (Baseline U-Net with DilatedBlock×2 replaced by Transformer×4 + DilatedBlock)

Loss = mean((FD_residual / RMS(f))^2)   — purely unsupervised.

Recommended:
    python TransUNet.py                      # N=32, 40 epoch (demo)
    python TransUNet.py --epochs 120          # longer training
    python TransUNet.py --N 64 --epochs 120   # finer grid
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse all PDE / grid / evaluation infrastructure from Baseline.py
from Baseline import (
    set_seed, select_device, get_run_id,
    make_grid, build_input, enforce_dirichlet,
    _safe_pow, power_coefficient, flux_divergence, pde_residual,
    K_DEFAULT,
    u_exact, forcing, generate_dataset, generate_random_dataset,
    compute_stats, InputStats,
    _take_batch, _forward, _error_tensors,
    evaluate, per_sample_metrics,
    _write_run_info, _update_latest_symlink, migrate_legacy_checkpoint,
    _write_csv, _save_convergence_plot, _save_panel, _get_plt,
    archive_existing_results,
)

# =========================================================================
# 1. Transformer components
# =========================================================================


class TransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer.

    LN → MultiheadAttention → + residual
    LN → MLP → + residual
    """

    def __init__(self, d_model: int, n_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBottleneck(nn.Module):
    """Flatten 2D feature map → Transformer encoder → reshape back.

    For N=32 the bottleneck spatial size is 5×5 = 25 tokens.
    For N=64 it's 9×9 = 81 tokens.
    """

    def __init__(self, ch: int = 256, n_layers: int = 4, n_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 max_tokens: int = 256):
        super().__init__()
        self.ch = ch
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, ch))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(ch, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(ch)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N_tokens = H * W

        # spatial → sequence
        tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]

        # position embedding
        if N_tokens <= self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed[:, :N_tokens, :]
        else:
            # interpolate if grid larger than max_tokens
            pe = F.interpolate(
                self.pos_embed[:, :N_tokens, :].transpose(1, 2),
                size=N_tokens, mode="linear", align_corners=False,
            ).transpose(1, 2)
            tokens = tokens + pe

        tokens = self.dropout(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)

        # sequence → spatial
        return tokens.transpose(1, 2).view(B, C, H, W)


# =========================================================================
# 2. TransUNet model
# =========================================================================


def _group_count(ch: int, max_groups: int = 8) -> int:
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return g


class ConvBlock(nn.Module):
    """Double 3×3 conv with GroupNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g = _group_count(out_ch, groups)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(g, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(g, out_ch), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedBlock(nn.Module):
    """Residual block with dilated first convolution."""

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


class TransUNet(nn.Module):
    """U-Net with Transformer bottleneck.

    Encoder (CNN) → Transformer bottleneck → Decoder (CNN).

    Baseline bottleneck:  ConvBlock + DilatedBlock(d=2) + DilatedBlock(d=4)
    TransUNet bottleneck: ConvBlock + Transformer×N + DilatedBlock(d=2)
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 32,
                 tf_layers: int = 4, tf_heads: int = 8,
                 tf_dropout: float = 0.1):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, 2 * base_ch)
        self.enc3 = ConvBlock(2 * base_ch, 4 * base_ch)
        # Transformer bottleneck
        self.bottleneck_conv = ConvBlock(4 * base_ch, 8 * base_ch)
        self.bottleneck_tf = TransformerBottleneck(
            ch=8 * base_ch, n_layers=tf_layers, n_heads=tf_heads,
            dropout=tf_dropout, max_tokens=256)
        self.bottleneck_dilated = DilatedBlock(8 * base_ch, dilation=2)
        # decoder
        self.up3 = nn.Conv2d(12 * base_ch, 4 * base_ch, 3,
                             padding=1, padding_mode="replicate")
        self.dec3 = ConvBlock(4 * base_ch, 4 * base_ch)
        self.up2 = nn.Conv2d(6 * base_ch, 2 * base_ch, 3,
                             padding=1, padding_mode="replicate")
        self.dec2 = ConvBlock(2 * base_ch, 2 * base_ch)
        self.up1 = nn.Conv2d(3 * base_ch, base_ch, 3,
                             padding=1, padding_mode="replicate")
        self.dec1 = ConvBlock(base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad = _pad_to_multiple(x, 8)

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))

        b = self.bottleneck_conv(F.avg_pool2d(e3, 2))
        b = self.bottleneck_tf(b)              # ← Transformer
        b = self.bottleneck_dilated(b)

        d3 = F.interpolate(b, size=e3.shape[-2:],
                           mode="bilinear", align_corners=False)
        d3 = self.dec3(self.up3(torch.cat([d3, e3], dim=1)))
        d2 = F.interpolate(d3, size=e2.shape[-2:],
                           mode="bilinear", align_corners=False)
        d2 = self.dec2(self.up2(torch.cat([d2, e2], dim=1)))
        d1 = F.interpolate(d2, size=e1.shape[-2:],
                           mode="bilinear", align_corners=False)
        d1 = self.dec1(self.up1(torch.cat([d1, e1], dim=1)))
        return _unpad(self.head(d1), pad)


# =========================================================================
# 3. Config-key storage (adds _trans suffix)
# =========================================================================


def config_key(N: int, epochs: int, k: float, base_ch: int = 32,
               tf_layers: int = 4, random_u: bool = False,
               train_mode: str = "physics") -> str:
    k_str = f"k{k:.1f}".rstrip("0").rstrip(".") if k == int(k) else f"k{k:.2f}"
    key = f"N{N}_e{epochs}_{k_str}_tf{tf_layers}"
    if base_ch != 32:
        key += f"_c{base_ch}"
    if random_u:
        key += "_rand"
    if train_mode != "physics":
        key += f"_{train_mode}"
    return key


def resolve_paths(N: int, epochs: int, k: float, base_ch: int = 32,
                  tf_layers: int = 4, random_u: bool = False,
                  train_mode: str = "physics",
                  model_type: str = "transunet") -> dict:
    ckey = config_key(N, epochs, k, base_ch, tf_layers, random_u, train_mode)
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
    best = ckpt_dir / "best.pt"
    if not best.exists():
        return None
    try:
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        if isinstance(ckpt_args, argparse.Namespace):
            ckpt_args = vars(ckpt_args)
        if ckpt_args.get("N") == N and ckpt_args.get("epochs") == epochs:
            return best
    except Exception:
        pass
    return None


def update_registry(model_type: str, ckey: str, info: dict) -> None:
    reg_path = Path("checkpoints") / "registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    registry: dict = {}
    if reg_path.exists():
        try:
            registry = json.loads(reg_path.read_text())
        except (json.JSONDecodeError, ValueError):
            registry = {}
    registry.setdefault(model_type, {})[ckey] = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **info}
    reg_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


# =========================================================================
# 4. Training
# =========================================================================


def train(args: argparse.Namespace, best_path: Path, last_path: Path,
          history_path: Path, convergence_path: Path
          ) -> tuple[nn.Module, float]:
    set_seed(args.seed)
    device = select_device(args.device)
    t0 = time.time()

    # ---- datasets ----
    random_u = getattr(args, "random_u", False)
    rhs_mode = getattr(args, "rhs_mode", "continuous")
    train_mode = getattr(args, "train_mode", "physics")
    pretrain_epochs = getattr(args, "pretrain_epochs", 0)
    k_range = getattr(args, "k_range", (0.5, 2.5))

    if random_u:
        print(f"generating {args.train_samples} RANDOM PDE train samples "
              f"(N={args.N}, k~U{k_range}, rhs={rhs_mode})", flush=True)
        train_data = generate_random_dataset(
            args.N, args.train_samples, device,
            k_range=k_range, rhs_mode=rhs_mode)
        print(f"generating {args.val_samples} RANDOM PDE val samples",
              flush=True)
        val_data = generate_random_dataset(
            args.N, args.val_samples, device,
            k_range=k_range, rhs_mode=rhs_mode)
    else:
        print(f"generating {args.train_samples} train samples (N={args.N})",
              flush=True)
        train_data = generate_dataset(args.N, args.train_samples, device)
        print(f"generating {args.val_samples} val samples", flush=True)
        val_data = generate_dataset(args.N, args.val_samples, device)

    stats = compute_stats(train_data)

    with torch.no_grad():
        res_exact = pde_residual(val_data["u_exact"], val_data["f"],
                                 val_data["k"], val_data["h"])
        oracle_floor = float(res_exact.pow(2).mean().sqrt())

    model = TransUNet(
        in_ch=6, base_ch=args.base_ch,
        tf_layers=args.tf_layers, tf_heads=args.tf_heads,
        tf_dropout=args.tf_dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransUNet params: {n_params:,}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith(".bias") or p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    steps_per_epoch = math.ceil(args.train_samples / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_val_relL2 = math.inf
    eval_bs = args.eval_batch_size or args.batch_size

    print(f"device={device}  N={args.N}  "
          f"train={args.train_samples}  val={args.val_samples}  "
          f"gen_time={time.time() - t0:.1f}s")
    print(f"oracle floor={oracle_floor:.4f}  "
          f"steps/epoch={steps_per_epoch}  total_steps={total_steps}")
    print(f"tf_layers={args.tf_layers}  tf_heads={args.tf_heads}  "
          f"tf_dropout={args.tf_dropout}")
    print(f"random_u={random_u}  rhs_mode={rhs_mode}  "
          f"train_mode={train_mode}  pretrain_epochs={pretrain_epochs}")

    def _supervised_loss(u_pred: torch.Tensor, batch: dict) -> torch.Tensor:
        interior = batch["interior_mask"].expand_as(u_pred)
        return ((u_pred - batch["u_exact"]) * interior).pow(2).mean()

    history: list[dict] = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.train_samples, device=device)

        for start in range(0, args.train_samples, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch = _take_batch(train_data, idx)
            u_pred = _forward(model, batch, stats, args.normalize_k)

            res = pde_residual(u_pred, batch["f"], batch["k"], batch["h"])
            f_int = batch["f"][:, :, 1:-1, 1:-1]
            denom = f_int.pow(2).mean(dim=(1, 2, 3), keepdim=True) \
                           .sqrt().clamp_min(1e-3)
            loss_pde = (res / denom).pow(2).mean()
            loss_sup = _supervised_loss(u_pred, batch)

            if train_mode == "physics":
                loss = loss_pde
            elif train_mode == "supervised":
                loss = loss_sup
            elif train_mode == "hybrid":
                if epoch < pretrain_epochs:
                    loss = loss_sup
                else:
                    loss = 0.1 * loss_sup + 1.0 * loss_pde
            else:
                raise ValueError(f"Unknown train_mode: {train_mode}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            train_res_rmse = float(res.pow(2).mean().sqrt())
            train_bmax = float(
                ((u_pred - batch["g"]).abs() * batch["mask"]).max())
            val_m = evaluate(model, val_data, stats, eval_bs,
                            args.normalize_k)

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

            print(f"epoch={epoch:03d}  loss={float(loss):.3e}  "
                  f"loss_pde={float(loss_pde):.3e}  "
                  f"loss_sup={float(loss_sup):.3e}  "
                  f"bmax={train_bmax:.2e} | "
                  f"val relL2={val_m['relL2']:.4e}  "
                  f"linf={val_m['linf']:.4e}  "
                  f"resRMSE={val_m['resRMSE']:.4f}  "
                  f"gradRelL2={val_m['gradRelL2']:.4e}")
            history.append({
                "epoch": epoch, "loss": float(loss),
                "train_resRMSE": train_res_rmse,
                "val_relL2": val_m["relL2"],
                "val_resRMSE": val_m["resRMSE"],
                "val_linf": val_m["linf"],
                "val_gradRelL2": val_m["gradRelL2"],
            })

    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "input_stats": vars(stats), "args": vars(args)}, last_path)
    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, oracle_floor)
    print(f"Saved: {last_path}, {history_path}, {convergence_path}")
    return model, oracle_floor


# =========================================================================
# 5. Prediction
# =========================================================================


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_args = ckpt.get("args", {})
    if isinstance(raw_args, argparse.Namespace):
        raw_args = vars(raw_args)
    base_ch = raw_args.get("base_ch", 32)
    tf_layers = raw_args.get("tf_layers", 4)
    tf_heads = raw_args.get("tf_heads", 8)
    tf_dropout = raw_args.get("tf_dropout", 0.1)
    model = TransUNet(in_ch=6, base_ch=base_ch,
                      tf_layers=tf_layers, tf_heads=tf_heads,
                      tf_dropout=tf_dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    s = ckpt["input_stats"]
    stats = InputStats(f_mean=s["f_mean"], f_std=s["f_std"],
                       k_mean=s["k_mean"], k_std=s["k_std"])
    return model, stats, argparse.Namespace(**raw_args)


def run_prediction(args: argparse.Namespace) -> None:
    if not args.load_path:
        raise ValueError("--load-path is required for prediction")
    set_seed(args.seed)
    device = select_device(args.device)
    model, stats, ckpt_args = load_checkpoint(args.load_path, device)

    N = args.predict_N or getattr(ckpt_args, "N", 32)
    k_val = getattr(ckpt_args, "k_min", 2.0)
    norm_k = getattr(ckpt_args, "normalize_k", True)

    print(f"prediction: N={N}  k={k_val}  samples={args.predict_samples}")
    set_seed(args.test_seed)
    batch = generate_dataset(N, args.predict_samples, device, k_val)

    model.eval()
    with torch.no_grad():
        u_pred = _forward(model, batch, stats, norm_k)

    vm = evaluate(model, batch, stats, args.predict_samples, norm_k)
    print(f"relL2={vm['relL2']:.4e}  linf={vm['linf']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  bmax={vm['bmax']:.2e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")

    rows = per_sample_metrics(u_pred, batch)

    if args.predict_save:
        path = Path(args.predict_save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "k": batch["k"].cpu(), "u_exact": batch["u_exact"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]),
            "metrics": vm, "config": {"N": N, "k_val": k_val}}, path)
        print(f"saved prediction: {path}")

    if args.plot:
        out = Path(args.plot_dir)
        archive_existing_results(out)
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
                        (f"sample {sid} | k={row['k']:.4g}  "
                         f"relL2={row['relL2']:.3e}  "
                         f"Linf={row['Linf']:.3e}  "
                         f"pred_resRMSE={row['pred_flux_resRMSE']:.3e}"),
                        full_ext, resid_ext)
        _write_csv(rows, out / "metrics.csv")
        print(f"saved panels + metrics: {out}")


# =========================================================================
# 6. CLI & main
# =========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # data
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--train-samples", type=int, default=2048)
    p.add_argument("--val-samples", type=int, default=128)
    # training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--normalize-k", action=argparse.BooleanOptionalAction,
                   default=True)
    # data generation mode
    p.add_argument("--random-u", action="store_true",
                   help="Use random manufactured-solution family")
    p.add_argument("--rhs-mode", default="continuous",
                   choices=["continuous", "discrete"])
    p.add_argument("--k-min", type=float, default=0.5)
    p.add_argument("--k-max", type=float, default=2.5)
    # training strategy
    p.add_argument("--train-mode", default="physics",
                   choices=["physics", "supervised", "hybrid"])
    p.add_argument("--pretrain-epochs", type=int, default=0)
    # transformer
    p.add_argument("--tf-layers", type=int, default=4)
    p.add_argument("--tf-heads", type=int, default=8)
    p.add_argument("--tf-dropout", type=float, default=0.1)
    # system
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    # I/O
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--load-path", default="")
    # prediction
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="")
    p.add_argument("--predict-N", type=int, default=None)
    p.add_argument("--test-seed", type=int, default=1234)
    # visualization
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-dir", default="results/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.predict or args.load_path:
        return run_prediction(args)

    args.demo = len(sys.argv) == 1
    model_type = "transunet"

    if args.demo:
        args.N = 32
        args.epochs = 40
        args.train_samples = 2048
        args.val_samples = 128
        args.batch_size = 16
        args.eval_batch_size = 0
        args.eval_every = 1

    paths = resolve_paths(args.N, args.epochs, K_DEFAULT,
                          args.base_ch, args.tf_layers,
                          getattr(args, "random_u", False),
                          getattr(args, "train_mode", "physics"),
                          model_type)
    migrate_legacy_checkpoint(paths["ckpt_dir"])
    existing = find_existing(paths["ckpt_dir"], args.N, args.epochs)
    ckey = config_key(args.N, args.epochs, K_DEFAULT,
                      args.base_ch, args.tf_layers,
                      getattr(args, "random_u", False),
                      getattr(args, "train_mode", "physics"))

    if existing and not args.retrain:
        print("=" * 60)
        print(f"  Config:  {paths['config_name']}")
        print(f"  ✓ Found trained model: {existing}")
        print(f"  Skipping training. Use --retrain to force retraining.")
        print("=" * 60)
        update_registry(model_type, ckey, {
            "N": args.N, "epochs": args.epochs, "k": K_DEFAULT,
            "tf_layers": args.tf_layers, "tf_heads": args.tf_heads})
    else:
        if args.retrain and existing:
            print(f"⟳ Retraining requested for {paths['config_name']}")
        print("=" * 60)
        print(f"  Config:  {paths['config_name']}")
        print("=" * 60)
        train_args = argparse.Namespace(**vars(args))
        if not hasattr(train_args, "random_u"):
            train_args.random_u = False
        if not hasattr(train_args, "rhs_mode"):
            train_args.rhs_mode = "continuous"
        if not hasattr(train_args, "train_mode"):
            train_args.train_mode = "physics"
        if not hasattr(train_args, "pretrain_epochs"):
            train_args.pretrain_epochs = 0
        if not hasattr(train_args, "k_range"):
            train_args.k_range = (getattr(args, "k_min", 0.5),
                                  getattr(args, "k_max", 2.5))
        model, oracle_floor = train(
            train_args, paths["best_path"], paths["last_path"],
            paths["history_path"], paths["convergence_path"])
        existing = paths["best_path"]
        update_registry(model_type, ckey, {
            "N": args.N, "epochs": args.epochs, "k": K_DEFAULT,
            "tf_layers": args.tf_layers, "tf_heads": args.tf_heads,
            "oracle_floor": oracle_floor})

    if args.demo or args.plot:
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

    run_params = {
        "run_id": get_run_id(), "model": "TransUNet",
        "N": args.N, "epochs": args.epochs,
        "train_samples": args.train_samples, "k": K_DEFAULT,
        "tf_layers": args.tf_layers, "tf_heads": args.tf_heads}
    _write_run_info(paths["run_info_path"], run_params)
    _update_latest_symlink(Path("checkpoints") / model_type, ckey)
    _update_latest_symlink(Path("results") / model_type, ckey)

    print(f"\nDone. Config: {paths['config_name']}")
    print(f"Checkpoints: {paths['ckpt_dir']}")
    print(f"Figures:     {paths['figure_dir']}")


if __name__ == "__main__":
    main()
