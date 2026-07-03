"""
CNN + Transformer solver for -div(u^k grad u) = f on [-1,1]^2.

Architecture: SimpleCNN body → Transformer bottleneck → SimpleCNN body → output.
All training, evaluation, and data infrastructure reused from Baseline.py.

Usage:
    python TransUNet.py                              # demo
    python TransUNet.py --N 32 --epochs 40           # standard
    python TransUNet.py --train-mode hybrid --pretrain-epochs 20
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F

from Baseline import (
    set_seed, select_device, run_id, get_run_id,
    build_input, enforce_dirichlet,
    pde_residual,
    K_DEFAULT, K_RANGE,
    generate_random_dataset, compute_stats, InputStats,
    train as baseline_train,
    evaluate, _forward, per_sample_metrics,
    config_key, resolve_paths, find_existing, update_registry,
    migrate_legacy_checkpoint,
    _write_run_info, _update_latest_symlink,
    _write_csv, _save_panel, _get_plt,
    archive_existing_results,
)

# ==============================================================================
# 1. Transformer bottleneck
# ==============================================================================

class TransformerEncoderLayer(nn.Module):
    """Pre-LN: LN → Attention → +residual → LN → MLP → +residual."""
    def __init__(self, d_model, n_heads=8, mlp_ratio=4., dropout=.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, d_model), nn.Dropout(dropout))

    def forward(self, x):
        xn = self.norm1(x); x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))

class TransformerBottleneck(nn.Module):
    """Flatten 2D → Transformer → reshape back to 2D."""
    def __init__(self, ch=256, n_layers=4, n_heads=8, dropout=.1, max_tokens=256):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, ch))
        self.layers = nn.ModuleList([TransformerEncoderLayer(ch, n_heads, 4., dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(ch); self.drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape; N = H * W
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + (self.pos[:, :N] if N <= self.pos.shape[1]
                           else F.interpolate(self.pos[:, :N].transpose(1, 2), size=N, mode="linear").transpose(1, 2))
        tokens = self.drop(tokens)
        for l in self.layers: tokens = l(tokens)
        return self.norm(tokens).transpose(1, 2).view(B, C, H, W)

# ==============================================================================
# 2. CNN + Transformer model
# ==============================================================================

def _make_conv_layers(n_layers, in_ch, hidden_ch):
    layers = []
    for i in range(n_layers):
        c_in = in_ch if i == 0 else hidden_ch
        g = min(8, hidden_ch)
        while hidden_ch % g: g -= 1
        layers += [nn.Conv2d(c_in, hidden_ch, 3, padding=1, padding_mode="replicate"),
                   nn.GroupNorm(g, hidden_ch), nn.GELU()]
    return nn.Sequential(*layers)

class CNNTransformer(nn.Module):
    """Conv stack → Transformer bottleneck → Conv stack → output head."""
    def __init__(self, in_ch=6, out_ch=1, hidden_ch=32, n_layers=6,
                 tf_layers=4, tf_heads=8, tf_dropout=.1):
        super().__init__()
        half = max(1, n_layers // 2)
        self.encoder = _make_conv_layers(half, in_ch, hidden_ch)
        self.tf = TransformerBottleneck(hidden_ch, tf_layers, tf_heads, tf_dropout)
        self.decoder = _make_conv_layers(n_layers - half, hidden_ch, hidden_ch)
        self.head = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x):
        f = self.encoder(x)
        f = self.tf(f)
        f = self.decoder(f)
        return self.head(f)

# ==============================================================================
# 3. Prediction (own load_checkpoint for CNNTransformer)
# ==============================================================================

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {}); a = vars(a) if isinstance(a, argparse.Namespace) else a
    model = CNNTransformer(hidden_ch=a.get("base_ch",32), n_layers=a.get("n_layers",6),
                           tf_layers=a.get("tf_layers",4), tf_heads=a.get("tf_heads",8),
                           tf_dropout=a.get("tf_dropout",.1)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    s = ckpt["input_stats"]
    return model, InputStats(**s), argparse.Namespace(**a)

def run_prediction(args):
    if not args.load_path: raise ValueError("--load-path required")
    set_seed(args.seed); device = select_device(args.device)
    model, stats, ckpt_args = load_checkpoint(args.load_path, device)
    N = args.predict_N or getattr(ckpt_args, "N", 32)
    norm_k = getattr(ckpt_args, "normalize_k", True)
    print(f"prediction: N={N}  samples={args.predict_samples}")
    set_seed(args.test_seed)
    batch = generate_random_dataset(N, args.predict_samples, device, rhs_mode="continuous")
    model.eval()
    with torch.no_grad(): u_pred = _forward(model, batch, stats, norm_k)
    vm = evaluate(model, batch, stats, args.predict_samples, norm_k)
    print(f"relL2={vm['relL2']:.4e}  linf={vm['linf']:.4e}  resRMSE={vm['resRMSE']:.4f}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")
    rows = per_sample_metrics(u_pred, batch)
    if args.predict_save:
        p = Path(args.predict_save); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(f=batch["f"].cpu(), g=batch["g"].cpu(), k=batch["k"].cpu(),
                        u_exact=batch["u_exact"].cpu(), u_pred=u_pred.cpu(),
                        h=float(batch["h"]), metrics=vm, config=dict(N=N)), p)
    if args.plot:
        out = Path(args.plot_dir); archive_existing_results(out)
        h = float(batch["h"]); fe = [-1.,1.,-1.,1.]; re = [-1.+h,1.-h,-1.+h,1.-h]
        for i, row in enumerate(rows):
            ui = batch["u_exact"][i,0]; up = u_pred[i,0]
            rp = pde_residual(u_pred[i:i+1], batch["f"][i:i+1], batch["k"][i:i+1], h)[0,0]
            re = pde_residual(batch["u_exact"][i:i+1], batch["f"][i:i+1], batch["k"][i:i+1], h)[0,0]
            _save_panel(ui, up, up-ui, batch["f"][i,0], rp, re,
                        out / f"sample_{row['sample_id']:03d}_panel.png",
                        f"sample {row['sample_id']} | k={row['k']:.4g}  "
                        f"relL2={row['relL2']:.3e}  pred_res={row['pred_flux_resRMSE']:.3e}", fe, re)
        _write_csv(rows, out/"metrics.csv"); print(f"saved: {out}")

# ==============================================================================
# 4. CLI & main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--train-samples", type=int, default=2048)
    p.add_argument("--val-samples", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--base-ch", type=int, default=32, help="hidden channels")
    p.add_argument("--n-layers", type=int, default=6, help="total conv layers")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--normalize-k", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--k-min", type=float, default=0.5)
    p.add_argument("--k-max", type=float, default=2.5)
    p.add_argument("--rhs-mode", default="continuous", choices=["continuous","discrete"])
    p.add_argument("--train-mode", default="physics", choices=["physics","supervised","hybrid"])
    p.add_argument("--pretrain-epochs", type=int, default=0)
    p.add_argument("--tf-layers", type=int, default=4, help="Transformer layers")
    p.add_argument("--tf-heads", type=int, default=8, help="Attention heads")
    p.add_argument("--tf-dropout", type=float, default=.1)
    p.add_argument("--device", default="auto"); p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--retrain", action="store_true"); p.add_argument("--load-path", default="")
    p.add_argument("--predict", action="store_true"); p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default=""); p.add_argument("--predict-N", type=int, default=None)
    p.add_argument("--test-seed", type=int, default=1234)
    p.add_argument("--plot", action="store_true"); p.add_argument("--plot-dir", default="results/figures")
    return p.parse_args()

def main():
    args = parse_args()
    if args.predict or args.load_path: return run_prediction(args)

    if len(sys.argv) == 1:
        args.N = 32; args.epochs = 40; args.train_samples = 2048; args.val_samples = 128
        args.batch_size = 16; args.eval_batch_size = 0; args.eval_every = 1

    mt = "transunet"
    paths = resolve_paths(args.N, args.epochs, K_DEFAULT, args.base_ch, 6,
                          getattr(args,"train_mode","physics"),
                          getattr(args,"rhs_mode","continuous"), mt)
    ckey = config_key(args.N, args.epochs, K_DEFAULT, args.base_ch, 6,
                      getattr(args,"train_mode","physics"),
                      getattr(args,"rhs_mode","continuous"))

    migrate_legacy_checkpoint(paths["ckpt_dir"])
    existing = find_existing(paths["ckpt_dir"], args)

    if existing and not args.retrain:
        print("=" * 60); print(f"  Config:  {paths['config_name']}")
        print(f"  Found trained model: {existing}"); print("=" * 60)
        update_registry(mt, ckey, {"N": args.N, "epochs": args.epochs, "k": K_DEFAULT})
    else:
        if args.retrain and existing: print(f"Retraining {paths['config_name']}")
        print("=" * 60); print(f"  Config:  {paths['config_name']}"); print("=" * 60)
        ta = argparse.Namespace(**vars(args))
        for attr in ["rhs_mode","train_mode","pretrain_epochs"]:
            if not hasattr(ta, attr): setattr(ta, attr, {"rhs_mode":"continuous","train_mode":"physics","pretrain_epochs":0}[attr])
        model = CNNTransformer(hidden_ch=args.base_ch, n_layers=args.n_layers,
                                tf_layers=args.tf_layers, tf_heads=args.tf_heads,
                                tf_dropout=args.tf_dropout)
        baseline_train(ta, paths["best_path"], paths["last_path"],
                        paths["history_path"], paths["convergence_path"], model=model)
        existing = paths["best_path"]
        update_registry(mt, ckey, {"N": args.N, "epochs": args.epochs, "k": K_DEFAULT})

    if getattr(args, "demo", False) or args.plot:
        pred_ckpt = existing or paths["best_path"]
        pa = argparse.Namespace(**vars(args)); pa.load_path = str(pred_ckpt)
        pa.predict_samples = 1 if getattr(args, "demo", False) else args.predict_samples
        pa.predict_save = ""; pa.plot = True; pa.plot_dir = str(paths["figure_dir"])
        pa.predict_N = None; pa.test_seed = args.seed; run_prediction(pa)

    _write_run_info(paths["run_info_path"], {"run_id": run_id(), "model": "CNN+Transformer",
                   "N": args.N, "epochs": args.epochs, "train_samples": args.train_samples, "k": K_DEFAULT,
                   "tf_layers": args.tf_layers, "tf_heads": args.tf_heads})
    _update_latest_symlink(Path("checkpoints")/mt, ckey); _update_latest_symlink(Path("results")/mt, ckey)
    print(f"\nDone. Config: {paths['config_name']}")

if __name__ == "__main__": main()
