"""
PINN solver for nonlinear Poisson equation.

Follows Raissi et al. (2017) design:
  - MLP with Tanh, Xavier init, input normalised to [-1,1]
  - Adam + L-BFGS fine-tuning
  - PDE residual via autodiff (strong form)
  - Boundary conditions via penalty (soft constraint)

Same PDE as Baseline.py:
    -div(u^k grad u) = f,  a(u) = u^k,  u = g on boundary
    Manufactured solution: u = sin(pi*x)*cos(pi*y),  k = 2

Recommended:
    python PINN.py --N 32 --epochs 5000
    python PINN.py --N 32 --epochs 5000 --bc-weight 100  # stronger BC
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

K_DEFAULT = 2.0


# -- inline utilities (no external dependencies) -----------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
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


def make_grid(N: int, device="cpu", dtype=torch.float32):
    H = W = N + 2
    h = 2.0 / (N + 1)
    xs = torch.linspace(-1., 1., W, device=device, dtype=dtype)
    ys = torch.linspace(-1., 1., H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = xx[None, None, :, :]
    y = yy[None, None, :, :]
    bnd = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    bnd[:, :,  0, :] = 1.
    bnd[:, :, -1, :] = 1.
    bnd[:, :,  :,  0] = 1.
    bnd[:, :,  :, -1] = 1.
    return x, y, bnd, 1. - bnd, h


def _flux_div(u, k, h, eps=1e-4):
    """5-point flux-form FD: -div(u^k grad u)."""
    if isinstance(k, (int, float)):
        k = torch.tensor([k], device=u.device)
    if k.ndim == 1:
        k = k.view(-1, 1, 1, 1)
    # raw integer-k coefficient: a = u^k exactly (k = K_DEFAULT = 2, integer),
    # signed u allowed; matches the signed analytic forcing / autodiff loss.
    # |u|^k * sign(u)^k form is backend-safe for negative bases.
    k_int = k.round()
    a = u.abs().pow(k_int)
    odd = (k_int.long() % 2 == 1)
    a = torch.where(odd & (u < 0), -a, a)
    uc = u[:, :, 1:-1, 1:-1]
    ac = a[:, :, 1:-1, 1:-1]
    ae = 0.5 * (ac + a[:, :, 1:-1, 2:])
    aw = 0.5 * (ac + a[:, :, 1:-1, :-2])
    an = 0.5 * (ac + a[:, :, 2:, 1:-1])
    as_ = 0.5 * (ac + a[:, :, :-2, 1:-1])
    fe = ae * (u[:, :, 1:-1, 2:] - uc) / h
    fw = aw * (uc - u[:, :, 1:-1, :-2]) / h
    fn = an * (u[:, :, 2:, 1:-1] - uc) / h
    fs = as_ * (uc - u[:, :, :-2, 1:-1]) / h
    return -(fe - fw + fn - fs) / h


def fd_residual(u, f, k, h, eps=1e-4):
    """R(u) = -div_h(u^k grad_h u) - f on interior nodes."""
    return _flux_div(u, k, h, eps) - f[:, :, 1:-1, 1:-1]


# ===========================================================================
# 0. Utilities
# ===========================================================================

def get_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_run_info(path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Run: {info.get('run_id', 'unknown')}",
             f"Model: {info.get('model', 'unknown')}",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    for k in ("N", "epochs", "hidden_dim", "n_layers", "lr", "bc_weight", "k"):
        if k in info:
            lines.append(f"{k} = {info[k]}")
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
    link_path = link_dir / "latest"
    link_dir.mkdir(parents=True, exist_ok=True)
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(run_id, target_is_directory=True)
    except OSError:
        link_path.with_suffix(".txt").write_text(run_id)


def _get_plt():
    cache = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ===========================================================================
# 0b. Config-key storage
# ===========================================================================

def config_key(N: int, epochs: int, hidden_dim: int, n_layers: int,
               bc_weight: float) -> str:
    bc_str = f"bc{bc_weight:.0f}" if bc_weight == int(bc_weight) else f"bc{bc_weight:.1f}"
    return f"N{N}_e{epochs}_h{hidden_dim}_l{n_layers}_{bc_str}"


def resolve_paths(N: int, epochs: int, hidden_dim: int, n_layers: int,
                  bc_weight: float, model_type: str = "pinn") -> dict:
    ckey = config_key(N, epochs, hidden_dim, n_layers, bc_weight)
    ckpt_dir = Path("checkpoints") / model_type / ckey
    results_dir = Path("results") / model_type / ckey
    return {
        "ckpt_dir": ckpt_dir,
        "best_path": ckpt_dir / "best.pt",
        "last_path": ckpt_dir / "last.pt",
        "history_path": ckpt_dir / "history.csv",
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
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **info,
    }
    reg_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


# ===========================================================================
# 1. Manufactured solution
# ===========================================================================

def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """u(x,y) = sin(pi*x) * cos(pi*y)."""
    return torch.sin(math.pi * x) * torch.cos(math.pi * y)


def forcing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Analytic f = -div(u^2 grad u) for u = sin(pi*x)*cos(pi*y)."""
    pi = math.pi
    u = u_exact(x, y)
    u_x = pi * torch.cos(pi * x) * torch.cos(pi * y)
    u_y = -pi * torch.sin(pi * x) * torch.sin(pi * y)
    laplacian = -2.0 * pi * pi * u
    grad_sq = u_x * u_x + u_y * u_y
    # a(u) = u^2,  a'(u) = 2u
    return -(2.0 * u * grad_sq + u * u * laplacian)


# ===========================================================================
# 2. MLP model  (Raissi-style: Tanh, Xavier init)
# ===========================================================================

class MLP(nn.Module):
    """Fully-connected network: (x,y) -> u(x,y)."""
    def __init__(self, hidden_dim: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


# ===========================================================================
# 3. Collocation point sampling
# ===========================================================================

def sample_points(n_int: int, n_bnd: int, device: torch.device):
    """Sample random collocation points each epoch (standard PINN).

    Interior: uniform in [-1,1]^2. Boundary: equal split among 4 edges.
    """
    x_int = torch.rand(n_int, 1, device=device) * 2.0 - 1.0
    y_int = torch.rand(n_int, 1, device=device) * 2.0 - 1.0

    n_edge = max(n_bnd // 4, 1)
    t = torch.rand(n_edge, 1, device=device) * 2.0 - 1.0
    one = torch.ones(n_edge, 1, device=device)

    x_bnd = torch.cat([t, t, -one, one], dim=0)
    y_bnd = torch.cat([-one, one, t, t], dim=0)

    return x_int, y_int, x_bnd, y_bnd


# ===========================================================================
# 4. PINN loss (autodiff PDE residual + boundary penalty)
# ===========================================================================

def pinn_loss(model: nn.Module,
              x_int: torch.Tensor, y_int: torch.Tensor,
              x_bnd: torch.Tensor, y_bnd: torch.Tensor,
              bc_weight: float = 1.0):
    """Compute physics-informed loss.

    Returns (total_loss, pde_loss, bc_loss) — all scalars.
    """
    # ---- PDE residual on interior ----
    x = x_int.detach().requires_grad_(True)
    y = y_int.detach().requires_grad_(True)

    u = model(x, y)
    ones = torch.ones_like(u)

    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True)[0]

    laplacian = u_xx + u_yy
    grad_sq = u_x * u_x + u_y * u_y
    residual = -(2.0 * u * grad_sq + u * u * laplacian) - forcing(x, y)

    loss_pde = residual.pow(2).mean()

    # ---- Boundary condition ----
    u_bnd = model(x_bnd, y_bnd)
    g_bnd = u_exact(x_bnd, y_bnd)
    loss_bc = (u_bnd - g_bnd).pow(2).mean()

    return loss_pde + bc_weight * loss_bc, loss_pde.detach(), loss_bc.detach()


# ===========================================================================
# 5. Grid evaluation  (shared by training and final eval)
# ===========================================================================

@torch.no_grad()
def _eval_grid(model, x, y, boundary, interior, h, f, k, g, need_gradient=False):
    xp = x.reshape(-1, 1)
    yp = y.reshape(-1, 1)
    u_pred_flat = model(xp, yp)
    u_pred = u_pred_flat.reshape(1, 1, x.shape[-2], x.shape[-1])

    ue = u_exact(x, y)
    err = (u_pred - ue) * interior

    m = {
        "relL2": float(err.norm(p=2) / ue.mul(interior).norm(p=2).clamp_min(1e-12)),
        "linf": float(err.abs().max()),
        "bmax": float(((u_pred - g) * boundary).abs().max()),
    }

    res_pred = fd_residual(u_pred, f, k, h)
    res_exact = fd_residual(ue, f, k, h)
    m["resRMSE"] = float(res_pred.pow(2).mean().sqrt())
    m["exact_resRMSE"] = float(res_exact.pow(2).mean().sqrt())

    if need_gradient:
        du = lambda v: (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2.0 * h)
        dv = lambda v: (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2.0 * h)
        p_grad = torch.cat([du(u_pred), dv(u_pred)], dim=1)
        t_grad = torch.cat([du(ue), dv(ue)], dim=1)
        m["gradRelL2"] = float(
            (p_grad - t_grad).norm(p=2) / t_grad.norm(p=2).clamp_min(1e-12)
        )
    else:
        m["gradRelL2"] = 0.0

    return m, u_pred


# ===========================================================================
# 6. Training  (Adam + L-BFGS, per Raissi et al. 2017)
# ===========================================================================

def train(args: argparse.Namespace, best_path: Path, last_path: Path,
          history_path: Path) -> tuple[nn.Module, list[dict], float]:
    set_seed(args.seed)
    device = select_device(args.device)
    t0 = time.time()

    model = MLP(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ---- evaluation grid (same as CNN for fair metric comparison) ----
    x_grid, y_grid, boundary, interior, h = make_grid(args.N, device)
    k_grid = torch.full((1, 1), K_DEFAULT, device=device)
    f_grid = forcing(x_grid, y_grid).detach()
    g_grid = boundary * u_exact(x_grid, y_grid)

    print(f"device={device}  N={args.N}  epochs={args.epochs}  "
          f"collocation: interior={args.n_int}  boundary={args.n_bnd}  "
          f"bc_weight={args.bc_weight}")
    print(f"hidden_dim={args.hidden_dim}  n_layers={args.n_layers}")
    print(f"{'epoch':>6s}  {'loss_pde':>10s}  {'loss_bc':>10s}  "
          f"{'relL2':>10s}  {'resRMSE':>10s}  {'time':>8s}")

    best_relL2 = math.inf
    best_state = None
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        x_int, y_int, x_bnd, y_bnd = sample_points(
            args.n_int, args.n_bnd, device,
        )

        loss, loss_pde, loss_bc = pinn_loss(
            model, x_int, y_int, x_bnd, y_bnd, args.bc_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ---- evaluation ----
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            vm, _ = _eval_grid(model, x_grid, y_grid, boundary, interior,
                               h, f_grid, k_grid, g_grid)
            relL2 = vm["relL2"]

            if relL2 < best_relL2:
                best_relL2 = relL2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": best_state,
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val_relL2": best_relL2,
                }, best_path)

            elapsed = time.time() - t0
            print(f"{epoch:6d}  {float(loss_pde):10.3e}  {float(loss_bc):10.3e}  "
                  f"{relL2:10.3e}  {vm['resRMSE']:10.3e}  {elapsed:7.1f}s")

            history.append({
                "epoch": epoch,
                "loss_pde": float(loss_pde),
                "loss_bc": float(loss_bc),
                "relL2": relL2,
                "resRMSE": vm["resRMSE"],
                "bmax": vm["bmax"],
            })

    # restore best Adam state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"best Adam model (val relL2={best_relL2:.4e})")

    # ---- L-BFGS fine-tuning (reuses the last Adam epoch's collocation pts) ----
    if args.lbfgs_epochs > 0 and args.epochs > 0:
        print(f"\nL-BFGS fine-tuning ({args.lbfgs_epochs} steps)...")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=20,
            max_eval=25,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def lbfgs_closure():
            optimizer_lbfgs.zero_grad()
            loss_total, _, _ = pinn_loss(
                model, x_int, y_int, x_bnd, y_bnd, args.bc_weight,
            )
            loss_total.backward()
            return loss_total

        for i in range(args.lbfgs_epochs):
            model.train()
            loss_lbfgs = optimizer_lbfgs.step(lbfgs_closure)
            model.eval()

            if (i + 1) % 20 == 0 or i == 0 or i == args.lbfgs_epochs - 1:
                vm, _ = _eval_grid(model, x_grid, y_grid, boundary, interior,
                                   h, f_grid, k_grid, g_grid)
                elapsed = time.time() - t0
                print(f"  lbfgs {i+1:4d}  loss={float(loss_lbfgs):.3e}  "
                      f"relL2={vm['relL2']:.3e}  resRMSE={vm['resRMSE']:.3e}  "
                      f"bmax={vm['bmax']:.3e}  {elapsed:.0f}s")

                if vm["relL2"] < best_relL2:
                    best_relL2 = vm["relL2"]
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": best_state,
                        "args": vars(args),
                        "best_val_relL2": best_relL2,
                    }, best_path)
                    history.append({
                        "epoch": f"lbfgs_{i+1}",
                        "loss_pde": 0, "loss_bc": 0,
                        "relL2": vm["relL2"],
                        "resRMSE": vm["resRMSE"],
                        "bmax": vm["bmax"],
                    })

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"best overall model (val relL2={best_relL2:.4e})")

    return model, history, best_relL2


# ===========================================================================
# 7. Final evaluation
# ===========================================================================

@torch.no_grad()
def evaluate_final(model: nn.Module, N: int, device: torch.device):
    x, y, boundary, interior, h = make_grid(N, device)
    k = torch.full((1, 1), K_DEFAULT, device=device)
    f = forcing(x, y)
    g = boundary * u_exact(x, y)

    vm, u_pred = _eval_grid(model, x, y, boundary, interior,
                            h, f, k, g, need_gradient=True)

    data = {"u_pred": u_pred, "u_exact": u_exact(x, y),
            "f": f, "h": h, "boundary": boundary, "interior": interior}
    return data, vm


# ===========================================================================
# 8. Visualization
# ===========================================================================

def _save_panel(u_true, u_pred, error, f_map, res_pred, res_exact,
                path, title, full_ext, resid_ext, dpi=200):
    """2×3 panel: true u | PINN pred | error || forcing | pred residual | exact residual."""
    plt = _get_plt()

    def _sym(v):
        mx = max(float(v.abs().max().cpu()), 1e-12)
        return -mx, mx

    u_min = min(float(u_true.min()), float(u_pred.min()))
    u_max = max(float(u_true.max()), float(u_pred.max()))
    err_lim = _sym(error)
    r_max = max(float(res_pred.abs().max().cpu()),
                float(res_exact.abs().max().cpu()), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (0, 0, u_true,     "true u",                  "viridis",  full_ext,  u_min, u_max),
        (0, 1, u_pred,     "PINN prediction",          "viridis",  full_ext,  u_min, u_max),
        (0, 2, error,      "signed error",             "coolwarm", full_ext,  *err_lim),
        (1, 0, f_map,      "forcing f",                "coolwarm", full_ext,  *_sym(f_map)),
        (1, 1, res_pred,   "pred PDE residual",        "coolwarm", resid_ext, -r_max, r_max),
        (1, 2, res_exact,  "exact PDE residual (floor)","coolwarm", resid_ext, -r_max, r_max),
    ]
    for r, c, vals, lbl, cmap, ext, v0, v1 in panels:
        ax = axes[r][c]
        im = ax.imshow(vals.detach().cpu().numpy(), origin="lower", extent=ext,
                       cmap=cmap, vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(lbl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 9. CLI & main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    # data
    p.add_argument("--N", type=int, default=32,
                   help="Evaluation grid size (same as CNN for metrics)")
    # training
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--n-int", type=int, default=1024,
                   help="Interior collocation points per epoch")
    p.add_argument("--n-bnd", type=int, default=256,
                   help="Boundary collocation points per epoch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bc-weight", type=float, default=50.0,
                   help="Boundary loss weight (higher = stronger BC enforcement)")
    p.add_argument("--eval-every", type=int, default=500)
    # model
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    # L-BFGS
    p.add_argument("--lbfgs-epochs", type=int, default=100,
                   help="L-BFGS fine-tuning steps (0 to skip)")
    # system
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    # output
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining even if a matching checkpoint exists")
    p.add_argument("--no-plot", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    model_type = "pinn"
    paths = resolve_paths(args.N, args.epochs, args.hidden_dim,
                          args.n_layers, args.bc_weight, model_type)

    ckey = config_key(args.N, args.epochs, args.hidden_dim,
                      args.n_layers, args.bc_weight)

    # ── Check for existing trained model ──
    existing = find_existing(paths["ckpt_dir"], args.N, args.epochs)

    if existing and not args.retrain:
        print("=" * 60)
        print(f"  PINN Config:  {paths['config_name']}")
        print(f"  ✓ Found trained model: {existing}")
        print(f"  Skipping training. Use --retrain to force retraining.")
        print("=" * 60)
        update_registry(model_type, ckey, {
            "N": args.N, "epochs": args.epochs,
            "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
            "bc_weight": args.bc_weight, "k": K_DEFAULT,
        })
        ckpt = torch.load(existing, map_location="cpu", weights_only=False)
        model = MLP(hidden_dim=args.hidden_dim, n_layers=args.n_layers)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(select_device(args.device))
        model.eval()
    else:
        if args.retrain and existing:
            print(f"⟳ Retraining requested for {paths['config_name']}")

        print("=" * 60)
        print(f"  PINN Config:  {paths['config_name']}")
        print(f"  Output:       {paths['ckpt_dir']}")
        print("=" * 60)

        model, history, best_relL2 = train(
            args, paths["best_path"], paths["last_path"], paths["history_path"],
        )

        paths["last_path"].parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        }, paths["last_path"])
        print(f"saved final: {paths['last_path']}")

        with open(str(paths["history_path"]), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        print(f"saved history: {paths['history_path']}")

        update_registry(model_type, ckey, {
            "N": args.N, "epochs": args.epochs,
            "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
            "bc_weight": args.bc_weight, "k": K_DEFAULT,
            "best_val_relL2": best_relL2,
        })

    # ── Evaluate & plot ──
    device = select_device(args.device)
    data, metrics = evaluate_final(model, args.N, device)

    print(f"\nFinal metrics (N={args.N} grid):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4e}")

    run_params = {
        "run_id": get_run_id(), "model": "PINN (MLP)",
        "N": args.N, "epochs": args.epochs,
        "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
        "lr": args.lr, "bc_weight": args.bc_weight, "k": K_DEFAULT,
        "metrics": {k: f"{v:.4e}" for k, v in metrics.items()},
    }
    _write_run_info(paths["run_info_path"], run_params)
    _update_latest_symlink(Path("checkpoints") / model_type, ckey)
    _update_latest_symlink(Path("results") / model_type, ckey)

    if not args.no_plot:
        out = paths["figure_dir"]
        out.mkdir(parents=True, exist_ok=True)
        h = float(data["h"])
        full_ext = [-1.0, 1.0, -1.0, 1.0]
        resid_ext = [-1.0 + h, 1.0 - h, -1.0 + h, 1.0 - h]

        ui = data["u_exact"][0, 0]
        up = data["u_pred"][0, 0]
        err = up - ui
        res_pred = fd_residual(
            data["u_pred"], data["f"],
            torch.full((1, 1), K_DEFAULT, device=device), h,
        )[0, 0]
        res_exact = fd_residual(
            data["u_exact"], data["f"],
            torch.full((1, 1), K_DEFAULT, device=device), h,
        )[0, 0]

        title = (f"PINN | relL2={metrics['relL2']:.3e}  "
                 f"Linf={metrics['linf']:.3e}  "
                 f"resRMSE={metrics['resRMSE']:.3e}  "
                 f"exact_resRMSE={metrics['exact_resRMSE']:.3e}")

        _save_panel(ui, up, err, data["f"][0, 0], res_pred, res_exact,
                    out / "pinn_panel.png", title, full_ext, resid_ext)

        csv_path = out / "metrics.csv"
        with open(str(csv_path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            w.writeheader()
            w.writerow(metrics)
        print(f"saved panel + metrics: {out}")

    print(f"\nDone. Config: {paths['config_name']}")
    print(f"Checkpoints: {paths['ckpt_dir']}")
    print(f"Figures:     {paths['figure_dir']}")
    print(f"Latest:      checkpoints/{model_type}/latest -> {ckey}")


if __name__ == "__main__":
    main()
