"""
PINN solver for nonlinear Poisson equation.

Same PDE as Baseline.py:
    -div(u^k grad u) = f,  a(u) = u^k,  u = g on boundary
    Manufactured solution: u = sin(pi*x)*cos(pi*y),  k = 2

Uses the SAME fixed grid as Baseline.py for training — the grid points
are extracted from make_grid(N) and used as collocation points every epoch.
No random sampling, so both methods see identical training data.

Difference from Baseline.py:
    PINN:  MLP + autodiff derivatives + penalty BC
    CNN:   U-Net + finite differences + hard BC enforcement

Recommended:
    python PINN.py --N 32 --epochs 5000
"""

import argparse
import csv
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

# Shared with Baseline.py for fair metric comparison
from Baseline import (
    set_seed, select_device, make_grid,
    pde_residual as fd_residual,
)

# ===========================================================================
# 1. Manufactured solution (scalar version for PINN collocation points)
# ===========================================================================

def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """u(x,y) = sin(pi*x) * cos(pi*y) — scalar (N,1) version."""
    return torch.sin(math.pi * x) * torch.cos(math.pi * y)


def forcing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Analytic f = -div(u^2 grad u) for u = sin(pi*x)*cos(pi*y).

    f = -(2*u*|grad u|^2 + u^2 * laplacian_u)
      = 2*pi^2 * sin(pi*x)*cos(pi*y) * (cos^2(pi*x)*cos^2(pi*y) + sin^2(pi*x)*sin^2(pi*y))
        + 2*pi^2 * sin^2(pi*x)*cos^2(pi*y) * sin(pi*x)*cos(pi*y)
    """
    pi = math.pi
    u = u_exact(x, y)
    u_x = pi * torch.cos(pi * x) * torch.cos(pi * y)
    u_y = -pi * torch.sin(pi * x) * torch.sin(pi * y)
    laplacian = -2.0 * pi * pi * u
    grad_sq = u_x * u_x + u_y * u_y

    # a(u) = u^2,  a'(u) = 2u
    # -div(u^2 grad u) = -(a'(u)*|grad u|^2 + a(u)*laplacian)
    return -(2.0 * u * grad_sq + u * u * laplacian)


# ===========================================================================
# 2. MLP model
# ===========================================================================

class MLP(nn.Module):
    """Fully-connected network for PINN: (x,y) -> u(x,y)."""
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
# 3. Grid-point extraction (same grid as CNN baseline)
# ===========================================================================

def get_grid_points(x: torch.Tensor, y: torch.Tensor,
                    interior: torch.Tensor, boundary: torch.Tensor):
    """Extract interior and boundary points from the same grid CNN uses.

    Returns (x_int, y_int, x_bnd, y_bnd) as [N,1] tensors.
    These are fixed — no random sampling, so both methods see identical data.
    """
    mask_int = interior.bool()
    mask_bnd = boundary.bool()

    x_int = x[mask_int].reshape(-1, 1)
    y_int = y[mask_int].reshape(-1, 1)
    x_bnd = x[mask_bnd].reshape(-1, 1)
    y_bnd = y[mask_bnd].reshape(-1, 1)

    return x_int, y_int, x_bnd, y_bnd


# ===========================================================================
# 4. PINN loss (autodiff PDE residual + boundary penalty)
# ===========================================================================

def pinn_loss(model: nn.Module,
              x_int: torch.Tensor, y_int: torch.Tensor,
              x_bnd: torch.Tensor, y_bnd: torch.Tensor,
              bc_weight: float = 1.0):
    """Compute physics-informed loss.

    PDE residual:  R = -div(u^2 grad u) - f   (autodiff, strong form)
    Boundary:      (u - g)^2                  (penalty)

    Returns (total_loss, pde_loss, bc_loss) — all scalars.
    """
    # ---- PDE residual on interior ----
    x = x_int.detach().requires_grad_(True)
    y = y_int.detach().requires_grad_(True)

    u = model(x, y)
    ones = torch.ones_like(u)

    # first derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]

    # second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True)[0]

    # strong form: -div(u^2 grad u) = -(2u*(u_x^2+u_y^2) + u^2*(u_xx+u_yy))
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
# 5. Training
# ===========================================================================

def train(args: argparse.Namespace) -> nn.Module:
    set_seed(args.seed)
    device = select_device(args.device)
    t0 = time.time()

    model = MLP(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ---- fixed grid: identical to CNN baseline's training grid ----
    x_grid, y_grid, boundary, interior, h = make_grid(args.N, device)
    ue_grid = u_exact(x_grid, y_grid)
    k_grid = torch.full((1, 1), 2.0, device=device)
    f_grid = forcing(x_grid, y_grid).detach()
    g_grid = boundary * ue_grid

    # extract collocation points from the grid (NOT random sampling)
    x_int, y_int, x_bnd, y_bnd = get_grid_points(x_grid, y_grid, interior, boundary)
    n_int = x_int.shape[0]
    n_bnd = x_bnd.shape[0]

    print(f"device={device}  N={args.N}  epochs={args.epochs}  "
          f"grid_points: interior={n_int}  boundary={n_bnd}  "
          f"(same grid as CNN baseline)")
    print(f"hidden_dim={args.hidden_dim}  n_layers={args.n_layers}")
    print(f"{'epoch':>6s}  {'loss_pde':>10s}  {'loss_bc':>10s}  "
          f"{'relL2':>10s}  {'resRMSE':>10s}  {'time':>8s}")

    best_relL2 = math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        loss, loss_pde, loss_bc = pinn_loss(
            model, x_int, y_int, x_bnd, y_bnd, args.bc_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ---- evaluation on fixed grid ----
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                # evaluate on grid: need [N,1] point format
                xp = x_grid.reshape(-1, 1)
                yp = y_grid.reshape(-1, 1)
                u_pred_flat = model(xp, yp)
                u_pred = u_pred_flat.reshape(1, 1, args.N + 2, args.N + 2)

                # relative L2 error (interior)
                err = (u_pred - ue_grid) * interior
                relL2 = float(
                    err.norm(p=2) / ue_grid.mul(interior).norm(p=2).clamp_min(1e-12)
                )

                # PDE residual on grid (FD, same as Baseline for comparison)
                res = fd_residual(u_pred, f_grid, k_grid, h)
                res_rmse = float(res.pow(2).mean().sqrt())

                # boundary max error
                bmax = float(((u_pred - g_grid) * boundary).abs().max())

            if relL2 < best_relL2:
                best_relL2 = relL2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            elapsed = time.time() - t0
            print(f"{epoch:6d}  {float(loss_pde):10.3e}  {float(loss_bc):10.3e}  "
                  f"{relL2:10.3e}  {res_rmse:10.3e}  {elapsed:7.1f}s")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best model (val relL2={best_relL2:.4e})")

    return model


# ===========================================================================
# 6. Evaluation & prediction on uniform grid
# ===========================================================================

@torch.no_grad()
def evaluate_on_grid(model: nn.Module, N: int, device: torch.device) -> dict:
    """Evaluate PINN on the same uniform grid as the CNN baseline.

    Returns dict of tensors suitable for visualization and metric dict.
    """
    x, y, boundary, interior, h = make_grid(N, device)
    ue = u_exact(x, y)
    g = boundary * ue
    k = torch.full((1, 1), 2.0, device=device)
    f = forcing(x, y)

    # PINN prediction on grid
    xp = x.reshape(-1, 1)
    yp = y.reshape(-1, 1)
    u_pred_flat = model(xp, yp)
    u_pred = u_pred_flat.reshape(1, 1, N + 2, N + 2)

    # metrics
    err = (u_pred - ue) * interior
    relL2 = float(err.norm(p=2) / ue.mul(interior).norm(p=2).clamp_min(1e-12))
    linf = float((err).abs().max())
    bmax = float(((u_pred - g) * boundary).abs().max())

    res_pred = fd_residual(u_pred, f, k, h)
    res_exact = fd_residual(ue, f, k, h)
    res_rmse = float(res_pred.pow(2).mean().sqrt())
    exact_res_rmse = float(res_exact.pow(2).mean().sqrt())

    # gradient error
    du = lambda v: (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2.0 * h)
    dv = lambda v: (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2.0 * h)
    p_grad = torch.cat([du(u_pred), dv(u_pred)], dim=1)
    t_grad = torch.cat([du(ue), dv(ue)], dim=1)
    grad_relL2 = float(
        ((p_grad - t_grad).norm(p=2) / t_grad.norm(p=2).clamp_min(1e-12))
    )

    return {
        "u_pred": u_pred,
        "u_exact": ue,
        "f": f,
        "h": h,
        "boundary": boundary,
        "interior": interior,
    }, {
        "relL2": relL2,
        "linf": linf,
        "resRMSE": res_rmse,
        "exact_resRMSE": exact_res_rmse,
        "bmax": bmax,
        "gradRelL2": grad_relL2,
    }


# ===========================================================================
# 7. Visualization
# ===========================================================================

def _get_plt():
    cache = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_panel(u_true, u_pred, error, abs_error, res_pred, res_exact,
                path, title, full_ext, resid_ext, dpi=200):
    """2×3 panel same layout as Baseline.py."""
    plt = _get_plt()

    def _sym(v):
        mx = max(float(v.abs().max().cpu()), 1e-12)
        return -mx, mx

    u_min = min(float(u_true.min()), float(u_pred.min()))
    u_max = max(float(u_true.max()), float(u_pred.max()))
    err_lim = _sym(error)
    abs_max = max(float(abs_error.max().cpu()), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (0, 0, u_true,     "true u",                  "viridis",  full_ext,  u_min, u_max),
        (0, 1, u_pred,     "PINN prediction",          "viridis",  full_ext,  u_min, u_max),
        (0, 2, error,      "signed error",             "coolwarm", full_ext,  *err_lim),
        (1, 0, abs_error,  "absolute error",           "magma",    full_ext,  0.0, abs_max),
        (1, 1, res_pred,   "pred PDE residual",        "coolwarm", resid_ext, *_sym(res_pred)),
        (1, 2, res_exact,  "exact PDE residual (floor)","coolwarm", resid_ext, *_sym(res_exact)),
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
# 8. CLI & main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    # data
    p.add_argument("--N", type=int, default=32,
                   help="Grid size (same as CNN baseline, N+2 points per side)")
    # training
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bc-weight", type=float, default=1.0,
                   help="Boundary loss weight")
    p.add_argument("--eval-every", type=int, default=500)
    # model
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    # system
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    # output
    p.add_argument("--save-path", default="checkpoints/pinn_demo.pt")
    p.add_argument("--plot-dir", default="results/pinn_demo")
    p.add_argument("--no-plot", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # train
    model = train(args)

    # save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, save_path)
    print(f"saved: {save_path}")

    # evaluate & plot
    device = select_device(args.device)
    data, metrics = evaluate_on_grid(model, args.N, device)

    print(f"\nFinal metrics (N={args.N} grid):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4e}")

    if not args.no_plot:
        out = Path(args.plot_dir)
        out.mkdir(parents=True, exist_ok=True)
        h = float(data["h"])
        full_ext = [-1.0, 1.0, -1.0, 1.0]
        resid_ext = [-1.0 + h, 1.0 - h, -1.0 + h, 1.0 - h]

        ui = data["u_exact"][0, 0]
        up = data["u_pred"][0, 0]
        err = up - ui
        res_pred = fd_residual(
            data["u_pred"], data["f"],
            torch.full((1, 1), 2.0, device=device), h,
        )[0, 0]
        res_exact = fd_residual(
            data["u_exact"], data["f"],
            torch.full((1, 1), 2.0, device=device), h,
        )[0, 0]

        title = (f"PINN | relL2={metrics['relL2']:.3e}  "
                 f"Linf={metrics['linf']:.3e}  "
                 f"resRMSE={metrics['resRMSE']:.3e}  "
                 f"exact_resRMSE={metrics['exact_resRMSE']:.3e}")

        _save_panel(ui, up, err, err.abs(), res_pred, res_exact,
                    out / "pinn_panel.png", title, full_ext, resid_ext)

        # metrics CSV
        csv_path = out / "metrics.csv"
        with open(str(csv_path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            w.writeheader()
            w.writerow(metrics)
        print(f"saved panel + metrics: {out}")


if __name__ == "__main__":
    main()
