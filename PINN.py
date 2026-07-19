"""
PINN solver for -div(a(x,y) grad u) = f on [-1,1]^2, Dirichlet BC u=g.
a(x,y) = exp(sigma*psi) is the same log-normal Fourier coefficient field as
Baseline_mms.py's random_positive_field; u is one of the u1-u4 analytic
targets (Baseline_mms.ANALYTIC_CASES, signed, no shift).

Follows Raissi et al. (2017) design:
  - MLP with Tanh, Xavier init
  - Adam + L-BFGS fine-tuning
  - PDE residual via autodiff (strong/flux form: -div(a grad u) computed as
    two nested autograd passes, so the product-rule term grad(a).grad(u)
    falls out automatically — no hand-derived formula needed)
  - Boundary conditions via penalty (soft constraint)

Mesh-free: training never touches a grid (collocation points are sampled
continuously in [-1,1]^2), so a single trained model can be re-evaluated at
any --N without retraining. The (case, sigma) pair defines the coefficient
field via a seeded RNG; the checkpoint key does not include N.

Recommended:
    python PINN.py --case u1 --sigma 0.5 --N 128 --epochs 5000 --retrain
    python PINN.py --case u1 --sigma 0.5 --N 32     # reuses the same checkpoint
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
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Reference grid resolution used only to compute the a(x,y) normalization
# denominator (see random_fourier_field), so training (mesh-free) and
# evaluation at any --N see numerically the same continuous a(x,y).
DENOM_GRID_N = 256


# -- inline utilities (self-contained, no cross-file imports; mirrors
#    Baseline_mms.py's low-level numerics by design) -------------------------

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
    bnd = torch.ones(1, 1, H, W, device=device, dtype=dtype)
    bnd[:, :, 1:-1, 1:-1] = 0.
    return x, y, bnd, 1. - bnd, h


# -- variable-coefficient flux-form operator (FD, grid-based; used only for
#    reporting metrics on the evaluation grid, matching Baseline_mms.py) ----

def flux_divergence(u, a, h):
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


def pde_residual(u, f, a, h):
    return flux_divergence(u, a, h) - f[:, :, 1:-1, 1:-1]


def _assemble_linear_system(a, g, f, h):
    """5-point FD system for -div(a grad u) = f with Dirichlet BC u=g."""
    N = a.shape[0] - 2
    n = N * N
    h2 = 1.0 / (h * h)

    ac = a[1:N+1, 1:N+1]
    ae = 0.5 * (ac + a[1:N+1, 2:N+2])
    aw = 0.5 * (ac + a[1:N+1, 0:N])
    a_n = 0.5 * (ac + a[2:N+2, 1:N+1])
    a_s = 0.5 * (ac + a[0:N, 1:N+1])

    b = f[1:N+1, 1:N+1].ravel().copy()
    b[N-1::N] += ae[:, N-1] * h2 * g[1:N+1, N+1]
    b[0::N]   += aw[:, 0]   * h2 * g[1:N+1, 0]
    b[(N-1)*N:] += a_n[N-1, :] * h2 * g[N+1, 1:N+1]
    b[:N]     += a_s[0, :]  * h2 * g[0, 1:N+1]

    diag = (ae + aw + a_n + a_s).ravel() * h2
    idx = np.arange(n)

    mask_e = (idx % N) != (N - 1)
    mask_w = (idx % N) != 0
    mask_n = idx < (n - N)
    mask_s = idx >= N

    rows = np.concatenate([idx,
        idx[mask_e], idx[mask_w], idx[mask_n], idx[mask_s]])
    cols = np.concatenate([idx,
        idx[mask_e] + 1, idx[mask_w] - 1, idx[mask_n] + N, idx[mask_s] - N])
    data = np.concatenate([diag,
        -ae.ravel()[mask_e] * h2, -aw.ravel()[mask_w] * h2,
        -a_n.ravel()[mask_n] * h2, -a_s.ravel()[mask_s] * h2])

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return A, b


def solve_linear_fd(f, g, a, h):
    A, b = _assemble_linear_system(a, g, f, h)
    u_int = spla.spsolve(A, b)
    u = g.copy()
    u[1:-1, 1:-1] = u_int.reshape(a.shape[0] - 2, a.shape[1] - 2)
    return u


# -- Fourier coefficient field: random draw + two evaluators (numpy grid,
#    torch continuous/autodiff) sharing the exact same coefficients --------

def _fourier_coeffs(M, rng):
    """Draw the (M,M,4) random coefficients random_fourier_field consumes,
    in the same per-(m,n) draw order, without evaluating on a grid."""
    return np.array([[[rng.uniform(-1, 1) for _ in range(4)]
                       for _ in range(M)] for _ in range(M)])


def _eval_fourier_np(coeffs, xx, yy):
    """Evaluate the Fourier series (numpy, any-shape grid)."""
    field = np.zeros_like(xx, dtype=np.float64)
    M = coeffs.shape[0]
    for m in range(1, M + 1):
        for n in range(1, M + 1):
            c_ss, c_sc, c_cs, c_cc = coeffs[m - 1, n - 1]
            sx = np.sin(m * np.pi * xx); cx = np.cos(m * np.pi * xx)
            sy = np.sin(n * np.pi * yy); cy = np.cos(n * np.pi * yy)
            field += (c_ss * sx * sy + c_sc * sx * cy
                      + c_cs * cx * sy + c_cc * cx * cy)
    return field


def _eval_fourier_torch(coeffs, x, y):
    """Same series, evaluated on torch tensors (arbitrary points, autodiff-
    friendly)."""
    field = torch.zeros_like(x)
    M = coeffs.shape[0]
    for m in range(1, M + 1):
        for n in range(1, M + 1):
            c_ss, c_sc, c_cs, c_cc = (float(v) for v in coeffs[m - 1, n - 1])
            sx = torch.sin(m * math.pi * x); cx = torch.cos(m * math.pi * x)
            sy = torch.sin(n * math.pi * y); cy = torch.cos(n * math.pi * y)
            field = field + (c_ss * sx * sy + c_sc * sx * cy
                             + c_cs * cx * sy + c_cc * cx * cy)
    return field


def _psi_denom(coeffs):
    """Normalization denominator max(|psi|) on a fixed fine reference grid
    (independent of any --N), so a(x,y) is numerically the same field
    whether it's evaluated during mesh-free training or on a coarse/fine
    evaluation grid."""
    xs = np.linspace(-1, 1, DENOM_GRID_N + 2)
    ys = np.linspace(-1, 1, DENOM_GRID_N + 2)
    xx, yy = np.meshgrid(xs, ys)
    psi = _eval_fourier_np(coeffs, xx, yy)
    return max(np.abs(psi).max(), 1e-12)


def a_field_torch(coeffs, sigma, denom, x, y):
    """a(x,y) = exp(sigma * psi(x,y)/denom), autodiff-friendly."""
    return torch.exp(sigma * _eval_fourier_torch(coeffs, x, y) / denom)


# -- analytic targets u1-u4 (mirrors Baseline_mms.ANALYTIC_CASES exactly;
#    numpy version for the FD evaluation grid, torch version for the
#    continuous/autodiff training loss) --------------------------------------

ANALYTIC_CASES_NP = {
    "u1": lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y),
    "u2": lambda x, y: (1. - x ** 2) * (1. - y ** 2),
    "u3": lambda x, y: np.cos(np.pi * x) * np.cos(2. * np.pi * y),
    "u4": lambda x, y: x ** 2 * (1. - y) + np.sin(np.pi * x) * y,
}

ANALYTIC_CASES_TORCH = {
    "u1": lambda x, y: torch.sin(math.pi * x) * torch.cos(math.pi * y),
    "u2": lambda x, y: (1. - x ** 2) * (1. - y ** 2),
    "u3": lambda x, y: torch.cos(math.pi * x) * torch.cos(2. * math.pi * y),
    "u4": lambda x, y: x ** 2 * (1. - y) + torch.sin(math.pi * x) * y,
}


def build_problem_def(case, sigma, M, seed):
    """Problem instance, independent of any evaluation grid: RNG-seeded
    Fourier coefficients for a(x,y) + their fixed-reference-grid
    normalization denominator."""
    rng = np.random.RandomState(seed)
    coeffs = _fourier_coeffs(M, rng)
    denom = _psi_denom(coeffs)
    return coeffs, denom


def build_eval_grid(N, case, sigma, coeffs, denom, device):
    """FD reference (a_grid, u_target_grid, f_grid, g_grid) at grid size N,
    for a fixed problem instance (coeffs, denom). f is the discrete
    flux-divergence of u_target under a_grid — same discretization
    Baseline_mms.py's analytic suite uses, so predResRMSE/refResRMSE are
    directly comparable to the CNN's numbers."""
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    xs = np.linspace(-1, 1, N + 2); ys = np.linspace(-1, 1, N + 2)
    xx, yy = np.meshgrid(xs, ys)
    psi_grid = _eval_fourier_np(coeffs, xx, yy)
    a_np = np.exp(sigma * psi_grid / denom)
    u_np = ANALYTIC_CASES_NP[case](xx, yy).astype(np.float64)
    g_np = u_np.copy()
    g_np[1:-1, 1:-1] = 0.

    a64 = torch.from_numpy(a_np)[None, None]
    u64 = torch.from_numpy(u_np)[None, None]
    f64 = torch.zeros_like(u64)
    f64[:, :, 1:-1, 1:-1] = flux_divergence(u64, a64, h)

    return {
        "x": x_t, "y": y_t, "bnd": bnd, "interior": interior, "h": h,
        "a_grid": a64.float().to(device),
        "u_target_grid": u64.float().to(device),
        "f_grid": f64.float().to(device),
        "g_grid": torch.from_numpy(g_np).float()[None, None].to(device),
    }


# ===========================================================================
# 0. Utilities
# ===========================================================================

def get_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_plt():
    cache = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ===========================================================================
# 0b. Config-key storage (excludes N: training is mesh-free, so one
#     checkpoint is re-evaluated at any --N without retraining)
# ===========================================================================

def config_key(case: str, sigma: float, M: int, epochs: int, hidden_dim: int,
               n_layers: int, bc_weight: float) -> str:
    bc_str = f"bc{bc_weight:g}"
    return f"{case}_sig{sigma:g}_M{M}_e{epochs}_h{hidden_dim}_l{n_layers}_{bc_str}"


def resolve_paths(ckey: str, N: int, model_type: str = "pinn") -> dict:
    ckpt_dir = Path("checkpoints") / model_type / ckey
    results_dir = Path("results") / model_type / ckey / f"N{N}"
    return {
        "ckpt_dir": ckpt_dir,
        "best_path": ckpt_dir / "best.pt",
        "last_path": ckpt_dir / "last.pt",
        "history_path": ckpt_dir / "history.csv",
        "figure_dir": results_dir / "figures",
        "results_dir": results_dir,
        "config_name": f"{model_type}/{ckey}",
    }


# ===========================================================================
# 1. MLP model (Raissi-style: Tanh, Xavier init)
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
# 2. Collocation point sampling
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
# 3. PINN loss (autodiff PDE residual + boundary penalty)
# ===========================================================================

def _flux_div_continuous(u, a, x, y):
    """-div(a grad u) via two nested autodiff passes; x, y must have
    requires_grad=True and u, a must be functions of x, y with the
    computation graph intact. The product-rule term grad(a).grad(u) falls
    out automatically through the second differentiation."""
    ones_u = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=ones_u, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones_u, create_graph=True)[0]
    qx = a * u_x
    qy = a * u_y
    qx_x = torch.autograd.grad(qx, x, grad_outputs=torch.ones_like(qx), create_graph=True)[0]
    qy_y = torch.autograd.grad(qy, y, grad_outputs=torch.ones_like(qy), create_graph=True)[0]
    return -(qx_x + qy_y)


def pinn_loss(model, x_int, y_int, x_bnd, y_bnd, coeffs, sigma, denom,
              u_target_fn, bc_weight):
    """Compute physics-informed loss. Returns (total_loss, pde_loss, bc_loss)."""
    x = x_int.detach().requires_grad_(True)
    y = y_int.detach().requires_grad_(True)

    # target forcing f = -div(a grad u_target), continuous (no grid) — does
    # not depend on the model, detach once computed.
    u_true = u_target_fn(x, y)
    a_true = a_field_torch(coeffs, sigma, denom, x, y)
    f = _flux_div_continuous(u_true, a_true, x, y).detach()

    # ---- PDE residual on interior ----
    u_pred = model(x, y)
    a_pred = a_field_torch(coeffs, sigma, denom, x, y)
    residual = _flux_div_continuous(u_pred, a_pred, x, y) - f
    loss_pde = residual.pow(2).mean()

    # ---- Boundary condition ----
    u_bnd = model(x_bnd, y_bnd)
    g_bnd = u_target_fn(x_bnd, y_bnd)
    loss_bc = (u_bnd - g_bnd).pow(2).mean()

    return loss_pde + bc_weight * loss_bc, loss_pde.detach(), loss_bc.detach()


def validation_loss(model, x_int, y_int, x_bnd, y_bnd, coeffs, sigma, denom,
                    u_target_fn, bc_weight) -> float:
    """PINN loss on a fixed validation collocation set, for model
    selection."""
    model.eval()
    loss, _, _ = pinn_loss(model, x_int, y_int, x_bnd, y_bnd, coeffs, sigma,
                           denom, u_target_fn, bc_weight)
    return float(loss)


# ===========================================================================
# 4. Grid evaluation (FD-based, matches Baseline_mms.py's analytic suite)
# ===========================================================================

@torch.no_grad()
def _eval_grid(model, grid):
    """Evaluate PINN on a grid. Returns (metrics_dict, u_pred_tensor)."""
    x, y = grid["x"], grid["y"]
    boundary, interior, h = grid["bnd"], grid["interior"], grid["h"]
    f, a, g, ue = grid["f_grid"], grid["a_grid"], grid["g_grid"], grid["u_target_grid"]

    xp = x.reshape(-1, 1)
    yp = y.reshape(-1, 1)
    u_pred_flat = model(xp, yp)
    u_pred = u_pred_flat.reshape(1, 1, x.shape[-2], x.shape[-1])
    err = (u_pred - ue) * interior
    err_flat = err[:, :, 1:-1, 1:-1]
    u_flat = ue[:, :, 1:-1, 1:-1]

    res_pred = pde_residual(u_pred, f, a, h)
    res_exact = pde_residual(ue, f, a, h)

    du = lambda v: (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2.0 * h)
    dv = lambda v: (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2.0 * h)
    p_grad = torch.cat([du(u_pred), dv(u_pred)], dim=1).flatten(2)
    t_grad = torch.cat([du(ue), dv(ue)], dim=1).flatten(2)

    const_val = u_flat.mean()
    cpred = boundary * g + (1. - boundary) * const_val
    const_err = ((cpred - ue) * interior)[:, :, 1:-1, 1:-1]

    u_lin = solve_linear_fd(
        f[0, 0].cpu().numpy(), g[0, 0].cpu().numpy(), np.ones((f.shape[-2], f.shape[-1])), float(h))
    lin_err = (torch.from_numpy(u_lin).to(u_pred) - ue[0, 0]) * interior[0, 0]
    lin_err_flat = lin_err[1:-1, 1:-1]

    m = {
        "relL2": float(err_flat.norm(p=2) / u_flat.norm(p=2).clamp_min(1e-12)),
        "RMSE": float(err_flat.pow(2).mean().sqrt()),
        "MAE": float(err_flat.abs().mean()),
        "maxError": float(err_flat.abs().max()),
        "predResRMSE": float(res_pred.pow(2).mean().sqrt()),
        "refResRMSE": float(res_exact.pow(2).mean().sqrt()),
        "bndErr": float(((u_pred - g) * boundary).abs().max()),
        "gradRelL2": float((p_grad - t_grad).norm(p=2)
                            / t_grad.norm(p=2).clamp_min(1e-12)),
        "gradLinf": float((p_grad - t_grad).abs().max()),
        "constBaselineRelL2": float(const_err.norm(p=2)
                                     / u_flat.norm(p=2).clamp_min(1e-12)),
        "linearBaselineRelL2": float(lin_err_flat.norm(p=2)
                                      / u_flat.norm(p=2).clamp_min(1e-12)),
    }
    return m, u_pred


def save_summary_json(summary, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ===========================================================================
# 5. Training (Adam + L-BFGS, per Raissi et al. 2017)
# ===========================================================================

def train(args, coeffs, denom, u_target_fn, best_path, last_path, history_path):
    set_seed(args.seed)
    device = select_device(args.device)
    t0 = time.time()

    model = MLP(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ---- evaluation grid (headline N, for periodic monitoring) ----
    eval_grid = build_eval_grid(args.N, args.case, args.sigma, coeffs, denom, device)

    # ---- fixed validation collocation set (model selection, no u_target leakage) ----
    xv_int, yv_int, xv_bnd, yv_bnd = sample_points(args.n_int, args.n_bnd, device)

    print(f"device={device}  case={args.case}  sigma={args.sigma}  N={args.N}  epochs={args.epochs}  "
          f"collocation: interior={args.n_int}  boundary={args.n_bnd}  "
          f"bc_weight={args.bc_weight}")
    print(f"hidden_dim={args.hidden_dim}  n_layers={args.n_layers}")
    print(f"{'epoch':>6s}  {'loss_pde':>10s}  {'loss_bc':>10s}  {'val_loss':>10s}  "
          f"{'relL2':>10s}  {'resRMSE':>10s}  {'time':>8s}")

    best_val_loss = math.inf
    best_state = None
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        x_int, y_int, x_bnd, y_bnd = sample_points(
            args.n_int, args.n_bnd, device,
        )

        loss, loss_pde, loss_bc = pinn_loss(
            model, x_int, y_int, x_bnd, y_bnd, coeffs, args.sigma, denom,
            u_target_fn, args.bc_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ---- evaluation ----
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            vm, _ = _eval_grid(model, eval_grid)
            relL2 = vm["relL2"]
            val_loss = validation_loss(model, xv_int, yv_int, xv_bnd, yv_bnd,
                                       coeffs, args.sigma, denom, u_target_fn,
                                       args.bc_weight)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": best_state,
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }, best_path)

            elapsed = time.time() - t0
            print(f"{epoch:6d}  {float(loss_pde):10.3e}  {float(loss_bc):10.3e}  "
                  f"{val_loss:10.3e}  "
                  f"{relL2:10.3e}  {vm['predResRMSE']:10.3e}  {elapsed:7.1f}s")

            history.append({
                "epoch": epoch,
                "loss_pde": float(loss_pde),
                "loss_bc": float(loss_bc),
                "val_loss": val_loss,
                "relL2": relL2,
                "resRMSE": vm["predResRMSE"],
                "bmax": vm["bndErr"],
            })

    # restore best Adam state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"best Adam model (val loss={best_val_loss:.4e})")

    # ---- L-BFGS fine-tuning (reuses the last Adam epoch's collocation pts:
    #      L-BFGS's line search needs a fixed objective, not a resampled one) ----
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
                model, x_int, y_int, x_bnd, y_bnd, coeffs, args.sigma, denom,
                u_target_fn, args.bc_weight,
            )
            loss_total.backward()
            return loss_total

        for i in range(args.lbfgs_epochs):
            model.train()
            loss_lbfgs = optimizer_lbfgs.step(lbfgs_closure)
            model.eval()

            if (i + 1) % 20 == 0 or i == 0 or i == args.lbfgs_epochs - 1:
                vm, _ = _eval_grid(model, eval_grid)
                val_loss = validation_loss(model, xv_int, yv_int, xv_bnd, yv_bnd,
                                           coeffs, args.sigma, denom, u_target_fn,
                                           args.bc_weight)
                _, cur_loss_pde, cur_loss_bc = pinn_loss(
                    model, x_int, y_int, x_bnd, y_bnd, coeffs, args.sigma, denom,
                    u_target_fn, args.bc_weight,
                )
                elapsed = time.time() - t0
                print(f"  lbfgs {i+1:4d}  loss={float(loss_lbfgs):.3e}  "
                      f"val_loss={val_loss:.3e}  "
                      f"relL2={vm['relL2']:.3e}  resRMSE={vm['predResRMSE']:.3e}  "
                      f"bndErr={vm['bndErr']:.3e}  {elapsed:.0f}s")

                # numeric epoch (continues the Adam-phase axis) so the training
                # curve stays on a single numeric x-axis instead of mixing in
                # string labels, which forces matplotlib to treat epoch as
                # categorical and garbles the tick labels.
                history.append({
                    "epoch": args.epochs + i + 1,
                    "loss_pde": float(cur_loss_pde), "loss_bc": float(cur_loss_bc),
                    "val_loss": val_loss,
                    "relL2": vm["relL2"],
                    "resRMSE": vm["predResRMSE"],
                    "bmax": vm["bndErr"],
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": best_state,
                        "args": vars(args),
                        "best_val_loss": best_val_loss,
                    }, best_path)

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"best overall model (val loss={best_val_loss:.4e})")

    return model, history, best_val_loss


# ===========================================================================
# 6. Visualization
# ===========================================================================

def _save_panel(u_true, u_pred, error, f_map, res_pred, res_exact,
                path, title, full_ext, resid_ext, dpi=200):
    """2x3 panel: true u | PINN pred | error || forcing | pred residual | exact residual."""
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
        (0, 0, u_true,     "true u",                    "viridis",  full_ext,  u_min, u_max),
        (0, 1, u_pred,     "PINN prediction",            "viridis",  full_ext,  u_min, u_max),
        (0, 2, error,      "signed error",               "coolwarm", full_ext,  *err_lim),
        (1, 0, f_map,      "forcing f",                  "coolwarm", full_ext,  *_sym(f_map)),
        (1, 1, res_pred,   "pred PDE residual",          "coolwarm", resid_ext, -r_max, r_max),
        (1, 2, res_exact,  "exact PDE residual (floor)", "coolwarm", resid_ext, -r_max, r_max),
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


def _save_centerline(ue, up, path, title, dpi=150):
    """1D cut along y≈0: u_ref vs u_pred."""
    plt = _get_plt()

    H, W = ue.shape[-2:]
    iy = H // 2                       # grid row closest to y = 0
    xs = np.linspace(-1., 1., W)
    y_val = np.linspace(-1., 1., H)[iy]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(xs, ue[iy].detach().cpu().numpy(), "b-", label="u_ref")
    ax.plot(xs, up[iy].detach().cpu().numpy(), "r--", label="u_pred")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.set_title(f"{title}  (centerline y={y_val:.3f})")
    ax.grid(True, alpha=.3); ax.legend()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_training_curve(history, path, title, dpi=150):
    """2-panel training curve: losses (log) | relL2 + resRMSE with best checkpoint."""
    if len(history) < 2:
        return
    plt = _get_plt()

    epochs = [h["epoch"] for h in history]
    val_losses = [h.get("val_loss", float("inf")) for h in history]
    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    best_epoch = epochs[best_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(title)

    ax1.plot(epochs, [h["loss_pde"] for h in history], "b-", alpha=0.7, label="PDE loss")
    ax1.plot(epochs, [h["loss_bc"] for h in history], "c-", alpha=0.7, label="BC loss")
    ax1.plot(epochs, val_losses, "r-", label="val loss")
    ax1.axvline(best_epoch, color="gray", ls=":", lw=1,
                label=f"best (epoch {best_epoch})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.set_title("Training & validation loss")
    ax1.legend(fontsize="small")
    ax1.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    l1, = ax2.plot(epochs, [h["relL2"] for h in history], "g-", label="relL2")
    l2, = ax2b.plot(epochs, [h["resRMSE"] for h in history], "m--", label="resRMSE")
    ax2.scatter(best_epoch, history[best_idx]["relL2"],
                color="gray", s=60, zorder=5, marker="o",
                label=f"best (epoch {best_epoch})")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("relL2")
    ax2b.set_ylabel("resRMSE")
    handles = [l1, l2, ax2.collections[0]] if ax2.collections else [l1, l2]
    ax2.legend(handles=handles, loc="upper right", fontsize="small")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Validation metrics (monitoring only)")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 7. CLI & main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    # problem
    p.add_argument("--case", default="u1", choices=list(ANALYTIC_CASES_NP),
                   help="analytic target u1-u4 (Baseline_mms.ANALYTIC_CASES)")
    p.add_argument("--sigma", type=float, default=0.5,
                   help="a(x,y)=exp(sigma*psi) contrast")
    p.add_argument("--fourier-m", type=int, default=6,
                   help="Fourier modes per dimension for a(x,y)")
    p.add_argument("--N", type=int, default=32,
                   help="evaluation grid size (training is mesh-free and "
                        "independent of N)")
    # training
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--n-int", type=int, default=1024,
                   help="interior collocation points per epoch")
    p.add_argument("--n-bnd", type=int, default=256,
                   help="boundary collocation points per epoch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bc-weight", type=float, default=50.0,
                   help="boundary loss weight (higher = stronger BC enforcement)")
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
                   help="force retraining even if a matching checkpoint exists")
    p.add_argument("--no-plot", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    ckey = config_key(args.case, args.sigma, args.fourier_m, args.epochs,
                      args.hidden_dim, args.n_layers, args.bc_weight)
    paths = resolve_paths(ckey, args.N)

    # ---- problem instance: a(x,y) coefficients, independent of N ----
    coeffs, denom = build_problem_def(args.case, args.sigma, args.fourier_m, args.seed)
    u_target_fn = ANALYTIC_CASES_TORCH[args.case]

    existing = paths["best_path"] if paths["best_path"].exists() else None

    if existing and not args.retrain:
        print("=" * 60)
        print(f"  PINN Config:  {paths['config_name']}")
        print(f"  ✓ Found trained model: {existing}")
        print(f"  Skipping training. Use --retrain to force retraining.")
        print("=" * 60)
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

        model, history, best_val_loss = train(
            args, coeffs, denom, u_target_fn,
            paths["best_path"], paths["last_path"], paths["history_path"],
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

        if not args.no_plot:
            out_dir = paths["figure_dir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            _save_training_curve(history, out_dir / "training_curve.png",
                                 f"PINN training — {ckey}")
            print(f"saved training curve: {out_dir / 'training_curve.png'}")

    # ---- Evaluate & plot (at --N, independent of training) ----
    device = select_device(args.device)
    eval_grid = build_eval_grid(args.N, args.case, args.sigma, coeffs, denom, device)
    t0 = time.time()
    with torch.no_grad():
        metrics, u_pred = _eval_grid(model, eval_grid)
    solve_time = time.time() - t0
    metrics["solve_time_sec"] = solve_time

    print(f"\nFinal metrics (N={args.N} grid, {solve_time:.1f}s):")
    for k in sorted(metrics.keys()):
        print(f"  {k}: {metrics[k]:.4e}")

    rows = [{"sample_id": 0, "case": args.case, "sigma": args.sigma, **metrics}]
    metric_keys = [k for k in sorted(metrics.keys()) if k != "solve_time_sec"]
    summary = {}
    for k in metric_keys:
        summary[k] = {"mean": metrics[k], "std": 0.0,
                       "min": metrics[k], "max": metrics[k]}

    out = paths["figure_dir"]
    out.mkdir(parents=True, exist_ok=True)
    with open(str(out / "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_summary_json(summary, out / "summary.json")
    print(f"saved metrics + summary: {out}")

    if not args.no_plot:
        h = float(eval_grid["h"])
        full_ext = [-1.0, 1.0, -1.0, 1.0]
        resid_ext = [-1.0 + h, 1.0 - h, -1.0 + h, 1.0 - h]

        ui = eval_grid["u_target_grid"][0, 0]
        up = u_pred[0, 0]
        err = up - ui
        res_pred = pde_residual(u_pred, eval_grid["f_grid"], eval_grid["a_grid"], h)[0, 0]
        res_exact = pde_residual(eval_grid["u_target_grid"], eval_grid["f_grid"],
                                 eval_grid["a_grid"], h)[0, 0]

        title = (f"PINN {args.case} sigma={args.sigma:g} N={args.N} | "
                 f"relL2={metrics['relL2']:.3e}  "
                 f"maxErr={metrics['maxError']:.3e}  "
                 f"resRMSE={metrics['predResRMSE']:.3e}  "
                 f"refResRMSE={metrics['refResRMSE']:.3e}")

        _save_panel(ui, up, err, eval_grid["f_grid"][0, 0], res_pred, res_exact,
                    out / "pinn_panel.png", title, full_ext, resid_ext)
        _save_centerline(ui, up, out / "pinn_centerline.png", title)
        print(f"saved panel + centerline: {out}")

    print(f"\nDone. Config: {paths['config_name']}")
    print(f"Checkpoints: {paths['ckpt_dir']}")
    print(f"Figures:     {paths['figure_dir']}")


if __name__ == "__main__":
    main()
