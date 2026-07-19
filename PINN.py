"""
PINN solver for -div(a(x,y) grad u) = f on [-1,1]^2, Dirichlet BC u=g.
a(x,y) = exp(sigma*psi) is the same log-normal Fourier coefficient field as
Baseline_mms.py's random_positive_field; u is one of the u1-u4 analytic
targets (Baseline_mms.ANALYTIC_CASES, signed, no shift).

Uses the Raissi et al. (2017) PINN method with training hyperparameters
borrowed from their Burgers experiment. This is NOT a strict reproduction of
any Raissi benchmark: the problem here is a custom 2D variable-coefficient
Poisson equation, LINEAR in u (despite the repo name, it is not a nonlinear
Poisson). What is shared with the paper is the method and setup, not the PDE:
  - MLP with Tanh, Xavier init (Burgers architecture: 8 hidden layers x 20)
  - collocation points drawn ONCE via Latin Hypercube Sampling and held fixed
    (N_f interior + N_u boundary), as in the paper's fixed-point training
  - equal-weight composite loss MSE_f + MSE_u (bc_weight = 1)
  - Adam warm-up + a single L-BFGS run to convergence
  - PDE residual via autodiff (strong/flux form: -div(a grad u) computed as
    two nested autograd passes, so the product-rule term grad(a).grad(u)
    falls out automatically — no hand-derived formula needed)
  - Boundary conditions via penalty (soft constraint)

Mesh-free: training never touches a grid (collocation points live in the
continuous [-1,1]^2), so a single trained model can be re-evaluated at any
--N without retraining; the checkpoint key does not include N. The coefficient
field a(x,y) is fixed by --seed (and the other PDE parameters), all of which
enter the checkpoint key, and the actual Fourier coefficients are stored in the
checkpoint so evaluation always uses exactly the field the model trained on.

Recommended:
    python PINN.py --case u1 --sigma 0.5 --N 128 --retrain
    python PINN.py --case u1 --sigma 0.5 --N 32     # reuses the same checkpoint
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import qmc

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


def build_problem_def(M, seed):
    """Problem instance, independent of any evaluation grid: RNG-seeded
    Fourier coefficients for a(x,y) + their fixed-reference-grid
    normalization denominator. Only M and seed affect the field; case/sigma
    are applied later (case selects the analytic target, sigma scales psi)."""
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

# Every parameter that changes the PDE instance or the trained weights. --N is
# excluded (mesh-free) and so are monitoring-only knobs (--eval-every) and
# runtime flags (--device/--retrain/--no-plot), which do not affect the model.
RESULT_AFFECTING_KEYS = [
    "case", "sigma", "fourier_m", "seed", "epochs", "lr",
    "n_int", "n_bnd", "bc_weight", "hidden_dim", "n_layers", "lbfgs_iters",
]


def config_key(args) -> str:
    """Readable prefix + short hash of every result-affecting parameter, so
    changing any of them (crucially --seed, which picks the a(x,y) field) can
    never silently reuse a mismatched checkpoint."""
    prefix = (f"{args.case}_sig{args.sigma:g}_M{args.fourier_m}"
              f"_h{args.hidden_dim}_l{args.n_layers}_s{args.seed}")
    payload = {k: getattr(args, k) for k in RESULT_AFFECTING_KEYS}
    payload["denom_grid_n"] = DENOM_GRID_N   # coefficient-field normalization grid
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def resolve_paths(ckey: str, N: int, model_type: str = "pinn") -> dict:
    ckpt_dir = Path("checkpoints") / model_type / ckey
    results_dir = Path("results") / model_type / ckey / f"N{N}"
    return {
        "ckpt_dir": ckpt_dir,
        "adam_path": ckpt_dir / "adam.pt",     # post-Adam weights
        "last_path": ckpt_dir / "last.pt",     # final (post-L-BFGS) weights — the one reused
        "history_path": ckpt_dir / "history.csv",
        "figure_dir": results_dir / "figures",
        "results_dir": results_dir,
        "config_name": f"{model_type}/{ckey}",
    }


# ===========================================================================
# 1. MLP model (Raissi-style: Tanh, Xavier init)
# ===========================================================================

class MLP(nn.Module):
    """Fully-connected network: (x,y) -> u(x,y). Raissi et al. (2017) Burgers
    architecture by default: n_layers hidden Tanh layers of width hidden_dim
    (8 x 20)."""
    def __init__(self, hidden_dim: int = 20, n_layers: int = 8):
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

def sample_points(n_int: int, n_bnd: int, seed: int, device: torch.device):
    """Latin Hypercube collocation points, sampled ONCE and held fixed for the
    whole run (Raissi et al. 2017 use a fixed LHS point set, not per-epoch
    resampling).

    Interior: LHS in [-1,1]^2, exactly n_int points. Boundary: exactly n_bnd
    points spread over the 4 edges — the remainder n_bnd % 4 is handed to the
    first edges (so n_bnd is honoured exactly, unlike a bare n_bnd // 4), and
    each edge draws its free coordinate from an independent 1D LHS (no shared
    coordinates across edges).
    """
    eng = qmc.LatinHypercube(d=2, seed=seed)
    pts = eng.random(n_int) * 2.0 - 1.0
    x_int = torch.tensor(pts[:, 0:1], dtype=torch.float32, device=device)
    y_int = torch.tensor(pts[:, 1:2], dtype=torch.float32, device=device)

    base, rem = divmod(n_bnd, 4)
    counts = [base + (1 if e < rem else 0) for e in range(4)]
    eng1 = qmc.LatinHypercube(d=1, seed=seed + 1)
    xs, ys = [], []
    for edge, c in enumerate(counts):
        if c == 0:
            continue
        t = eng1.random(c)[:, 0] * 2.0 - 1.0          # free coordinate in [-1,1]
        fixed = np.full(c, -1.0 if edge in (0, 2) else 1.0)
        if edge in (0, 1):      # bottom (y=-1) / top (y=+1)
            xs.append(t);     ys.append(fixed)
        else:                   # left (x=-1) / right (x=+1)
            xs.append(fixed); ys.append(t)
    x_bnd = torch.tensor(np.concatenate(xs)[:, None], dtype=torch.float32, device=device)
    y_bnd = torch.tensor(np.concatenate(ys)[:, None], dtype=torch.float32, device=device)

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


def manufactured_forcing(x_int, y_int, coeffs, sigma, denom, u_target_fn):
    """Precompute the manufactured forcing f = -div(a grad u_target) at the
    (fixed) interior collocation points. f depends only on the analytic target
    and a(x,y), not on the network, so with a fixed point set it is computed
    ONCE and reused every optimisation step (the old code recomputed it — two
    nested autograd passes on the analytic solution — inside every loss call,
    which the L-BFGS closure hits many times per iteration)."""
    x = x_int.detach().clone().requires_grad_(True)
    y = y_int.detach().clone().requires_grad_(True)
    u_true = u_target_fn(x, y)
    a_true = a_field_torch(coeffs, sigma, denom, x, y)
    return _flux_div_continuous(u_true, a_true, x, y).detach()


def pinn_loss(model, x_int, y_int, f_int, x_bnd, y_bnd, g_bnd,
              coeffs, sigma, denom, bc_weight):
    """Equal-weight composite loss MSE_f + bc_weight * MSE_u (Raissi et al.
    2017 use bc_weight = 1). x_int/y_int are the fixed interior points (plain
    value tensors); f_int and g_bnd are precomputed constants. Returns
    (total, pde, bc)."""
    # Fresh grad leaves each call so input-derivative graphs don't accumulate
    # onto a persistent leaf's .grad across optimisation steps.
    x = x_int.detach().requires_grad_(True)
    y = y_int.detach().requires_grad_(True)

    # ---- PDE residual on interior (a_pred must keep its graph in x,y so the
    #      grad(a).grad(u) product-rule term survives the nested autodiff) ----
    u_pred = model(x, y)
    a_pred = a_field_torch(coeffs, sigma, denom, x, y)
    residual = _flux_div_continuous(u_pred, a_pred, x, y) - f_int
    loss_pde = residual.pow(2).mean()

    # ---- Boundary condition (soft penalty) ----
    u_bnd = model(x_bnd, y_bnd)
    loss_bc = (u_bnd - g_bnd).pow(2).mean()

    return loss_pde + bc_weight * loss_bc, loss_pde.detach(), loss_bc.detach()


# ===========================================================================
# 4. Grid evaluation (FD-based, matches Baseline_mms.py's analytic suite)
# ===========================================================================

@torch.no_grad()
def _eval_grid_light(model, grid):
    """Cheap training-monitor metrics (relL2, predResRMSE, bndErr) — no SciPy
    sparse solve and no constant/linear baselines, since those are independent
    of the network and only belong in the final evaluation (they were being
    recomputed on every periodic monitoring call)."""
    x, y = grid["x"], grid["y"]
    boundary, interior, h = grid["bnd"], grid["interior"], grid["h"]
    f, a, g, ue = grid["f_grid"], grid["a_grid"], grid["g_grid"], grid["u_target_grid"]

    u_pred = model(x.reshape(-1, 1), y.reshape(-1, 1)).reshape(
        1, 1, x.shape[-2], x.shape[-1])
    err_flat = ((u_pred - ue) * interior)[:, :, 1:-1, 1:-1]
    u_flat = ue[:, :, 1:-1, 1:-1]
    res_pred = pde_residual(u_pred, f, a, h)
    return {
        "relL2": float(err_flat.norm(p=2) / u_flat.norm(p=2).clamp_min(1e-12)),
        "predResRMSE": float(res_pred.pow(2).mean().sqrt()),
        "bndErr": float(((u_pred - g) * boundary).abs().max()),
    }, u_pred


@torch.no_grad()
def _eval_grid(model, grid):
    """Full final evaluation (all metrics + const/linear baselines).
    Returns (metrics_dict, u_pred_tensor)."""
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

def train(args, coeffs, denom, u_target_fn):
    set_seed(args.seed)
    device = select_device(args.device)
    t0 = time.time()

    model = MLP(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01,
    )

    # ---- evaluation grid (headline N, for monitoring only) ----
    eval_grid = build_eval_grid(args.N, args.case, args.sigma, coeffs, denom, device)

    # ---- fixed collocation set (drawn ONCE via LHS, held for the whole run).
    #      Manufactured forcing f and boundary data g are precomputed once —
    #      with fixed points they never change, so the two nested autodiff
    #      passes that build f run a single time instead of every loss call. ----
    x_int, y_int, x_bnd, y_bnd = sample_points(args.n_int, args.n_bnd, args.seed, device)
    f_int = manufactured_forcing(x_int, y_int, coeffs, args.sigma, denom, u_target_fn)
    g_bnd = u_target_fn(x_bnd, y_bnd).detach()
    colloc = (x_int, y_int, f_int, x_bnd, y_bnd, g_bnd)

    print(f"device={device}  case={args.case}  sigma={args.sigma}  N={args.N}  epochs={args.epochs}  "
          f"collocation (fixed LHS): interior={args.n_int}  boundary={x_bnd.shape[0]}  "
          f"bc_weight={args.bc_weight}")
    print(f"hidden_dim={args.hidden_dim}  n_layers={args.n_layers}  "
          f"lbfgs_iters={args.lbfgs_iters}")
    print(f"{'step':>8s}  {'loss_pde':>10s}  {'loss_bc':>10s}  "
          f"{'relL2':>10s}  {'resRMSE':>10s}  {'time':>8s}")

    history: list[dict] = []

    def _log(step):
        """Append a monitoring point and print it. Loss and grid metrics are
        both read from the CURRENT model state, so the loss on each training-
        curve row matches the relL2 on the same row (no off-by-one-update gap).
        Pure monitoring — it never selects or saves a model, so eval_every only
        sets the curve resolution, not the result."""
        model.eval()
        with torch.enable_grad():
            _, lp, lb = pinn_loss(model, *colloc, coeffs, args.sigma, denom, args.bc_weight)
        vm, _ = _eval_grid_light(model, eval_grid)
        history.append({
            "step": step, "loss_pde": float(lp), "loss_bc": float(lb),
            "relL2": vm["relL2"], "resRMSE": vm["predResRMSE"], "bmax": vm["bndErr"],
        })
        print(f"{step:8d}  {float(lp):10.3e}  {float(lb):10.3e}  "
              f"{vm['relL2']:10.3e}  {vm['predResRMSE']:10.3e}  {time.time() - t0:7.1f}s")

    # ---- Adam warm-up (full-batch on the fixed collocation set) ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss, _, _ = pinn_loss(model, *colloc, coeffs, args.sigma, denom, args.bc_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # periodic monitoring only; the final epoch is covered by the log below
        if epoch % args.eval_every == 0 and epoch != args.epochs:
            _log(epoch)

    if args.epochs > 0:
        _log(args.epochs)                 # one evaluation at the end of Adam
    adam_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- single L-BFGS run to convergence. The closure does ONLY zero_grad /
    #      loss / backward: strong-Wolfe calls it many times per accepted
    #      iteration with trial parameters, so monitoring or checkpointing
    #      inside it would record line-search probes, not real iterations. All
    #      evaluation happens once, after step() returns. ----
    if args.lbfgs_iters > 0:
        print(f"\nL-BFGS to convergence (max_iter={args.lbfgs_iters})...")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=args.lbfgs_iters,
            max_eval=int(args.lbfgs_iters * 1.25),
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def lbfgs_closure():
            optimizer_lbfgs.zero_grad()
            loss_total, _, _ = pinn_loss(model, *colloc, coeffs, args.sigma,
                                         denom, args.bc_weight)
            loss_total.backward()
            return loss_total

        optimizer_lbfgs.step(lbfgs_closure)
        _log(args.epochs + 1)             # one evaluation at the end of L-BFGS

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, history, adam_state, last_state, coeffs, denom


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
    """2-panel training curve: PDE/BC loss (log) | relL2 + resRMSE. Monitoring
    only — the final model is the converged L-BFGS model, not a selected best,
    so there is no best-checkpoint marker."""
    if len(history) < 2:
        return
    plt = _get_plt()

    steps = [h["step"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(title)

    ax1.plot(steps, [h["loss_pde"] for h in history], "b-", alpha=0.7, label="PDE loss")
    ax1.plot(steps, [h["loss_bc"] for h in history], "c-", alpha=0.7, label="BC loss")
    ax1.set_xlabel("step (Adam epoch → L-BFGS end)")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.set_title("Training loss")
    ax1.legend(fontsize="small")
    ax1.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    l1, = ax2.plot(steps, [h["relL2"] for h in history], "g-", label="relL2")
    l2, = ax2b.plot(steps, [h["resRMSE"] for h in history], "m--", label="resRMSE")
    ax2.set_xlabel("step (Adam epoch → L-BFGS end)")
    ax2.set_ylabel("relL2")
    ax2b.set_ylabel("resRMSE")
    ax2.legend(handles=[l1, l2], loc="upper right", fontsize="small")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Grid metrics (monitoring only)")

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
    p.add_argument("--epochs", type=int, default=5000,
                   help="Adam warm-up epochs (full-batch on the fixed points)")
    p.add_argument("--n-int", type=int, default=10000,
                   help="interior collocation points (fixed LHS; Raissi Burgers N_f)")
    p.add_argument("--n-bnd", type=int, default=100,
                   help="boundary collocation points (fixed LHS; Raissi Burgers N_u)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bc-weight", type=float, default=1.0,
                   help="boundary loss weight (Raissi: equal-weight MSE_f+MSE_u = 1)")
    p.add_argument("--eval-every", type=int, default=500,
                   help="monitoring cadence (Adam); does not affect the model")
    # model (Raissi Burgers architecture: 8 hidden layers x 20, Tanh)
    p.add_argument("--hidden-dim", type=int, default=20)
    p.add_argument("--n-layers", type=int, default=8)
    # L-BFGS (single run to convergence)
    p.add_argument("--lbfgs-iters", type=int, default=50000,
                   help="L-BFGS max iterations for the single to-convergence run "
                        "(tolerances stop it earlier; 0 to skip)")
    # system
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234,
                   help="picks the a(x,y) Fourier field AND the LHS points; "
                        "part of the checkpoint key")
    # output
    p.add_argument("--retrain", action="store_true",
                   help="force retraining even if a matching checkpoint exists")
    p.add_argument("--no-plot", action="store_true")

    args = p.parse_args()
    if args.epochs < 1 and args.lbfgs_iters < 1:
        p.error("nothing to train: need --epochs >= 1 or --lbfgs-iters >= 1")
    if args.epochs < 0:
        p.error("--epochs must be >= 0")
    if args.lbfgs_iters < 0:
        p.error("--lbfgs-iters must be >= 0")
    # strictly-positive structural / cadence params (0 would divide-by-zero in
    # the eval cadence or build a degenerate grid / network)
    for name, val in [("--n-int", args.n_int), ("--n-bnd", args.n_bnd),
                      ("--eval-every", args.eval_every), ("--fourier-m", args.fourier_m),
                      ("--N", args.N), ("--n-layers", args.n_layers),
                      ("--hidden-dim", args.hidden_dim)]:
        if val < 1:
            p.error(f"{name} must be >= 1")
    return args


def _verify_ckpt_args(ckpt, args, path):
    """Sanity-check that a reused checkpoint was trained with the same
    result-affecting parameters. The config-key hash already prevents cross-run
    collisions, but this catches a hand-edited/renamed checkpoint directory."""
    saved = ckpt.get("args")
    if not saved:
        return
    mismatched = {k: (saved.get(k), getattr(args, k))
                  for k in RESULT_AFFECTING_KEYS
                  if k in saved and saved[k] != getattr(args, k)}
    if mismatched:
        raise SystemExit(
            f"checkpoint {path} was trained with different parameters "
            f"{mismatched} — refusing to reuse it (use --retrain).")


def main():
    args = parse_args()

    ckey = config_key(args)
    paths = resolve_paths(ckey, args.N)

    # ---- problem instance: a(x,y) coefficients, independent of N ----
    coeffs, denom = build_problem_def(args.fourier_m, args.seed)
    u_target_fn = ANALYTIC_CASES_TORCH[args.case]

    existing = paths["last_path"] if paths["last_path"].exists() else None

    if existing and not args.retrain:
        print("=" * 60)
        print(f"  PINN Config:  {paths['config_name']}")
        print(f"  ✓ Found trained model: {existing}")
        print(f"  Skipping training. Use --retrain to force retraining.")
        print("=" * 60)
        ckpt = torch.load(existing, map_location="cpu", weights_only=False)
        _verify_ckpt_args(ckpt, args, existing)
        model = MLP(hidden_dim=args.hidden_dim, n_layers=args.n_layers)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(select_device(args.device))
        model.eval()
        # evaluate on exactly the field the model trained on (stored in ckpt),
        # not a freshly reseeded one — belt-and-suspenders against any key/seed
        # drift.
        if "coeffs" in ckpt and ckpt["coeffs"] is not None:
            coeffs, denom = ckpt["coeffs"], ckpt["denom"]
    else:
        if args.retrain and existing:
            print(f"⟳ Retraining requested for {paths['config_name']}")

        print("=" * 60)
        print(f"  PINN Config:  {paths['config_name']}")
        print(f"  Output:       {paths['ckpt_dir']}")
        print("=" * 60)

        model, history, adam_state, last_state, coeffs, denom = train(
            args, coeffs, denom, u_target_fn,
        )

        # Save both phase-end models directly (no best-by-validation selection):
        # adam.pt = post-Adam, last.pt = final post-L-BFGS (the one reused).
        paths["last_path"].parent.mkdir(parents=True, exist_ok=True)
        for path, state in [(paths["adam_path"], adam_state),
                            (paths["last_path"], last_state)]:
            torch.save({
                "model_state_dict": state, "args": vars(args),
                "coeffs": coeffs, "denom": denom,
            }, path)
        print(f"saved: {paths['adam_path'].name} (post-Adam), "
              f"{paths['last_path'].name} (final)")

        if history:
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
