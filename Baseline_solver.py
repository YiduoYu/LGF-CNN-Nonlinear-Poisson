"""
CNN solver for -div(u^k grad u) = f on [-1,1]^2, Dirichlet BC u=g.

Data generation: random f, g, k -> nonlinear finite-difference solver -> u_ref.
Supports --loss {supervised, physics}.

  f, g generated via full 2D Fourier series (--fourier-m).
  Nonlinear FD solver: red-black SOR relaxation directly on the 5-point
  flux-form difference equations (a = clamp(u,eps)^k re-evaluated from the
  current iterate every half-sweep).  Pure stencil arithmetic — no Picard
  linearization, no inner linear-system solves.
  Reject if solver fails (not converged, NaN, u_min <= eps, large residual).

  k ~ U(0, 5), u > 0.
  No FNO/PINN/FEM/spectral solver.
"""

import argparse, csv, json, math, os, random, tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -- grid -------------------------------------------------------------------

def make_grid(N, device="cpu", dtype=torch.float32):
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


def enforce_dirichlet(u_raw, g, bnd):
    """u = g on boundary, u_raw on interior."""
    return bnd * g + (1. - bnd) * u_raw


# -- nonlinear flux-form operator -------------------------------------------

def _coeff(u, k, eps, raw_power):
    """a(u) for the flux operator.

    Default (training): a = clamp(u,eps)^k — k ~ U(0,5) may be non-integer,
    so negative bases must be clamped to avoid NaN.
    raw_power (integer-k test mode): a = u^k exactly, signed u allowed.
    Implemented as |u|^k * sign(u)^k so it is backend-safe (MPS pow returns
    NaN for negative float bases)."""
    if raw_power:
        k_int = k.round()
        a = u.abs().pow(k_int)
        odd = (k_int.long() % 2 == 1)
        return torch.where(odd & (u < 0), -a, a)
    return torch.clamp(u, min=eps).pow(k)


def nonlinear_flux_divergence(u, k, h, eps=1e-4, raw_power=False):
    """5-point flux-form FD: -div(u^k grad u)."""
    if isinstance(k, (int, float)):
        k = torch.tensor([k], device=u.device)
    if k.ndim == 1:
        k = k.view(-1, 1, 1, 1)
    a = _coeff(u, k, eps, raw_power)

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


def nonlinear_pde_residual(u, f, k, h, eps=1e-4, raw_power=False):
    """R(u) = -div_h(u^k grad_h u) - f  on interior nodes.  [B,1,N,N]."""
    return (nonlinear_flux_divergence(u, k, h, eps, raw_power)
            - f[:, :, 1:-1, 1:-1])


# -- CNN --------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=6, out_ch=1, hidden_ch=32, n_layers=6):
        super().__init__()
        g = min(8, hidden_ch)
        while hidden_ch % g:
            g -= 1
        layers = []
        for i in range(n_layers):
            c_in = in_ch if i == 0 else hidden_ch
            layers += [
                nn.Conv2d(c_in, hidden_ch, 3, padding=1, padding_mode="replicate"),
                nn.GroupNorm(g, hidden_ch),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x):
        return self.head(self.body(x))


# -- general 2D Fourier series generator --------------------------------------

def random_fourier_field(N, M, rng=np.random, normalize=True):
    """2D Fourier series: sum_{m,n=1}^M of sin/cos products with random coefficients."""
    xs = np.linspace(-1, 1, N + 2)
    ys = np.linspace(-1, 1, N + 2)
    xx, yy = np.meshgrid(xs, ys)

    field = np.zeros((N + 2, N + 2), dtype=np.float64)
    for m in range(1, M + 1):
        for n in range(1, M + 1):
            sx = np.sin(m * np.pi * xx); cx = np.cos(m * np.pi * xx)
            sy = np.sin(n * np.pi * yy); cy = np.cos(n * np.pi * yy)
            field += (rng.uniform(-1, 1) * sx * sy
                      + rng.uniform(-1, 1) * sx * cy
                      + rng.uniform(-1, 1) * cx * sy
                      + rng.uniform(-1, 1) * cx * cy)

    if normalize:
        denom = max(np.abs(field).max(), 1e-12)
        field /= denom
    return field


# -- random f and boundary generators ----------------------------------------

def _random_f(N, f_scale, M, rng=np.random):
    """Random forcing f via full 2D Fourier series, boundary = 0."""
    f = random_fourier_field(N, M, rng) * f_scale
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.
    return f


def _random_positive_g(N, g_base=1.0, g_scale=0.4, M=6, rng=np.random):
    """Random positive Dirichlet boundary via Fourier series. Interior = 0."""
    raw = random_fourier_field(N, M, rng)
    g = np.zeros((N + 2, N + 2), dtype=np.float64)
    g[0, :] = g_base + g_scale * raw[0, :]
    g[-1, :] = g_base + g_scale * raw[-1, :]
    g[:, 0] = g_base + g_scale * raw[:, 0]
    g[:, -1] = g_base + g_scale * raw[:, -1]
    g = np.maximum(g, 0.5)
    g[1:-1, 1:-1] = 0.
    return g


# -- sparse linear FD system assembly ---------------------------------------

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
    """Direct solve: -div(a grad u) = f, Dirichlet BC u = g.
    Inputs/outputs are numpy arrays [H,W]."""
    A, b = _assemble_linear_system(a, g, f, h)
    u_int = spla.spsolve(A, b)
    u = g.copy()
    u[1:-1, 1:-1] = u_int.reshape(a.shape[0] - 2, a.shape[1] - 2)
    return u


def trivial_linear_baseline(data):
    """relL2 of the a≡1 linear Poisson solution vs the nonlinear u_ref.

    Solves -div(grad u) = f (ignoring the u^k diffusivity entirely) per sample.
    This is the must-report trivial baseline: if the CNN cannot beat it, then a
    good relL2 only reflects that the problem is nearly linear, not that the
    model learned the u^k structure."""
    h = float(data["h"])
    f = data["f"].detach().cpu().numpy()
    g = data["g"].detach().cpu().numpy()
    u_ref = data["u_ref"].detach().cpu().numpy()
    interior = data["interior_mask"].detach().cpu().numpy()[0, 0]
    rl2 = []
    for i in range(f.shape[0]):
        u_lin = solve_linear_fd(f[i, 0], g[i, 0], np.ones_like(g[i, 0]), h)
        err = (u_lin - u_ref[i, 0]) * interior
        ref = u_ref[i, 0] * interior
        rl2.append(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-12))
    return float(np.mean(rl2))


# -- nonlinear finite-difference solver ---------------------------------------

def _np_flux_residual(u, f, k, h, eps, raw_power=False):
    """R(u) = -div_h(a(u) grad_h u) - f on interior nodes (numpy).

    a = clamp(u,eps)^k by default; a = u^k in raw integer-k mode."""
    a = u ** k if raw_power else np.maximum(u, eps) ** k
    uc = u[1:-1, 1:-1]
    ac = a[1:-1, 1:-1]
    ae = 0.5 * (ac + a[1:-1, 2:])
    aw = 0.5 * (ac + a[1:-1, :-2])
    an = 0.5 * (ac + a[2:, 1:-1])
    as_ = 0.5 * (ac + a[:-2, 1:-1])
    fe = ae * (u[1:-1, 2:] - uc) / h
    fw = aw * (uc - u[1:-1, :-2]) / h
    fn = an * (u[2:, 1:-1] - uc) / h
    fs = as_ * (uc - u[:-2, 1:-1]) / h
    return -(fe - fw + fn - fs) / h - f[1:-1, 1:-1]


def solve_nonlinear_fd(f, g, k, h, eps, max_iter, tol, omega, raw_power=False):
    """Nonlinear red-black SOR directly on the 5-point flux-form FD equations.

    Each half-sweep re-evaluates a(u) from the current iterate and relaxes
    every grid point with its own difference equation:
      u_C <- (1-omega) u_C + omega (ae u_E + aw u_W + an u_N + as u_S + h^2 f) / (ae+aw+an+as)
    a = clamp(u,eps)^k by default; a = u^k in raw integer-k mode (signed u).
    Pure stencil arithmetic on the FD discretization — no Picard outer
    linearization; the a≡1 linear FD solve is used only as the initial guess.
    Converged when residual RMSE < tol * max(1, ||f||_rms)."""
    u = solve_linear_fd(f, g, np.ones_like(g), h)
    f_int = f[1:-1, 1:-1]
    f_scale = max(float(np.sqrt((f_int ** 2).mean())), 1.0)
    h2 = h * h

    ni, nj = u.shape[0] - 2, u.shape[1] - 2
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    colors = [(ii + jj) % 2 == 0, (ii + jj) % 2 == 1]

    converged = False
    n_iter = max_iter
    for it in range(max_iter):
        for mask in colors:
            a = u ** k if raw_power else np.maximum(u, eps) ** k
            ac = a[1:-1, 1:-1]
            ae = 0.5 * (ac + a[1:-1, 2:])
            aw = 0.5 * (ac + a[1:-1, :-2])
            an = 0.5 * (ac + a[2:, 1:-1])
            as_ = 0.5 * (ac + a[:-2, 1:-1])
            gs = ((ae * u[1:-1, 2:] + aw * u[1:-1, :-2]
                   + an * u[2:, 1:-1] + as_ * u[:-2, 1:-1] + h2 * f_int)
                  / (ae + aw + an + as_))
            ui = u[1:-1, 1:-1]
            ui[mask] = (1. - omega) * ui[mask] + omega * gs[mask]

        res_rmse = float(np.sqrt(
            (_np_flux_residual(u, f, k, h, eps, raw_power) ** 2).mean()))
        if not np.isfinite(res_rmse):
            break                       # diverged (e.g. infeasible sample)
        if res_rmse < tol * f_scale:
            converged = True
            n_iter = it + 1
            break

    # Final residual in float64 via the shared discrete flux-form operator
    u_t = torch.tensor(u, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    f_t = torch.tensor(f, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    k_t = torch.tensor([k], dtype=torch.float64)
    with torch.no_grad():
        final_res = float(nonlinear_pde_residual(
            u_t, f_t, k_t, h, eps=eps,
            raw_power=raw_power).pow(2).mean().sqrt())

    return u, {"num_iter": n_iter, "converged": converged,
               "u_min": float(u.min()), "u_max": float(u.max()),
               "residual_rmse": final_res}


# -- forward solver dataset --------------------------------------------------

def make_solver_dataset(N, n_samples, device, args, seed):
    """Random f,g,k -> nonlinear FD solve -> u_ref.  Rejects if no positive solution."""
    rng = np.random.RandomState(seed)
    x_t, y_t, bnd, interior, h_grid = make_grid(N, device)
    M = getattr(args, "fourier_m", 6)

    f_list, g_list, k_list, u_list, info_list = [], [], [], [], []
    res64_sq = 0.
    max_attempts = args.max_attempt_factor * n_samples
    attempts = 0

    print(f"generating {n_samples} solver samples (N={N})  "
          f"f_scale={args.f_scale}  g_base={args.g_base}  g_scale={args.g_scale}  "
          f"FD (red-black SOR): max_iter={args.max_fd_iter}  tol={args.fd_tol:.1e}  "
          f"omega={args.fd_omega}")

    while len(u_list) < n_samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"too many failed solver attempts ({attempts}) — "
                "try smaller --f-scale, larger --g-base, or looser --fd-tol.")
        attempts += 1

        if getattr(args, "random_k", False):
            k_i = float(rng.uniform(args.k_min, args.k_max))
        else:
            k_i = float(args.k)

        f_np = _random_f(N, args.f_scale, M, rng)
        g_np = _random_positive_g(N, args.g_base, args.g_scale, M, rng)

        u_np, info = solve_nonlinear_fd(
            f_np, g_np, k_i, h_grid, args.coeff_eps,
            args.max_fd_iter, args.fd_tol, args.fd_omega)

        # Acceptance: converged, positive, reasonable residual
        f_rms = float(np.sqrt(np.mean(f_np[1:-1, 1:-1] ** 2)))
        if (info["converged"]
                and info["u_min"] > args.coeff_eps
                and np.isfinite(info["residual_rmse"])
                and info["residual_rmse"] < 1e-3 * max(1.0, f_rms)):

            # Reference residual in float64
            u64 = torch.from_numpy(u_np).double().unsqueeze(0).unsqueeze(0)
            f64 = torch.from_numpy(f_np).double().unsqueeze(0).unsqueeze(0)
            k64 = torch.tensor([k_i], dtype=torch.float64)
            res64_sq += float(nonlinear_pde_residual(
                u64, f64, k64, h_grid, eps=args.coeff_eps).pow(2).sum())

            f_list.append(torch.from_numpy(f_np).float().unsqueeze(0).unsqueeze(0).to(device))
            g_list.append(torch.from_numpy(g_np).float().unsqueeze(0).unsqueeze(0).to(device))
            k_list.append(torch.tensor([k_i], dtype=torch.float32, device=device))
            u_list.append(torch.from_numpy(u_np).float().unsqueeze(0).unsqueeze(0).to(device))
            info_list.append(info)

    iters = [x["num_iter"] for x in info_list]
    res_list = [x["residual_rmse"] for x in info_list]
    ref_res64 = math.sqrt(res64_sq / (n_samples * N * N))
    print(f"  acceptance: {len(u_list)}/{attempts} "
          f"({100*len(u_list)/max(1,attempts):.1f}%)")
    print(f"  FD iterations: mean={np.mean(iters):.1f}  max={np.max(iters)}")
    print(f"  solver residual: mean={np.mean(res_list):.4e}  max={np.max(res_list):.4e}")
    print(f"  ref residual (float64) = {ref_res64:.2e}")
    print(f"  u range: [{min(x['u_min'] for x in info_list):.4f}, "
          f"{max(x['u_max'] for x in info_list):.4f}]")

    return {
        "f": torch.cat(f_list), "g": torch.cat(g_list),
        "k": torch.cat(k_list), "u_ref": torch.cat(u_list),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h_grid,
        "ref_res64": ref_res64, "n_rejected": attempts - len(u_list),
        "solver_info": info_list,
    }


def ref_residual_rmse64(data, eps):
    """Residual RMSE of the stored (float32) reference data, computed in
    float64, so the ~h^-2*eps32 rounding floor is measured, not hidden.
    Training itself stays float32."""
    u = data["u_ref"].detach().cpu().double()
    f = data["f"].detach().cpu().double()
    k = data["k"].detach().cpu().double()
    return float(nonlinear_pde_residual(u, f, k, data["h"], eps=eps)
                 .pow(2).mean().sqrt())


# -- input / stats / batching ------------------------------------------------

def build_input(f_norm, g, bnd, k_norm, x, y):
    """Assemble 6-channel CNN input: [f_norm, g, bnd, k_ch, x, y]."""
    B = f_norm.shape[0]
    def _expand(t):
        return t.expand(B, -1, -1, -1) if t.shape[0] == 1 else t
    k_ch = k_norm.view(B, 1, 1, 1).expand(-1, 1, *bnd.shape[-2:])
    return torch.cat([f_norm, g, _expand(bnd), k_ch, _expand(x), _expand(y)], dim=1)


def compute_stats(data):
    fi = data["f"][:, :, 1:-1, 1:-1]
    k = data["k"]
    return (fi.mean().item(), fi.std(unbiased=False).clamp_min(1e-8).item(),
            k.mean().item(), k.std(unbiased=False).clamp_min(1e-8).item())


def _take_batch(data, idx):
    return {"h": data["h"],
            **{k: data[k][idx] for k in ("f", "g", "k", "u_ref")},
            **{k: data[k][:1] for k in ("mask", "interior_mask", "x", "y")}}


def _forward(model, batch, stats):
    f_mean, f_std, k_mean, k_std = stats
    fn = (batch["f"] - f_mean) / f_std
    kn = (batch["k"] - k_mean) / k_std
    inp = build_input(fn, batch["g"], batch["mask"], kn, batch["x"], batch["y"])
    return enforce_dirichlet(model(inp), batch["g"], batch["mask"])


def _select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -- training ---------------------------------------------------------------

def train(args, best_path, last_path, history_path, convergence_path, model=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _select_device(args.device)

    print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}  |  data=solver")
    if getattr(args, "random_k", False):
        print(f"  random k in [{args.k_min}, {args.k_max}]")

    train_data = make_solver_dataset(args.N, args.train_samples, device,
                                      args, seed=args.seed)
    val_data = make_solver_dataset(args.N, args.val_samples, device,
                                    args, seed=args.seed + 1)
    stats = compute_stats(train_data)
    print(f"f stats: mean={stats[0]:.4e}  std={stats[1]:.4e}  "
          f"k stats: mean={stats[2]:.4e}  std={stats[3]:.4e}")

    # Reference residual, reported in float64 (float32 rounding floor
    # ~h^-2*eps32 must not be mistaken for solver accuracy)
    ref_res = val_data["ref_res64"]
    with torch.no_grad():
        u_min = float(val_data["u_ref"].min())
        u_max = float(val_data["u_ref"].max())
        bnd_err = float(
            ((val_data["u_ref"] - val_data["g"]).abs()
             * val_data["mask"].expand_as(val_data["u_ref"])).max())
    print(f"ref residual RMSE (float64) = {ref_res:.4e}  (nonlinear FD solver residual)")
    print(f"u_ref range: [{u_min:.4f}, {u_max:.4f}]  "
          f"boundary error = {bnd_err:.2e}")
    oracle = ref_res

    triv_rl2 = trivial_linear_baseline(val_data)
    print(f"trivial baseline (a≡1 linear Poisson) relL2 = {triv_rl2:.4e}  "
          f"(CNN must beat this to have learned u^k)")

    n_layers = getattr(args, "n_layers", 6)
    if model is None:
        model = SimpleCNN(hidden_ch=args.base_ch, n_layers=n_layers).to(device)
    else:
        model = model.to(device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith(".bias") or p.ndim <= 1 else decay).append(p)

    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.},
    ], lr=args.lr)

    steps_per_ep = math.ceil(args.train_samples / args.batch_size)
    total_steps = steps_per_ep * args.epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_fn(s):
        if s < warmup:
            return s / warmup
        return 0.5 * (1. + math.cos(
            math.pi * (s - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    n_train = train_data["f"].shape[0]
    n_val = val_data["f"].shape[0]
    print(f"device={device}  oracle={oracle:.4e}  "
          f"steps/epoch={steps_per_ep}  total={total_steps}  "
          f"n_train={n_train}  n_val={n_val}")

    best_metric = math.inf
    eval_bs = args.eval_batch_size or args.batch_size
    history = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)

        for start in range(0, n_train, args.batch_size):
            batch = _take_batch(train_data, perm[start:start + args.batch_size])
            u_pred = _forward(model, batch, stats)

            if args.loss == "supervised":
                interior = batch["interior_mask"].expand_as(u_pred)
                err = (u_pred - batch["u_ref"]) * interior
                loss = err.pow(2).mean()
            else:
                # physics: per-sample normalized PDE residual
                res = nonlinear_pde_residual(
                    u_pred, batch["f"], batch["k"], batch["h"], eps=args.coeff_eps)
                f_int = batch["f"][:, :, 1:-1, 1:-1]
                scale = f_int.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1.0)
                loss = ((res / scale) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            val_m = evaluate(model, val_data, stats, eval_bs, args)
            val_key = "relL2" if args.loss == "supervised" else "resRMSE"
            if val_m[val_key] < best_metric:
                best_metric = val_m[val_key]
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "stats": stats,
                    "args": vars(args),
                    "epoch": epoch,
                    f"best_val_{val_key}": best_metric,
                }, best_path)

            print(f"epoch {epoch:03d}  loss={loss.detach():.3e}  "
                  f"resRMSE={val_m['resRMSE']:.4f}  "
                  f"relL2={val_m['relL2']:.4e}  maxErr={val_m['maxError']:.4e}  "
                  f"gradRelL2={val_m['gradRelL2']:.4e}  "
                  f"linBL={val_m['linearBaselineRelL2']:.4e}")

            history.append({
                "epoch": epoch,
                "loss": loss.detach().item(),
                "val_resRMSE": val_m["resRMSE"],
                "val_relL2": val_m["relL2"],
                "val_maxError": val_m["maxError"],
                "val_gradRelL2": val_m["gradRelL2"],
                "val_gradLinf": val_m["gradLinf"],
                "val_meanMaxError": val_m["meanMaxError"],
                "val_maxMaxError": val_m["maxMaxError"],
                "val_meanBoundaryError": val_m["meanBoundaryError"],
                "val_maxBoundaryError": val_m["maxBoundaryError"],
                "val_meanGradLinf": val_m["meanGradLinf"],
                "val_maxGradLinf": val_m["maxGradLinf"],
                "val_refResRMSE": val_m["refResRMSE"],
                "val_linearBaselineRelL2": val_m["linearBaselineRelL2"],
            })

    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                 "stats": stats, "args": vars(args)}, last_path)
    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, oracle, args.loss)
    return model, oracle, best_metric


# -- evaluation -------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, stats, batch_size, args):
    model.eval()
    n = data["f"].shape[0]
    dev = data["f"].device
    H, W = data["f"].shape[-2:]
    eps = getattr(args, "coeff_eps", 1e-4)
    all_rows = []

    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=dev)
        b = _take_batch(data, idx)
        up = _forward(model, b, stats)
        rows, _ = per_sample_metrics(up, b, eps)
        all_rows.extend(rows)

    n_total = len(all_rows)
    n_pts = (H - 2) * (W - 2)          # interior points per sample
    n_int = n_total * n_pts
    return {"relL2": sum(r["relL2"] for r in all_rows) / n_total,
            "maxError": max(r["maxError"] for r in all_rows),
            "meanMaxError": sum(r["maxError"] for r in all_rows) / n_total,
            "maxMaxError": max(r["maxError"] for r in all_rows),
            "meanBoundaryError": sum(r["bndErr"] for r in all_rows) / n_total,
            "maxBoundaryError": max(r["bndErr"] for r in all_rows),
            "meanGradLinf": sum(r["gradLinf"] for r in all_rows) / n_total,
            "maxGradLinf": max(r["gradLinf"] for r in all_rows),
            "resRMSE": math.sqrt(sum(r["predResRMSE"] ** 2 * n_pts for r in all_rows) / n_int),
            "refResRMSE": math.sqrt(sum(r["refResRMSE"] ** 2 * n_pts for r in all_rows) / n_int),
            "bmax": max(r["bndErr"] for r in all_rows),
            "gradRelL2": sum(r["gradRelL2"] for r in all_rows) / n_total,
            "gradLinf": max(r["gradLinf"] for r in all_rows),
            "linearBaselineRelL2": sum(r["linearBaselineRelL2"] for r in all_rows) / n_total,
            "constBaselineRelL2": sum(r["constBaselineRelL2"] for r in all_rows) / n_total,
            "per_sample": all_rows}


def per_sample_metrics(u_pred, batch, eps):
    """Per-sample metrics: all 12 quantities computed sample-by-sample then averaged.

    Returns (list_of_dicts, summary_dict)."""
    ue = batch["u_ref"]
    interior = batch["interior_mask"].expand_as(u_pred)
    bnd_mask = batch["mask"].expand_as(u_pred)
    h = float(batch["h"])
    n_samples = u_pred.shape[0]
    raw = bool(batch.get("raw_power", False))

    err = (u_pred - ue) * interior
    res = nonlinear_pde_residual(u_pred, batch["f"], batch["k"], batch["h"],
                                 eps=eps, raw_power=raw)
    res_ref = nonlinear_pde_residual(ue, batch["f"], batch["k"], batch["h"],
                                     eps=eps, raw_power=raw)

    err_flat = err[:, :, 1:-1, 1:-1].flatten(1)
    u_flat = ue[:, :, 1:-1, 1:-1].flatten(1)
    res_flat = res.flatten(1)
    res_ref_flat = res_ref.flatten(1)
    f_flat = batch["f"][:, :, 1:-1, 1:-1].flatten(1)

    du = lambda v: (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2. * h)
    dv = lambda v: (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2. * h)
    grad_du = torch.cat([du(u_pred), dv(u_pred)], 1).flatten(2)
    grad_true = torch.cat([du(ue), dv(ue)], 1).flatten(2)
    grad_err = grad_du - grad_true

    rows = []
    for i in range(n_samples):
        rl2 = (err_flat[i].norm() / u_flat[i].norm().clamp_min(1e-12)).item()
        const_pred = enforce_dirichlet(
            torch.full_like(ue[i:i+1], u_flat[i].mean()),
            batch["g"][i:i+1], batch["mask"][:1])
        const_err = ((const_pred - ue[i:i+1]) * interior[i:i+1])[:, :, 1:-1, 1:-1].flatten(1)
        const_rl2 = (const_err[0].norm() / u_flat[i].norm().clamp_min(1e-12)).item()

        rows.append({
            "sample_id": i, "k": batch["k"][i].item(),
            "relL2": rl2,
            "RMSE": err_flat[i].pow(2).mean().sqrt().item(),
            "MAE": err_flat[i].abs().mean().item(),
            "maxError": err_flat[i].abs().max().item(),
            "predResRMSE": res_flat[i].pow(2).mean().sqrt().item(),
            "refResRMSE": res_ref_flat[i].pow(2).mean().sqrt().item(),
            "bndErr": ((u_pred[i:i+1] - batch["g"][i:i+1]).abs()
                       * bnd_mask[i:i+1]).max().item(),
            "gradRelL2": (grad_err[i].flatten().norm()
                          / grad_true[i].flatten().norm().clamp_min(1e-12)).item(),
            "gradLinf": grad_err[i].abs().max().item(),
            "constBaselineRelL2": const_rl2,
            "f_RMSE": f_flat[i].pow(2).mean().sqrt().item(),
        })

    f_np = batch["f"].detach().cpu().numpy()
    g_np = batch["g"].detach().cpu().numpy()
    ue_np = ue.detach().cpu().numpy()
    interior_np = interior.detach().cpu().numpy()
    for i, row in enumerate(rows):
        u_lin = solve_linear_fd(f_np[i, 0], g_np[i, 0], np.ones_like(g_np[i, 0]), h)
        err_lin = (u_lin - ue_np[i, 0]) * interior_np[i, 0]
        ref_lin = ue_np[i, 0] * interior_np[i, 0]
        row["linearBaselineRelL2"] = float(
            np.linalg.norm(err_lin) / max(np.linalg.norm(ref_lin), 1e-12))

    metric_keys = ["relL2", "RMSE", "MAE", "maxError", "predResRMSE", "refResRMSE",
                   "bndErr", "gradRelL2", "gradLinf", "constBaselineRelL2",
                   "linearBaselineRelL2", "f_RMSE"]
    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in rows]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "min": float(np.min(vals)), "max": float(np.max(vals))}
    return rows, summary


def save_summary_json(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# -- visualization ----------------------------------------------------------

def _setup_plot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _save_panel(ue, up, err, fm, rp, re, path, title, fext, rext, dpi=200):
    plt = _setup_plot()

    def sym(v):
        m = max(v.abs().max().cpu().item(), 1e-12)
        return -m, m

    umin = min(ue.min().item(), up.min().item())
    umax = max(ue.max().item(), up.max().item())
    rmax = max(rp.abs().max().cpu().item(), re.abs().max().cpu().item(), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (0, 0, ue,  "u_ref",          "viridis",  fext, umin, umax),
        (0, 1, up,  "CNN prediction",  "viridis",  fext, umin, umax),
        (0, 2, err, "signed error",    "coolwarm", fext, *sym(err)),
        (1, 0, fm,  "forcing f",       "coolwarm", fext, *sym(fm)),
        (1, 1, rp,  "pred PDE res",    "coolwarm", rext, -rmax, rmax),
        (1, 2, re,  "ref PDE res",     "coolwarm", rext, -rmax, rmax),
    ]
    for r, c, v, label, cmap, ext, v0, v1 in panels:
        ax = axes[r][c]
        im = ax.imshow(v.detach().cpu().numpy(), origin="lower", extent=ext,
                       cmap=cmap, vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_centerline(ue, up, path, title, dpi=150):
    """1D cut along y≈0: u_ref vs u_pred."""
    plt = _setup_plot()

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

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_plot(history, path, oracle=0., loss_type="physics"):
    if len(history) < 2:
        return
    plt = _setup_plot()

    epochs_list = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"Training convergence — {loss_type}")

    ax1.plot(epochs_list, [h["loss"] for h in history], "b-")
    ax1.set_xlabel("epoch"); ax1.set_yscale("log")
    ax1.set_title("Loss"); ax1.grid(True, alpha=.3)

    ax2b = ax2.twinx()
    l1, = ax2.plot(epochs_list, [h["val_relL2"] for h in history], "g-", label="relL2")
    l2, = ax2b.plot(epochs_list, [h["val_resRMSE"] for h in history], "m--", label="resRMSE")
    handles = [l1, l2]
    if oracle > 0:
        handles.append(ax2b.axhline(oracle, color="gray", ls=":", label=f"oracle {oracle:.1e}"))
    ax2.set_xlabel("epoch"); ax2.legend(handles=handles, loc="upper right")
    ax2.grid(True, alpha=.3); ax2.set_title("Validation")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- fixed-solver test suite --------------------------------------------------
#
# predict-case semantics:
#   data:         random f,g,k -> nonlinear FD solver -> u_ref (training family)
#   fixed-solver: manufactured u_target only builds f and g; u_ref is
#                 recomputed by the nonlinear FD solver, so the test stays
#                 strictly in the solver pipeline and is comparable with the
#                 MMS analytic cases (which live in Baseline_mms.py).

# Same signed functions as the MMS ANALYTIC_CASES u1-u4.
# Pipeline uses a = u^k exactly (raw_power=True) — k must be 2.
FIXED_SOLVER_CASES = {
    "u1_mms_sincos": lambda x, y: (1. - x ** 2) * (1. - y ** 2) * np.sin(np.pi * x) * np.cos(np.pi * y),
    "u2_mms_poly":   lambda x, y: x * (1. - x) * y * (1. - y),
    "u3_mms_cos":    lambda x, y: np.cos(np.pi * x) * np.cos(2. * np.pi * y),
    "u4_mms_mixed":  lambda x, y: x ** 2 * (1. - y) + np.sin(np.pi * x) * y,
}


def make_fixed_solver_dataset(N, device, args, k_val):
    """Deterministic solver test: u_target -> f = -div_h(u^k grad u), then
    solve_nonlinear_fd(f,g,k) -> u_ref.  Signed u allowed; uses raw u^k
    (raw_power=True), so k must be 2."""
    k_val = float(k_val)
    if abs(k_val - 2.0) > 1e-12:
        raise ValueError(
            f"fixed-solver requires k=2 (signed u, raw u^k), got {k_val}")
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    x64, y64, *_ = make_grid(N, "cpu", torch.float64)
    x_np = x64[0, 0].numpy()
    y_np = y64[0, 0].numpy()
    eps = args.coeff_eps
    # Conservative SOR for signed u: under-relaxation prevents divergence
    max_fd_iter = max(getattr(args, "max_fd_iter", 5000), 10000)
    fd_tol = min(getattr(args, "fd_tol", 1e-6), 1e-8)
    fd_omega = 0.8

    f_list, g_list, k_list, u_list, t_list = [], [], [], [], []
    info_list, names = [], []
    res64_sq = 0.

    for case_name, u_func in FIXED_SOLVER_CASES.items():
        u_target = u_func(x_np, y_np).astype(np.float64)

        g_np = u_target.copy()
        g_np[1:-1, 1:-1] = 0.

        t64 = torch.from_numpy(u_target)[None, None]      # CPU float64
        k64 = torch.tensor([k_val], dtype=torch.float64)
        f64 = torch.zeros_like(t64)
        f64[:, :, 1:-1, 1:-1] = nonlinear_flux_divergence(
            t64, k64, h, eps, raw_power=True)
        f_np = f64[0, 0].numpy()

        u_ref, info = solve_nonlinear_fd(
            f_np, g_np, k_val, h, eps, max_fd_iter, fd_tol, fd_omega,
            raw_power=True)

        # Acceptance: converged + small residual
        f_rms = float(np.sqrt(np.mean(f_np[1:-1, 1:-1] ** 2)))
        if not (info["converged"]
                and np.isfinite(info["residual_rmse"])
                and info["residual_rmse"] < 1e-3 * max(1.0, f_rms)):
            raise RuntimeError(
                f"fixed-solver case {case_name} rejected by solver: {info}")

        u64 = torch.from_numpy(u_ref).double()[None, None]
        res64_sq += float(nonlinear_pde_residual(
            u64, f64, k64, h, eps=eps, raw_power=True).pow(2).sum())

        f_list.append(f64.float().to(device))
        g_list.append(torch.from_numpy(g_np).float()[None, None].to(device))
        k_list.append(torch.tensor([k_val], dtype=torch.float32, device=device))
        u_list.append(u64.float().to(device))
        t_list.append(t64.float().to(device))
        info_list.append(info)
        names.append(case_name)

    return {
        "f": torch.cat(f_list), "g": torch.cat(g_list),
        "k": torch.cat(k_list), "u_ref": torch.cat(u_list),
        "u_target": torch.cat(t_list),
        "case_names": names, "solver_info": info_list,
        "raw_power": True,
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h,
        "ref_res64": math.sqrt(res64_sq / (len(names) * N * N)),
    }


def _save_target_comparison(u_target, u_solver, path, title, dpi=150):
    """1x3 panel: MMS u_target | solver u_ref | (solver - target)."""
    plt = _setup_plot()
    ut = u_target.detach().cpu().numpy()
    us = u_solver.detach().cpu().numpy()
    diff = us - ut
    vmin = min(ut.min(), us.min())
    vmax = max(ut.max(), us.max())
    dmax = max(abs(diff).max(), 1e-12)
    ext = [-1., 1., -1., 1.]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    fig.suptitle(title)
    panels = [(ut,   "u_target (MMS)",   "viridis",  vmin,  vmax),
              (us,   "u_solver_ref",     "viridis",  vmin,  vmax),
              (diff, "solver - target",  "coolwarm", -dmax, dmax)]
    for ax, (v, lbl, cmap, v0, v1) in zip(axes, panels):
        im = ax.imshow(v, origin="lower", extent=ext, cmap=cmap,
                       vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(lbl)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_fixed_solver_suite(model, stats, device, args, data_args, out_dir):
    """Run the 4 fixed-solver cases, save metrics CSV/JSON + figures.

    CNN metrics (via per_sample_metrics) compare u_pred against the solver
    reference; target_vs_solver_ref_* checks whether the solver solution
    matches the MMS target, i.e. whether the case is a fair MMS comparison."""
    eps = data_args.coeff_eps
    N = data_args.N
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"prediction (fixed-solver): N={N}  k={args.predict_k:g}  "
          f"4 MMS-matched cases, u_ref from nonlinear FD solver")
    batch = make_fixed_solver_dataset(N, device, data_args, args.predict_k)
    print(f"  ref residual (float64) = {batch['ref_res64']:.4e}")

    with torch.no_grad():
        u_pred = _forward(model, batch, stats)
    rows, _ = per_sample_metrics(u_pred, batch, eps)

    interior = batch["interior_mask"][0, 0].detach().cpu().numpy()
    ur_np = batch["u_ref"].detach().cpu().numpy()
    ut_np = batch["u_target"].detach().cpu().numpy()

    h = float(batch["h"])
    fext = [-1., 1., -1., 1.]
    rext = [-1. + h, 1. - h, -1. + h, 1. - h]

    out_rows = []
    for i, (case, info) in enumerate(zip(batch["case_names"],
                                         batch["solver_info"])):
        diff = (ur_np[i, 0] - ut_np[i, 0]) * interior
        ref = ur_np[i, 0] * interior
        t_rl2 = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12))
        t_max = float(np.abs(diff).max())

        r = rows[i]
        out_rows.append({
            "case": case,
            "k": r["k"],
            "cnn_vs_solver_ref_relL2": r["relL2"],
            "cnn_vs_solver_ref_maxError": r["maxError"],
            "predResRMSE": r["predResRMSE"],
            "refResRMSE": r["refResRMSE"],
            "boundaryError": r["bndErr"],
            "gradRelL2": r["gradRelL2"],
            "constBaselineRelL2": r["constBaselineRelL2"],
            "linearBaselineRelL2": r["linearBaselineRelL2"],
            "target_vs_solver_ref_relL2": t_rl2,
            "target_vs_solver_ref_maxError": t_max,
            "solver_num_iter": info["num_iter"],
            "solver_residual_rmse": info["residual_rmse"],
            "solver_u_min": info["u_min"],
            "solver_u_max": info["u_max"],
        })

        print(f"\n  --- fixed-solver case: {case}  (k={r['k']:g}) ---")
        print(f"    relL2={r['relL2']:.4e}  maxErr={r['maxError']:.4e}  "
              f"resRMSE={r['predResRMSE']:.4e}  "
              f"linBL={r['linearBaselineRelL2']:.4e}")
        print(f"    target_vs_solver relL2={t_rl2:.2e}  maxErr={t_max:.2e}  "
              f"solver: iter={info['num_iter']}  res={info['residual_rmse']:.2e}")
        if t_rl2 > 1e-3:
            print(f"    WARNING: target_vs_solver_ref_relL2={t_rl2:.2e} — "
                  f"solver reference deviates from the MMS target; "
                  f"this case is NOT a fair MMS comparison")

        ui = batch["u_ref"][i, 0]
        up = u_pred[i, 0]
        pr = nonlinear_pde_residual(u_pred[i:i+1], batch["f"][i:i+1],
                                    batch["k"][i:i+1], batch["h"], eps=eps,
                                    raw_power=True)[0, 0]
        er = nonlinear_pde_residual(batch["u_ref"][i:i+1], batch["f"][i:i+1],
                                    batch["k"][i:i+1], batch["h"], eps=eps,
                                    raw_power=True)[0, 0]
        _save_panel(ui, up, up - ui, batch["f"][i, 0], pr, er,
                    out_dir / f"fixed_solver_{case}_panel.png",
                    f"fixed-solver {case}  k={r['k']:.2f}  "
                    f"relL2={r['relL2']:.3e}", fext, rext)
        _save_centerline(ui, up, out_dir / f"fixed_solver_{case}_centerline.png",
                         f"fixed-solver {case}  k={r['k']:.2f}  "
                         f"relL2={r['relL2']:.3e}")
        _save_target_comparison(batch["u_target"][i, 0], ui,
                                out_dir / f"fixed_solver_{case}_target_vs_solver.png",
                                f"{case}  k={r['k']:.2f}  "
                                f"target_vs_solver relL2={t_rl2:.2e}")

    _write_csv(out_rows, out_dir / "fixed_solver_metrics.csv")
    summary = {}
    for key in out_rows[0]:
        if key == "case":
            continue
        vals = [r[key] for r in out_rows]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
    save_summary_json(summary, out_dir / "fixed_solver_summary.json")
    print(f"\n  fixed-solver suite saved → {out_dir}")
    return out_rows


# -- prediction -------------------------------------------------------------

def run_prediction(args):
    if not args.predict_from:
        raise ValueError("--predict-from required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _select_device(args.device)

    ckpt = torch.load(args.predict_from, map_location=device, weights_only=False)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = SimpleCNN(
        hidden_ch=ckpt_args.base_ch,
        n_layers=getattr(ckpt_args, "n_layers", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = ckpt["stats"]

    N = ckpt_args.N
    data_args = argparse.Namespace(**vars(ckpt_args))
    data_args.device = args.device
    data_args.seed = args.test_seed

    random.seed(args.test_seed)
    np.random.seed(args.test_seed)

    pred_case = getattr(args, "predict_case", "data")
    if pred_case == "fixed-solver":
        out = (Path(getattr(args, "plot_dir", None))
               if getattr(args, "plot_dir", None)
               else Path("results") / "predict" / "fixed_solver")
        run_fixed_solver_suite(model, stats, device, args, data_args, out)
        return
    else:
        assert args.test_seed not in (ckpt_args.seed, ckpt_args.seed + 1), (
            "test seed collides with a train/val seed! prediction data would "
            "overlap the training run (evaluation pollution)."
        )
        data_args.random_k = False
        data_args.k = args.predict_k
        print(f"prediction (solver): N={N}  samples={args.predict_samples}  "
              f"k={args.predict_k:g} (fixed)")
        batch = make_solver_dataset(N, args.predict_samples, device,
                                     data_args, seed=args.test_seed)

    with torch.no_grad():
        u_pred = _forward(model, batch, stats)

    vm = evaluate(model, batch, stats, args.predict_samples, data_args)
    triv_rl2 = trivial_linear_baseline(batch)
    print(f"relL2={vm['relL2']:.4e}  maxErr={vm['maxError']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  refRMSE={vm['refResRMSE']:.4e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")
    print(f"meanMaxError={vm['meanMaxError']:.4e}  maxMaxError={vm['maxMaxError']:.4e}")
    print(f"meanBoundaryError={vm['meanBoundaryError']:.2e}  maxBoundaryError={vm['maxBoundaryError']:.2e}")
    print(f"meanGradLinf={vm['meanGradLinf']:.4e}  maxGradLinf={vm['maxGradLinf']:.4e}")
    if "ref_res64" in batch:
        print(f"reference residual RMSE (float64) = {batch['ref_res64']:.4e}")
    print(f"trivial baseline (a≡1 linear Poisson) relL2 = {triv_rl2:.4e}  "
          f"(CNN must beat this to have learned u^k)")

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "k": batch["k"].cpu(), "u_ref": batch["u_ref"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]), "metrics": vm,
        }, p)
        print(f"saved predictions → {p}")

    rows, summary = per_sample_metrics(u_pred, batch, eps=data_args.coeff_eps)

    # Always save metrics — use --predict-save or default dir
    save_dir = getattr(args, "predict_save", "")
    if save_dir:
        out = Path(save_dir).parent
    else:
        out = Path(getattr(args, "plot_dir", None)) if getattr(args, "plot_dir", None) else Path("results") / "predict"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out / "metrics.csv")
    save_summary_json(summary, out / "summary.json")

    if args.plot:
        h = float(batch["h"])
        fext = [-1., 1., -1., 1.]
        rext = [-1. + h, 1. - h, -1. + h, 1. - h]
        eps = data_args.coeff_eps

        for i, row in enumerate(rows):
            ui = batch["u_ref"][i, 0]
            up = u_pred[i, 0]

            pr = nonlinear_pde_residual(
                u_pred[i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h, eps=eps)[0, 0]
            er = nonlinear_pde_residual(
                batch["u_ref"][i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h, eps=eps)[0, 0]

            _save_panel(ui, up, up - ui, batch["f"][i, 0], pr, er,
                        out / f"sample_{i:03d}_panel.png",
                        f"sample {i}  k={row['k']:.2f}  relL2={row['relL2']:.3e}",
                        fext, rext)
            _save_centerline(ui, up, out / f"sample_{i:03d}_centerline.png",
                             f"sample {i}  k={row['k']:.2f}  "
                             f"relL2={row['relL2']:.3e}")
    print(f"saved → {out}")


# -- CLI --------------------------------------------------------------------

def ckpt_key(args):
    """Experiment key encoding all data/model parameters that affect results."""
    if getattr(args, "random_k", False):
        k_str = f"k{args.k_min:g}-{args.k_max:g}"
    else:
        k_str = f"k{args.k:g}"
    n_layers = getattr(args, "n_layers", 6)
    loss_suffix = "_sup" if getattr(args, "loss", "physics") == "supervised" else ""
    return (f"N{args.N}_e{args.epochs}_tr{args.train_samples}"
            f"_{k_str}_M{getattr(args, 'fourier_m', 6)}"
            f"_fs{args.f_scale:g}_gb{args.g_base:g}_gs{args.g_scale:g}"
            f"_ch{args.base_ch}_L{n_layers}{loss_suffix}")


def _write_run_info(path, args, best_metric, oracle):
    k_str = (f"{args.k_min:.4g}–{args.k_max:.4g}" if getattr(args, "random_k", False)
             else f"{args.k:.4g}")
    metric_name = "relL2" if args.loss == "supervised" else "resRMSE"
    solver_line = (f"Solver: nonlinear finite-difference, red-black SOR "
                   f"(max_iter={args.max_fd_iter}, "
                   f"tol={args.fd_tol:.1e}, omega={args.fd_omega})\n")
    path.write_text(
        f"Experiment: CNN Nonlinear Poisson (loss={args.loss})\n"
        f"Key: {ckpt_key(args)}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"N: {args.N}  Epochs: {args.epochs}  k: {k_str}  Seed: {args.seed}\n"
        f"Train: {args.train_samples}  Val: {args.val_samples}\n"
        f"ch={args.base_ch}  L={getattr(args, 'n_layers', 6)}\n"
        f"f_scale={args.f_scale:.4g}  g_base={args.g_base:.4g}  g_scale={args.g_scale:.4g}\n"
        f"Fourier M={getattr(args, 'fourier_m', 6)}\n"
        f"{solver_line}"
        f"Best val {metric_name}: {best_metric:.4e}  Oracle: {oracle:.4e}\n")


def _update_registry(base_dir, key, args, best_metric, oracle):
    """Upsert experiment entry in registry.json."""
    reg_path = base_dir / "registry.json"
    try:
        reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    except json.JSONDecodeError:
        reg = {}
    metric_key = "best_val_relL2" if args.loss == "supervised" else "best_val_resRMSE"
    reg.setdefault("cnn", {})[key] = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "N": args.N,
        "epochs": args.epochs,
        "k": [args.k_min, args.k_max] if getattr(args, "random_k", False) else args.k,
        "train_samples": args.train_samples,
        "data": "solver",
        "loss": args.loss,
        metric_key: best_metric,
        "oracle": oracle,
    }
    reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")


def _link_latest(base_dir, key):
    """Update cnn/latest symlink to point to the most recent run."""
    latest = base_dir / "cnn" / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(key)


def _archive_old_figures(figures_dir):
    """Move existing figures (excluding history/) into history/{timestamp}/."""
    if not figures_dir.exists():
        return
    old = [p for p in figures_dir.iterdir() if p.name != "history"]
    if not old:
        return
    archive = figures_dir / "history" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=True)
    for p in old:
        p.rename(archive / p.name)
    print(f"  archived {len(old)} files → {archive}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--train-samples", type=int, default=128)
    p.add_argument("--val-samples", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)

    # Data generation
    p.add_argument("--data-sanity", action="store_true",
                   help="generate solver data, print stats and FD solver demo, then exit")

    # Nonlinear PDE
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--random-k", action=argparse.BooleanOptionalAction, default=True,
                   help="sample k uniformly from [k_min, k_max] (default: True)")
    p.add_argument("--k-min", type=float, default=0.0)
    p.add_argument("--k-max", type=float, default=5.0)

    # Fourier series
    p.add_argument("--fourier-m", type=int, default=6,
                   help="number of Fourier modes per dimension for random fields")

    # Random field scales
    p.add_argument("--f-scale", type=float, default=5.0,
                   help="scale of random forcing")
    p.add_argument("--g-base", type=float, default=1.0)
    p.add_argument("--g-scale", type=float, default=0.4)

    # Nonlinear FD solver (red-black SOR relaxation)
    p.add_argument("--max-fd-iter", type=int, default=5000,
                   help="max SOR sweeps for nonlinear finite-difference solver")
    p.add_argument("--fd-tol", type=float, default=1e-6,
                   help="convergence tolerance: residual RMSE < tol * max(1, ||f||_rms)")
    p.add_argument("--fd-omega", type=float, default=1.5,
                   help="SOR relaxation factor (1 = Gauss-Seidel, must be < 2)")
    p.add_argument("--coeff-eps", type=float, default=1e-4,
                   help="epsilon for u^k clamp")
    p.add_argument("--max-attempt-factor", type=int, default=20,
                   help="max attempts = factor * n_samples")

    # Forward sanity
    p.add_argument("--solver-sanity", action="store_true",
                   help="generate solver-forward data, print stats, and exit")

    # Loss
    p.add_argument("--loss", default="physics",
                   choices=["physics", "supervised"],
                   help="physics: PDE residual only | supervised: |u_pred - u_ref|^2")

    # Training control
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--predict-from", default="",
                   help="load checkpoint for prediction (not for resume training)")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="")
    p.add_argument("--test-seed", type=int, default=1234)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--predict-case", default="data",
                   choices=["data", "fixed-solver"],
                   help="'data': random f,g,k -> solver -> u_ref | "
                        "'fixed-solver': manufactured u_target only builds f,g; "
                        "u_ref recomputed by the nonlinear FD solver (strict "
                        "solver test, comparable with MMS)")
    p.add_argument("--plot-dir", default="",
                   help="output dir for prediction figures "
                        "(default: results/predict/<case>)")
    p.add_argument("--predict-k", type=float, default=2.0,
                   help="fixed k for prediction (--predict-case fixed-solver requires k=2)")
    return p.parse_args()


# -- main -------------------------------------------------------------------

def main():
    args = parse_args()

    if args.solver_sanity:
        print("=" * 60)
        print("NONLINEAR FD SOLVER SANITY CHECK")
        print("=" * 60)
        print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")
        print(f"Solver: nonlinear finite-difference, red-black SOR "
              f"(max_iter={args.max_fd_iter}, "
              f"tol={args.fd_tol:.1e}, omega={args.fd_omega})")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = _select_device(args.device)
        data = make_solver_dataset(args.N, args.train_samples, device,
                                    args, seed=args.seed)

        with torch.no_grad():
            res_e = nonlinear_pde_residual(
                data["u_ref"], data["f"], data["k"], data["h"], eps=args.coeff_eps)
            ref_rmse = float(res_e.pow(2).mean().sqrt())
            u_ex = data["u_ref"]
            bnd_err = float(
                ((u_ex - data["g"]).abs()
                 * data["mask"].expand_as(u_ex)).max())

        B = data["f"].shape[0]
        print(f"  accepted: {B}/{args.train_samples}")
        print(f"  ref residual RMSE (float64) = {data['ref_res64']:.4e}")
        print(f"  ref residual RMSE (float32) = {ref_rmse:.4e}")
        print(f"  boundary error    = {bnd_err:.2e}  (should be ~0)")

        if data.get("solver_info"):
            iters = [x["num_iter"] for x in data["solver_info"]]
            res_rmses = [x["residual_rmse"] for x in data["solver_info"]]
            print(f"  mean FD iter = {np.mean(iters):.1f}  max={np.max(iters)}")
            print(f"  mean solver res = {np.mean(res_rmses):.4e}  "
                  f"max={np.max(res_rmses):.4e}")

        u_min, u_max = float(u_ex.min()), float(u_ex.max())
        fi = data["f"][:, :, 1:-1, 1:-1]
        print(f"  u_ref range = [{u_min:.4f}, {u_max:.4f}]")
        print(f"  f range     = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")
        print(f"  OK — nonlinear FD solver completed.")
        print("=" * 60)
        return

    if args.data_sanity:
        print("=" * 60)
        print("DATA SANITY CHECK (data=solver)")
        print("=" * 60)
        print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")

        device = _select_device(args.device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        data = make_solver_dataset(args.N, args.train_samples, device,
                                    args, seed=args.seed)

        ref32in64 = ref_residual_rmse64(data, args.coeff_eps)
        with torch.no_grad():
            u_ex = data["u_ref"]
            bnd_err = float(((u_ex - data["g"]).abs()
                             * data["mask"].expand_as(u_ex)).max())
            u_min, u_max = float(u_ex.min()), float(u_ex.max())
            fi = data["f"][:, :, 1:-1, 1:-1]

        if "ref_res64" in data:
            print(f"  ref residual (float64 pipeline) = {data['ref_res64']:.4e}")
        print(f"  stored float32 data residual    = {ref32in64:.4e}  "
              f"(~h^-2·eps32 rounding floor)")
        print(f"  boundary error = {bnd_err:.2e}  (should be 0)")
        print(f"  u_ref range    = [{u_min:.4f}, {u_max:.4f}]  (positive)")
        print(f"  f range        = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")
        print(f"  trivial baseline (a≡1) relL2    = "
              f"{trivial_linear_baseline(data):.4e}")
        # FD solver feasibility demo: try a hard case
        H = args.N + 2
        h_demo = 2.0 / (args.N + 1)
        g_bad = np.zeros((H, H))
        g_bad[0, :] = g_bad[-1, :] = g_bad[:, 0] = g_bad[:, -1] = 0.1
        f_bad = np.zeros((H, H))
        f_bad[1:-1, 1:-1] = -50.0
        u_bad, info_bad = solve_nonlinear_fd(
            f_bad, g_bad, args.k, h_demo, args.coeff_eps,
            args.max_fd_iter, args.fd_tol, args.fd_omega)
        verdict = ("rejected" if (not info_bad["converged"] or info_bad["u_min"] <= args.coeff_eps)
                   else "accepted — may be suspicious")
        print(f"  FD solver feasibility demo (f=-50, g=0.1): "
              f"converged={info_bad['converged']}  u_min={info_bad['u_min']:.3e}  → {verdict}")
        print("=" * 60)
        return

    if args.predict:
        return run_prediction(args)

    if args.epochs == 0:
        print("epochs=0, nothing to train.")
        return

    key = ckpt_key(args)
    base_dir = Path("checkpoints")
    ckpt_dir = base_dir / "cnn" / key
    best_path = ckpt_dir / "best.pt"

    if best_path.exists() and not args.retrain:
        print(f"found {best_path}, use --retrain to retrain")
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"training {key}")
        _, oracle, best_metric = train(args, best_path,
                                       ckpt_dir / "last.pt",
                                       ckpt_dir / "history.csv",
                                       ckpt_dir / "convergence.png")
        _write_run_info(ckpt_dir / "run_info.txt", args, best_metric, oracle)
        _update_registry(base_dir, key, args, best_metric, oracle)
        _link_latest(base_dir, key)
        print(f"saved → {ckpt_dir}")

    if args.plot:
        figures_dir = Path("results") / "cnn" / key / "figures"
        _archive_old_figures(figures_dir)
        pa = argparse.Namespace(**vars(args))
        pa.predict_from = str(best_path)
        pa.predict_save = ""
        pa.plot = True
        pa.plot_dir = str(figures_dir)
        pa.test_seed = args.seed + 100000   # disjoint from train RNG stream
        run_prediction(pa)

    print(f"done: checkpoints/cnn/{key}")


if __name__ == "__main__":
    main()
