"""
CNN solver for -div(|u|^k grad u) = f on [-1,1]^2, Dirichlet BC u=g.
a(u,k) = |u|^k.

Data generation: random f,g,k -> damped Newton FD solver -> u_ref.
k integer >= 0, signed u. Supports --loss {supervised, physics}.
"""

import argparse, csv, json, math, os, random, tempfile, time
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

def _abs_pow(x, k):
    """a(u,k) = |u|^k.

    k == 0  →  a = 1  (constant diffusivity)
    k >= 1  →  a = |u|^k
    Tensor k: uses abs(u)^k.
    """
    if isinstance(k, (int, float)):
        ki = int(k)
        if ki == 0:
            return torch.ones_like(x)
        return x.abs().pow(ki)

    # batched k: one k per sample, broadcast to u shape
    while k.ndim < x.ndim:
        k = k.unsqueeze(-1)
    return x.abs().pow(k.float())


def _coeff_np(u, k):
    """a(u,k) = |u|^k for numpy arrays (integer k only)."""
    ki = int(k)
    if ki == 0:
        return np.ones_like(u)
    return np.abs(u) ** ki


def _coeff_prime_np(u, k):
    """da/du = k * |u|^(k-1) * sign(u) for numpy arrays. k=0 → 0."""
    ki = int(k)
    if ki == 0:
        return np.zeros_like(u)
    return ki * (np.abs(u) ** (ki - 1)) * np.sign(u)


def nonlinear_flux_divergence(u, k, h):
    """5-point flux-form FD: -div(a(u,k) grad u) with arithmetic-mean
    interface coefficients a = |u|^k."""
    if isinstance(k, (int, float)):
        k = torch.tensor([k], device=u.device)
    if k.ndim == 1:
        k = k.view(-1, 1, 1, 1)

    a = _abs_pow(u, k)

    # cell-centered coefficient and u
    uc = u[:, :, 1:-1, 1:-1]
    ac = a[:, :, 1:-1, 1:-1]

    # arithmetic-mean interface coefficients
    ae = 0.5 * (ac + a[:, :, 1:-1, 2:])
    aw = 0.5 * (ac + a[:, :, 1:-1, :-2])
    an = 0.5 * (ac + a[:, :, 2:, 1:-1])
    as_ = 0.5 * (ac + a[:, :, :-2, 1:-1])

    # flux on each face (east, west, north, south)
    fe = ae * (u[:, :, 1:-1, 2:] - uc) / h
    fw = aw * (uc - u[:, :, 1:-1, :-2]) / h
    fn = an * (u[:, :, 2:, 1:-1] - uc) / h
    fs = as_ * (uc - u[:, :, :-2, 1:-1]) / h

    return -(fe - fw + fn - fs) / h


def nonlinear_pde_residual(u, f, k, h):
    """PDE residual R = -div_h(a(u,k) grad_h u) - f, interior nodes only."""
    return nonlinear_flux_divergence(u, k, h) - f[:, :, 1:-1, 1:-1]

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

def _random_g(N, g_base=0.0, g_scale=0.4, M=6, rng=np.random):
    """Random signed Dirichlet boundary via Fourier series. Interior = 0."""
    raw = random_fourier_field(N, M, rng)
    g = np.zeros((N + 2, N + 2), dtype=np.float64)
    g[0, :] = g_base + g_scale * raw[0, :]
    g[-1, :] = g_base + g_scale * raw[-1, :]
    g[:, 0] = g_base + g_scale * raw[:, 0]
    g[:, -1] = g_base + g_scale * raw[:, -1]
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

    Solves -div(grad u) = f (ignoring the nonlinear diffusivity entirely)
    per sample.  This is the must-report trivial baseline: if the CNN cannot
    beat it, then a good relL2 only reflects that the problem is nearly
    linear, not that the model learned a(u,k)."""
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

def _np_flux_residual(u, f, k, h):
    """R(u) = -div_h(a(u,k) grad_h u) - f on interior nodes (numpy)."""
    a = _coeff_np(u, k)
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

def _assemble_newton_jacobian(u, k, h):
    """Jacobian of -div(|u|^k grad u) w.r.t. u for damped Newton."""
    N = u.shape[0] - 2
    n = N * N
    h2 = 1.0 / (h * h)

    a = _coeff_np(u, k)
    ap = _coeff_prime_np(u, k)

    # interface coefficients (arithmetic mean)
    ac = a[1:N+1, 1:N+1]
    ae = 0.5 * (ac + a[1:N+1, 2:N+2])
    aw = 0.5 * (ac + a[1:N+1, 0:N])
    an_ = 0.5 * (ac + a[2:N+2, 1:N+1])
    as_ = 0.5 * (ac + a[0:N, 1:N+1])

    # u and a'(u) at cell centers and neighbours
    uc = u[1:N+1, 1:N+1]
    apc = ap[1:N+1, 1:N+1]
    ue = u[1:N+1, 2:N+2]; ape = ap[1:N+1, 2:N+2]
    uw = u[1:N+1, 0:N];   apw = ap[1:N+1, 0:N]
    un = u[2:N+2, 1:N+1]; apn = ap[2:N+2, 1:N+1]
    us = u[0:N, 1:N+1];   aps = ap[0:N, 1:N+1]

    # diagonal: sum of interface coeffs + derivative term
    diag = (ae + aw + an_ + as_) * h2 \
           - 0.5 * apc * (ue + uw + un + us - 4. * uc) * h2
    diag = np.where(np.abs(diag) < 1e-14, np.sign(diag + 1e-30) * 1e-14, diag)

    # off-diagonal entries (east, west, north, south)
    val_e = -(ae + 0.5 * ape * (ue - uc)) * h2
    val_w = -(aw + 0.5 * apw * (uw - uc)) * h2
    val_n = -(an_ + 0.5 * apn * (un - uc)) * h2
    val_s = -(as_ + 0.5 * aps * (us - uc)) * h2

    # assemble COO sparse matrix
    idx = np.arange(n)
    mask_e = (idx % N) != (N - 1)
    mask_w = (idx % N) != 0
    mask_n = idx < (n - N)
    mask_s = idx >= N

    rows = [idx]
    cols = [idx]
    data = [diag.ravel()]
    if mask_e.any():
        rows.append(idx[mask_e]); cols.append(idx[mask_e] + 1)
        data.append(val_e.ravel()[mask_e])
    if mask_w.any():
        rows.append(idx[mask_w]); cols.append(idx[mask_w] - 1)
        data.append(val_w.ravel()[mask_w])
    if mask_n.any():
        rows.append(idx[mask_n]); cols.append(idx[mask_n] + N)
        data.append(val_n.ravel()[mask_n])
    if mask_s.any():
        rows.append(idx[mask_s]); cols.append(idx[mask_s] - N)
        data.append(val_s.ravel()[mask_s])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

def _damped_newton_stage(u, f, g, k, h, f_scale, max_iter, tol):
    """Single-k damped Newton: minimize ||R(u)|| on the interior.
    Returns (u, converged, num_iter)."""
    N = f.shape[0] - 2

    def pack(v):
        return v[1:-1, 1:-1].ravel()

    def unpack(vf):
        v = g.copy()
        v[1:-1, 1:-1] = vf.reshape(N, N)
        return v

    for it in range(max_iter):
        res = _np_flux_residual(u, f, k, h)
        res_rms = float(np.sqrt(np.mean(res.ravel() ** 2)))
        if res_rms < tol * f_scale:
            return u, True, it + 1
        J = _assemble_newton_jacobian(u, k, h)
        try:
            delta = spla.spsolve(J, -res.ravel())
        except Exception:
            break
        uf = pack(u)
        accepted = False
        for lp in range(16):
            lam = 0.5 ** lp
            unew = unpack(uf + lam * delta)
            new_res = _np_flux_residual(unew, f, k, h)
            if float(np.sqrt(np.mean(new_res.ravel() ** 2))) < res_rms:
                u = unew
                accepted = True
                break
        if not accepted:
            break
    return u, False, max_iter


def solve_nonlinear_fd(f, g, k, h, max_iter=50, tol=1e-10):
    """Damped Newton with k-continuation: k=0 (linear) → k=1 → ... → target k.

    Always starts from the a≡1 linear solve.  Each stage uses the converged
    solution of the previous stage as the initial guess, avoiding the
    near-singular region around |u|≈0 that appears for k≥3 when starting
    from the a≡1 linear guess."""
    t_start = time.perf_counter()
    k = int(k)
    f_int = f[1:-1, 1:-1]
    f_scale = max(float(np.sqrt((f_int ** 2).mean())), 1.0)

    # stage 0: linear (a≡1) initial guess
    u = solve_linear_fd(f, g, np.ones_like(g), h)

    if k == 0:
        u_t = torch.tensor(u, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        f_t = torch.tensor(f, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        k_t = torch.tensor([0], dtype=torch.float64)
        with torch.no_grad():
            final_res64 = float(nonlinear_pde_residual(u_t, f_t, k_t, h).pow(2).mean().sqrt())
        return u, {"num_iter": 0, "converged": True,
                   "u_min": float(u.min()), "u_max": float(u.max()),
                   "residual_rmse": final_res64, "stage_iters": [],
                   "solve_time_ms": (time.perf_counter() - t_start) * 1e3}

    # continuation: kk = 1, 2, ..., k
    total_iter = 0
    stage_iters = []
    converged = True
    for kk in range(1, k + 1):
        u, stage_conv, stage_it = _damped_newton_stage(
            u, f, g, kk, h, f_scale, max_iter, tol,
        )
        total_iter += stage_it
        stage_iters.append(stage_it)
        if not stage_conv:
            converged = False
            break

    u_t = torch.tensor(u, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    f_t = torch.tensor(f, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    k_t = torch.tensor([k], dtype=torch.float64)
    with torch.no_grad():
        final_res64 = float(nonlinear_pde_residual(u_t, f_t, k_t, h).pow(2).mean().sqrt())
    return u, {"num_iter": total_iter, "converged": converged,
               "u_min": float(u.min()), "u_max": float(u.max()),
               "residual_rmse": final_res64, "stage_iters": stage_iters,
               "solve_time_ms": (time.perf_counter() - t_start) * 1e3}

# -- forward solver dataset --------------------------------------------------

def make_solver_dataset(N, n_samples, device, args, seed):
    """Random f,g,k -> nonlinear FD solve -> u_ref.
    a(u,k) = |u|^k >= 0, so sign of u is irrelevant — no positivity check needed."""
    rng = np.random.RandomState(seed)
    x_t, y_t, bnd, interior, h_grid = make_grid(N, device)
    M = getattr(args, "fourier_m", 6)

    f_list, g_list, k_list, u_list, info_list = [], [], [], [], []
    res64_sq = 0.
    max_attempts = args.max_attempt_factor * n_samples
    attempts = 0

    print(f"generating {n_samples} solver samples (N={N})  "
          f"f_scale={args.f_scale}  g_base={args.g_base}  g_scale={args.g_scale}  "
          f"FD (damped Newton): max_iter={args.newton_max_iter}  tol={args.fd_tol:.1e}")

    while len(u_list) < n_samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"too many failed solver attempts ({attempts}) — "
                "try smaller --f-scale or looser --fd-tol.")
        attempts += 1

        if getattr(args, "random_k", False):
            k_i = int(rng.randint(args.k_min, args.k_max + 1))
        else:
            k_i = int(args.k)

        f_np = _random_f(N, args.f_scale, M, rng)
        g_np = _random_g(N, args.g_base, args.g_scale, M, rng)

        u_np, info = solve_nonlinear_fd(f_np, g_np, k_i, h_grid, max_iter=args.newton_max_iter, tol=args.fd_tol)

        # Acceptance: converged, reasonable residual
        # a(u,k) = |u|^k >= 0, no positivity constraint needed
        f_rms = float(np.sqrt(np.mean(f_np[1:-1, 1:-1] ** 2)))
        if (info["converged"]
                and np.isfinite(info["residual_rmse"])
                and info["residual_rmse"] < 1e-3 * max(1.0, f_rms)):

            # Reference residual in float64
            u64 = torch.from_numpy(u_np).double().unsqueeze(0).unsqueeze(0)
            f64 = torch.from_numpy(f_np).double().unsqueeze(0).unsqueeze(0)
            k64 = torch.tensor([k_i], dtype=torch.float64)
            res64_sq += float(nonlinear_pde_residual(
                u64, f64, k64, h_grid).pow(2).sum())

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
    u_mins = [x['u_min'] for x in info_list]
    u_maxs = [x['u_max'] for x in info_list]
    sign_change = sum(1 for mn, mx in zip(u_mins, u_maxs) if mn <= 0 <= mx)
    print(f"  u range: [{min(u_mins):.4f}, {max(u_maxs):.4f}]")
    print(f"  sign-changing samples: {sign_change}/{len(u_list)} "
          f"({100*sign_change/max(1,len(u_list)):.0f}%)")

    return {
        "f": torch.cat(f_list), "g": torch.cat(g_list),
        "k": torch.cat(k_list), "u_ref": torch.cat(u_list),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h_grid,
        "ref_res64": ref_res64, "n_rejected": attempts - len(u_list),
        "solver_info": info_list,
    }

def ref_residual_rmse64(data):
    """Residual RMSE of the stored (float32) reference data, computed in
    float64, so the ~h^-2*eps32 rounding floor is measured, not hidden.
    Training itself stays float32."""
    u = data["u_ref"].detach().cpu().double()
    f = data["f"].detach().cpu().double()
    k = data["k"].detach().cpu().double()
    return float(nonlinear_pde_residual(u, f, k, data["h"])
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

def _sync(device):
    """Block until pending GPU kernels finish (for wall-time measurement)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

def _time_forward(model, batch, stats):
    """CNN inference wall time (model forward only, after one warm-up forward).
    Returns (u_pred, ms_per_batch, ms_per_sample)."""
    device = batch["f"].device
    with torch.no_grad():
        _forward(model, batch, stats)          # warm-up
        _sync(device)
        t0 = time.perf_counter()
        u_pred = _forward(model, batch, stats)
        _sync(device)
        ms = (time.perf_counter() - t0) * 1e3
    return u_pred, ms, ms / batch["f"].shape[0]

# -- training ---------------------------------------------------------------

def train(args, best_path, last_path, history_path, convergence_path, model=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _select_device(args.device)

    print(f"Equation: -div(|u|^k grad u) = f  |  k = {args.k}  |  data=solver")
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

    triv_rl2 = trivial_linear_baseline(val_data)
    print(f"trivial baseline (a≡1 linear Poisson) relL2 = {triv_rl2:.4e}  "
          f"(CNN must beat this to have learned a(u,k))")

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
    print(f"device={device}  ref_res={ref_res:.4e}  "
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
                    u_pred, batch["f"], batch["k"], batch["h"])
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
                    "model_type": "cnn",
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
                "val_meanMaxError": val_m["meanMaxError"],
                "val_meanBoundaryError": val_m["meanBoundaryError"],
                "val_maxBoundaryError": val_m["maxBoundaryError"],
                "val_meanGradLinf": val_m["meanGradLinf"],
                "val_maxGradLinf": val_m["maxGradLinf"],
                "val_refResRMSE": val_m["refResRMSE"],
                "val_linearBaselineRelL2": val_m["linearBaselineRelL2"],
            })

    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "cnn",
                 "model_state_dict": model.state_dict(),
                 "stats": stats, "args": vars(args)}, last_path)
    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, args.loss)
    return model, best_metric

# -- evaluation -------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, stats, batch_size, args):
    model.eval()
    n = data["f"].shape[0]
    dev = data["f"].device
    H, W = data["f"].shape[-2:]
    all_rows = []

    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=dev)
        b = _take_batch(data, idx)
        up = _forward(model, b, stats)
        rows, _ = per_sample_metrics(up, b, start_id=start)
        all_rows.extend(rows)

    n_total = len(all_rows)
    n_pts = (H - 2) * (W - 2)          # interior points per sample
    n_int = n_total * n_pts
    return {"relL2": sum(r["relL2"] for r in all_rows) / n_total,
            "maxError": max(r["maxError"] for r in all_rows),
            "meanMaxError": sum(r["maxError"] for r in all_rows) / n_total,
            "meanBoundaryError": sum(r["bndErr"] for r in all_rows) / n_total,
            "maxBoundaryError": max(r["bndErr"] for r in all_rows),
            "meanGradLinf": sum(r["gradLinf"] for r in all_rows) / n_total,
            "maxGradLinf": max(r["gradLinf"] for r in all_rows),
            "resRMSE": math.sqrt(sum(r["predResRMSE"] ** 2 * n_pts for r in all_rows) / n_int),
            "refResRMSE": math.sqrt(sum(r["refResRMSE"] ** 2 * n_pts for r in all_rows) / n_int),
            "gradRelL2": sum(r["gradRelL2"] for r in all_rows) / n_total,
            "linearBaselineRelL2": sum(r["linearBaselineRelL2"] for r in all_rows) / n_total,
            "constBaselineRelL2": sum(r["constBaselineRelL2"] for r in all_rows) / n_total,
            "per_sample": all_rows}

def per_sample_metrics(u_pred, batch, start_id=0):
    """Per-sample metrics: all 12 quantities computed sample-by-sample then averaged.

    Returns (list_of_dicts, summary_dict)."""
    ue = batch["u_ref"]
    interior = batch["interior_mask"].expand_as(u_pred)
    bnd_mask = batch["mask"].expand_as(u_pred)
    h = float(batch["h"])
    n_samples = u_pred.shape[0]

    err = (u_pred - ue) * interior
    res = nonlinear_pde_residual(u_pred, batch["f"], batch["k"], batch["h"])
    res_ref = nonlinear_pde_residual(ue, batch["f"], batch["k"], batch["h"])

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
            "sample_id": start_id + i, "k": batch["k"][i].item(),
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

def _save_convergence_plot(history, path, loss_type="physics"):
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
    ax2.set_xlabel("epoch")
    ax2.legend(handles=[l1, l2], loc="upper right")
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

# Same signed functions as the MMS ANALYTIC_CASES u0-u4.
# a(u,k) = |u|^k, so any integer k >= 0 is safe.
FIXED_SOLVER_CASES = {
    "u1_sincos": lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y),
    "u2_poly":   lambda x, y: x * (1. - x) * y * (1. - y),
    "u3_cos":    lambda x, y: np.cos(np.pi * x) * np.cos(2. * np.pi * y),
    "u4_mixed":  lambda x, y: x ** 2 * (1. - y) + np.sin(np.pi * x) * y,
}

def make_fixed_solver_dataset(N, device, args, k_val):
    """Deterministic solver test: u_target -> f = -div_h(|u|^k grad u), then
    solve_nonlinear_fd(f,g,k) -> u_ref.  a(u,k) = |u|^k,
    so signed u with any integer k >= 0 is safe."""
    k_val = int(k_val)
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    x64, y64, *_ = make_grid(N, "cpu", torch.float64)
    x_np = x64[0, 0].numpy()
    y_np = y64[0, 0].numpy()
    # Newton solver for signed u: under-relaxation prevents divergence
    newton_max_iter = getattr(args, "newton_max_iter", 50)
    fd_tol = getattr(args, "fd_tol", 1e-10)

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
            t64, k64, h)
        f_np = f64[0, 0].numpy()

        u_ref, info = solve_nonlinear_fd(f_np, g_np, k_val, h, max_iter=newton_max_iter, tol=fd_tol)

        # Acceptance: converged + small residual
        f_rms = float(np.sqrt(np.mean(f_np[1:-1, 1:-1] ** 2)))
        if not (info["converged"]
                and np.isfinite(info["residual_rmse"])
                and info["residual_rmse"] < 1e-3 * max(1.0, f_rms)):
            raise RuntimeError(
                f"fixed-solver case {case_name} rejected by solver: {info}")

        u64 = torch.from_numpy(u_ref).double()[None, None]
        res64_sq += float(nonlinear_pde_residual(
            u64, f64, k64, h).pow(2).sum())

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
    """Run the fixed-solver cases, save metrics CSV/JSON + figures.

    CNN metrics (via per_sample_metrics) compare u_pred against the solver
    reference; target_vs_solver_ref_* checks whether the solver solution
    matches the MMS target, i.e. whether the case is a fair MMS comparison."""
    N = data_args.N
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"prediction (fixed-solver): N={N}  k={args.predict_k:g}  "
          f"{len(FIXED_SOLVER_CASES)} cases, u_ref from nonlinear FD solver")
    batch = make_fixed_solver_dataset(N, device, data_args, args.predict_k)
    print(f"  ref residual (float64) = {batch['ref_res64']:.4e}")

    u_pred, cnn_ms_batch, cnn_ms_sample = _time_forward(model, batch, stats)
    rows, _ = per_sample_metrics(u_pred, batch)

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
            "cnn_ms_per_sample": cnn_ms_sample,
            "fd_solver_ms_per_sample": info["solve_time_ms"],
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
        if args.plot:
            pr = nonlinear_pde_residual(u_pred[i:i+1], batch["f"][i:i+1],
                                        batch["k"][i:i+1], batch["h"])[0, 0]
            er = nonlinear_pde_residual(batch["u_ref"][i:i+1], batch["f"][i:i+1],
                                        batch["k"][i:i+1], batch["h"])[0, 0]
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

    fd_ms_mean = float(np.mean([x["solve_time_ms"] for x in batch["solver_info"]]))
    print(f"\n  timing: cnn_ms_per_sample={cnn_ms_sample:.3f}  "
          f"fd_solver_ms_per_sample={fd_ms_mean:.3f}")
    _write_csv(out_rows, out_dir / "fixed_solver_metrics.csv")
    summary = {}
    for key in out_rows[0]:
        if key == "case":
            continue
        vals = [r[key] for r in out_rows]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
    with open(out_dir / "fixed_solver_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
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
    ckpt_type = ckpt.get("model_type")
    if ckpt_type and ckpt_type != "cnn":
        raise ValueError(
            f"checkpoint type mismatch: expected 'cnn', got '{ckpt_type}'. "
            f"Use the correct prediction script (Baseline_solver.py for "
            f"'cnn', Baseline_mms.py for 'cnn_mms').")
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

    u_pred, cnn_ms_batch, cnn_ms_sample = _time_forward(model, batch, stats)
    fd_ms_sample = float(np.mean(
        [x["solve_time_ms"] for x in batch["solver_info"]]))

    vm = evaluate(model, batch, stats, args.predict_samples, data_args)
    triv_rl2 = trivial_linear_baseline(batch)
    print(f"relL2={vm['relL2']:.4e}  maxErr={vm['maxError']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  refRMSE={vm['refResRMSE']:.4e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")
    print(f"meanMaxError={vm['meanMaxError']:.4e}")
    print(f"meanBoundaryError={vm['meanBoundaryError']:.2e}  maxBoundaryError={vm['maxBoundaryError']:.2e}")
    print(f"meanGradLinf={vm['meanGradLinf']:.4e}  maxGradLinf={vm['maxGradLinf']:.4e}")
    if "ref_res64" in batch:
        print(f"reference residual RMSE (float64) = {batch['ref_res64']:.4e}")
    print(f"trivial baseline (a≡1 linear Poisson) relL2 = {triv_rl2:.4e}  "
          f"(CNN must beat this to have learned a(u,k))")
    print(f"timing: cnn={cnn_ms_batch:.2f} ms/batch  "
          f"cnn_ms_per_sample={cnn_ms_sample:.3f}  "
          f"fd_solver_ms_per_sample={fd_ms_sample:.3f}")

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "k": batch["k"].cpu(), "u_ref": batch["u_ref"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]), "metrics": vm,
        }, p)
        print(f"saved predictions → {p}")

    rows, summary = per_sample_metrics(u_pred, batch)
    for r, info in zip(rows, batch["solver_info"]):
        r["cnn_ms_per_sample"] = cnn_ms_sample
        r["fd_solver_ms_per_sample"] = info["solve_time_ms"]

    # Always save metrics — use --predict-save or default dir
    save_dir = getattr(args, "predict_save", "")
    if save_dir:
        out = Path(save_dir).parent
    else:
        out = Path(getattr(args, "plot_dir", None)) if getattr(args, "plot_dir", None) else Path("results") / "predict"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out / "metrics.csv")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.plot:
        h = float(batch["h"])
        fext = [-1., 1., -1., 1.]
        rext = [-1. + h, 1. - h, -1. + h, 1. - h]

        for i, row in enumerate(rows):
            ui = batch["u_ref"][i, 0]
            up = u_pred[i, 0]

            pr = nonlinear_pde_residual(
                u_pred[i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h)[0, 0]
            er = nonlinear_pde_residual(
                batch["u_ref"][i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h)[0, 0]

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
    p.add_argument("--sanity", action="store_true",
                   help="generate solver dataset, print acceptance/iteration/"
                        "residual/data stats, then exit")

    # Nonlinear PDE
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--random-k", action=argparse.BooleanOptionalAction, default=True,
                   help="sample k uniformly from [k_min, k_max] (default: True)")
    p.add_argument("--k-min", type=int, default=0)
    p.add_argument("--k-max", type=int, default=5)

    # Fourier series
    p.add_argument("--fourier-m", type=int, default=6,
                   help="number of Fourier modes per dimension for random fields")

    # Random field scales
    p.add_argument("--f-scale", type=float, default=5.0,
                   help="scale of random forcing")
    p.add_argument("--g-base", type=float, default=0.0)
    p.add_argument("--g-scale", type=float, default=0.4)

    # Nonlinear FD solver (damped Newton)
    p.add_argument("--newton-max-iter", type=int, default=50)
    p.add_argument("--fd-tol", type=float, default=1e-10)
    p.add_argument("--max-attempt-factor", type=int, default=20,
                   help="max attempts = factor * n_samples")

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
    p.add_argument("--test-seed", type=int, default=9999)
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
    p.add_argument("--predict-k", type=int, default=2,
                   help="fixed integer k for prediction")
    return p.parse_args()

# -- main -------------------------------------------------------------------

def _validate_args(args):
    """Reject invalid argument combinations early."""
    if args.N < 4:
        raise ValueError(f"N must be >= 4, got {args.N}")
    if args.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {args.batch_size}")
    if args.train_samples < 1:
        raise ValueError(f"train_samples must be >= 1, got {args.train_samples}")
    if args.k_min > args.k_max:
        raise ValueError(f"k_min ({args.k_min}) > k_max ({args.k_max})")
    if args.k < 0 and not args.random_k:
        raise ValueError(f"--k must be >= 0, got {args.k}")

def main():
    args = parse_args()
    _validate_args(args)

    if args.sanity:
        print("=" * 60)
        print("SANITY CHECK (data=solver, damped Newton)")
        print("=" * 60)
        print(f"Equation: -div(|u|^k grad u) = f  |  k = {args.k}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")
        print(f"Solver: damped Newton "
              f"(max_iter={args.newton_max_iter}, tol={args.fd_tol:.1e})")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = _select_device(args.device)
        data = make_solver_dataset(args.N, args.train_samples, device,
                                    args, seed=args.seed)

        ref32in64 = ref_residual_rmse64(data)
        with torch.no_grad():
            u_ex = data["u_ref"]
            bnd_err = float(((u_ex - data["g"]).abs()
                             * data["mask"].expand_as(u_ex)).max())
            u_min, u_max = float(u_ex.min()), float(u_ex.max())
            fi = data["f"][:, :, 1:-1, 1:-1]

        stage_lists = [x["stage_iters"] for x in data["solver_info"]
                       if x["stage_iters"]]
        if stage_lists:
            max_stages = max(len(s) for s in stage_lists)
            per_stage = "  ".join(
                f"k{j+1}:{np.mean([s[j] for s in stage_lists if len(s) > j]):.1f}"
                for j in range(max_stages))
            print(f"  per-stage mean iters: {per_stage}")
        print(f"  ref residual (float64 pipeline) = {data['ref_res64']:.4e}")
        print(f"  stored float32 data residual    = {ref32in64:.4e}  "
              f"(~h^-2·eps32 rounding floor)")
        print(f"  boundary error = {bnd_err:.2e}  (should be 0)")
        print(f"  u_ref range    = [{u_min:.4f}, {u_max:.4f}]  (signed allowed)")
        print(f"  f range        = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")
        print(f"  trivial baseline (a≡1) relL2    = "
              f"{trivial_linear_baseline(data):.4e}")
        print("=" * 60)
        return

    if args.predict:
        return run_prediction(args)

    key = ckpt_key(args)
    base_dir = Path("checkpoints")
    ckpt_dir = base_dir / "cnn" / key
    best_path = ckpt_dir / "best.pt"

    if best_path.exists() and not args.retrain:
        print(f"found {best_path}, use --retrain to retrain")
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"training {key}")
        _, best_metric = train(args, best_path,
                               ckpt_dir / "last.pt",
                               ckpt_dir / "history.csv",
                               ckpt_dir / "convergence.png")
        print(f"saved → {ckpt_dir}")

    if args.plot:
        figures_dir = Path("results") / "cnn" / key / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
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
