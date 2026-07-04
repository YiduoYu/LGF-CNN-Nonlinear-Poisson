"""
CNN solver for nonlinear flux-form Poisson equation: -div(u^k grad u) = f on [-1,1]^2.
Dirichlet BC: u = g on boundary.

ROUTE B: forward data-generation using Picard iteration.
  - Randomly sample f, g, k.
  - Solve -div(u^k grad u) = f via Picard (frozen-coefficient linear solves).
  - Obtain reference solution u_ref (Picard).
  - Train CNN to learn (f, g, k) -> u_ref.

  Equation: -div(u^k grad u) = f,  u > 0,  k > 0.
  No external a(x,y) field — the diffusivity is u^k itself.
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

def power_coefficient(u, k, eps=1e-6):
    """Compute a = u^k.  k may be scalar or shape [B].  u assumed positive."""
    if isinstance(k, (int, float)):
        k = torch.tensor([k], device=u.device)
    if k.ndim == 1:
        k = k.view(-1, 1, 1, 1)
    return torch.clamp(u, min=eps).pow(k)


def nonlinear_flux_divergence(u, k, h, eps=1e-6):
    """5-point flux-form FD: -div(u^k grad u).
    Interface coefficient = arithmetic average of neighbour values.
    Args:
        u: [B, 1, H, W]  solution field
        k: [B] or scalar   exponent
        h: grid spacing
    Returns:
        [B, 1, H-2, W-2]  -div_h(u^k grad_h u) on interior
    """
    a = power_coefficient(u, k, eps)

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


def nonlinear_pde_residual(u, f, k, h, eps=1e-6):
    """R(u) = -div_h(u^k grad_h u) - f  on interior nodes.  [B,1,N,N]."""
    return nonlinear_flux_divergence(u, k, h, eps) - f[:, :, 1:-1, 1:-1]


# -- CNN --------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=6, out_ch=1, hidden_ch=32, n_layers=6):
        super().__init__()
        layers = []
        for i in range(n_layers):
            c_in = in_ch if i == 0 else hidden_ch
            g = min(8, hidden_ch)
            while hidden_ch % g:
                g -= 1
            layers += [
                nn.Conv2d(c_in, hidden_ch, 3, padding=1, padding_mode="replicate"),
                nn.GroupNorm(g, hidden_ch),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x):
        return self.head(self.body(x))


# -- smooth field generator -------------------------------------------------

def _smooth_field(N, seed=None):
    """Random smooth field on (N+2)x(N+2) grid."""
    if seed is not None:
        np.random.seed(seed)
    xs = np.linspace(-1, 1, N + 2)
    ys = np.linspace(-1, 1, N + 2)
    xx, yy = np.meshgrid(xs, ys)

    field = np.zeros((N + 2, N + 2))
    for _ in range(np.random.randint(2, 5)):
        kx = np.random.randint(1, 4)
        ky = np.random.randint(1, 4)
        amp = np.random.uniform(-1, 1)
        px = np.random.uniform(0, 2 * np.pi)
        py = np.random.uniform(0, 2 * np.pi)
        field += amp * np.sin(kx * np.pi * xx + px) * np.sin(ky * np.pi * yy + py)

    return field / max(np.abs(field).max(), 1e-12)


# -- random data generators -------------------------------------------------

def _random_f(N, f_scale):
    """Random forcing f with boundary = 0."""
    f = _smooth_field(N) * f_scale
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.
    return f


def _random_positive_g(N, g_base=1.0, g_scale=0.05):
    """Random positive Dirichlet boundary.  Interior = 0.  Boundary >= 0.5."""
    raw = _smooth_field(N)
    g = np.zeros((N + 2, N + 2), dtype=np.float64)
    g[0, :] = g_base + g_scale * raw[0, :]
    g[-1, :] = g_base + g_scale * raw[-1, :]
    g[:, 0] = g_base + g_scale * raw[:, 0]
    g[:, -1] = g_base + g_scale * raw[:, -1]
    # Safety: ensure boundary >= 0.5
    g = np.maximum(g, 0.5)
    g[1:-1, 1:-1] = 0.
    return g


# -- linear Darcy solver (for Picard inner loop) ----------------------------

def _assemble_darcy_system_slow_reference(a, g, f, h):
    """Sparse 5-point flux-form system for -div(a grad u) = f, Dirichlet BC u=g.
    Reference implementation: Python for-loop + lil_matrix. Keep for testing."""
    N = a.shape[0] - 2
    n = N * N

    A = sp.lil_matrix((n, n))
    b = np.zeros(n)
    h2 = 1.0 / (h * h)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            row = (i - 1) * N + (j - 1)
            ac = a[i, j]

            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                coeff = 0.5 * (ac + a[ni, nj])
                A[row, row] += coeff * h2
                if 1 <= ni <= N and 1 <= nj <= N:
                    A[row, (ni - 1) * N + (nj - 1)] -= coeff * h2
                else:
                    b[row] += coeff * h2 * g[ni, nj]

            b[row] += f[i, j]

    return A.tocsr(), b


def _assemble_darcy_system(a, g, f, h):
    """Vectorized 5-point flux-form assembly using COO matrix.
    Discretisation identical to _assemble_darcy_system_slow_reference."""
    N = a.shape[0] - 2
    n = N * N
    h2 = 1.0 / (h * h)

    # Interface coefficients (arithmetic average)
    ac = a[1:N+1, 1:N+1]                    # (N, N)
    ae = 0.5 * (ac + a[1:N+1, 2:N+2])       # east
    aw = 0.5 * (ac + a[1:N+1, 0:N])         # west
    a_n = 0.5 * (ac + a[2:N+2, 1:N+1])      # north
    a_s = 0.5 * (ac + a[0:N, 1:N+1])        # south

    # RHS: forcing + boundary contributions
    b = f[1:N+1, 1:N+1].ravel().copy()
    b[N-1::N] += ae[:, N-1] * h2 * g[1:N+1, N+1]     # east boundary
    b[0::N]   += aw[:, 0]   * h2 * g[1:N+1, 0]       # west boundary
    b[(N-1)*N:] += a_n[N-1, :] * h2 * g[N+1, 1:N+1]  # north boundary
    b[:N]     += a_s[0, :]  * h2 * g[0, 1:N+1]       # south boundary

    # Diagonal
    diag = (ae + aw + a_n + a_s).ravel() * h2

    idx = np.arange(n)

    # East connections: row -> col=row+1  (skip last column)
    mask_e = (idx % N) != (N - 1)
    r_e, c_e = idx[mask_e], idx[mask_e] + 1
    d_e = -ae.ravel()[mask_e] * h2

    # West connections: row -> col=row-1  (skip first column)
    mask_w = (idx % N) != 0
    r_w, c_w = idx[mask_w], idx[mask_w] - 1
    d_w = -aw.ravel()[mask_w] * h2

    # North connections: row -> col=row+N  (skip last row)
    mask_n = idx < (n - N)
    r_n, c_n = idx[mask_n], idx[mask_n] + N
    d_n = -a_n.ravel()[mask_n] * h2

    # South connections: row -> col=row-N  (skip first row)
    mask_s = idx >= N
    r_s, c_s = idx[mask_s], idx[mask_s] - N
    d_s = -a_s.ravel()[mask_s] * h2

    rows = np.concatenate([idx, r_e, r_w, r_n, r_s])
    cols = np.concatenate([idx, c_e, c_w, c_n, c_s])
    data = np.concatenate([diag, d_e, d_w, d_n, d_s])

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return A, b


def solve_darcy_fd(f, g, a, h):
    """Direct solve: -div(a grad u) = f, Dirichlet BC u = g.
    Inputs/outputs are numpy arrays [H,W]."""
    A, b = _assemble_darcy_system(a, g, f, h)
    u_int = spla.spsolve(A, b)
    u = g.copy()
    u[1:-1, 1:-1] = u_int.reshape(a.shape[0] - 2, a.shape[1] - 2)
    return u


# -- Picard solver ----------------------------------------------------------

def initial_guess_from_boundary(g, h):
    """Solve -div(1 * grad u0) = 0 with boundary g.  Harmonic, smooth, positive."""
    f0 = np.zeros_like(g)
    a0 = np.ones_like(g)
    return solve_darcy_fd(f0, g, a0, h)


def solve_nonlinear_picard(f, g, k, h, args):
    """Solve -div(u^k grad u) = f via Picard iteration.

    Args:
        f, g: numpy arrays [H, W]
        k: scalar float  exponent
        h: grid spacing
        args: CLI args with picard params

    Returns:
        u: numpy array [H, W]  converged solution
        info: dict with converged, num_iter, rel_change, u_min, u_max, residual_rmse
    """
    u = initial_guess_from_boundary(g, h)

    converged = False
    info = {
        "converged": False,
        "num_iter": 0,
        "rel_change": math.inf,
        "u_min": 0.,
        "u_max": 0.,
        "residual_rmse": math.inf,
    }

    for it in range(args.picard_max_iter):
        a_eff = np.maximum(u, args.coeff_eps) ** k

        u_lin = solve_darcy_fd(f, g, a_eff, h)

        u_new = args.picard_damping * u_lin + (1. - args.picard_damping) * u

        u_norm = np.linalg.norm(u_new)
        rel_change = np.linalg.norm(u_new - u) / max(u_norm, 1e-12)

        u = u_new

        if rel_change < args.picard_tol:
            converged = True
            info["num_iter"] = it + 1
            info["rel_change"] = rel_change
            break

    if not converged:
        info["num_iter"] = args.picard_max_iter

    info["converged"] = converged
    info["u_min"] = float(u.min())
    info["u_max"] = float(u.max())

    # Compute residual via torch (reuse operator)
    u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    f_t = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    k_t = torch.tensor([k], dtype=torch.float32)
    with torch.no_grad():
        res = nonlinear_pde_residual(u_t, f_t, k_t, h, eps=args.coeff_eps)
        info["residual_rmse"] = float(res.pow(2).mean().sqrt())

    return u, info


# -- Picard dataset ---------------------------------------------------------

def make_picard_dataset(N, n_samples, device, args):
    """Generate dataset by sampling f,g,k and solving nonlinear PDE via Picard.

    Returns dict with same keys as make_mms_dataset, plus "solver_info".
    """
    x_t, y_t, bnd, interior, h_grid = make_grid(N, device)

    f_list, g_list, k_list, u_list, info_list = [], [], [], [], []

    max_attempts = args.max_attempt_factor * n_samples
    attempts = 0
    accepted = 0

    print(f"generating {n_samples} Picard samples (N={N})  "
          f"k={args.k}  damping={args.picard_damping}  tol={args.picard_tol:.1e}")

    while accepted < n_samples and attempts < max_attempts:
        attempts += 1

        # Sample k
        if getattr(args, "random_k", False):
            k_i = float(np.random.uniform(args.k_min, args.k_max))
        else:
            k_i = float(args.k)

        f_np = _random_f(N, args.f_scale)
        g_np = _random_positive_g(N, args.g_base, args.g_scale)

        u_np, info = solve_nonlinear_picard(f_np, g_np, k_i, h_grid, args)

        # Acceptance criteria
        if (info["converged"]
                and info["u_min"] > args.u_min_accept
                and info["residual_rmse"] < args.max_residual_rmse):
            accepted += 1

            f_t = torch.tensor(f_np, dtype=torch.float32, device=device)
            g_t = torch.tensor(g_np, dtype=torch.float32, device=device)
            u_t = torch.tensor(u_np, dtype=torch.float32, device=device)

            f_list.append(f_t.unsqueeze(0).unsqueeze(0))
            g_list.append(g_t.unsqueeze(0).unsqueeze(0))
            k_list.append(torch.tensor([k_i], device=device))
            u_list.append(u_t.unsqueeze(0).unsqueeze(0))
            info_list.append(info)

        if attempts % 50 == 0:
            print(f"  attempts {attempts}/{max_attempts}  accepted {accepted}/{n_samples}")

    if accepted == 0:
        raise RuntimeError(
            "No Picard samples accepted. Try: smaller --f-scale, larger --picard-damping, "
            "larger --picard-max-iter, or looser --max-residual-rmse."
        )

    # Summary
    iterations = [x["num_iter"] for x in info_list]
    residuals = [x["residual_rmse"] for x in info_list]
    u_mins = [x["u_min"] for x in info_list]
    u_maxs = [x["u_max"] for x in info_list]

    print(f"  acceptance: {accepted}/{attempts} ({100*accepted/max(1,attempts):.1f}%)")
    print(f"  Picard iterations: mean={np.mean(iterations):.1f}  max={np.max(iterations)}")
    print(f"  solver residual:   mean={np.mean(residuals):.4e}  "
          f"max={np.max(residuals):.4e}")
    print(f"  u range: [{min(u_mins):.4f}, {max(u_maxs):.4f}]")

    if accepted < n_samples:
        raise RuntimeError(
            f"Only accepted {accepted}/{n_samples} Picard samples. "
            "Relax acceptance criteria (e.g. --max-residual-rmse, --u-min-accept) "
            "or reduce --f-scale."
        )

    return {
        "f": torch.cat(f_list),
        "g": torch.cat(g_list),
        "k": torch.cat(k_list),
        "u_ref": torch.cat(u_list),         # Picard reference solution
        "mask": bnd,
        "interior_mask": interior,
        "x": x_t,
        "y": y_t,
        "h": h_grid,
        "solver_info": info_list,
    }


# -- input / stats / batching ------------------------------------------------

def build_input(f_norm, g, bnd, k_norm, x, y):
    """Build 6-channel CNN input: [f_norm, g, boundary_mask, k_channel, x, y]."""
    B = f_norm.shape[0]
    if bnd.shape[0] == 1:
        bnd = bnd.expand(B, -1, -1, -1)
    if x.shape[0] == 1:
        x = x.expand(B, -1, -1, -1)
        y = y.expand(B, -1, -1, -1)
    if k_norm.ndim == 1:
        k_ch = k_norm.view(-1, 1, 1, 1).expand(B, 1, *bnd.shape[-2:])
    else:
        k_ch = torch.full(
            (B, 1, *bnd.shape[-2:]), float(k_norm),
            device=bnd.device, dtype=bnd.dtype)
    return torch.cat([f_norm, g, bnd, k_ch, x, y], dim=1)


def compute_stats(data):
    """Compute z-score stats for f (interior) and k.
    Uses unbiased=False to avoid NaN on single-sample datasets."""
    fi = data["f"][:, :, 1:-1, 1:-1]
    k = data["k"]
    return (
        float(fi.mean()),
        float(fi.std(unbiased=False).clamp_min(1e-8)),
        float(k.mean()),
        float(k.std(unbiased=False).clamp_min(1e-8)),
    )


def _take_batch(data, idx):
    batch = {k: data[k][idx] for k in ("f", "g", "k", "u_ref")}
    batch.update({k: data[k][:1] for k in ("mask", "interior_mask", "x", "y")})
    batch["h"] = data["h"]
    return batch


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

    print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}  |  data=picard")
    if getattr(args, "random_k", False):
        print(f"  random k in [{args.k_min}, {args.k_max}]")

    print(f"generating {args.train_samples} Picard samples (N={args.N})")
    train_data = make_picard_dataset(args.N, args.train_samples, device, args)
    val_data = make_picard_dataset(args.N, args.val_samples, device, args)
    stats = compute_stats(train_data)
    print(f"f stats: mean={stats[0]:.4e}  std={stats[1]:.4e}  "
          f"k stats: mean={stats[2]:.4e}  std={stats[3]:.4e}")

    # Reference residual
    with torch.no_grad():
        ref_res = float(nonlinear_pde_residual(
            val_data["u_ref"], val_data["f"], val_data["k"],
            val_data["h"], eps=args.coeff_eps).pow(2).mean().sqrt())
        u_min = float(val_data["u_ref"].min())
        u_max = float(val_data["u_ref"].max())
        f_vals = val_data["f"][:, :, 1:-1, 1:-1]
        bnd_err = float(
            ((val_data["u_ref"] - val_data["g"]).abs()
             * val_data["mask"].expand_as(val_data["u_ref"])).max())
        print(f"ref residual RMSE = {ref_res:.4e}")
        print(f"u_ref range: [{u_min:.4f}, {u_max:.4f}]  "
              f"boundary error = {bnd_err:.2e}")
        oracle = ref_res

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
          f"lambda_sup={args.lambda_sup}  lambda_pde={args.lambda_pde}  "
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

            interior = batch["interior_mask"].expand_as(u_pred)
            err = (u_pred - batch["u_ref"]) * interior
            loss_sup = err.pow(2).mean()

            res = nonlinear_pde_residual(
                u_pred, batch["f"], batch["k"], batch["h"], eps=args.coeff_eps)
            h = batch["h"]
            loss_pde = (h * h * res).pow(2).mean()

            loss = args.lambda_sup * loss_sup + args.lambda_pde * loss_pde

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            val_m = evaluate(model, val_data, stats, eval_bs, args)
            if val_m["relL2"] < best_metric:
                best_metric = val_m["relL2"]
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "stats": stats,
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val_relL2": best_metric,
                }, best_path)

            print(f"epoch {epoch:03d}  loss={loss.detach():.3e}  "
                  f"sup={loss_sup.detach():.3e}  pde={loss_pde.detach():.3e}  "
                  f"relL2={val_m['relL2']:.4e}  linf={val_m['linf']:.4e}  "
                  f"resRMSE={val_m['resRMSE']:.4f}  "
                  f"refRMSE={val_m['refResRMSE']:.4e}  "
                  f"gradRelL2={val_m['gradRelL2']:.4e}")

            history.append({
                "epoch": epoch,
                "loss": loss.detach().item(),
                "loss_sup": loss_sup.detach().item(),
                "loss_pde": loss_pde.detach().item(),
                "val_relL2": val_m["relL2"],
                "val_resRMSE": val_m["resRMSE"],
                "val_linf": val_m["linf"],
                "val_gradRelL2": val_m["gradRelL2"],
                "val_refResRMSE": val_m["refResRMSE"],
            })

    for p, d in [
        (last_path, {"model_state_dict": model.state_dict(),
                      "stats": stats, "args": vars(args)}),
        (history_path, None),
        (convergence_path, None),
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        if d:
            torch.save(d, p)

    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, oracle)
    return model, oracle, best_metric


# -- evaluation -------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, stats, batch_size, args):
    model.eval()
    n = data["f"].shape[0]
    dev = data["f"].device
    H, W = data["f"].shape[-2], data["f"].shape[-1]
    ac = {"rl2": 0., "li": 0., "rs": 0., "rli": 0., "bm": 0., "gs": 0.,
          "ers": 0., "c": 0}

    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=dev)
        b = _take_batch(data, idx)

        up = _forward(model, b, stats)
        ue = b["u_ref"]
        interior = b["interior_mask"].expand_as(up)
        h = float(b["h"])

        err = (up - ue) * interior
        res = nonlinear_pde_residual(up, b["f"], b["k"], b["h"], eps=args.coeff_eps)
        res_e = nonlinear_pde_residual(ue, b["f"], b["k"], b["h"], eps=args.coeff_eps)

        m = up.shape[0]
        ac["c"] += m

        ac["rl2"] += float(
            (err.flatten(1).norm(dim=1)
             / ue.mul(interior).flatten(1).norm(dim=1).clamp_min(1e-12)).sum())
        ac["li"] = max(ac["li"], float(err.abs().max()))
        ac["rs"] += float(res.pow(2).sum())
        ac["ers"] += float(res_e.pow(2).sum())
        ac["rli"] = max(ac["rli"], float(res.abs().max()))
        ac["bm"] = max(ac["bm"],
                       float(((up - b["g"]).abs()
                              * b["mask"].expand_as(up)).max()))

        du = lambda v: (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2. * h)
        dv = lambda v: (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2. * h)
        pg = torch.cat([du(up), dv(up)], 1)
        tg = torch.cat([du(ue), dv(ue)], 1)
        ac["gs"] += float(
            ((pg - tg).flatten(1).norm(dim=1)
             / tg.flatten(1).norm(dim=1).clamp_min(1e-12)).sum())

    c = ac["c"]
    ni = c * (H - 2) * (W - 2)
    return {
        "relL2": ac["rl2"] / c,
        "linf": ac["li"],
        "resRMSE": math.sqrt(ac["rs"] / ni),
        "refResRMSE": math.sqrt(ac["ers"] / ni),
        "resLinf": ac["rli"],
        "bmax": ac["bm"],
        "gradRelL2": ac["gs"] / c,
    }


def per_sample_metrics(u_pred, batch, eps):
    ue = batch["u_ref"]
    interior = batch["interior_mask"].expand_as(u_pred)
    err = (u_pred - ue) * interior

    res = nonlinear_pde_residual(u_pred, batch["f"], batch["k"], batch["h"], eps=eps)
    res_e = nonlinear_pde_residual(ue, batch["f"], batch["k"], batch["h"], eps=eps)

    ie = err[:, :, 1:-1, 1:-1].flatten(1)
    iu = ue[:, :, 1:-1, 1:-1].flatten(1)
    rf = res.flatten(1)
    ref = res_e.flatten(1)
    fi = batch["f"][:, :, 1:-1, 1:-1].flatten(1)
    rl2 = ie.norm(dim=1) / iu.norm(dim=1).clamp_min(1e-12)

    rows = []
    for i in range(u_pred.shape[0]):
        k_i = batch["k"][i].item()
        rows.append({
            "sample_id": i,
            "k": float(k_i),
            "u_min": float(ue[i].min()),
            "u_max": float(ue[i].max()),
            "relL2": float(rl2[i]),
            "RMSE": float(ie[i].pow(2).mean().sqrt()),
            "MAE": float(ie[i].abs().mean()),
            "Linf": float(ie[i].abs().max()),
            "pred_flux_resRMSE": float(rf[i].pow(2).mean().sqrt()),
            "ref_flux_resRMSE": float(ref[i].pow(2).mean().sqrt()),
            "f_RMSE": float(fi[i].pow(2).mean().sqrt()),
        })
    return rows


# -- visualization ----------------------------------------------------------

def _write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _save_panel(ue, up, err, fm, rp, re, path, title, fext, rext, dpi=200):
    os.environ.setdefault("MPLCONFIGDIR",
                          str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sym(v):
        mx = max(float(v.abs().max().cpu()), 1e-12)
        return -mx, mx

    umin = min(float(ue.min()), float(up.min()))
    umax = max(float(ue.max()), float(up.max()))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (0, 0, ue,  "reference u",            "viridis",  fext, umin, umax),
        (0, 1, up,  "CNN prediction",          "viridis",  fext, umin, umax),
        (0, 2, err, "signed error",            "coolwarm", fext, *sym(err)),
        (1, 0, fm,  "forcing f",               "coolwarm", fext, *sym(fm)),
        (1, 1, rp,  "pred PDE residual",       "coolwarm", rext, *sym(rp)),
        (1, 2, re,  "FD residual of ref u",    "coolwarm", rext, *sym(re)),
    ]
    for r, c, v, label, cmap, ext, v0, v1 in panels:
        ax = axes[r][c]
        im = ax.imshow(v.detach().cpu().numpy(), origin="lower", extent=ext,
                       cmap=cmap, vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_plot(history, path, oracle=0.):
    if len(history) < 2:
        return

    os.environ.setdefault("MPLCONFIGDIR",
                          str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_list = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle("Training convergence — Nonlinear Picard")

    ax1 = axes[0]
    ax1.plot(epochs_list, [h["loss"] for h in history], "b-", label="total")
    ax1.plot(epochs_list, [h["loss_sup"] for h in history], "r--", label="sup")
    ax1.plot(epochs_list, [h["loss_pde"] for h in history], "g--", label="pde")
    ax1.set_xlabel("epoch")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=.3)
    ax1.set_title("Training loss")

    ax2 = axes[1]
    ax2b = ax2.twinx()
    l3, = ax2.plot(epochs_list, [h["val_relL2"] for h in history],
                   "g-", label="val relL2")
    l4, = ax2b.plot(epochs_list, [h["val_resRMSE"] for h in history],
                    "m--", label="val resRMSE")
    handles = [l3, l4]
    if oracle > 0:
        l5 = ax2b.axhline(oracle, color="gray", ls=":",
                          label=f"oracle ({oracle:.4f})")
        handles.append(l5)
    ax2.set_xlabel("epoch")
    ax2.legend(handles=handles, loc="upper right")
    ax2.grid(True, alpha=.3)
    ax2.set_title("Validation metrics")

    ax3 = axes[2]
    ax3.plot(epochs_list, [h["val_linf"] for h in history], "c-", label="linf")
    ax3.set_xlabel("epoch")
    ax3.set_yscale("log")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=.3)
    ax3.set_title("Max error")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- prediction -------------------------------------------------------------

def run_prediction(args):
    if not args.predict_from:
        raise ValueError("--predict-from required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    N = args.predict_N or ckpt_args.N
    print(f"prediction (picard): N={N}  samples={args.predict_samples}")

    # Use training data distribution; only override prediction-specific params
    data_args = argparse.Namespace(**vars(ckpt_args))
    data_args.device = args.device
    data_args.seed = args.test_seed

    random.seed(args.test_seed)
    np.random.seed(args.test_seed)
    batch = make_picard_dataset(N, args.predict_samples, device, data_args)

    with torch.no_grad():
        u_pred = _forward(model, batch, stats)

    vm = evaluate(model, batch, stats, args.predict_samples, args)
    print(f"relL2={vm['relL2']:.4e}  linf={vm['linf']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  refRMSE={vm['refResRMSE']:.4e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(),
            "g": batch["g"].cpu(),
            "k": batch["k"].cpu(),
            "u_ref": batch["u_ref"].cpu(),
            "u_pred": u_pred.cpu(),
            "h": float(batch["h"]),
            "metrics": vm,
        }, p)
        print(f"saved predictions → {p}")

    rows = per_sample_metrics(u_pred, batch, eps=data_args.coeff_eps)

    if args.plot:
        out = Path(args.plot_dir)
        out.mkdir(parents=True, exist_ok=True)

        h = float(batch["h"])
        fext = [-1., 1., -1., 1.]
        res_extent = [-1. + h, 1. - h, -1. + h, 1. - h]

        for i, row in enumerate(rows):
            ui = batch["u_ref"][i, 0]
            up = u_pred[i, 0]

            pr = nonlinear_pde_residual(
                u_pred[i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h, eps=data_args.coeff_eps)[0, 0]
            er = nonlinear_pde_residual(
                batch["u_ref"][i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h, eps=data_args.coeff_eps)[0, 0]

            title = (f"sample {row['sample_id']}  k={row['k']:.2f}  "
                     f"relL2={row['relL2']:.3e}  "
                     f"pred_res={row['pred_flux_resRMSE']:.3e}")
            _save_panel(ui, up, up - ui, batch["f"][i, 0],
                        pr, er,
                        out / f"sample_{row['sample_id']:03d}_panel.png",
                        title, fext, res_extent)

        _write_csv(rows, out / "metrics.csv")
        print(f"saved -> {out}")


# -- CLI --------------------------------------------------------------------

def ckpt_key(args):
    """Generate compact experiment key: N{N}_e{epochs}_k{k} or N{N}_e{epochs}_k{min}-{max}."""
    if getattr(args, "random_k", False):
        k_str = f"k{args.k_min:g}-{args.k_max:g}"
    else:
        k_str = f"k{args.k:g}"
    return f"N{args.N}_e{args.epochs}_{k_str}"


def _write_run_info(path, args, best_metric, oracle):
    """Write human-readable run metadata."""
    path.write_text("\n".join([
        f"Experiment: CNN Nonlinear Poisson (Picard, physics mode)",
        f"Key: {ckpt_key(args)}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"N: {args.N}",
        f"Epochs: {args.epochs}",
        f"k: {args.k_min:.4g}–{args.k_max:.4g}" if getattr(args, "random_k", False)
        else f"k: {args.k:.4g}",
        f"Train samples: {args.train_samples}",
        f"Val samples: {args.val_samples}",
        f"Base channels: {args.base_ch}",
        f"Layers: {getattr(args, 'n_layers', 6)}",
        f"f_scale: {args.f_scale:.4g}",
        f"g_base: {args.g_base:.4g}  g_scale: {args.g_scale:.4g}",
        f"Picard damping: {args.picard_damping}  tol: {args.picard_tol:.1e}",
        f"Best val relL2: {best_metric:.4e}",
        f"Oracle residual: {oracle:.4e}",
        f"Seed: {args.seed}",
    ]) + "\n")


def _update_registry(base_dir, key, args, best_metric, oracle):
    """Upsert experiment entry in registry.json."""
    reg_path = base_dir / "registry.json"
    try:
        reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    except json.JSONDecodeError:
        reg = {}
    reg.setdefault("cnn", {})[key] = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "N": args.N,
        "epochs": args.epochs,
        "k": [args.k_min, args.k_max] if getattr(args, "random_k", False) else args.k,
        "train_samples": args.train_samples,
        "best_val_relL2": best_metric,
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

    # Nonlinear PDE
    p.add_argument("--k", type=float, default=2.0,
                   help="exponent in u^k (diffusivity)")
    p.add_argument("--random-k", action="store_true",
                   help="sample k uniformly from [k_min, k_max]")
    p.add_argument("--k-min", type=float, default=1.0)
    p.add_argument("--k-max", type=float, default=3.0)

    # Random field scales
    p.add_argument("--f-scale", type=float, default=0.1,
                   help="scale of random forcing")
    p.add_argument("--g-base", type=float, default=1.0,
                   help="base value for boundary data")
    p.add_argument("--g-scale", type=float, default=0.05,
                   help="scale of random boundary variation")

    # Picard solver
    p.add_argument("--picard-max-iter", type=int, default=100)
    p.add_argument("--picard-tol", type=float, default=1e-8)
    p.add_argument("--picard-damping", type=float, default=0.5)
    p.add_argument("--coeff-eps", type=float, default=1e-4,
                   help="epsilon for u^k clamp in Picard")
    p.add_argument("--u-min-accept", type=float, default=1e-3,
                   help="minimum u_val for sample acceptance")
    p.add_argument("--max-residual-rmse", type=float, default=1e-3,
                   help="maximum solver residual to accept")
    p.add_argument("--max-attempt-factor", type=int, default=20,
                   help="max attempts = factor * n_samples")

    # Picard sanity
    p.add_argument("--picard-sanity", action="store_true",
                   help="generate data and print solver stats, then exit if epochs==0")

    # Loss
    p.add_argument("--lambda-sup", type=float, default=1.0)
    p.add_argument("--lambda-pde", type=float, default=0.1)

    # Training control
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--predict-from", default="",
                   help="load checkpoint for prediction (not for resume training)")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="")
    p.add_argument("--predict-N", type=int, default=None)
    p.add_argument("--test-seed", type=int, default=1234)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-dir", default="results/figures")
    p.add_argument("--check-assembly", action="store_true",
                   help="compare fast vs slow assembly and exit")
    return p.parse_args()


# -- main -------------------------------------------------------------------

def main():
    args = parse_args()

    if args.check_assembly:
        print("=== assembly check: fast (COO) vs slow (lil) ===")
        N = 32
        h = 2.0 / (N + 1)
        rng = np.random.RandomState(42)
        a = 0.5 + rng.rand(N + 2, N + 2)
        g = 0.5 + rng.rand(N + 2, N + 2)
        f = rng.randn(N + 2, N + 2) * 0.1
        # Zero out interior of g (only boundary matters) and boundary of f
        g[1:-1, 1:-1] = 0.
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.

        A_fast, b_fast = _assemble_darcy_system(a, g, f, h)
        A_slow, b_slow = _assemble_darcy_system_slow_reference(a, g, f, h)

        A_err = abs(A_fast - A_slow).max()
        b_err = abs(b_fast - b_slow).max()
        print(f"  |A_fast - A_slow|_max = {A_err:.2e}")
        print(f"  |b_fast - b_slow|_max = {b_err:.2e}")

        u_fast = spla.spsolve(A_fast, b_fast)
        u_slow = spla.spsolve(A_slow, b_slow)
        u_err = abs(u_fast - u_slow).max()
        print(f"  |u_fast - u_slow|_max = {u_err:.2e}")

        tol = 1e-9
        ok = A_err < tol and b_err < tol and u_err < tol
        print(f"  {'PASS' if ok else 'FAIL'}  (threshold {tol:.0e})")
        return
        print("=" * 60)
        print("PICARD SANITY CHECK")
        print("=" * 60)
        print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")
        print(f"Picard: damping={args.picard_damping}  tol={args.picard_tol:.1e}  "
              f"max_iter={args.picard_max_iter}")
        print(f"Accept: u_min > {args.u_min_accept}  "
              f"residual < {args.max_residual_rmse}")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = _select_device(args.device)
        data = make_picard_dataset(args.N, args.train_samples, device, args)

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
        print(f"  ref residual RMSE = {ref_rmse:.4e}  (should be < {args.max_residual_rmse})")
        print(f"  boundary error    = {bnd_err:.2e}  (should be ~0)")

        if data.get("solver_info"):
            iters = [x["num_iter"] for x in data["solver_info"]]
            res_rmses = [x["residual_rmse"] for x in data["solver_info"]]
            print(f"  mean Picard iter  = {np.mean(iters):.1f}  max={np.max(iters)}")
            print(f"  mean solver res   = {np.mean(res_rmses):.4e}  "
                  f"max={np.max(res_rmses):.4e}")

        u_min, u_max = float(u_ex.min()), float(u_ex.max())
        fi = data["f"][:, :, 1:-1, 1:-1]
        print(f"  u_ref range = [{u_min:.4f}, {u_max:.4f}]")
        print(f"  f range     = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")

        if ref_rmse > args.max_residual_rmse:
            print(f"  WARNING: ref residual too large, adjust Picard params!")
        else:
            print(f"  OK — Picard solutions are accurate.")
        print("=" * 60)
        return  # sanity check done, skip training

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
        pa.predict_N = None
        pa.test_seed = args.seed
        run_prediction(pa)

    print(f"done: checkpoints/cnn/{key}")


if __name__ == "__main__":
    main()
