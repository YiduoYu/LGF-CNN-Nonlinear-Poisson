"""
CNN solver for -div(u^k grad u) = f on [-1,1]^2, Dirichlet BC u=g.
a(u,k) = u^k.  MMS data: u_ref -> f = -div_h(a grad_h u_ref).
k integer in {0,1,2,3}, u strictly positive (u >= U_EPS). Supports
--loss {supervised, physics}.
"""

import argparse, csv, json, math, os, random, tempfile, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla

U_EPS = 1e-15  # strict positivity floor: every constructed u has min(u) == U_EPS

# per-sample metric columns written to metrics.csv / summary.json
METRIC_KEYS = ["relL2", "RMSE", "MAE", "maxError", "predResRMSE", "refResRMSE",
               "bndErr", "gradRelL2", "gradLinf", "constBaselineRelL2",
               "linearBaselineRelL2", "f_RMSE"]

# -- grid -------------------------------------------------------------------

def make_grid(N, device="cpu", dtype=torch.float32):
    """Uniform grid on [-1,1]^2: returns x, y, boundary_mask, interior_mask, h."""
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

def enforce_dirichlet(u_raw, g, bnd, u_floor=1e-4):
    """Hard Dirichlet projection: u = bnd * g + (1-bnd) * u_raw. Interior
    values are forced positive via softplus so a(u,k) = u^k stays well-defined."""
    u_int = u_floor + F.softplus(u_raw)
    return bnd * g + (1. - bnd) * u_int

# -- nonlinear flux-form operator -------------------------------------------

def nonlinear_flux_divergence(u, k, h):
    """5-point flux-form FD: -div(a(u,k) grad u) with arithmetic-mean
    interface coefficients a = u^k."""
    if isinstance(k, (int, float)):
        k = torch.tensor([k], device=u.device)
    if k.ndim == 1:
        k = k.view(-1, 1, 1, 1)

    a = u.pow(k.float())    # a(u,k) = u^k (u > 0)

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
            d = 2 ** i
            layers += [
                nn.Conv2d(c_in, hidden_ch, 3, padding=d, dilation=d,
                          padding_mode="replicate"),
                nn.GroupNorm(g, hidden_ch),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x):
        return self.head(self.body(x))

# -- Fourier field generator ------------------------------------------------

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

# -- manufactured solution --------------------------------------------------

def manufactured_u(x, y, M, rng=np.random):
    """u = b(x,y) + (1-x^2)(1-y^2)*q(x,y), shifted so min(u) == U_EPS.

    q(x,y) is a normalized 2D Fourier series (interior perturbation).
    x, y are torch tensors; returns a CPU float64 tensor."""
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    N = x_np.shape[-1] - 2

    phi = (1. - x_np ** 2) * (1. - y_np ** 2)

    # Boundary polynomial: b(x,y) = b0 + b1*x + b2*y + b3*x*y
    b0 = rng.uniform(-0.25, 0.25)
    b1, b2 = rng.uniform(-0.15, 0.15, 2)
    b3 = rng.uniform(-0.1, 0.1)

    boundary_part = b0 + b1 * x_np + b2 * y_np + b3 * x_np * y_np

    fourier_part = random_fourier_field(N, M, rng, normalize=True)

    u_signed = boundary_part + 0.36 * phi * fourier_part
    u = u_signed - u_signed.min() + U_EPS   # strictly positive, min == U_EPS
    return torch.from_numpy(u)

# -- MMS dataset ------------------------------------------------------------

def make_mms_dataset(N, n_samples, device, args, seed):
    """Build MMS dataset: u_ref -> f = -div_h(u_ref^k grad_h u_ref) in float64, store float32."""
    rng = np.random.RandomState(seed)
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    x64, y64, *_ = make_grid(N, "cpu", torch.float64)
    f_list, g_list, k_list, u_list = [], [], [], []
    res64_sq = 0.

    M = getattr(args, "fourier_m", 6)
    for _ in range(n_samples):
        if getattr(args, "random_k", False):
            k_i = int(rng.randint(args.k_min, args.k_max + 1))
        else:
            k_i = int(args.k)

        u64 = manufactured_u(x64[0, 0], y64[0, 0], M, rng)
        u64 = u64.unsqueeze(0).unsqueeze(0)          # CPU float64
        k64 = torch.tensor([k_i], dtype=torch.float64)
        f64 = torch.zeros_like(u64)
        f64[:, :, 1:-1, 1:-1] = nonlinear_flux_divergence(u64, k64, h)
        g64 = u64.clone()
        g64[:, :, 1:-1, 1:-1] = 0.
        res64_sq += float(nonlinear_pde_residual(
            u64, f64, k64, h).pow(2).sum())

        f_list.append(f64.float().to(device))
        g_list.append(g64.float().to(device))
        k_list.append(torch.tensor([k_i], dtype=torch.float32, device=device))
        u_list.append(u64.float().to(device))

    return {
        "f": torch.cat(f_list), "g": torch.cat(g_list),
        "k": torch.cat(k_list), "u_ref": torch.cat(u_list),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h,
        "ref_res64": math.sqrt(res64_sq / (n_samples * N * N)),
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

# -- sparse linear FD solver / trivial baseline ------------------------------

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

def _linear_baseline_rl2(f, g, u_ref, interior, h):
    """relL2 of the a≡1 linear Poisson solve vs u_ref (numpy, one sample)."""
    u_lin = solve_linear_fd(f, g, np.ones_like(g), h)
    err = (u_lin - u_ref) * interior
    ref = u_ref * interior
    return float(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-12))

def trivial_linear_baseline(data):
    """Mean relL2 of a≡1 linear solve vs nonlinear u_ref.  CNN must beat this."""
    h = float(data["h"])
    f = data["f"].detach().cpu().numpy()
    g = data["g"].detach().cpu().numpy()
    u_ref = data["u_ref"].detach().cpu().numpy()
    interior = data["interior_mask"].detach().cpu().numpy()[0, 0]
    rl2 = [_linear_baseline_rl2(f[i, 0], g[i, 0], u_ref[i, 0], interior, h)
           for i in range(f.shape[0])]
    return float(np.mean(rl2))

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

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    _set_seed(args.seed)
    device = _select_device(args.device)

    print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}  |  data=mms")
    if args.random_k:
        print(f"  random k in [{args.k_min}, {args.k_max}]")

    print(f"generating {args.train_samples} MMS samples (N={args.N})")
    train_data = make_mms_dataset(args.N, args.train_samples, device, args,
                                  seed=args.seed)
    val_data = make_mms_dataset(args.N, args.val_samples, device, args,
                                seed=args.seed + 1)
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
    print(f"MMS residual RMSE (float64) = {ref_res:.4e}  (should be near 0)")
    print(f"u_ref range: [{u_min:.4f}, {u_max:.4f}]  "
          f"boundary error = {bnd_err:.2e}")

    triv_rl2 = trivial_linear_baseline(val_data)
    print(f"trivial baseline (a≡1 linear Poisson) relL2 = {triv_rl2:.4e}  "
          f"(CNN must beat this to have learned a(u,k))")

    if model is None:
        model = SimpleCNN(hidden_ch=args.base_ch, n_layers=args.n_layers).to(device)
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
            val_m = evaluate(model, val_data, stats, eval_bs)
            val_key = "relL2" if args.loss == "supervised" else "resRMSE"
            if val_m[val_key] < best_metric:
                best_metric = val_m[val_key]
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_type": "cnn_mms",
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

            history.append({"epoch": epoch, "loss": loss.detach().item(),
                            **{f"val_{key}": val_m[key] for key in (
                                "resRMSE", "relL2", "maxError", "gradRelL2",
                                "meanMaxError", "meanBoundaryError",
                                "maxBoundaryError", "meanGradLinf",
                                "maxGradLinf", "refResRMSE",
                                "linearBaselineRelL2")}})

    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "cnn_mms",
                 "model_state_dict": model.state_dict(),
                 "stats": stats, "args": vars(args)}, last_path)
    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, args.loss)
    return model, best_metric

# -- evaluation -------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, stats, batch_size):
    model.eval()
    n = data["f"].shape[0]
    all_rows = []

    for start in range(0, n, batch_size):
        b = _take_batch(data, slice(start, start + batch_size))
        up = _forward(model, b, stats)
        rows, _ = per_sample_metrics(up, b, start_id=start)
        all_rows.extend(rows)

    # aggregate per-sample rows; residuals use RMS (all samples share one grid)
    mean = lambda key: sum(r[key] for r in all_rows) / len(all_rows)
    rms = lambda key: math.sqrt(sum(r[key] ** 2 for r in all_rows) / len(all_rows))
    vmax = lambda key: max(r[key] for r in all_rows)
    return {"relL2": mean("relL2"),
            "maxError": vmax("maxError"), "meanMaxError": mean("maxError"),
            "meanBoundaryError": mean("bndErr"), "maxBoundaryError": vmax("bndErr"),
            "meanGradLinf": mean("gradLinf"), "maxGradLinf": vmax("gradLinf"),
            "resRMSE": rms("predResRMSE"), "refResRMSE": rms("refResRMSE"),
            "gradRelL2": mean("gradRelL2"),
            "linearBaselineRelL2": mean("linearBaselineRelL2"),
            "constBaselineRelL2": mean("constBaselineRelL2"),
            "per_sample": all_rows}

def summarize_rows(rows, metric_keys):
    """Aggregate per-sample metric rows into mean/std/min/max per metric."""
    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in rows]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "min": float(np.min(vals)), "max": float(np.max(vals))}
    return summary

def per_sample_metrics(u_pred, batch, start_id=0):
    """Per-sample metrics: all 12 quantities computed sample-by-sample then averaged.

    Returns (list_of_dicts, summary_dict) where summary = {metric: {mean, std, min, max}}."""
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
        # constant baseline: predict mean(u_ref) on interior (direct
        # projection — enforce_dirichlet would distort the constant via softplus)
        m01 = batch["mask"][:1]
        const_pred = m01 * batch["g"][i:i+1] + (1. - m01) * u_flat[i].mean()
        const_err = ((const_pred - ue[i:i+1]) * interior[i:i+1])[:, :, 1:-1, 1:-1].flatten(1)
        const_rl2 = (const_err[0].norm() / u_flat[i].norm().clamp_min(1e-12)).item()

        rows.append({
            "sample_id": start_id + i,
            "k": batch["k"][i].item(),
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

    # linear baseline: a=1 Poisson solve per sample
    f_np = batch["f"].detach().cpu().numpy()
    g_np = batch["g"].detach().cpu().numpy()
    ue_np = ue.detach().cpu().numpy()
    interior_np = interior.detach().cpu().numpy()
    for i, row in enumerate(rows):
        row["linearBaselineRelL2"] = _linear_baseline_rl2(
            f_np[i, 0], g_np[i, 0], ue_np[i, 0], interior_np[i, 0], h)

    return rows, summarize_rows(rows, METRIC_KEYS)


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
    fig.suptitle(f"Training convergence — MMS ({loss_type})")

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

# -- analytic test suite ----------------------------------------------------

# Raw analytic formulas (signed); shifted+rescaled to strictly positive u
# below via ANALYTIC_RANGE so a(u,k) = u^k stays well-defined for any k >= 0.
ANALYTIC_CASES = {
    "u1": lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y),
    "u2": lambda x, y: (1. - x ** 2) * (1. - y ** 2),
    "u3": lambda x, y: np.cos(np.pi * x) * np.cos(2. * np.pi * y),
    "u4": lambda x, y: x ** 2 * (1. - y) + np.sin(np.pi * x) * y,
}

# Exact (min, max) of the raw formulas above over [-1,1]^2 (analytically
# verified). Used to shift+rescale to [U_EPS, 1+U_EPS]: shifting alone would
# leave some cases spanning up to 2, which blows up a=u^k's dynamic range
# (up to 2^k vs the intended ~1^k) and destabilizes CNN training.
ANALYTIC_RANGE = {"u1": (-1.0, 1.0), "u2": (0.0, 1.0), "u3": (-1.0, 1.0), "u4": (-1.0, 2.0)}

def make_analytic_dataset(N, case, k_val, device):
    """Build a test dataset from an analytic u(x,y). k integer."""
    k_val = int(k_val)
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    x_np = x_t[0, 0].cpu().numpy()
    y_np = y_t[0, 0].cpu().numpy()
    u_func = ANALYTIC_CASES[case]

    u_np = u_func(x_np, y_np).astype(np.float64)
    lo, hi = ANALYTIC_RANGE[case]
    u_np = (u_np - lo) / (hi - lo) + U_EPS   # shift+rescale to [U_EPS, 1+U_EPS]
    u64 = torch.from_numpy(u_np)[None, None]          # CPU float64
    k64 = torch.tensor([k_val], dtype=torch.float64)
    f64 = torch.zeros_like(u64)
    f64[:, :, 1:-1, 1:-1] = nonlinear_flux_divergence(
        u64, k64, h)
    g_np = u_np.copy()
    g_np[1:-1, 1:-1] = 0.

    res64 = float(nonlinear_pde_residual(u64, f64, k64, h).pow(2).mean().sqrt())

    return {
        "f": f64.float().to(device), "g": torch.from_numpy(g_np).float()[None, None].to(device),
        "k": torch.tensor([k_val], dtype=torch.float32, device=device),
        "u_ref": u64.float().to(device),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h,
        "ref_res64": res64,   # float64 residual of (u_ref, f): machine precision by construction
    }

def run_analytic_suite(model, stats, device, args, ckpt_args, out_dir):
    """Run all analytic test cases, save metrics + figures."""
    N = ckpt_args.N
    all_rows = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = 2.0 / (N + 1)
    fext = [-1., 1., -1., 1.]
    rext = [-1. + h, 1. - h, -1. + h, 1. - h]

    k_test = int(args.predict_k)
    cnn_ms_all = []
    for case_name in ANALYTIC_CASES:
        print(f"\n  --- analytic case: {case_name}  (k={k_test:g}) ---")
        batch = make_analytic_dataset(N, case_name, k_test, device)
        u_pred, _, cnn_ms_sample = _time_forward(model, batch, stats)
        cnn_ms_all.append(cnn_ms_sample)
        rows, summary = per_sample_metrics(u_pred, batch)
        for r in rows:
            r["case"] = case_name
            r["cnn_ms_per_sample"] = cnn_ms_sample
            r["fd_solver_ms_per_sample"] = float("nan")   # no FD solve in MMS pipeline
        all_rows.extend(rows)

        print(f"    relL2={rows[0]['relL2']:.4e}  maxErr={rows[0]['maxError']:.4e}  "
              f"resRMSE={rows[0]['predResRMSE']:.4e}  "
              f"linBL={rows[0]['linearBaselineRelL2']:.4e}")
        print(f"    ref residual (float64) = {batch['ref_res64']:.4e}")

        # analytic suite always saves figures (only reached with predict_case=analytic)
        ui = batch["u_ref"][0, 0]
        up = u_pred[0, 0]
        pr = nonlinear_pde_residual(
            u_pred[0:1], batch["f"][0:1], batch["k"][0:1], batch["h"])[0, 0]
        er = nonlinear_pde_residual(
            batch["u_ref"][0:1], batch["f"][0:1], batch["k"][0:1], batch["h"])[0, 0]
        _save_panel(ui, up, up - ui, batch["f"][0, 0], pr, er,
                    out_dir / f"analytic_{case_name}_panel.png",
                    f"analytic {case_name}  k={rows[0]['k']:.2f}  "
                    f"relL2={rows[0]['relL2']:.3e}", fext, rext)
        _save_centerline(ui, up, out_dir / f"analytic_{case_name}_centerline.png",
                         f"analytic {case_name}  k={rows[0]['k']:.2f}  "
                         f"relL2={rows[0]['relL2']:.3e}")

    print(f"\n  timing: cnn_ms_per_sample={np.mean(cnn_ms_all):.3f}  "
          f"fd_solver_ms_per_sample=nan (no FD solve in MMS pipeline)")
    _write_csv(all_rows, out_dir / "analytic_metrics.csv")
    metric_keys = ["relL2", "maxError", "predResRMSE", "refResRMSE",
                   "bndErr", "gradRelL2", "constBaselineRelL2", "linearBaselineRelL2"]
    summary_all = summarize_rows(all_rows, metric_keys)
    with open(out_dir / "analytic_summary.json", "w") as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)
    print(f"  analytic suite saved → {out_dir}")
    return all_rows

# -- prediction -------------------------------------------------------------

def run_prediction(args):
    if not args.predict_from:
        raise ValueError("--predict-from required")

    _set_seed(args.seed)
    device = _select_device(args.device)

    ckpt = torch.load(args.predict_from, map_location=device, weights_only=False)
    ckpt_type = ckpt.get("model_type")
    if ckpt_type and ckpt_type != "cnn_mms":
        raise ValueError(
            f"checkpoint type mismatch: expected 'cnn_mms', got '{ckpt_type}'. "
            f"Use the correct prediction script (Baseline_mms.py for "
            f"'cnn_mms', Baseline_solver.py for 'cnn').")
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
    data_args.random_k = False
    data_args.k = args.predict_k

    if args.predict_case == "analytic":
        out = (Path(args.plot_dir) if args.plot_dir
               else Path("results") / "predict_mms" / "analytic")
        run_analytic_suite(model, stats, device, args, ckpt_args, out)
        return

    assert args.test_seed not in (ckpt_args.seed, ckpt_args.seed + 1), (
        "test seed collides with a train/val seed! prediction data would "
        "overlap the training run (evaluation pollution)."
    )

    print(f"prediction (MMS): N={N}  samples={args.predict_samples}  "
          f"k={args.predict_k:g} (fixed)")

    batch = make_mms_dataset(N, args.predict_samples, device, data_args,
                             seed=args.test_seed)

    u_pred, cnn_ms_batch, cnn_ms_sample = _time_forward(model, batch, stats)

    vm = evaluate(model, batch, stats, args.predict_samples)
    print(f"relL2={vm['relL2']:.4e}  maxErr={vm['maxError']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  refRMSE={vm['refResRMSE']:.4e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")
    print(f"meanMaxError={vm['meanMaxError']:.4e}")
    print(f"meanBoundaryError={vm['meanBoundaryError']:.2e}  maxBoundaryError={vm['maxBoundaryError']:.2e}")
    print(f"meanGradLinf={vm['meanGradLinf']:.4e}  maxGradLinf={vm['maxGradLinf']:.4e}")
    print(f"reference residual RMSE (float64) = {batch['ref_res64']:.4e}")
    print(f"linear baseline (a≡1) relL2 = {vm['linearBaselineRelL2']:.4e}  "
          f"(CNN must beat this to have learned a(u,k))  "
          f"const baseline relL2 = {vm['constBaselineRelL2']:.4e}")
    print(f"timing: cnn={cnn_ms_batch:.2f} ms/batch  "
          f"cnn_ms_per_sample={cnn_ms_sample:.3f}  "
          f"fd_solver_ms_per_sample=nan (no FD solve in MMS pipeline)")

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "k": batch["k"].cpu(), "u_ref": batch["u_ref"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]), "metrics": vm,
        }, p)
        print(f"saved predictions → {p}")

    rows = vm["per_sample"]
    for r in rows:
        r["cnn_ms_per_sample"] = cnn_ms_sample
        r["fd_solver_ms_per_sample"] = float("nan")   # no FD solve in MMS pipeline
    summary = summarize_rows(rows, METRIC_KEYS)

    # Always save metrics — use save dir parent, plot-dir, or default
    if args.predict_save:
        out = Path(args.predict_save).parent
    elif args.plot_dir:
        out = Path(args.plot_dir)
    else:
        out = Path("results") / "predict_mms"
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
    if getattr(args, "random_k", False):
        k_str = f"k{args.k_min:g}-{args.k_max:g}"
    else:
        k_str = f"k{args.k:g}"
    n_layers = getattr(args, "n_layers", 6)
    loss_suffix = "_sup" if getattr(args, "loss", "physics") == "supervised" else ""
    return (f"N{args.N}_e{args.epochs}_tr{args.train_samples}"
            f"_{k_str}_M{getattr(args, 'fourier_m', 6)}"
            f"_ch{args.base_ch}_L{n_layers}{loss_suffix}_mms")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--train-samples", type=int, default=32)
    p.add_argument("--val-samples", type=int, default=8)
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
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--random-k", action=argparse.BooleanOptionalAction, default=True,
                   help="sample k uniformly from [k_min, k_max] (default: True; use --no-random-k to fix)")
    p.add_argument("--k-min", type=int, default=0)
    p.add_argument("--k-max", type=int, default=3)
    p.add_argument("--fourier-m", type=int, default=6,
                   help="number of Fourier modes per dimension for random fields")

    # Loss
    p.add_argument("--loss", default="physics",
                   choices=["physics", "supervised"],
                   help="physics: PDE residual only | supervised: |u_pred - u_ref|^2")

    # MMS sanity
    p.add_argument("--mms-sanity", action="store_true",
                   help="verify MMS residual, then exit")

    # Training control
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--predict-from", default="")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="")
    p.add_argument("--test-seed", type=int, default=9999)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--predict-case", default="data",
                   choices=["data", "analytic"],
                   help="'data': same MMS family | 'analytic': 4 shifted-positive analytic test cases")
    p.add_argument("--predict-k", type=int, default=2,
                   help="fixed integer k for prediction")
    p.add_argument("--plot-dir", default="",
                   help="directory for output figures and metrics CSVs")
    return p.parse_args()

# -- main -------------------------------------------------------------------

def main():
    args = parse_args()

    if args.mms_sanity:
        print("=" * 60)
        print("MMS SANITY CHECK")
        print("=" * 60)
        print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")

        _set_seed(args.seed)
        device = _select_device(args.device)
        data = make_mms_dataset(args.N, args.train_samples, device, args,
                                seed=args.seed)

        exact_rmse = data["ref_res64"]
        ref32in64 = ref_residual_rmse64(data)
        with torch.no_grad():
            u_ex = data["u_ref"]
            bnd_err = float(
                ((u_ex - data["g"]).abs()
                 * data["mask"].expand_as(u_ex)).max())
            u_min, u_max = float(u_ex.min()), float(u_ex.max())
            fi = data["f"][:, :, 1:-1, 1:-1]

        print(f"  exact_resRMSE (float64) = {exact_rmse:.4e}  (should be near 0)")
        print(f"  float32 data residual   = {ref32in64:.4e}  "
              f"(~h^-2·eps32 rounding floor)")
        print(f"  boundary error  = {bnd_err:.2e}  (should be 0)")
        print(f"  u_ref range     = [{u_min:.4f}, {u_max:.4f}]")
        print(f"  f range         = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")
        print(f"  trivial baseline (a≡1) relL2 = {trivial_linear_baseline(data):.4e}")

        if exact_rmse > 1e-8:
            print(f"  WARNING: residual = {exact_rmse:.4e} > 1e-8, check FD!")
        else:
            print("  OK — MMS residual is near machine precision.")
        print("=" * 60)
        return

    if args.predict:
        return run_prediction(args)

    key = ckpt_key(args)
    ckpt_dir = Path("checkpoints") / "cnn_mms" / key
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
        figures_dir = Path("results") / "cnn_mms" / key / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        pa = argparse.Namespace(**vars(args))
        pa.predict_from = str(best_path)
        pa.predict_save = ""
        pa.plot = True
        pa.plot_dir = str(figures_dir)
        pa.test_seed = args.seed + 100000   # disjoint from train RNG stream
        run_prediction(pa)

    print(f"done: checkpoints/cnn_mms/{key}")

if __name__ == "__main__":
    main()
