"""
Shared components for the variable-coefficient Poisson CNN pipelines
(Baseline_mms.py and Baseline_solver.py):

  -div(a(x,y) grad u) = f  on [-1,1]^2,  Dirichlet BC u = g,
  a(x,y) = exp(sigma*psi) a given smooth positive coefficient field.

Contains the grid / flux-form FD operator, the sparse FD solver and linear
baseline, the SimpleCNN and multi-kappa LGF unrolled models, normalization /
batching / forward helpers, training helpers, metrics, and generic plotting.
Dataset generation, experiment orchestration, CLI, and checkpoint layout stay
in the two entry scripts. Importing this module has no side effects (no
argument parsing, no data generation, no directory creation, no RNG seeding).
"""

import csv, math, os, random, tempfile, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from LGF_python.lgf import LGF_FFT

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

def enforce_dirichlet(u_raw, g, bnd):
    """Hard Dirichlet projection: u = bnd * g + (1-bnd) * u_raw (u may take
    either sign)."""
    return bnd * g + (1. - bnd) * u_raw

# -- variable-coefficient flux-form operator ---------------------------------

def flux_divergence(u, a, h):
    """5-point flux-form FD: -div(a(x,y) grad u), arithmetic-mean interface
    coefficients. u, a same shape (B,1,H,W); a is a given positive field.
    Returns interior nodes (B,1,H-2,W-2)."""
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
    """R = -div_h(a grad_h u) - f, interior nodes only."""
    return flux_divergence(u, a, h) - f[:, :, 1:-1, 1:-1]

# -- CNN --------------------------------------------------------------------

def _make_groupnorm(channels):
    """GroupNorm with up to 8 groups, reduced until it divides `channels`."""
    g = min(8, channels)
    while channels % g:
        g -= 1
    return nn.GroupNorm(g, channels)

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=6, out_ch=1, hidden_ch=32, n_layers=6):
        super().__init__()
        layers = []
        for i in range(n_layers):
            c_in = in_ch if i == 0 else hidden_ch
            d = 2 ** i
            layers += [
                nn.Conv2d(c_in, hidden_ch, 3, padding=d, dilation=d,
                          padding_mode="replicate"),
                _make_groupnorm(hidden_ch),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x):
        return self.head(self.body(x))

# -- LGFNet: multi-kappa LGF unrolled model ----------------------------------

LGF_BANK_DIR = Path("checkpoints") / "lgf_bank"
LGF_LOG_A_RANGE = (-1.5, 1.5)   # a_m in [exp(-1.5), exp(1.5)], covers sigma in [0.5, 1.5]

def build_lgf_kernel_bank(N, h, M, N_fft=512, cache_dir=LGF_BANK_DIR):
    """Build the fixed bank of M 3x3 LGF stencils K_m = h^2 * G_m[center-1:center+2,
    center-1:center+2], one per node a_m (log-equispaced on LGF_LOG_A_RANGE).
    LGF_FFT returns only the non-negative quadrant G[i,j] = G(i,j) (i,j >= 0); the
    LGF is symmetric under i -> -i and j -> -j, so the 3x3 neighborhood around the
    zero offset is recovered via K[di+1,dj+1] = G[|di|,|dj|]. Cached to .npz keyed
    by (N, M, N_fft, LGF_LOG_A_RANGE, h) so repeated runs skip regeneration, and
    changing N_fft/the node range/the physical mesh size h for a given N can't
    silently reuse a stale cache (h is always 2/(N+1) for the [-1,1]^2 domain
    every caller uses, but build_lgf_kernel_bank accepts h independently, so
    the key covers it too).
    Returns (bank (M,1,3,3) float32, a_nodes (M,) float32)."""
    cache_dir = Path(cache_dir)
    lo, hi = LGF_LOG_A_RANGE
    cache_path = cache_dir / f"bank_N{N}_M{M}_fft{N_fft}_r{lo:g}_{hi:g}_h{h:.8g}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return (torch.from_numpy(d["kernel_bank"]).float().unsqueeze(1),
                torch.from_numpy(d["a_nodes"]).float())

    log_a_nodes = np.linspace(lo, hi, M)
    a_nodes = np.exp(log_a_nodes)
    kernels = np.zeros((M, 3, 3), dtype=np.float64)
    for m, a_m in enumerate(a_nodes):
        G = LGF_FFT(h, float(a_m), N_fft)
        assert np.isclose(G[0, 1], G[1, 0], rtol=1e-8, atol=1e-10), (
            "LGF quadrant not symmetric about the zero offset")
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                kernels[m, di + 1, dj + 1] = G[abs(di), abs(dj)]
    kernels *= h * h

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, kernel_bank=kernels, a_nodes=a_nodes)
    return (torch.from_numpy(kernels).float().unsqueeze(1),
            torch.from_numpy(a_nodes).float())

def _select_kernel_weights(a, log_a_nodes):
    """Per-point bracket + log-space interpolation weight against the kernel-bank
    nodes. a: (B,1,H,W) raw coefficient field. log_a_nodes: (M,) ascending.
    Returns idx_lo, idx_hi (int64, shape of a), w (float, shape of a)."""
    M = log_a_nodes.shape[0]
    a_min, a_max = log_a_nodes[0].exp(), log_a_nodes[-1].exp()
    log_a = a.clamp(a_min, a_max).log()
    idx_lo = torch.searchsorted(log_a_nodes, log_a.contiguous()).clamp(1, M - 1) - 1
    idx_hi = (idx_lo + 1).clamp(max=M - 1)
    lo_val = log_a_nodes[idx_lo]
    hi_val = log_a_nodes[idx_hi]
    w = ((log_a - lo_val) / (hi_val - lo_val).clamp_min(1e-12)).clamp(0., 1.)
    return idx_lo, idx_hi, w

def _apply_kernel_bank(X, kernel_bank):
    """Apply all M fixed 3x3 kernels to every channel of X (B,C,H,W), zero padding.
    Returns (B,C,M,H,W)."""
    B, C, H, W = X.shape
    M = kernel_bank.shape[0]
    Zr = F.conv2d(X.reshape(B * C, 1, H, W), kernel_bank, padding=1)
    return Zr.view(B, C, M, H, W)

def _interp_kernel_response(Z, idx_lo, idx_hi, w):
    """Gather + log-space-weighted combine of the bracketing kernel responses.
    Z: (B,C,M,H,W); idx_lo/idx_hi/w: (B,1,H,W). Returns (B,C,H,W)."""
    C = Z.shape[1]
    idx_lo_e = idx_lo.unsqueeze(2).expand(-1, C, -1, -1, -1)
    idx_hi_e = idx_hi.unsqueeze(2).expand(-1, C, -1, -1, -1)
    Z_lo = torch.gather(Z, 2, idx_lo_e).squeeze(2)
    Z_hi = torch.gather(Z, 2, idx_hi_e).squeeze(2)
    return (1. - w) * Z_lo + w * Z_hi

class MultiKappaLGFBlock(nn.Module):
    """Fixed multi-kappa LGF spatial propagation + learnable 1x1 conv -> GN -> GELU
    -> residual. No learnable 3x3 spatial convs."""
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = _make_groupnorm(channels)
        self.act = nn.GELU()

    def forward(self, X, kernel_bank, idx_lo, idx_hi, w):
        Z = _apply_kernel_bank(X, kernel_bank)
        Zc = _interp_kernel_response(Z, idx_lo, idx_hi, w)
        return X + self.act(self.norm(self.proj(Zc)))

class LGFPhi(nn.Module):
    """1x1 lift -> [MultiKappaLGFBlock]^L -> 1x1 head."""
    def __init__(self, in_ch, hidden_ch, L):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, hidden_ch, 1)
        self.blocks = nn.ModuleList([MultiKappaLGFBlock(hidden_ch) for _ in range(L)])
        self.head = nn.Conv2d(hidden_ch, 1, 1)
        # zero-init: du=Phi(X)=0 at step 0, so the unrolled iteration starts as
        # an identity map (u_t raw-fed back each step would otherwise compound
        # through T steps and diverge before training has a chance to correct it)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, X, kernel_bank, idx_lo, idx_hi, w):
        h = self.lift(X)
        for blk in self.blocks:
            h = blk(h, kernel_bank, idx_lo, idx_hi, w)
        return self.head(h)

class LGFNet(nn.Module):
    """Unrolled residual iteration: spatial propagation via a fixed bank of 3x3
    LGF stencils of P_h(kappa) = -Delta_h + kappa^2 I, selected per grid point by
    the local coefficient kappa_ij^2 = a_ij. Only the 1x1 convs / GroupNorm /
    per-step step-size eta_t are learnable."""
    def __init__(self, hidden_ch=32, T=12, M=8, L=4, N=64, N_fft=512):
        super().__init__()
        self.T, self.M, self.L = T, M, L
        h = 2.0 / (N + 1)
        bank, a_nodes = build_lgf_kernel_bank(N, h, M, N_fft)
        assert torch.isfinite(bank).all(), "LGF kernel bank contains non-finite values"
        assert not bank.requires_grad
        self.register_buffer("kernel_bank", bank)          # (M,1,3,3), fixed
        self.register_buffer("a_nodes", a_nodes)            # (M,)
        self.register_buffer("log_a_nodes", a_nodes.log())
        self._bank_snapshot = bank.clone()

        self.phi = LGFPhi(in_ch=9, hidden_ch=hidden_ch, L=L)
        self.eta = nn.Parameter(torch.ones(T))

    def forward(self, batch, stats, collect_diagnostics=False):
        f, g, a, bnd = batch["f"], batch["g"], batch["a"], batch["mask"]
        x, y, h = batch["x"], batch["y"], batch["h"]
        B = f.shape[0]
        bnd_b, x_b, y_b = (_expand_to_batch(bnd, B), _expand_to_batch(x, B),
                          _expand_to_batch(y, B))
        f_norm = (f - stats.f_mean) / stats.f_std
        log_a_norm = (a.clamp_min(1e-12).log() - stats.loga_mean) / stats.loga_std
        idx_lo, idx_hi, w = _select_kernel_weights(a, self.log_a_nodes)

        u = enforce_dirichlet(torch.zeros_like(g), g, bnd)
        # res_trace is only computed when requested, and kept as tensors (no
        # per-step .item()) so it costs at most one device sync at the end
        # instead of T — .item() every step would force a GPU/MPS sync each
        # time, contaminating timed forward passes and wasting cost the
        # training loop never needs at all.
        res_trace = [] if collect_diagnostics else None
        for t in range(self.T):
            res = pde_residual(u, f, a, h)
            if collect_diagnostics:
                res_trace.append(res.detach().pow(2).mean().sqrt())
            r_full = torch.zeros_like(u)
            r_full[:, :, 1:-1, 1:-1] = -res
            q_norm = (r_full / a.clamp_min(1e-12)) / stats.f_std

            t_channel = torch.full_like(f, t / self.T)
            X = torch.cat([u, q_norm, f_norm, g, bnd_b, log_a_norm, x_b, y_b,
                           t_channel], dim=1)
            du = (1. - bnd) * self.phi(X, self.kernel_bank, idx_lo, idx_hi, w)
            u = enforce_dirichlet(u + self.eta[t] * du, g, bnd)

        if collect_diagnostics:
            # stays a device tensor here — callers that time this forward pass
            # convert to Python floats themselves, after timing stops, so the
            # host sync never lands inside a measured region.
            res_trace = torch.stack(res_trace)
        return (u, res_trace) if collect_diagnostics else u

def _lgf_const_coeff_sanity_check(N, M, N_fft=512, seed=0):
    """Constant-coefficient consistency: for a ≡ a_m (an interior bank node), the
    interpolated multi-kappa response must equal the single-kernel K_m response."""
    h = 2.0 / (N + 1)
    bank, a_nodes = build_lgf_kernel_bank(N, h, M, N_fft)
    log_a_nodes = a_nodes.log()
    m_test = M // 2
    a_const = a_nodes[m_test].item()

    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(1, 3, 16, 16, generator=gen)
    a_field = torch.full((1, 1, 16, 16), a_const)

    idx_lo, idx_hi, w = _select_kernel_weights(a_field, log_a_nodes)
    Z = _apply_kernel_bank(X, bank)
    Z_interp = _interp_kernel_response(Z, idx_lo, idx_hi, w)
    Z_direct = F.conv2d(X.reshape(3, 1, 16, 16), bank[m_test:m_test + 1],
                        padding=1).view(1, 3, 16, 16)

    diff = (Z_interp - Z_direct).abs().max().item()
    ok = torch.allclose(Z_interp, Z_direct, atol=1e-5, rtol=1e-5)
    print(f"  constant-coefficient consistency (a={a_const:.4f}, node {m_test}): "
          f"{'OK' if ok else 'FAILED'}  max diff={diff:.2e}")
    assert ok, "LGF multi-kappa interpolation does not reduce to the single-kernel response at a node"

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

def random_positive_field(N, sigma, M, rng=np.random):
    """Smooth log-normal positive coefficient field a(x,y)=exp(sigma*psi),
    psi a 2D Fourier field normalized to [-1,1]. sigma controls contrast:
    a in [exp(-sigma), exp(sigma)], dynamic range exp(2*sigma). sigma=0 ->
    a≡1 (degenerates to constant-coefficient Poisson). Returns full grid
    (N+2, N+2) float64."""
    psi = random_fourier_field(N, M, rng, normalize=True)
    return np.exp(sigma * psi)

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

def compute_linear_baselines(data):
    """Per-sample relL2 of the a≡1 linear solve vs u_ref (one sparse solve per
    sample)."""
    h = float(data["h"])
    f = data["f"].detach().cpu().numpy()
    g = data["g"].detach().cpu().numpy()
    u_ref = data["u_ref"].detach().cpu().numpy()
    interior = data["interior_mask"].detach().cpu().numpy()[0, 0]
    return np.array([_linear_baseline_rl2(f[i, 0], g[i, 0], u_ref[i, 0], interior, h)
                      for i in range(f.shape[0])], dtype=np.float64)

def ensure_linear_baselines(data):
    """Lazily compute + cache the per-sample linear (a≡1) baseline on this
    dataset dict, keyed by object identity (mutates `data` in place). Only
    called where the baseline is actually consumed (val/predict/test suites);
    training data never reads it, so it's never computed for it. Cached so
    repeated evaluate() calls on the same val_data don't re-run the sparse
    solve every time."""
    if "linear_baseline_rl2" not in data:
        data["linear_baseline_rl2"] = torch.from_numpy(
            compute_linear_baselines(data)).float().to(data["f"].device)
    return data["linear_baseline_rl2"]

def trivial_linear_baseline(data):
    """Mean relL2 of a≡1 linear solve vs u_ref.  CNN must beat this."""
    return float(ensure_linear_baselines(data).mean().item())

# -- dataset sanity helpers ---------------------------------------------------

def ref_residual_rmse64(data):
    """Residual RMSE of the stored (float32) reference data, computed in
    float64, so the ~h^-2*eps32 rounding floor is measured, not hidden.
    Training itself stays float32."""
    u = data["u_ref"].detach().cpu().double()
    f = data["f"].detach().cpu().double()
    a = data["a"].detach().cpu().double()
    return float(pde_residual(u, f, a, data["h"])
                 .pow(2).mean().sqrt())

def _uref_range_and_bnd_err(data):
    """(min, max) of u_ref and max |u_ref - g| on the boundary — dataset
    sanity numbers printed by train() and the sanity checks."""
    u = data["u_ref"]
    bnd_err = float(((u - data["g"]).abs() * data["mask"].expand_as(u)).max())
    return float(u.min()), float(u.max()), bnd_err

# -- input / stats / batching ------------------------------------------------

def _expand_to_batch(t, B):
    """Broadcast a shared (1,C,H,W) field (mask/x/y/...) to batch size B;
    no-op if it's already batched."""
    return t.expand(B, -1, -1, -1) if t.shape[0] == 1 else t

def build_input(f_norm, g, bnd, a_norm, x, y):
    """Assemble 6-channel CNN input: [f_norm, g, bnd, log_a_norm, x, y].
    a_norm is a full-grid field (B,1,H,W)."""
    B = f_norm.shape[0]
    return torch.cat([f_norm, g, _expand_to_batch(bnd, B), _expand_to_batch(a_norm, B),
                      _expand_to_batch(x, B), _expand_to_batch(y, B)], dim=1)

@dataclass
class NormalizationStats:
    f_mean: float
    f_std: float
    loga_mean: float
    loga_std: float

def compute_stats(data):
    fi = data["f"][:, :, 1:-1, 1:-1]
    loga = data["a"].clamp_min(1e-12).log()
    return NormalizationStats(
        f_mean=fi.mean().item(),
        f_std=fi.std(unbiased=False).clamp_min(1e-8).item(),
        loga_mean=loga.mean().item(),
        loga_std=loga.std(unbiased=False).clamp_min(1e-8).item())

def _take_batch(data, idx):
    keys = ("f", "g", "a", "sigma", "u_ref")
    if "linear_baseline_rl2" in data:   # absent for training data, which never reads it
        keys = keys + ("linear_baseline_rl2",)
    return {"h": data["h"],
            **{k: data[k][idx] for k in keys},
            **{k: data[k][:1] for k in ("mask", "interior_mask", "x", "y")}}

def _forward(model, batch, stats, return_diagnostics=False):
    if isinstance(model, LGFNet):
        return model(batch, stats, collect_diagnostics=return_diagnostics)
    fn = (batch["f"] - stats.f_mean) / stats.f_std
    loga_n = (batch["a"].clamp_min(1e-12).log() - stats.loga_mean) / stats.loga_std
    inp = build_input(fn, batch["g"], batch["mask"], loga_n, batch["x"], batch["y"])
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

def _time_forward(model, batch, stats, return_diagnostics=False):
    """CNN inference wall time (model forward only, after one warm-up forward).
    Returns (u_pred, ms_per_batch, ms_per_sample), or with
    return_diagnostics=True on an LGFNet, (u_pred, ms_per_batch, ms_per_sample,
    res_trace). Note: with return_diagnostics=True the timed region also
    computes the per-step residual-RMS trace, so it measures inference +
    diagnostics, not pure inference."""
    device = batch["f"].device
    with torch.no_grad():
        _forward(model, batch, stats)          # warm-up
        _sync(device)
        t0 = time.perf_counter()
        result = _forward(model, batch, stats, return_diagnostics=return_diagnostics)
        _sync(device)
        ms = (time.perf_counter() - t0) * 1e3
    ms_per_sample = ms / batch["f"].shape[0]
    if return_diagnostics and isinstance(model, LGFNet):
        u_pred, res_trace = result
        return u_pred, ms, ms_per_sample, res_trace.cpu().tolist()   # after timing stops
    return result, ms, ms_per_sample

# -- training helpers ---------------------------------------------------------

def create_model(args, device):
    if getattr(args, "model", "cnn") == "lgf":
        return LGFNet(hidden_ch=args.base_ch, T=args.lgf_T, M=args.lgf_M,
                      L=args.lgf_L, N=args.N).to(device)
    return SimpleCNN(hidden_ch=args.base_ch,
                      n_layers=getattr(args, "n_layers", 6)).to(device)

def create_optimizer(model, args):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith(".bias") or p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.},
    ], lr=args.lr)

def compute_loss(u_pred, batch, args):
    if args.loss == "supervised":
        interior = batch["interior_mask"].expand_as(u_pred)
        err = (u_pred - batch["u_ref"]) * interior
        return err.pow(2).mean()
    # physics: per-sample normalized PDE residual
    res = pde_residual(u_pred, batch["f"], batch["a"], batch["h"])
    f_int = batch["f"][:, :, 1:-1, 1:-1]
    scale = f_int.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1.0)
    return ((res / scale) ** 2).mean()

def train_one_epoch(model, train_data, stats, opt, sched, args, device):
    model.train()
    n_train = train_data["f"].shape[0]
    perm = torch.randperm(n_train, device=device)
    loss_sum, sample_count = 0., 0

    for start in range(0, n_train, args.batch_size):
        batch = _take_batch(train_data, perm[start:start + args.batch_size])
        u_pred = _forward(model, batch, stats)
        loss = compute_loss(u_pred, batch, args)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        bsz = batch["f"].shape[0]
        loss_sum += loss.detach().item() * bsz   # sample-weighted: last (possibly
        sample_count += bsz                       # partial) batch doesn't get equal say

    return loss_sum / sample_count

# -- evaluation -------------------------------------------------------------

def _print_res_trace(trace, label="LGFNet per-step residual RMS"):
    print(f"  {label}: " + " ".join(f"{v:.3e}" for v in trace))

def _aggregate_metrics(all_rows):
    """Aggregate per-sample metric rows (residuals use RMS, since all samples
    share one grid). Shared by evaluate() and run_prediction() so a
    single-shot prediction call doesn't need a second forward pass just to
    reach this aggregation."""
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

@torch.no_grad()
def evaluate(model, data, stats, batch_size):
    model.eval()
    ensure_linear_baselines(data)   # cached on `data` — computed once, reused by every call
    n = data["f"].shape[0]
    all_rows = []
    res_traces = []

    for start in range(0, n, batch_size):
        b = _take_batch(data, slice(start, start + batch_size))
        if isinstance(model, LGFNet):
            up, res_trace = _forward(model, b, stats, return_diagnostics=True)
            res_traces.append((res_trace.cpu().tolist(), up.shape[0]))
        else:
            up = _forward(model, b, stats)
        rows = per_sample_metrics(up, b, start_id=start)
        all_rows.extend(rows)

    if res_traces:
        # RMS-of-RMS combine (each per-batch trace value is itself an RMS over
        # that batch), weighted by batch size so an uneven last batch doesn't
        # get equal say — same convention as _aggregate_metrics' rms().
        total = sum(bsz for _, bsz in res_traces)
        mean_trace = [math.sqrt(sum(bsz * trace[i] ** 2 for trace, bsz in res_traces) / total)
                      for i in range(len(res_traces[0][0]))]
        _print_res_trace(mean_trace)

    return _aggregate_metrics(all_rows)

def summarize_rows(rows, metric_keys):
    """Aggregate per-sample metric rows into mean/std/min/max per metric."""
    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in rows]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "min": float(np.min(vals)), "max": float(np.max(vals))}
    return summary

def per_sample_metrics(u_pred, batch, start_id=0):
    """Per-sample metrics: all 12 quantities computed sample-by-sample.
    Returns a list of per-sample dicts (use summarize_rows() to aggregate,
    if/when a caller actually needs the summary)."""
    ue = batch["u_ref"]
    interior = batch["interior_mask"].expand_as(u_pred)
    bnd_mask = batch["mask"].expand_as(u_pred)
    h = float(batch["h"])
    n_samples = u_pred.shape[0]

    err = (u_pred - ue) * interior
    res = pde_residual(u_pred, batch["f"], batch["a"], batch["h"])
    res_ref = pde_residual(ue, batch["f"], batch["a"], batch["h"])

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
        # projection — enforce_dirichlet would distort the constant otherwise)
        m01 = batch["mask"][:1]
        const_pred = m01 * batch["g"][i:i+1] + (1. - m01) * u_flat[i].mean()
        const_err = ((const_pred - ue[i:i+1]) * interior[i:i+1])[:, :, 1:-1, 1:-1].flatten(1)
        const_rl2 = (const_err[0].norm() / u_flat[i].norm().clamp_min(1e-12)).item()

        rows.append({
            "sample_id": start_id + i,
            "sigma": batch["sigma"][i].item(),
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
            "linearBaselineRelL2": batch["linear_baseline_rl2"][i].item(),
            "f_RMSE": f_flat[i].pow(2).mean().sqrt().item(),
        })

    return rows


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

def _save_sample_figures(batch, u_pred, i, out_dir, stem, title):
    """Panel + centerline figures for sample i of a batch (shared by the
    test suites and --plot prediction)."""
    h = float(batch["h"])
    fext = [-1., 1., -1., 1.]
    rext = [-1. + h, 1. - h, -1. + h, 1. - h]
    ui = batch["u_ref"][i, 0]
    up = u_pred[i, 0]
    pr = pde_residual(u_pred[i:i+1], batch["f"][i:i+1], batch["a"][i:i+1], h)[0, 0]
    er = pde_residual(batch["u_ref"][i:i+1], batch["f"][i:i+1],
                      batch["a"][i:i+1], h)[0, 0]
    _save_panel(ui, up, up - ui, batch["f"][i, 0], pr, er,
                out_dir / f"{stem}_panel.png", title, fext, rext)
    _save_centerline(ui, up, out_dir / f"{stem}_centerline.png", title)

def _save_convergence_plot(history, path, loss_type="physics", dataset_label=""):
    if len(history) < 2:
        return
    plt = _setup_plot()

    epochs_list = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"Training convergence — {dataset_label} ({loss_type})")

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
