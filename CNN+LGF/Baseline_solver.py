"""
CNN solver for -div(a(x,y) grad u) = f on [-1,1]^2, Dirichlet BC u=g.
a(x,y) = exp(sigma*psi) is a given smooth positive coefficient field
(sigma controls contrast). Linear elliptic problem: forward solves reduce
to a single sparse linear solve; u may take either sign.

Data generation: random f,g,a(sigma) -> direct sparse linear solve -> u_ref.
Supports --loss {supervised, physics} and --model {cnn, lgf}.

This is a self-contained build (see the import section below); it owns the
random-f,g solver data generation, the fixed-solver test suite, and the cnn
checkpoint/output layout.
"""

# Self-contained build: the shared grid/operator/model/metric/training helpers
# (factored into Baseline_core.py in the CNN+LGF layout) are inlined below, so
# this script has no Baseline_core dependency. LGFNet still needs LGF_python at
# the repo root; the sys.path line makes it importable from any working dir.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, csv, hashlib, json, math, random, tempfile, time
from dataclasses import dataclass, astuple
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


# -- random f and boundary generators ----------------------------------------

def _random_f(N, f_scale, M, rng=np.random):
    """Random forcing f via full 2D Fourier series, boundary = 0."""
    f = random_fourier_field(N, M, rng) * f_scale
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.
    return f

def _random_g(N, g_base=0.0, g_scale=0.6, M=6, rng=np.random):
    """Random Dirichlet boundary via Fourier series (g_base shifts it).
    Interior = 0."""
    raw = random_fourier_field(N, M, rng)
    g = np.zeros((N + 2, N + 2), dtype=np.float64)
    g[0, :] = g_base + g_scale * raw[0, :]
    g[-1, :] = g_base + g_scale * raw[-1, :]
    g[:, 0] = g_base + g_scale * raw[:, 0]
    g[:, -1] = g_base + g_scale * raw[:, -1]
    g[1:-1, 1:-1] = 0.
    return g

# -- forward solver dataset --------------------------------------------------

def make_solver_dataset(N, n_samples, device, args, seed):
    """Random f,g,a(sigma) -> single sparse linear FD solve -> u_ref."""
    rng = np.random.RandomState(seed)
    x_t, y_t, bnd, interior, h_grid = make_grid(N, device)
    M = getattr(args, "fourier_m", 6)

    f_list, g_list, a_list, sigma_list, u_list, info_list = [], [], [], [], [], []
    res64_sq = 0.

    print(f"generating {n_samples} solver samples (N={N})  "
          f"f_scale={args.f_scale}  g_base={args.g_base}  g_scale={args.g_scale}")

    for _ in range(n_samples):
        sigma_i = (rng.uniform(args.sigma_min, args.sigma_max)
                   if getattr(args, "random_sigma", False) else args.sigma)
        a_np = random_positive_field(N, sigma_i, M, rng)
        f_np = _random_f(N, args.f_scale, M, rng)
        g_np = _random_g(N, args.g_base, args.g_scale, M, rng)

        t0 = time.perf_counter()
        u_np = solve_linear_fd(f_np, g_np, a_np, h_grid)
        solve_ms = (time.perf_counter() - t0) * 1e3

        u64 = torch.from_numpy(u_np).double()[None, None]
        f64 = torch.from_numpy(f_np).double()[None, None]
        a64 = torch.from_numpy(a_np).double()[None, None]
        res64_sq += float(pde_residual(u64, f64, a64, h_grid).pow(2).sum())

        # accumulate on CPU; one bulk .to(device) after the cat below, instead
        # of one small host->device transfer per sample
        f_list.append(f64.float())
        g_list.append(torch.from_numpy(g_np).float().unsqueeze(0).unsqueeze(0))
        a_list.append(a64.float())
        sigma_list.append(torch.tensor([sigma_i], dtype=torch.float32))
        u_list.append(u64.float())
        info_list.append({"solve_time_ms": solve_ms})

    a_cat = torch.cat(a_list)
    sigma_cat = torch.cat(sigma_list)
    ref_res64 = math.sqrt(res64_sq / (n_samples * N * N))
    a_min, a_max = float(a_cat.min()), float(a_cat.max())
    print(f"  sigma range: [{float(sigma_cat.min()):.4f}, {float(sigma_cat.max()):.4f}]")
    print(f"  a range: [{a_min:.4f}, {a_max:.4f}]  "
          f"contrast a_max/a_min={a_max/a_min:.4f}")
    print(f"  ref residual (float64) = {ref_res64:.2e}")

    return {
        "f": torch.cat(f_list).to(device), "g": torch.cat(g_list).to(device),
        "a": a_cat.to(device), "sigma": sigma_cat.to(device),
        "u_ref": torch.cat(u_list).to(device),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h_grid,
        "ref_res64": ref_res64, "solver_info": info_list,
    }

# -- training ---------------------------------------------------------------

def validate_and_checkpoint(model, val_data, stats, args, epoch, epoch_loss,
                            best_metric, best_path, history):
    eval_bs = args.eval_batch_size or args.batch_size
    val_m = evaluate(model, val_data, stats, eval_bs)
    val_key = "relL2" if args.loss == "supervised" else "resRMSE"
    if val_m[val_key] < best_metric:
        best_metric = val_m[val_key]
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_type": "cnn",
            "model_state_dict": model.state_dict(),
            "stats": astuple(stats),   # plain tuple: robust to future
            "args": vars(args),        # NormalizationStats renames/moves
            "epoch": epoch,
            f"best_val_{val_key}": best_metric,
        }, best_path)

    print(f"epoch {epoch:03d}  loss={epoch_loss:.3e}  "
          f"resRMSE={val_m['resRMSE']:.4f}  "
          f"relL2={val_m['relL2']:.4e}  maxErr={val_m['maxError']:.4e}  "
          f"gradRelL2={val_m['gradRelL2']:.4e}  "
          f"linBL={val_m['linearBaselineRelL2']:.4e}")

    history.append({"epoch": epoch, "loss": epoch_loss,
                    **{f"val_{key}": val_m[key] for key in (
                        "resRMSE", "relL2", "maxError", "gradRelL2",
                        "meanMaxError", "meanBoundaryError",
                        "maxBoundaryError", "meanGradLinf",
                        "maxGradLinf", "refResRMSE",
                        "linearBaselineRelL2")}})
    return best_metric

def train(args, best_path, last_path, history_path, convergence_path, model=None):
    _set_seed(args.seed)
    device = _select_device(args.device)

    print(f"Equation: -div(a(x,y) grad u) = f  |  sigma = {args.sigma}  |  data=solver")
    if args.random_sigma:
        print(f"  random sigma in [{args.sigma_min}, {args.sigma_max}]")

    train_data = make_solver_dataset(args.N, args.train_samples, device,
                                      args, seed=args.seed)
    val_data = make_solver_dataset(args.N, args.val_samples, device,
                                    args, seed=args.seed + 1)
    stats = compute_stats(train_data)
    print(f"f stats: mean={stats.f_mean:.4e}  std={stats.f_std:.4e}  "
          f"log(a) stats: mean={stats.loga_mean:.4e}  std={stats.loga_std:.4e}")

    # Reference residual, reported in float64 (float32 rounding floor
    # ~h^-2*eps32 must not be mistaken for solver accuracy)
    ref_res = val_data["ref_res64"]
    u_min, u_max, bnd_err = _uref_range_and_bnd_err(val_data)
    print(f"ref residual RMSE (float64) = {ref_res:.4e}  (linear FD solver residual)")
    print(f"u_ref range: [{u_min:.4f}, {u_max:.4f}]  "
          f"boundary error = {bnd_err:.2e}")

    triv_rl2 = trivial_linear_baseline(val_data)
    print(f"trivial baseline (a≡1 linear Poisson) relL2 = {triv_rl2:.4e}  "
          f"(CNN must beat this to have learned a(x,y))")

    model = create_model(args, device) if model is None else model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  parameters={n_params:,}")

    opt = create_optimizer(model, args)

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
    history = []

    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(model, train_data, stats, opt, sched, args, device)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            best_metric = validate_and_checkpoint(
                model, val_data, stats, args, epoch, epoch_loss,
                best_metric, best_path, history)

    if isinstance(model, LGFNet):
        assert torch.equal(model.kernel_bank.cpu(), model._bank_snapshot.cpu()), (
            "LGF kernel bank changed during training (should be frozen)")

    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "cnn",
                 "model_state_dict": model.state_dict(),
                 "stats": astuple(stats), "args": vars(args)}, last_path)
    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, args.loss, dataset_label="solver")
    return model, best_metric

# -- fixed-solver test suite --------------------------------------------------
#
# predict-case semantics:
#   data:         random f,g,sigma -> linear FD solve -> u_ref (training family)
#   fixed-solver: manufactured DISCRETE test — u_target builds f and g through
#                 the discrete flux operator, then u_ref is recomputed by the
#                 sparse FD solver of that same discrete system. It checks that
#                 the torch flux operator and the scipy matrix assembly agree,
#                 that the sparse-solve residual is small, and how the CNN
#                 generalizes to manufactured analytic inputs. It does NOT
#                 measure the continuous-PDE discretization error (the forcing
#                 itself comes from the discrete operator): report it as a
#                 discrete consistency / manufactured discrete test, not as an
#                 independent continuous-accuracy validation. The analytic MMS
#                 cases live in Baseline_mms.py.

# Same raw formulas as the MMS ANALYTIC_CASES u1-u4 (signed, no shift).
FIXED_SOLVER_CASES = {
    "u1_sincos": lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y),
    "u2_poly":   lambda x, y: (1. - x ** 2) * (1. - y ** 2),
    "u3_cos":    lambda x, y: np.cos(np.pi * x) * np.cos(2. * np.pi * y),
    "u4_mixed":  lambda x, y: x ** 2 * (1. - y) + np.sin(np.pi * x) * y,
}

def make_fixed_solver_dataset(N, device, args, sigma_val, test_rng):
    """Deterministic solver test: u_target (signed) -> f = -div_h(a grad u),
    then u_ref = solve_linear_fd(f, g, a, h) (single linear solve) against
    a fresh test coefficient field per case."""
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    x64, y64, *_ = make_grid(N, "cpu", torch.float64)
    x_np = x64[0, 0].numpy()
    y_np = y64[0, 0].numpy()
    M = getattr(args, "fourier_m", 6)

    f_list, g_list, a_list, sigma_list, u_list, t_list = [], [], [], [], [], []
    info_list, names = [], []
    res64_sq = 0.

    for case_name, u_func in FIXED_SOLVER_CASES.items():
        u_target = u_func(x_np, y_np).astype(np.float64)
        a_np = random_positive_field(N, sigma_val, M, test_rng)

        g_np = u_target.copy()
        g_np[1:-1, 1:-1] = 0.

        t64 = torch.from_numpy(u_target)[None, None]      # CPU float64
        a64 = torch.from_numpy(a_np)[None, None]
        f64 = torch.zeros_like(t64)
        f64[:, :, 1:-1, 1:-1] = flux_divergence(t64, a64, h)
        f_np = f64[0, 0].numpy()

        t0 = time.perf_counter()
        u_ref = solve_linear_fd(f_np, g_np, a_np, h)
        solve_ms = (time.perf_counter() - t0) * 1e3

        u64 = torch.from_numpy(u_ref).double()[None, None]
        res64 = pde_residual(u64, f64, a64, h).pow(2)
        case_res64 = float(res64.mean().sqrt())
        res64_sq += float(res64.sum())

        f_list.append(f64.float())
        g_list.append(torch.from_numpy(g_np).float()[None, None])
        a_list.append(a64.float())
        sigma_list.append(torch.tensor([sigma_val], dtype=torch.float32))
        u_list.append(u64.float())
        t_list.append(t64.float())
        info_list.append({"residual_rmse": case_res64, "solve_time_ms": solve_ms})
        names.append(case_name)

    return {
        "f": torch.cat(f_list).to(device), "g": torch.cat(g_list).to(device),
        "a": torch.cat(a_list).to(device), "sigma": torch.cat(sigma_list).to(device),
        "u_ref": torch.cat(u_list).to(device), "u_target": torch.cat(t_list).to(device),
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
    panels = [(ut,   "u_target (manufactured)", "viridis",  vmin,  vmax),
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
    """Run the fixed-solver (manufactured discrete) cases, save metrics
    CSV/JSON + figures.

    CNN metrics (via per_sample_metrics) compare u_pred against the solver
    reference; target_vs_solver_ref_* checks that the sparse solve reproduces
    the manufactured discrete target, i.e. that the torch flux operator and
    the scipy assembly are consistent (discrete consistency — see the section
    comment above; this is not a continuous-accuracy test)."""
    N = data_args.N
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"prediction (fixed-solver): N={N}  sigma={args.predict_sigma:g}  "
          f"{len(FIXED_SOLVER_CASES)} cases, u_ref from single linear FD solve")
    test_rng = np.random.RandomState(args.test_seed)
    batch = make_fixed_solver_dataset(N, device, data_args, args.predict_sigma, test_rng)
    ensure_linear_baselines(batch)
    print(f"  ref residual (float64) = {batch['ref_res64']:.4e}")

    u_pred, _, cnn_ms_sample = _time_forward(model, batch, stats)
    rows = per_sample_metrics(u_pred, batch)

    interior = batch["interior_mask"][0, 0].detach().cpu().numpy()
    ur_np = batch["u_ref"].detach().cpu().numpy()
    ut_np = batch["u_target"].detach().cpu().numpy()

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
            "sigma": r["sigma"],
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
            "solver_residual_rmse": info["residual_rmse"],
            "cnn_ms_per_sample": cnn_ms_sample,
            "fd_solver_ms_per_sample": info["solve_time_ms"],
        })

        print(f"\n  --- fixed-solver case: {case}  (sigma={r['sigma']:g}) ---")
        print(f"    relL2={r['relL2']:.4e}  maxErr={r['maxError']:.4e}  "
              f"resRMSE={r['predResRMSE']:.4e}  "
              f"linBL={r['linearBaselineRelL2']:.4e}")
        print(f"    target_vs_solver relL2={t_rl2:.2e}  maxErr={t_max:.2e}  "
              f"solver: res={info['residual_rmse']:.2e}")
        if t_rl2 > 1e-8:
            print(f"    WARNING: target_vs_solver_ref_relL2={t_rl2:.2e} — "
                  f"solver reference deviates from the manufactured discrete "
                  f"target; torch flux operator and scipy assembly disagree "
                  f"(discrete consistency broken)")

        if args.plot:
            _save_sample_figures(batch, u_pred, i, out_dir, f"fixed_solver_{case}",
                                 f"fixed-solver {case}  sigma={r['sigma']:.2f}  "
                                 f"relL2={r['relL2']:.3e}")
            _save_target_comparison(batch["u_target"][i, 0], batch["u_ref"][i, 0],
                                    out_dir / f"fixed_solver_{case}_target_vs_solver.png",
                                    f"{case}  sigma={r['sigma']:.2f}  "
                                    f"target_vs_solver relL2={t_rl2:.2e}")

    fd_ms_mean = float(np.mean([x["solve_time_ms"] for x in batch["solver_info"]]))
    print(f"\n  timing: cnn_ms_per_sample={cnn_ms_sample:.3f}  "
          f"fd_solver_ms_per_sample={fd_ms_mean:.3f}")
    _write_csv(out_rows, out_dir / "fixed_solver_metrics.csv")
    metric_keys = [key for key in out_rows[0] if key != "case"]
    summary = summarize_rows(out_rows, metric_keys)
    with open(out_dir / "fixed_solver_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  fixed-solver suite saved → {out_dir}")
    return out_rows

# -- prediction -------------------------------------------------------------

def run_prediction(args):
    if not args.predict_from:
        raise ValueError("--predict-from required")

    _set_seed(args.seed)
    device = _select_device(args.device)

    ckpt = torch.load(args.predict_from, map_location=device, weights_only=False)
    ckpt_type = ckpt.get("model_type")
    if ckpt_type and ckpt_type != "cnn":
        raise ValueError(
            f"checkpoint type mismatch: expected 'cnn', got '{ckpt_type}'. "
            f"Use the correct prediction script (Baseline_solver.py for "
            f"'cnn', Baseline_mms.py for 'cnn_mms').")
    if "sigma" not in ckpt["args"]:
        raise ValueError(
            "checkpoint predates the a(x,y)/sigma refactor (has 'k', not "
            "'sigma') — retrain with the current script.")
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = create_model(ckpt_args, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = ckpt["stats"]
    if not isinstance(stats, NormalizationStats):
        stats = NormalizationStats(*stats)   # pre-refactor checkpoint: plain tuple

    N = ckpt_args.N
    data_args = argparse.Namespace(**vars(ckpt_args))
    data_args.device = args.device
    data_args.seed = args.test_seed
    data_args.random_sigma = False
    data_args.sigma = args.predict_sigma

    if args.predict_case == "fixed-solver":
        out = (Path(args.plot_dir) if args.plot_dir
               else Path("results") / "predict" / "fixed_solver")
        run_fixed_solver_suite(model, stats, device, args, data_args, out)
        return

    assert args.test_seed not in (ckpt_args.seed, ckpt_args.seed + 1), (
        "test seed collides with a train/val seed! prediction data would "
        "overlap the training run (evaluation pollution)."
    )

    print(f"prediction (solver): N={N}  samples={args.predict_samples}  "
          f"sigma={args.predict_sigma:g} (fixed)")
    batch = make_solver_dataset(N, args.predict_samples, device,
                                data_args, seed=args.test_seed)
    ensure_linear_baselines(batch)

    want_trace = isinstance(model, LGFNet)
    timing = _time_forward(model, batch, stats, return_diagnostics=want_trace)
    if want_trace:
        u_pred, cnn_ms_batch, cnn_ms_sample, res_trace = timing
        _print_res_trace(res_trace)
    else:
        u_pred, cnn_ms_batch, cnn_ms_sample = timing
    fd_ms_sample = float(np.mean(
        [x["solve_time_ms"] for x in batch["solver_info"]]))

    rows = per_sample_metrics(u_pred, batch)
    vm = _aggregate_metrics(rows)
    print(f"relL2={vm['relL2']:.4e}  maxErr={vm['maxError']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  refRMSE={vm['refResRMSE']:.4e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")
    print(f"meanMaxError={vm['meanMaxError']:.4e}")
    print(f"meanBoundaryError={vm['meanBoundaryError']:.2e}  maxBoundaryError={vm['maxBoundaryError']:.2e}")
    print(f"meanGradLinf={vm['meanGradLinf']:.4e}  maxGradLinf={vm['maxGradLinf']:.4e}")
    print(f"reference residual RMSE (float64) = {batch['ref_res64']:.4e}")
    print(f"linear baseline (a≡1) relL2 = {vm['linearBaselineRelL2']:.4e}  "
          f"(CNN must beat this to have learned a(x,y))  "
          f"const baseline relL2 = {vm['constBaselineRelL2']:.4e}")
    diag_note = " (incl. per-step residual diagnostics)" if want_trace else ""
    print(f"timing: cnn={cnn_ms_batch:.2f} ms/batch  "
          f"cnn_ms_per_sample={cnn_ms_sample:.3f}{diag_note}  "
          f"fd_solver_ms_per_sample={fd_ms_sample:.3f}")

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "a": batch["a"].cpu(), "u_ref": batch["u_ref"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]), "metrics": vm,
        }, p)
        print(f"saved predictions → {p}")

    for r, info in zip(rows, batch["solver_info"]):
        r["cnn_ms_per_sample"] = cnn_ms_sample
        r["fd_solver_ms_per_sample"] = info["solve_time_ms"]
    summary = summarize_rows(rows, METRIC_KEYS)

    # Always save metrics — use --predict-save, --plot-dir, or default dir
    if args.predict_save:
        out = Path(args.predict_save).parent
    elif args.plot_dir:
        out = Path(args.plot_dir)
    else:
        out = Path("results") / "predict"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out / "metrics.csv")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.plot:
        for i, row in enumerate(rows):
            _save_sample_figures(batch, u_pred, i, out, f"sample_{i:03d}",
                                 f"sample {i}  sigma={row['sigma']:.2f}  "
                                 f"relL2={row['relL2']:.3e}")
    print(f"saved → {out}")

# -- CLI --------------------------------------------------------------------

def ckpt_key(args):
    """Readable experiment key + short hash of the remaining result-affecting
    training args (seed, lr, weight decay, batch size, val samples, eval
    cadence, grad clip), so changing any of them can't silently reuse another
    run's checkpoint directory."""
    if getattr(args, "random_sigma", False):
        sig_str = f"sig{args.sigma_min:g}-{args.sigma_max:g}"
    else:
        sig_str = f"sig{args.sigma:g}"
    n_layers = getattr(args, "n_layers", 6)
    loss_suffix = "_sup" if getattr(args, "loss", "physics") == "supervised" else ""
    model_name = getattr(args, "model", "cnn")
    model_suffix = ""
    if model_name == "lgf":
        model_suffix = f"_lgf_T{args.lgf_T}_M{args.lgf_M}_L{args.lgf_L}"
    extra = (args.seed, args.lr, args.weight_decay, args.batch_size,
             args.val_samples, args.eval_every, args.grad_clip)
    cfg = hashlib.md5(repr(extra).encode()).hexdigest()[:8]
    return (f"N{args.N}_e{args.epochs}_tr{args.train_samples}"
            f"_{sig_str}_M{getattr(args, 'fourier_m', 6)}"
            f"_fs{args.f_scale:g}_gb{args.g_base:g}_gs{args.g_scale:g}"
            f"_ch{args.base_ch}_L{n_layers}{loss_suffix}{model_suffix}_{cfg}")

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

    # Model selection
    p.add_argument("--model", default="cnn", choices=["cnn", "lgf"],
                   help="cnn: SimpleCNN one-shot map | lgf: LGFNet multi-kappa "
                        "LGF unrolled iteration")
    p.add_argument("--lgf-T", type=int, default=12,
                   help="LGFNet: number of unrolled iteration steps")
    p.add_argument("--lgf-M", type=int, default=8,
                   help="LGFNet: number of kernel-bank nodes")
    p.add_argument("--lgf-L", type=int, default=4,
                   help="LGFNet: number of multi-kappa LGF blocks in Phi_theta")

    # Data generation
    p.add_argument("--sanity", action="store_true",
                   help="generate solver dataset, print residual/data stats, "
                        "then exit")

    # Linear variable coefficient a(x,y) = exp(sigma * psi)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--random-sigma", action=argparse.BooleanOptionalAction, default=False,
                   help="sample sigma uniformly from [sigma_min, sigma_max] "
                        "(default: False; use --random-sigma to enable)")
    p.add_argument("--sigma-min", type=float, default=0.5)
    p.add_argument("--sigma-max", type=float, default=1.5)

    # Fourier series
    p.add_argument("--fourier-m", type=int, default=6,
                   help="number of Fourier modes per dimension for random fields")

    # Random field scales
    p.add_argument("--f-scale", type=float, default=6.0,
                   help="scale of random forcing")
    p.add_argument("--g-base", type=float, default=0.0)
    p.add_argument("--g-scale", type=float, default=0.6)

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
                   help="'data': random f,g,sigma -> solver -> u_ref | "
                        "'fixed-solver': manufactured u_target builds f,g via "
                        "the discrete flux operator; u_ref recomputed by the "
                        "sparse FD solver (discrete consistency test, not a "
                        "continuous-accuracy test)")
    p.add_argument("--plot-dir", default="",
                   help="output dir for prediction figures "
                        "(default: results/predict/<case>)")
    p.add_argument("--predict-sigma", type=float, default=0.5,
                   help="fixed sigma (test coefficient-field contrast) for prediction")
    return p.parse_args()

# -- main -------------------------------------------------------------------

def main():
    args = parse_args()

    if args.sanity:
        print("=" * 60)
        print("SANITY CHECK (data=solver, single linear FD solve)")
        print("=" * 60)
        print(f"Equation: -div(a(x,y) grad u) = f  |  sigma = {args.sigma}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")

        _set_seed(args.seed)
        device = _select_device(args.device)
        data = make_solver_dataset(args.N, args.train_samples, device,
                                    args, seed=args.seed)

        ref32in64 = ref_residual_rmse64(data)
        u_min, u_max, bnd_err = _uref_range_and_bnd_err(data)
        fi = data["f"][:, :, 1:-1, 1:-1]

        print(f"  ref residual (float64 pipeline) = {data['ref_res64']:.4e}")
        print(f"  stored float32 data residual    = {ref32in64:.4e}  "
              f"(~h^-2·eps32 rounding floor)")
        print(f"  boundary error = {bnd_err:.2e}  (should be 0)")
        print(f"  u_ref range    = [{u_min:.4f}, {u_max:.4f}]")
        print(f"  f range        = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")
        print(f"  trivial baseline (a≡1) relL2    = "
              f"{trivial_linear_baseline(data):.4e}")

        if args.model == "lgf":
            print("-" * 60)
            print(f"LGF kernel bank (M={args.lgf_M}) sanity checks:")
            bank, _ = build_lgf_kernel_bank(args.N, data["h"], args.lgf_M)
            assert torch.isfinite(bank).all()
            print(f"  bank finite: OK  (shape={tuple(bank.shape)})")
            _lgf_const_coeff_sanity_check(args.N, args.lgf_M)

        print("=" * 60)
        return

    if args.predict:
        return run_prediction(args)

    key = ckpt_key(args)
    ckpt_dir = Path("checkpoints") / "cnn" / key
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
