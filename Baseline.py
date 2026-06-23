"""CNN baseline for a nonlinear Poisson equation in flux form.

The model learns a grid-to-grid map for

    -div(a(u) grad u) = f,  a(u) = u**k,  u = g on boundary,

on Omega = [-1, 1]^2.  The loss is the conservative finite-difference
flux residual on the interior grid nodes, with Dirichlet boundary values
enforced exactly by an output projection.

Example:
    python Baseline.py --check-residual
    python Baseline.py --N 32 --train-samples 2048 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F


SHARP_SOFTPLUS_BETA = 20.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Grid, masks, and input channels
# ---------------------------------------------------------------------------


def make_grid(
    N: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return x, y, boundary mask, interior mask, and grid spacing h.

    The full grid includes boundary nodes: H = W = N + 2.
    Tensor convention is [B, C, H, W], with H in the y direction and W in x.
    """

    H = W = N + 2
    h = 2.0 / (N + 1)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x = xx[None, None, :, :]
    y = yy[None, None, :, :]

    boundary_mask = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    boundary_mask[:, :, 0, :] = 1.0
    boundary_mask[:, :, -1, :] = 1.0
    boundary_mask[:, :, :, 0] = 1.0
    boundary_mask[:, :, :, -1] = 1.0
    interior_mask = 1.0 - boundary_mask
    return x, y, boundary_mask, interior_mask, h


def build_input_tensor(
    f: torch.Tensor,
    g_map: torch.Tensor,
    boundary_mask: torch.Tensor,
    k: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    k_mean: float | None = None,
    k_std: float | None = None,
) -> torch.Tensor:
    """Build the six-channel input [f, boundary g, mask, k, x, y]."""

    B, _, H, W = f.shape

    if boundary_mask.shape[0] == 1:
        boundary_mask = boundary_mask.expand(B, -1, -1, -1)
    if x.shape[0] == 1:
        x = x.expand(B, -1, -1, -1)
    if y.shape[0] == 1:
        y = y.expand(B, -1, -1, -1)

    k_channel = k.view(B, 1, 1, 1)
    if k_mean is not None and k_std is not None:
        k_channel = (k_channel - k_mean) / k_std
    k_channel = k_channel.expand(B, 1, H, W)

    return torch.cat([f, g_map, boundary_mask, k_channel, x, y], dim=1)


def enforce_dirichlet(
    u_raw: torch.Tensor,
    g_map: torch.Tensor,
    boundary_mask: torch.Tensor,
    positive_interior: bool = False,
    positive_eps: float = 1e-6,
) -> torch.Tensor:
    """Hard-impose boundary values on all edge nodes."""

    if positive_interior:
        u_raw = positive_eps + F.softplus(u_raw)
    return boundary_mask * g_map + (1.0 - boundary_mask) * u_raw


# ---------------------------------------------------------------------------
# Flux-form nonlinear residual, losses, and metrics
# ---------------------------------------------------------------------------


def _k_to_broadcast(k: torch.Tensor | float, u: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(k):
        k = torch.tensor(k, device=u.device, dtype=u.dtype)
    k = k.to(device=u.device, dtype=u.dtype)
    if k.ndim == 0:
        return k.view(1, 1, 1, 1)
    if k.ndim == 1:
        return k[:, None, None, None]
    return k


def _coefficient_base_and_derivative(
    u: torch.Tensor,
    mode: str,
    eps: float,
    softplus_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "pow":
        return torch.clamp(u, min=eps), (u > eps).to(dtype=u.dtype)
    if mode == "abs_pow":
        abs_u = u.abs()
        return torch.clamp(abs_u, min=eps), (abs_u > eps).to(dtype=u.dtype) * u.sign()
    if mode == "softplus_pow":
        beta = torch.as_tensor(softplus_beta, device=u.device, dtype=u.dtype)
        return eps + F.softplus(beta * u) / beta, torch.sigmoid(beta * u)
    raise ValueError(f"Unknown coeff_mode: {mode}")


def power_coefficient(
    u: torch.Tensor,
    k: torch.Tensor | float,
    mode: str = "pow",
    eps: float = 1e-6,
    softplus_beta: float = SHARP_SOFTPLUS_BETA,
) -> torch.Tensor:
    """Compute a(u).

    mode="pow" clamps only as a numerical guard. mode="softplus_pow" uses a
    sharp softplus base, which is a smooth clamp approximation rather than the
    broad F.softplus(u) smoothing.
    """

    k = _k_to_broadcast(k, u)
    base, _ = _coefficient_base_and_derivative(u, mode, eps, softplus_beta)
    return torch.pow(base, k)


def nonlinear_poisson_residual(
    u: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor | float,
    h: float,
    coeff_mode: str = "pow",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Interior residual R = -div_h(a(u) grad_h u) - f.

    Interface coefficients are arithmetic averages, so each face has one
    conservative flux shared by adjacent control volumes.
    """

    a = power_coefficient(u, k, mode=coeff_mode, eps=eps)

    uc = u[:, :, 1:-1, 1:-1]
    ue = u[:, :, 1:-1, 2:]
    uw = u[:, :, 1:-1, :-2]
    un = u[:, :, 2:, 1:-1]
    us = u[:, :, :-2, 1:-1]

    ac = a[:, :, 1:-1, 1:-1]
    ae = 0.5 * (ac + a[:, :, 1:-1, 2:])
    aw = 0.5 * (ac + a[:, :, 1:-1, :-2])
    an = 0.5 * (ac + a[:, :, 2:, 1:-1])
    a_s = 0.5 * (ac + a[:, :, :-2, 1:-1])

    flux_e = ae * (ue - uc) / h
    flux_w = aw * (uc - uw) / h
    flux_n = an * (un - uc) / h
    flux_s = a_s * (uc - us) / h

    div_h = (flux_e - flux_w + flux_n - flux_s) / h
    return -div_h - f[:, :, 1:-1, 1:-1]


def kirchhoff_transform(u: torch.Tensor, k: torch.Tensor | float) -> torch.Tensor:
    """Map positive u to w = u^(k+1)/(k+1)."""

    if bool((u <= 0).any().item()):
        raise ValueError("kirchhoff_transform requires strictly positive u values.")
    kp1 = _k_plus_one_to_broadcast(k, u)
    return torch.pow(u, kp1) / kp1


def inverse_kirchhoff_transform(
    w: torch.Tensor,
    k: torch.Tensor | float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Invert w = u^(k+1)/(k+1) with a small clamp against roundoff."""

    kp1 = _k_plus_one_to_broadcast(k, w)
    base = torch.clamp(kp1 * w, min=eps)
    return torch.pow(base, 1.0 / kp1)


def poisson_residual_from_w(w: torch.Tensor, f: torch.Tensor, h: float) -> torch.Tensor:
    wc = w[:, :, 1:-1, 1:-1]
    lap_w = (
        w[:, :, 1:-1, 2:]
        + w[:, :, 1:-1, :-2]
        + w[:, :, 2:, 1:-1]
        + w[:, :, :-2, 1:-1]
        - 4.0 * wc
    ) / (h * h)
    return -lap_w - f[:, :, 1:-1, 1:-1]


def kirchhoff_residual_from_u(
    u: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor | float,
    h: float,
) -> torch.Tensor:
    return poisson_residual_from_w(kirchhoff_transform(u, k), f, h)


def validate_discretization(discretization: str, coeff_mode: str) -> None:
    if discretization not in {"flux", "kirchhoff"}:
        raise ValueError("discretization must be 'flux' or 'kirchhoff'.")
    if discretization == "kirchhoff" and coeff_mode != "pow":
        raise ValueError("discretization='kirchhoff' requires coeff_mode='pow'.")


def compute_pde_residual(
    u: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor | float,
    h: float,
    discretization: str,
    coeff_mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    validate_discretization(discretization, coeff_mode)
    if discretization == "flux":
        return nonlinear_poisson_residual(u, f, k, h, coeff_mode=coeff_mode, eps=eps)
    return kirchhoff_residual_from_u(u, f, k, h)


def pde_loss(
    u: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor | float,
    h: float,
    scale_with_h: bool = True,
    coeff_mode: str = "pow",
    discretization: str = "kirchhoff",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    res = compute_pde_residual(
        u,
        f,
        k,
        h,
        discretization=discretization,
        coeff_mode=coeff_mode,
        eps=eps,
    )
    res_for_loss = (h * h) * res if scale_with_h else res
    return (res_for_loss * res_for_loss).mean(), res


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return (diff * diff).sum() / denom


def relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if mask is not None:
        pred = pred * mask
        target = target * mask
    num = torch.linalg.vector_norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)
    return (num / den).mean()


def boundary_max_error(pred: torch.Tensor, g_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - g_map).abs() * mask).amax()


def interior_linf_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs() * mask).amax()


def gradient_relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    h: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Centered-gradient relative L2 on stencils covered by the full grid."""

    pred_dx = (pred[:, :, 1:-1, 2:] - pred[:, :, 1:-1, :-2]) / (2.0 * h)
    pred_dy = (pred[:, :, 2:, 1:-1] - pred[:, :, :-2, 1:-1]) / (2.0 * h)
    target_dx = (target[:, :, 1:-1, 2:] - target[:, :, 1:-1, :-2]) / (2.0 * h)
    target_dy = (target[:, :, 2:, 1:-1] - target[:, :, :-2, 1:-1]) / (2.0 * h)

    diff = torch.cat([pred_dx - target_dx, pred_dy - target_dy], dim=1)
    ref = torch.cat([target_dx, target_dy], dim=1)
    num = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
    den = torch.linalg.vector_norm(ref.reshape(ref.shape[0], -1), dim=1).clamp_min(eps)
    return (num / den).mean()


# ---------------------------------------------------------------------------
# Transformed Poisson finite-difference baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoissonCGDiagnostics:
    iterations: int
    final_max_relative_residual: float
    converged: bool


def _k_plus_one_to_broadcast(k: torch.Tensor | float, u: torch.Tensor) -> torch.Tensor:
    kp1 = _k_to_broadcast(k, u) + 1.0
    if bool((kp1.abs() < 1e-8).any().item()):
        raise ValueError("The transform w = u^(k+1)/(k+1) requires k != -1.")
    return kp1


def u_to_w(u: torch.Tensor, k: torch.Tensor | float) -> torch.Tensor:
    return kirchhoff_transform(u, k)


def w_to_u(w: torch.Tensor, k: torch.Tensor | float, eps: float = 1e-8) -> torch.Tensor:
    return inverse_kirchhoff_transform(w, k, eps=eps)


def _negative_laplacian_interior_matvec(v: torch.Tensor) -> torch.Tensor:
    """Dimensionless interior operator A(v)=4v-v_e-v_w-v_n-v_s."""

    if v.ndim != 4 or v.shape[1] != 1:
        raise ValueError("v must have shape [B, 1, N, N].")
    out = 4.0 * v
    out[:, :, :, 1:] = out[:, :, :, 1:] - v[:, :, :, :-1]
    out[:, :, :, :-1] = out[:, :, :, :-1] - v[:, :, :, 1:]
    out[:, :, 1:, :] = out[:, :, 1:, :] - v[:, :, :-1, :]
    out[:, :, :-1, :] = out[:, :, :-1, :] - v[:, :, 1:, :]
    return out


def _poisson_dirichlet_rhs(
    f: torch.Tensor,
    w_boundary: torch.Tensor,
    h: float,
) -> torch.Tensor:
    if f.ndim != 4 or f.shape[1] != 1:
        raise ValueError("f must have shape [B, 1, H, W].")
    B, _, H, W = f.shape
    if H != W:
        raise ValueError("Only square grids are supported.")
    N = H - 2
    if N < 1:
        raise ValueError("Interior grid size N must be positive.")
    if w_boundary.shape[0] == 1 and B > 1:
        w_boundary = w_boundary.expand(B, -1, -1, -1)
    if w_boundary.shape != f.shape:
        raise ValueError("w_boundary must have shape [B, 1, H, W] or batch size 1.")

    w_boundary = w_boundary.to(device=f.device, dtype=f.dtype)
    rhs = (h * h) * f[:, :, 1:-1, 1:-1].clone()
    rhs[:, :, :, 0] = rhs[:, :, :, 0] + w_boundary[:, :, 1:-1, 0]
    rhs[:, :, :, -1] = rhs[:, :, :, -1] + w_boundary[:, :, 1:-1, -1]
    rhs[:, :, 0, :] = rhs[:, :, 0, :] + w_boundary[:, :, 0, 1:-1]
    rhs[:, :, -1, :] = rhs[:, :, -1, :] + w_boundary[:, :, -1, 1:-1]
    return rhs


def _interior_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).flatten(start_dim=1).sum(dim=1)


def _default_cg_tol(dtype: torch.dtype) -> float:
    return 1e-10 if dtype == torch.float64 else 1e-6


def _default_cg_max_iter(N: int, cg_max_iter: int | None) -> int:
    if cg_max_iter is None or cg_max_iter <= 0:
        return max(100, 10 * N)
    return int(cg_max_iter)


def solve_laplacian_interior_cg(
    rhs: torch.Tensor,
    cg_tol: float | None = None,
    cg_max_iter: int | None = None,
) -> tuple[torch.Tensor, PoissonCGDiagnostics]:
    """Solve A(x)=rhs for batched interior arrays with matrix-free CG."""

    if rhs.ndim != 4 or rhs.shape[1] != 1:
        raise ValueError("rhs must have shape [B, 1, N, N].")
    B, _, N, W = rhs.shape
    if N != W:
        raise ValueError("Only square interior grids are supported.")
    if N < 1:
        raise ValueError("Interior grid size N must be positive.")
    if not torch.isfinite(rhs).all():
        raise ValueError("Poisson CG inputs must be finite.")

    tol = _default_cg_tol(rhs.dtype) if cg_tol is None or cg_tol <= 0 else float(cg_tol)
    max_iter = _default_cg_max_iter(N, cg_max_iter)
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()

    b_norm = torch.linalg.vector_norm(rhs.flatten(start_dim=1), dim=1)
    denom = torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))
    rr = _interior_dot(r, r)
    rel = rr.sqrt() / denom
    if bool((rel <= tol).all().item()):
        return x, PoissonCGDiagnostics(
            iterations=0,
            final_max_relative_residual=float(rel.max().detach().cpu()),
            converged=True,
        )

    tiny = torch.finfo(rhs.dtype).tiny
    for iteration in range(1, max_iter + 1):
        active = rel > tol
        active4 = active.view(B, 1, 1, 1)
        Ap = _negative_laplacian_interior_matvec(p)
        if not torch.isfinite(Ap).all():
            raise ValueError("Poisson CG produced non-finite operator values.")
        denom_alpha = _interior_dot(p, Ap)
        if bool(((denom_alpha <= 0) & active).any().item()):
            raise RuntimeError("Poisson CG encountered a non-positive search direction.")

        alpha = torch.zeros_like(rr)
        alpha[active] = rr[active] / denom_alpha[active]
        alpha4 = alpha.view(B, 1, 1, 1)
        x = torch.where(active4, x + alpha4 * p, x)
        r_next = r - alpha4 * Ap
        r = torch.where(active4, r_next, r)
        if not torch.isfinite(x).all() or not torch.isfinite(r).all():
            raise ValueError("Poisson CG produced non-finite iterates.")

        rr_next = _interior_dot(r, r)
        rel = rr_next.sqrt() / denom
        if bool((rel <= tol).all().item()):
            return x, PoissonCGDiagnostics(
                iterations=iteration,
                final_max_relative_residual=float(rel.max().detach().cpu()),
                converged=True,
            )

        beta = torch.zeros_like(rr)
        beta[active] = rr_next[active] / rr[active].clamp_min(tiny)
        beta4 = beta.view(B, 1, 1, 1)
        p_next = r + beta4 * p
        p = torch.where(active4, p_next, p)
        rr = rr_next

    final_rel = float(rel.max().detach().cpu())
    raise RuntimeError(
        "Poisson CG failed to converge "
        f"after {max_iter} iterations (max relative residual {final_rel:.3e}, tol {tol:.3e})."
    )


def solve_poisson_dirichlet_cg(
    f: torch.Tensor,
    w_boundary: torch.Tensor,
    h: float,
    mask: torch.Tensor,
    cg_tol: float | None = None,
    cg_max_iter: int | None = None,
) -> tuple[torch.Tensor, PoissonCGDiagnostics]:
    """Matrix-free batched CG solve for -Delta w = f with Dirichlet data."""

    if f.ndim != 4 or f.shape[1] != 1:
        raise ValueError("f must have shape [B, 1, H, W].")
    B, _, H, W = f.shape
    if H != W:
        raise ValueError("Only square grids are supported.")
    N = H - 2
    if N < 1:
        raise ValueError("Interior grid size N must be positive.")
    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, -1, -1, -1)
    if w_boundary.shape[0] == 1 and B > 1:
        w_boundary = w_boundary.expand(B, -1, -1, -1)
    if mask.shape != f.shape or w_boundary.shape != f.shape:
        raise ValueError("mask and w_boundary must match f shape or use batch size 1.")

    w_boundary = w_boundary.to(device=f.device, dtype=f.dtype)
    mask = mask.to(device=f.device, dtype=f.dtype)
    b = _poisson_dirichlet_rhs(f, w_boundary, h)
    if not torch.isfinite(w_boundary).all():
        raise ValueError("Poisson CG inputs must be finite.")
    x, diagnostics = solve_laplacian_interior_cg(b, cg_tol=cg_tol, cg_max_iter=cg_max_iter)
    w = mask * w_boundary
    w[:, :, 1:-1, 1:-1] = x
    return w, diagnostics


def format_cg_diagnostics(label: str, diagnostics: PoissonCGDiagnostics) -> str:
    return (
        f"{label} cg_iters={diagnostics.iterations} "
        f"cg_relres={diagnostics.final_max_relative_residual:.3e} "
        f"cg_converged={diagnostics.converged}"
    )


def _solve_poisson_dirichlet_dense_reference(
    f: torch.Tensor,
    w_boundary: torch.Tensor,
    h: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Test-only dense reference; unsuitable for production-size grids."""

    if f.ndim != 4 or f.shape[1] != 1:
        raise ValueError("f must have shape [B, 1, H, W].")
    B, _, H, W = f.shape
    if H != W:
        raise ValueError("Only square grids are supported.")
    N = H - 2
    if N < 1:
        raise ValueError("Interior grid size N must be positive.")

    if w_boundary.shape[0] == 1 and B > 1:
        w_boundary = w_boundary.expand(B, -1, -1, -1)
    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, -1, -1, -1)
    w_boundary = w_boundary.to(device=f.device, dtype=f.dtype)
    mask = mask.to(device=f.device, dtype=f.dtype)

    n_unknowns = N * N
    A = torch.zeros((n_unknowns, n_unknowns), device=f.device, dtype=f.dtype)
    rhs = (h * h) * f[:, 0, 1:-1, 1:-1].reshape(B, n_unknowns)

    for iy in range(N):
        for ix in range(N):
            row = iy * N + ix
            A[row, row] = 4.0
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                jy = iy + dy
                jx = ix + dx
                if 0 <= jy < N and 0 <= jx < N:
                    A[row, jy * N + jx] = -1.0
                else:
                    by = iy + 1 + dy
                    bx = ix + 1 + dx
                    rhs[:, row] += w_boundary[:, 0, by, bx]

    sol = torch.linalg.solve(A, rhs.T).T.reshape(B, 1, N, N)
    w = mask * w_boundary
    w[:, :, 1:-1, 1:-1] = sol
    return w


def solve_transformed_poisson(
    f: torch.Tensor,
    g_map: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor,
    h: float,
    cg_tol: float | None = None,
    cg_max_iter: int | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PoissonCGDiagnostics]:
    """Solve via w = u^(k+1)/(k+1) using matrix-free Poisson CG."""

    if mask.shape[0] == 1 and g_map.shape[0] > 1:
        mask = mask.expand(g_map.shape[0], -1, -1, -1)
    safe_g = torch.where(mask > 0, g_map, torch.ones_like(g_map))
    w_boundary = mask * kirchhoff_transform(safe_g, k)
    w, diagnostics = solve_poisson_dirichlet_cg(
        f,
        w_boundary,
        h,
        mask,
        cg_tol=cg_tol,
        cg_max_iter=cg_max_iter,
    )
    u = inverse_kirchhoff_transform(w, k)
    u_full = mask * g_map + (1.0 - mask) * u
    if return_diagnostics:
        return u_full, diagnostics
    return u_full


# ---------------------------------------------------------------------------
# U-Net with GroupNorm and a dilated bottleneck
# ---------------------------------------------------------------------------


def _valid_group_count(channels: int, requested_groups: int) -> int:
    groups = min(requested_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        group_count = _valid_group_count(out_ch, groups)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(group_count, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(group_count, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DilatedResidualBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, groups: int = 8) -> None:
        super().__init__()
        group_count = _valid_group_count(ch, groups)
        self.conv1 = nn.Conv2d(
            ch,
            ch,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="replicate",
        )
        self.gn1 = nn.GroupNorm(group_count, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")
        self.gn2 = nn.GroupNorm(group_count, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.gn1(self.conv1(x)))
        z = self.gn2(self.conv2(z))
        return F.gelu(x + z)


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = 8,
    mode: str = "replicate",
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    pad = (0, pad_w, 0, pad_h)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, pad, mode=mode)
    return x, pad


def unpad(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    H = x.shape[-2] - top - bottom
    W = x.shape[-1] - left - right
    return x[:, :, top : top + H, left : left + W]


class UNet2d(nn.Module):
    def __init__(self, in_ch: int = 6, out_ch: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, 2 * base_ch)
        self.enc3 = ConvBlock(2 * base_ch, 4 * base_ch)
        self.bottleneck = nn.Sequential(
            ConvBlock(4 * base_ch, 8 * base_ch),
            DilatedResidualBlock(8 * base_ch, dilation=2),
            DilatedResidualBlock(8 * base_ch, dilation=4),
        )
        self.up3 = nn.Conv2d(
            8 * base_ch + 4 * base_ch,
            4 * base_ch,
            3,
            padding=1,
            padding_mode="replicate",
        )
        self.dec3 = ConvBlock(4 * base_ch, 4 * base_ch)
        self.up2 = nn.Conv2d(
            4 * base_ch + 2 * base_ch,
            2 * base_ch,
            3,
            padding=1,
            padding_mode="replicate",
        )
        self.dec2 = ConvBlock(2 * base_ch, 2 * base_ch)
        self.up1 = nn.Conv2d(
            2 * base_ch + base_ch,
            base_ch,
            3,
            padding=1,
            padding_mode="replicate",
        )
        self.dec1 = ConvBlock(base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, pad = pad_to_multiple(x, multiple=8, mode="replicate")

        e1 = self.enc1(x0)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(self.up3(torch.cat([d3, e3], dim=1)))

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(self.up2(torch.cat([d2, e2], dim=1)))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(self.up1(torch.cat([d1, e1], dim=1)))

        return unpad(self.head(d1), pad)


# ---------------------------------------------------------------------------
# Manufactured solutions
# ---------------------------------------------------------------------------


def manufactured_u(
    x: torch.Tensor,
    y: torch.Tensor,
    params: dict[str, torch.Tensor],
    solution_mode: str = "base",
) -> torch.Tensor:
    """A positive solution family with nontrivial boundary values."""

    phi = (1.0 - x * x) * (1.0 - y * y)
    boundary_part = (
        params["c0"] + params["c1"] * x + params["c2"] * y + params["c3"] * x * y
    )
    base_part = (
        params["a1"] * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        + params["a2"] * torch.cos(2.0 * math.pi * x) * torch.cos(math.pi * y)
    )

    if solution_mode == "base":
        interior_part = base_part
    elif solution_mode == "fourier":
        interior_part = base_part + (
            params["a3"] * torch.sin(2.0 * math.pi * x) * torch.sin(3.0 * math.pi * y)
            + params["a4"] * torch.cos(3.0 * math.pi * x) * torch.sin(2.0 * math.pi * y)
        )
    elif solution_mode == "gaussian":
        r1 = (x - params["gx1"]) ** 2 + (y - params["gy1"]) ** 2
        r2 = (x - params["gx2"]) ** 2 + (y - params["gy2"]) ** 2
        g1 = torch.exp(-r1 / (2.0 * params["gs1"] ** 2))
        g2 = torch.exp(-r2 / (2.0 * params["gs2"] ** 2))
        interior_part = params["ga1"] * g1 + params["ga2"] * g2
    elif solution_mode == "mixed":
        r1 = (x - params["gx1"]) ** 2 + (y - params["gy1"]) ** 2
        g1 = torch.exp(-r1 / (2.0 * params["gs1"] ** 2))
        interior_part = base_part + params["a3"] * torch.sin(
            2.0 * math.pi * x
        ) * torch.sin(3.0 * math.pi * y) + params["ga1"] * g1
    else:
        raise ValueError(f"Unknown solution_mode: {solution_mode}")
    return boundary_part + phi * interior_part


def continuous_rhs_from_params(
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
    params: dict[str, torch.Tensor],
    coeff_mode: str = "pow",
    eps: float = 1e-6,
    softplus_beta: float = SHARP_SOFTPLUS_BETA,
) -> torch.Tensor:
    """Compute f = -div(a(u) grad u) analytically for manufactured_u."""

    pi = math.pi
    sin_px = torch.sin(pi * x)
    cos_px = torch.cos(pi * x)
    sin_py = torch.sin(pi * y)
    cos_py = torch.cos(pi * y)
    sin_2px = torch.sin(2.0 * pi * x)
    cos_2px = torch.cos(2.0 * pi * x)

    one_minus_x2 = 1.0 - x * x
    one_minus_y2 = 1.0 - y * y
    phi = one_minus_x2 * one_minus_y2
    phi_x = -2.0 * x * one_minus_y2
    phi_y = -2.0 * y * one_minus_x2
    phi_xx = -2.0 * one_minus_y2
    phi_yy = -2.0 * one_minus_x2

    b = params["c0"] + params["c1"] * x + params["c2"] * y + params["c3"] * x * y
    b_x = params["c1"] + params["c3"] * y
    b_y = params["c2"] + params["c3"] * x

    s = params["a1"] * sin_px * sin_py + params["a2"] * cos_2px * cos_py
    s_x = params["a1"] * pi * cos_px * sin_py - 2.0 * pi * params["a2"] * sin_2px * cos_py
    s_y = params["a1"] * pi * sin_px * cos_py - pi * params["a2"] * cos_2px * sin_py
    s_xx = -params["a1"] * pi * pi * sin_px * sin_py - 4.0 * pi * pi * params["a2"] * cos_2px * cos_py
    s_yy = -params["a1"] * pi * pi * sin_px * sin_py - pi * pi * params["a2"] * cos_2px * cos_py

    u = b + phi * s
    u_x = b_x + phi_x * s + phi * s_x
    u_y = b_y + phi_y * s + phi * s_y
    u_xx = phi_xx * s + 2.0 * phi_x * s_x + phi * s_xx
    u_yy = phi_yy * s + 2.0 * phi_y * s_y + phi * s_yy

    grad_sq = u_x * u_x + u_y * u_y
    k4 = _k_to_broadcast(k, u)
    base, dbase_du = _coefficient_base_and_derivative(
        u,
        coeff_mode,
        eps,
        softplus_beta,
    )
    a = torch.pow(base, k4)
    da_du = k4 * torch.pow(base, k4 - 1.0) * dbase_du
    return -(da_du * grad_sq + a * (u_xx + u_yy))


def discrete_rhs_from_u(
    u_exact: torch.Tensor,
    k: torch.Tensor,
    h: float,
    coeff_mode: str = "pow",
    discretization: str = "kirchhoff",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build f so u_exact satisfies the same discrete residual to roundoff."""

    f = torch.zeros_like(u_exact)
    validate_discretization(discretization, coeff_mode)
    if discretization == "kirchhoff":
        w = kirchhoff_transform(u_exact, k)
        wc = w[:, :, 1:-1, 1:-1]
        lap_w = (
            w[:, :, 1:-1, 2:]
            + w[:, :, 1:-1, :-2]
            + w[:, :, 2:, 1:-1]
            + w[:, :, :-2, 1:-1]
            - 4.0 * wc
        ) / (h * h)
        f[:, :, 1:-1, 1:-1] = -lap_w
        return f

    a = power_coefficient(u_exact, k, mode=coeff_mode, eps=eps)

    uc = u_exact[:, :, 1:-1, 1:-1]
    ue = u_exact[:, :, 1:-1, 2:]
    uw = u_exact[:, :, 1:-1, :-2]
    un = u_exact[:, :, 2:, 1:-1]
    us = u_exact[:, :, :-2, 1:-1]

    ac = a[:, :, 1:-1, 1:-1]
    ae = 0.5 * (ac + a[:, :, 1:-1, 2:])
    aw = 0.5 * (ac + a[:, :, 1:-1, :-2])
    an = 0.5 * (ac + a[:, :, 2:, 1:-1])
    a_s = 0.5 * (ac + a[:, :, :-2, 1:-1])

    minus_div = -(
        ae * (ue - uc)
        - aw * (uc - uw)
        + an * (un - uc)
        - a_s * (uc - us)
    ) / (h * h)
    f[:, :, 1:-1, 1:-1] = minus_div
    return f


def _sample_params(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    shape = (1, 1, 1, 1)
    return {
        "c0": torch.empty(shape, device=device, dtype=dtype).uniform_(1.2, 1.8),
        "c1": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.15, 0.15),
        "c2": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.15, 0.15),
        "c3": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.10, 0.10),
        "a1": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.20, 0.20),
        "a2": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.20, 0.20),
        "a3": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.12, 0.12),
        "a4": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.12, 0.12),
        "gx1": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.55, 0.55),
        "gy1": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.55, 0.55),
        "gx2": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.55, 0.55),
        "gy2": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.55, 0.55),
        "gs1": torch.empty(shape, device=device, dtype=dtype).uniform_(0.18, 0.45),
        "gs2": torch.empty(shape, device=device, dtype=dtype).uniform_(0.18, 0.45),
        "ga1": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.35, 0.35),
        "ga2": torch.empty(shape, device=device, dtype=dtype).uniform_(-0.35, 0.35),
    }


def sample_manufactured_problem(
    N: int,
    device: torch.device,
    k_range: tuple[float, float] = (0.5, 2.5),
    rhs_mode: str = "continuous",
    positive_only: bool = True,
    coeff_mode: str = "pow",
    discretization: str = "kirchhoff",
    solution_mode: str = "base",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | float]:
    """Return one manufactured problem instance."""

    validate_discretization(discretization, coeff_mode)
    if rhs_mode == "continuous" and solution_mode != "base":
        raise ValueError(
            "rhs_mode='continuous' currently supports only solution_mode='base'; "
            "use rhs_mode='discrete' for richer manufactured modes."
        )

    x0, y0, boundary_mask, interior_mask, h = make_grid(N, device=device, dtype=dtype)
    x = x0.clone().detach()
    y = y0.clone().detach()

    params = _sample_params(device, dtype)
    u_exact = manufactured_u(x, y, params, solution_mode=solution_mode)
    if positive_only:
        margin = 0.2 - u_exact.detach().min()
        if margin.item() > 0:
            params["c0"] = params["c0"] + margin
            u_exact = manufactured_u(x, y, params, solution_mode=solution_mode)

    k = torch.empty((1,), device=device, dtype=dtype).uniform_(k_range[0], k_range[1])
    if rhs_mode == "continuous":
        f = continuous_rhs_from_params(x, y, k, params, coeff_mode=coeff_mode)
    elif rhs_mode == "discrete":
        f = discrete_rhs_from_u(
            u_exact.detach(),
            k,
            h,
            coeff_mode=coeff_mode,
            discretization=discretization,
        )
    else:
        raise ValueError("rhs_mode must be 'continuous' or 'discrete'")

    u_exact = u_exact.detach()
    g_map = boundary_mask * u_exact

    return {
        "f": f.detach(),
        "g": g_map.detach(),
        "mask": boundary_mask.detach(),
        "interior_mask": interior_mask.detach(),
        "x": x0.detach(),
        "y": y0.detach(),
        "k": k.detach(),
        "u_exact": u_exact.detach(),
        "h": h,
    }


def collate_problem_list(problem_list: list[dict[str, torch.Tensor | float]]) -> dict[str, torch.Tensor | float]:
    batch: dict[str, torch.Tensor | float] = {}
    for key in ["f", "g", "mask", "interior_mask", "x", "y", "u_exact"]:
        batch[key] = torch.cat([p[key] for p in problem_list], dim=0)  # type: ignore[arg-type]
    batch["k"] = torch.cat([p["k"] for p in problem_list], dim=0)  # type: ignore[arg-type]
    batch["h"] = problem_list[0]["h"]
    return batch


def make_batch(
    N: int,
    batch_size: int,
    device: torch.device,
    k_range: tuple[float, float],
    rhs_mode: str,
    coeff_mode: str,
    discretization: str,
    solution_mode: str = "base",
) -> dict[str, torch.Tensor | float]:
    problems = [
        sample_manufactured_problem(
            N=N,
            device=device,
            k_range=k_range,
            rhs_mode=rhs_mode,
            positive_only=True,
            coeff_mode=coeff_mode,
            discretization=discretization,
            solution_mode=solution_mode,
        )
        for _ in range(batch_size)
    ]
    return collate_problem_list(problems)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


@dataclass
class InputStats:
    f_mean: float
    f_std: float
    k_mean: float
    k_std: float


def build_dataset(
    N: int,
    num_samples: int,
    device: torch.device,
    k_range: tuple[float, float],
    rhs_mode: str,
    coeff_mode: str,
    discretization: str,
    solution_mode: str = "base",
) -> dict[str, torch.Tensor | float]:
    problems = [
        sample_manufactured_problem(
            N=N,
            device=device,
            k_range=k_range,
            rhs_mode=rhs_mode,
            positive_only=True,
            coeff_mode=coeff_mode,
            discretization=discretization,
            solution_mode=solution_mode,
        )
        for _ in range(num_samples)
    ]
    return collate_problem_list(problems)


def slice_batch(
    data: dict[str, torch.Tensor | float],
    idx: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    f = data["f"]
    if torch.is_tensor(f) and idx.device != f.device:
        idx = idx.to(f.device)

    batch: dict[str, torch.Tensor | float] = {}
    for key in ["f", "g", "u_exact", "k"]:
        batch[key] = data[key][idx]  # type: ignore[index]
    for key in ["mask", "interior_mask", "x", "y"]:
        batch[key] = data[key][:1]  # type: ignore[index]
    batch["h"] = data["h"]
    return batch


def compute_input_stats(
    data: dict[str, torch.Tensor | float],
    k_range: tuple[float, float],
) -> InputStats:
    f = data["f"]  # type: ignore[assignment]
    fi = f[:, :, 1:-1, 1:-1]
    return InputStats(
        f_mean=float(fi.mean()),
        f_std=float(fi.std().clamp_min(1e-8)),
        k_mean=0.5 * (k_range[0] + k_range[1]),
        k_std=max((k_range[1] - k_range[0]) / math.sqrt(12.0), 1e-8),
    )


def forward_normalized(
    model: nn.Module,
    batch: dict[str, torch.Tensor | float],
    stats: InputStats,
    normalize_k: bool,
    positive_interior: bool,
) -> torch.Tensor:
    f_norm = (batch["f"] - stats.f_mean) / stats.f_std  # type: ignore[operator]
    inp = build_input_tensor(
        f_norm,
        batch["g"],  # type: ignore[arg-type]
        batch["mask"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["x"],  # type: ignore[arg-type]
        batch["y"],  # type: ignore[arg-type]
        k_mean=stats.k_mean if normalize_k else None,
        k_std=stats.k_std if normalize_k else None,
    )
    u_raw = model(inp)
    return enforce_dirichlet(
        u_raw,
        batch["g"],  # type: ignore[arg-type]
        batch["mask"],  # type: ignore[arg-type]
        positive_interior=positive_interior,
    )


def relative_pde_loss(
    u: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor,
    h: float,
    coeff_mode: str = "pow",
    discretization: str = "kirchhoff",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean((R / rms(f))^2) and the raw residual R."""

    res = compute_pde_residual(
        u,
        f,
        k,
        h,
        discretization=discretization,
        coeff_mode=coeff_mode,
    )
    fi = f[:, :, 1:-1, 1:-1]
    denom = fi.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-3)
    return (res / denom).pow(2).mean(), res


@dataclass
class EvalMetrics:
    rel_l2: float
    linf: float
    res_rmse: float
    res_linf: float
    boundary_max: float
    grad_rel_l2: float | None = None


@dataclass(frozen=True)
class PredictionConfig:
    N: int
    rhs_mode: str
    solution_mode: str
    k_min: float
    k_max: float
    test_seed: int
    samples: int


@dataclass
class _MetricSums:
    samples: int = 0
    rel_l2_sum: float = 0.0
    linf: float = 0.0
    res_sq_sum: float = 0.0
    res_count: int = 0
    res_linf: float = 0.0
    boundary_max: float = 0.0
    grad_rel_l2_sum: float = 0.0
    grad_samples: int = 0


def _accumulate_metrics(
    sums: _MetricSums,
    u_pred: torch.Tensor,
    batch: dict[str, torch.Tensor | float],
    coeff_mode: str,
    discretization: str,
    include_gradient: bool = True,
) -> None:
    n = u_pred.shape[0]
    interior_mask = batch["interior_mask"].expand_as(u_pred)  # type: ignore[union-attr]
    rel_l2 = relative_l2_error(
        u_pred,
        batch["u_exact"],  # type: ignore[arg-type]
        mask=interior_mask,
    )
    linf = interior_linf_error(
        u_pred,
        batch["u_exact"],  # type: ignore[arg-type]
        interior_mask,
    )
    res = compute_pde_residual(
        u_pred,
        batch["f"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["h"],  # type: ignore[arg-type]
        discretization=discretization,
        coeff_mode=coeff_mode,
    )
    bmax = boundary_max_error(
        u_pred,
        batch["g"],  # type: ignore[arg-type]
        batch["mask"],  # type: ignore[arg-type]
    )

    sums.samples += n
    sums.rel_l2_sum += float(rel_l2) * n
    sums.linf = max(sums.linf, float(linf))
    sums.res_sq_sum += float(res.pow(2).sum())
    sums.res_count += res.numel()
    sums.res_linf = max(sums.res_linf, float(res.abs().max()))
    sums.boundary_max = max(sums.boundary_max, float(bmax))

    if include_gradient:
        grad_rel = gradient_relative_l2_error(
            u_pred,
            batch["u_exact"],  # type: ignore[arg-type]
            batch["h"],  # type: ignore[arg-type]
        )
        sums.grad_rel_l2_sum += float(grad_rel) * n
        sums.grad_samples += n


def _finalize_metrics(sums: _MetricSums) -> EvalMetrics:
    if sums.samples < 1 or sums.res_count < 1:
        raise ValueError("Cannot finalize empty metric sums.")
    grad_rel_l2 = None
    if sums.grad_samples > 0:
        grad_rel_l2 = sums.grad_rel_l2_sum / sums.grad_samples
    return EvalMetrics(
        rel_l2=sums.rel_l2_sum / sums.samples,
        linf=sums.linf,
        res_rmse=math.sqrt(sums.res_sq_sum / sums.res_count),
        res_linf=sums.res_linf,
        boundary_max=sums.boundary_max,
        grad_rel_l2=grad_rel_l2,
    )


def format_eval_metrics(label: str, metrics: EvalMetrics) -> str:
    grad = ""
    if metrics.grad_rel_l2 is not None:
        grad = f" gradRelL2={metrics.grad_rel_l2:.4e}"
    return (
        f"{label} relL2={metrics.rel_l2:.4e} linf={metrics.linf:.4e} "
        f"resRMSE={metrics.res_rmse:.4f} resLinf={metrics.res_linf:.4e} "
        f"bmax={metrics.boundary_max:.2e}{grad}"
    )


def evaluate_on(
    model: nn.Module,
    data: dict[str, torch.Tensor | float],
    stats: InputStats,
    batch_size: int,
    coeff_mode: str,
    discretization: str,
    normalize_k: bool,
    positive_interior: bool,
    compare_transform_solver: bool = False,
    include_gradient: bool = True,
    cg_tol: float | None = None,
    cg_max_iter: int | None = None,
) -> dict[str, EvalMetrics]:
    return _evaluate_on_impl(
        model=model,
        data=data,
        stats=stats,
        batch_size=batch_size,
        coeff_mode=coeff_mode,
        discretization=discretization,
        normalize_k=normalize_k,
        positive_interior=positive_interior,
        compare_transform_solver=compare_transform_solver,
        include_gradient=include_gradient,
        cg_tol=cg_tol,
        cg_max_iter=cg_max_iter,
    )


def _evaluate_on_impl(
    model: nn.Module,
    data: dict[str, torch.Tensor | float],
    stats: InputStats,
    batch_size: int,
    coeff_mode: str,
    discretization: str,
    normalize_k: bool,
    positive_interior: bool,
    compare_transform_solver: bool = False,
    include_gradient: bool = True,
    cg_tol: float | None = None,
    cg_max_iter: int | None = None,
) -> dict[str, EvalMetrics]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validate_discretization(discretization, coeff_mode)
    if compare_transform_solver and discretization != "kirchhoff":
        raise ValueError(
            "--compare-transform-solver requires discretization='kirchhoff'; "
            "the flux stencil and transformed Poisson solver are different discrete equations."
        )

    model.eval()
    n = data["f"].shape[0]  # type: ignore[union-attr]
    device = data["f"].device  # type: ignore[union-attr]
    totals = {"cnn": _MetricSums()}
    if compare_transform_solver:
        totals["transform"] = _MetricSums()

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = torch.arange(start, min(start + batch_size, n), device=device)
            batch = slice_batch(data, idx)
            u_pred = forward_normalized(
                model,
                batch,
                stats,
                normalize_k=normalize_k,
                positive_interior=positive_interior,
            )
            _accumulate_metrics(
                totals["cnn"],
                u_pred,
                batch,
                coeff_mode,
                discretization,
                include_gradient,
            )
            if compare_transform_solver:
                u_transform = solve_transformed_poisson(
                    batch["f"],  # type: ignore[arg-type]
                    batch["g"],  # type: ignore[arg-type]
                    batch["k"],  # type: ignore[arg-type]
                    batch["mask"],  # type: ignore[arg-type]
                    batch["h"],  # type: ignore[arg-type]
                    cg_tol=cg_tol,
                    cg_max_iter=cg_max_iter,
                )
                _accumulate_metrics(
                    totals["transform"],
                    u_transform,
                    batch,
                    coeff_mode,
                    discretization,
                    include_gradient,
                )

    return {name: _finalize_metrics(sums) for name, sums in totals.items()}


def evaluate_predictions(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | float],
    coeff_mode: str,
    discretization: str,
    include_gradient: bool = True,
) -> dict[str, EvalMetrics]:
    metrics: dict[str, EvalMetrics] = {}
    for name, u_pred in predictions.items():
        sums = _MetricSums()
        _accumulate_metrics(sums, u_pred, batch, coeff_mode, discretization, include_gradient)
        metrics[name] = _finalize_metrics(sums)
    return metrics


def oracle_floor(
    data: dict[str, torch.Tensor | float],
    coeff_mode: str,
    discretization: str,
) -> float:
    """Raw resRMSE of the exact solution on the validation set."""

    with torch.no_grad():
        res = compute_pde_residual(
            data["u_exact"],  # type: ignore[arg-type]
            data["f"],  # type: ignore[arg-type]
            data["k"],  # type: ignore[arg-type]
            data["h"],  # type: ignore[arg-type]
            discretization=discretization,
            coeff_mode=coeff_mode,
        )
        return float(res.pow(2).mean().sqrt())


def adamw_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def add_path_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    return path.with_name(f"{path.name}_{suffix}")


def apply_positive_interior_default(args: argparse.Namespace) -> None:
    if getattr(args, "positive_interior", None) is None:
        args.positive_interior = getattr(args, "coeff_mode", "pow") == "pow"


def train(args: argparse.Namespace) -> nn.Module:
    set_seed(args.seed)
    device = select_device(args.device)
    k_range = (args.k_min, args.k_max)
    validate_discretization(args.discretization, args.coeff_mode)
    if args.compare_transform_solver and args.discretization != "kirchhoff":
        raise ValueError(
            "--compare-transform-solver requires discretization='kirchhoff'; "
            "the flux stencil and transformed Poisson solver are different discrete equations."
        )
    eval_batch_size = args.eval_batch_size or args.batch_size
    if args.eval_batches is not None:
        args.val_samples = args.eval_batches * eval_batch_size
    if args.train_samples < 1 or args.val_samples < 1:
        raise ValueError("train_samples and val_samples must be positive")
    if args.batch_size < 1 or eval_batch_size < 1:
        raise ValueError("batch_size and eval_batch_size must be positive")
    if args.eval_every < 1:
        raise ValueError("eval_every must be positive")
    if args.pde_scale_batches < 1:
        raise ValueError("pde_scale_batches must be positive")
    apply_positive_interior_default(args)

    t0 = time.time()
    print(
        f"generating train set: {args.train_samples} samples "
        f"(N={args.N}, rhs_mode={args.rhs_mode}, "
        f"discretization={args.discretization}, solution_mode={args.solution_mode})",
        flush=True,
    )
    train_data = build_dataset(
        args.N,
        args.train_samples,
        device,
        k_range,
        args.rhs_mode,
        args.coeff_mode,
        args.discretization,
        args.solution_mode,
    )
    print(f"generating val set: {args.val_samples} samples", flush=True)
    val_data = build_dataset(
        args.N,
        args.val_samples,
        device,
        k_range,
        args.rhs_mode,
        args.coeff_mode,
        args.discretization,
        args.solution_mode,
    )
    stats = compute_input_stats(train_data, k_range)
    floor = oracle_floor(val_data, args.coeff_mode, args.discretization)

    model = UNet2d(in_ch=6, out_ch=1, base_ch=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(
        adamw_param_groups(model, args.weight_decay),
        lr=args.lr,
    )
    steps_per_epoch = math.ceil(args.train_samples / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    save_path = Path(args.save_path) if args.save_path else None
    best_save_path = add_path_suffix(save_path, "best") if save_path is not None else None
    best_val_rel_l2 = math.inf

    print(
        f"device={device} N={args.N} rhs_mode={args.rhs_mode} "
        f"discretization={args.discretization} "
        f"train={args.train_samples} val={args.val_samples} "
        f"gen_time={time.time() - t0:.1f}s"
    )
    print(f"f stats: mean={stats.f_mean:.3f} std={stats.f_std:.3f}")
    print(f"oracle resRMSE floor: {floor:.4f}")
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    step = 0
    pde_scale: float | None = None
    pde_scale_samples: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.train_samples, device=device)
        for start in range(0, args.train_samples, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch = slice_batch(train_data, idx)
            u_pred = forward_normalized(
                model,
                batch,
                stats,
                normalize_k=args.normalize_k,
                positive_interior=args.positive_interior,
            )
            loss_pde, res = relative_pde_loss(
                u_pred,
                batch["f"],  # type: ignore[arg-type]
                batch["k"],  # type: ignore[arg-type]
                batch["h"],  # type: ignore[arg-type]
                coeff_mode=args.coeff_mode,
                discretization=args.discretization,
            )
            loss_sup = masked_mse(
                u_pred,
                batch["u_exact"],  # type: ignore[arg-type]
                batch["interior_mask"].expand_as(u_pred),  # type: ignore[union-attr]
            )

            if args.pure_physics:
                loss = loss_pde
            elif epoch < args.pretrain_epochs:
                loss = loss_sup
            else:
                if pde_scale is None:
                    pde_scale_samples.append(
                        float(loss_sup.detach() / loss_pde.detach().clamp_min(1e-12))
                    )
                    current_pde_scale = float(median(pde_scale_samples))
                    if len(pde_scale_samples) >= args.pde_scale_batches:
                        pde_scale = current_pde_scale
                        print(
                            "[auto-balance] "
                            f"pde_scale={pde_scale:.3e} "
                            f"(median of {len(pde_scale_samples)} batches)"
                        )
                else:
                    current_pde_scale = pde_scale
                loss = (
                    args.supervised_weight * loss_sup
                    + args.pde_weight * current_pde_scale * loss_pde
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            train_res_rmse = float(res.pow(2).mean().sqrt().detach())
            train_boundary = float(
                boundary_max_error(
                    u_pred,
                    batch["g"],  # type: ignore[arg-type]
                    batch["mask"],  # type: ignore[arg-type]
                ).detach()
            )
            val_metrics = evaluate_on(
                model,
                val_data,
                stats,
                eval_batch_size,
                args.coeff_mode,
                args.discretization,
                normalize_k=args.normalize_k,
                positive_interior=args.positive_interior,
                compare_transform_solver=args.compare_transform_solver,
                cg_tol=args.cg_tol,
                cg_max_iter=args.cg_max_iter,
            )
            val_cnn = val_metrics["cnn"]
            if best_save_path is not None and val_cnn.rel_l2 < best_val_rel_l2:
                best_val_rel_l2 = val_cnn.rel_l2
                best_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_stats": vars(stats),
                        "args": vars(args),
                        "epoch": epoch,
                        "step": step,
                        "best_val_rel_l2": best_val_rel_l2,
                        "checkpoint_type": "best_val_rel_l2",
                    },
                    best_save_path,
                )
                print(f"saved best checkpoint: {best_save_path}")
            print(
                f"epoch={epoch:03d} step={step:05d} lr={scheduler.get_last_lr()[0]:.2e} "
                f"loss={float(loss.detach()):.3e} sup={float(loss_sup.detach()):.3e} "
                f"pde={float(loss_pde.detach()):.3e} train_resRMSE={train_res_rmse:.4f} "
                f"bmax={train_boundary:.2e} | {format_eval_metrics('val cnn', val_cnn)} "
                f"(floor {floor:.4f})"
            )
            if "transform" in val_metrics:
                print(f"  {format_eval_metrics('val transform', val_metrics['transform'])}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_stats": vars(stats),
                "args": vars(args),
                "checkpoint_type": "final",
            },
            save_path,
        )
        print(f"saved checkpoint: {save_path}")

    return model


def load_model_from_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, InputStats, argparse.Namespace]:
    with torch.serialization.safe_globals([argparse.Namespace, InputStats]):
        checkpoint = torch.load(path, map_location=device)
    raw_args = checkpoint.get("args", {})
    if isinstance(raw_args, argparse.Namespace):
        checkpoint_args = raw_args
    elif isinstance(raw_args, dict):
        checkpoint_args = argparse.Namespace(**raw_args)
    else:
        raise ValueError("Checkpoint args must be a dict or argparse.Namespace.")
    base_ch = int(getattr(checkpoint_args, "base_ch", 32))

    model = UNet2d(in_ch=6, out_ch=1, base_ch=base_ch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats_dict = checkpoint.get("input_stats")
    if isinstance(stats_dict, InputStats):
        stats = stats_dict
    elif isinstance(stats_dict, dict):
        stats = InputStats(
            f_mean=float(stats_dict["f_mean"]),
            f_std=float(stats_dict["f_std"]),
            k_mean=float(stats_dict["k_mean"]),
            k_std=float(stats_dict["k_std"]),
        )
    else:
        k_min = float(getattr(checkpoint_args, "k_min", 0.5))
        k_max = float(getattr(checkpoint_args, "k_max", 2.5))
        stats = InputStats(
            f_mean=0.0,
            f_std=1.0,
            k_mean=0.5 * (k_min + k_max),
            k_std=max((k_max - k_min) / math.sqrt(12.0), 1e-8),
        )
        print("checkpoint has no input_stats; using legacy identity f normalization")
    if not hasattr(checkpoint_args, "discretization"):
        checkpoint_args.discretization = "flux"
    apply_positive_interior_default(checkpoint_args)
    return model, stats, checkpoint_args


def predict_on_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor | float],
    stats: InputStats,
    args: argparse.Namespace,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return forward_normalized(
            model,
            batch,
            stats,
            normalize_k=getattr(args, "normalize_k", True),
            positive_interior=getattr(args, "positive_interior", False),
        )


def resolve_checkpoint_model_config(checkpoint_args: argparse.Namespace) -> argparse.Namespace:
    model_config = argparse.Namespace(
        base_ch=int(getattr(checkpoint_args, "base_ch", 32)),
        coeff_mode=getattr(checkpoint_args, "coeff_mode", "pow"),
        discretization=getattr(checkpoint_args, "discretization", "flux"),
        normalize_k=getattr(checkpoint_args, "normalize_k", True),
        positive_interior=getattr(checkpoint_args, "positive_interior", None),
    )
    validate_discretization(model_config.discretization, model_config.coeff_mode)
    apply_positive_interior_default(model_config)
    return model_config


def resolve_prediction_config(
    runtime_args: argparse.Namespace,
    checkpoint_args: argparse.Namespace,
) -> PredictionConfig:
    N = int(getattr(checkpoint_args, "N", 32))
    rhs_mode = getattr(checkpoint_args, "rhs_mode", "continuous")
    solution_mode = getattr(checkpoint_args, "solution_mode", "base")
    k_min = float(getattr(checkpoint_args, "k_min", 0.5))
    k_max = float(getattr(checkpoint_args, "k_max", 2.5))

    if getattr(runtime_args, "predict_N", None) is not None:
        N = int(runtime_args.predict_N)
    if getattr(runtime_args, "predict_rhs_mode", None) is not None:
        rhs_mode = runtime_args.predict_rhs_mode
    if getattr(runtime_args, "predict_solution_mode", None) is not None:
        solution_mode = runtime_args.predict_solution_mode
    if getattr(runtime_args, "predict_k_min", None) is not None:
        k_min = float(runtime_args.predict_k_min)
    if getattr(runtime_args, "predict_k_max", None) is not None:
        k_max = float(runtime_args.predict_k_max)

    return PredictionConfig(
        N=N,
        rhs_mode=rhs_mode,
        solution_mode=solution_mode,
        k_min=k_min,
        k_max=k_max,
        test_seed=int(getattr(runtime_args, "test_seed", getattr(runtime_args, "seed", 1234))),
        samples=int(runtime_args.predict_samples),
    )


def _save_prediction_payload(
    path: str | Path,
    batch: dict[str, torch.Tensor | float],
    u_pred: torch.Tensor,
    u_transform: torch.Tensor | None,
    metrics: dict[str, EvalMetrics],
    prediction_config: PredictionConfig,
    model_config: argparse.Namespace,
) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_metadata = asdict(prediction_config)
    args_metadata = {**vars(model_config), **prediction_metadata}

    tensors = {
        "f": batch["f"],
        "g": batch["g"],
        "k": batch["k"],
        "u_exact": batch["u_exact"],
        "u_pred": u_pred,
    }
    if u_transform is not None:
        tensors["u_transform"] = u_transform
    if save_path.suffix.lower() == ".npz":
        import numpy as np

        np_payload = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in tensors.items()
            if torch.is_tensor(tensor)
        }
        np_payload["h"] = np.array(float(batch["h"]))  # type: ignore[arg-type]
        np_payload["prediction_config"] = np.array(json.dumps(prediction_metadata))
        np.savez(save_path, **np_payload)
    else:
        torch_payload = {
            name: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
            for name, tensor in tensors.items()
        }
        torch_payload["h"] = float(batch["h"])  # type: ignore[arg-type]
        torch_payload["metrics"] = {name: vars(value) for name, value in metrics.items()}
        torch_payload["prediction_config"] = prediction_metadata
        torch_payload["args"] = args_metadata
        torch.save(torch_payload, save_path)
    print(f"saved predictions: {save_path}")


def run_prediction(args: argparse.Namespace) -> None:
    if not args.load_path:
        raise ValueError("--predict-manufactured requires --load-path")
    if args.predict_samples < 1:
        raise ValueError("--predict-samples must be positive")

    set_seed(args.seed)
    device = select_device(args.device)
    model, stats, checkpoint_args = load_model_from_checkpoint(args.load_path, device)
    model_config = resolve_checkpoint_model_config(checkpoint_args)
    prediction_config = resolve_prediction_config(args, checkpoint_args)
    k_range = (prediction_config.k_min, prediction_config.k_max)

    print(
        "resolved prediction config: "
        f"N={prediction_config.N} rhs_mode={prediction_config.rhs_mode} "
        f"solution_mode={prediction_config.solution_mode} "
        f"discretization={model_config.discretization} "
        f"k_range=({prediction_config.k_min:g}, {prediction_config.k_max:g}) "
        f"test_seed={prediction_config.test_seed} samples={prediction_config.samples}",
        flush=True,
    )
    set_seed(prediction_config.test_seed)
    batch = build_dataset(
        prediction_config.N,
        prediction_config.samples,
        device,
        k_range,
        prediction_config.rhs_mode,
        model_config.coeff_mode,
        model_config.discretization,
        prediction_config.solution_mode,
    )
    u_pred = predict_on_batch(model, batch, stats, model_config)

    u_transform: torch.Tensor | None = None
    if args.compare_transform_solver:
        if model_config.discretization != "kirchhoff":
            raise ValueError(
                "--compare-transform-solver requires discretization='kirchhoff'; "
                "the flux stencil and transformed Poisson solver are different discrete equations."
            )
        else:
            with torch.no_grad():
                u_transform, cg_diagnostics = solve_transformed_poisson(
                    batch["f"],  # type: ignore[arg-type]
                    batch["g"],  # type: ignore[arg-type]
                    batch["k"],  # type: ignore[arg-type]
                    batch["mask"],  # type: ignore[arg-type]
                    batch["h"],  # type: ignore[arg-type]
                    cg_tol=args.cg_tol,
                    cg_max_iter=args.cg_max_iter,
                    return_diagnostics=True,
                )
            print(format_cg_diagnostics("predict transform", cg_diagnostics))

    predictions = {"cnn": u_pred}
    if args.compare_transform_solver and u_transform is not None:
        predictions["transform"] = u_transform
    metrics = evaluate_predictions(
        predictions,
        batch,
        model_config.coeff_mode,
        model_config.discretization,
    )
    for name, metric in metrics.items():
        print(format_eval_metrics(f"predict {name}", metric))

    if args.predict_save:
        _save_prediction_payload(
            args.predict_save,
            batch,
            u_pred,
            u_transform,
            metrics,
            prediction_config,
            model_config,
        )


def run_grid_convergence(args: argparse.Namespace) -> None:
    if args.predict_samples < 1:
        raise ValueError("--predict-samples must be positive")
    if args.coeff_mode != "pow":
        raise ValueError("--grid-convergence uses the transformed solver and requires coeff_mode='pow'.")
    if args.discretization != "kirchhoff":
        raise ValueError(
            "--grid-convergence uses the transformed Poisson solver and requires "
            "discretization='kirchhoff'; the flux stencil solves a different discrete equation."
        )

    set_seed(args.seed)
    device = select_device(args.device)
    k_range = (args.k_min, args.k_max)
    previous_h: float | None = None
    previous_rel_l2: float | None = None

    print(
        "transformed Poisson grid convergence "
        f"(samples={args.predict_samples}, rhs_mode={args.rhs_mode}, "
        f"discretization={args.discretization}, solution_mode={args.solution_mode})"
    )
    if args.rhs_mode == "discrete":
        print(
            "note: rhs_mode='discrete' uses the Kirchhoff-transformed "
            "five-point Laplacian, so the transformed linear solve matches "
            "the selected discrete equation."
        )

    for N in args.grid_Ns:
        # Reuse the same manufactured parameters at each N so rates compare grids.
        set_seed(args.seed)
        data = build_dataset(
            N,
            args.predict_samples,
            device,
            k_range,
            args.rhs_mode,
            args.coeff_mode,
            args.discretization,
            args.solution_mode,
        )
        with torch.no_grad():
            u_transform, cg_diagnostics = solve_transformed_poisson(
                data["f"],  # type: ignore[arg-type]
                data["g"],  # type: ignore[arg-type]
                data["k"],  # type: ignore[arg-type]
                data["mask"],  # type: ignore[arg-type]
                data["h"],  # type: ignore[arg-type]
                cg_tol=args.cg_tol,
                cg_max_iter=args.cg_max_iter,
                return_diagnostics=True,
            )
        metrics = evaluate_predictions(
            {"transform": u_transform},
            data,
            args.coeff_mode,
            args.discretization,
        )["transform"]
        h = float(data["h"])  # type: ignore[arg-type]
        rate = "-"
        if previous_h is not None and previous_rel_l2 is not None and metrics.rel_l2 > 0:
            rate_value = math.log(previous_rel_l2 / metrics.rel_l2) / math.log(previous_h / h)
            rate = f"{rate_value:.2f}"
        print(
            f"N={N:4d} h={h:.4e} relL2={metrics.rel_l2:.4e} "
            f"rate={rate:>5} linf={metrics.linf:.4e} "
            f"resRMSE={metrics.res_rmse:.4f} "
            f"cg_iters={cg_diagnostics.iterations} "
            f"cg_relres={cg_diagnostics.final_max_relative_residual:.3e}"
        )
        previous_h = h
        previous_rel_l2 = metrics.rel_l2


def run_residual_check(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    k_range = (args.k_min, args.k_max)
    validate_discretization(args.discretization, args.coeff_mode)
    batch = make_batch(
        args.N,
        args.batch_size,
        device,
        k_range,
        "discrete",
        args.coeff_mode,
        args.discretization,
        args.solution_mode,
    )
    with torch.no_grad():
        _, res = pde_loss(
            batch["u_exact"],  # type: ignore[arg-type]
            batch["f"],  # type: ignore[arg-type]
            batch["k"],  # type: ignore[arg-type]
            batch["h"],  # type: ignore[arg-type]
            scale_with_h=False,
            coeff_mode=args.coeff_mode,
            discretization=args.discretization,
        )
        rmse = torch.sqrt((res * res).mean())
        max_abs = res.abs().max()
        scaled_rmse = torch.sqrt(((res * (batch["h"] ** 2)) ** 2).mean())  # type: ignore[operator]
    print(
        "discrete RHS residual check: "
        f"discretization={args.discretization}, "
        f"rmse={float(rmse.cpu()):.4e}, "
        f"scaled_rmse={float(scaled_rmse.cpu()):.4e}, "
        f"max_abs={float(max_abs.cpu()):.4e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=32, help="Interior grid size. Full grid is (N+2)^2.")
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40, help="Real epochs over the fixed dataset.")
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Defaults to --batch-size.")
    parser.add_argument("--eval-batches", type=int, default=None, help="Compatibility alias for val samples.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--rhs-mode", choices=["continuous", "discrete"], default="continuous")
    parser.add_argument("--discretization", choices=["flux", "kirchhoff"], default="kirchhoff")
    parser.add_argument(
        "--solution-mode",
        choices=["base", "fourier", "gaussian", "mixed"],
        default="base",
    )
    parser.add_argument("--coeff-mode", choices=["pow", "abs_pow", "softplus_pow"], default="pow")
    parser.add_argument("--k-min", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--pde-scale-batches", type=int, default=20)
    parser.add_argument("--pure-physics", action="store_true")
    parser.add_argument(
        "--positive-interior",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to on when --coeff-mode pow.",
    )
    parser.add_argument("--normalize-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-path", default="")
    parser.add_argument("--load-path", default="")
    parser.add_argument("--predict-manufactured", action="store_true")
    parser.add_argument("--predict-samples", type=int, default=8)
    parser.add_argument("--predict-save", default="")
    parser.add_argument("--predict-N", "--predict-n", dest="predict_N", type=int, default=None)
    parser.add_argument(
        "--predict-rhs-mode",
        choices=["continuous", "discrete"],
        default=None,
    )
    parser.add_argument(
        "--predict-solution-mode",
        choices=["base", "fourier", "gaussian", "mixed"],
        default=None,
    )
    parser.add_argument("--predict-k-min", type=float, default=None)
    parser.add_argument("--predict-k-max", type=float, default=None)
    parser.add_argument("--test-seed", type=int, default=1234)
    parser.add_argument("--compare-transform-solver", action="store_true")
    parser.add_argument("--cg-tol", type=float, default=None)
    parser.add_argument("--cg-max-iter", type=int, default=None)
    parser.add_argument("--grid-convergence", action="store_true")
    parser.add_argument("--grid-Ns", "--grid-ns", dest="grid_Ns", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--check-residual", action="store_true")
    parser.add_argument("--train-after-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_positive_interior_default(args)
    if args.check_residual:
        run_residual_check(args)
        if not args.train_after_check:
            return
    if args.grid_convergence:
        run_grid_convergence(args)
        return
    if args.predict_manufactured:
        run_prediction(args)
        return
    if args.load_path:
        raise ValueError("--load-path is only used with --predict-manufactured.")
    train(args)


if __name__ == "__main__":
    main()
