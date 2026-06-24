"""CNN baseline for a nonlinear Poisson equation in flux form.

The model learns a grid-to-grid map for

    -div(a(u) grad u) = f,  a(u) = u**k,  u = g on boundary,

on Omega = [-1, 1]^2.  The loss is the conservative finite-difference
flux residual on the interior grid nodes, with Dirichlet boundary values
enforced exactly by an output projection.

Recommended:
    python Baseline.py --N 32 --train-samples 2048 --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Build the six-channel input [f, boundary g, mask, k, x, y].

    Interior u_exact values are intentionally excluded from the CNN input.
    """

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


def power_coefficient(
    u: torch.Tensor,
    k: torch.Tensor | float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a(u) = u^k, with a small clamp as a numerical guard."""

    k = _k_to_broadcast(k, u)
    if torch.allclose(k, k.round(), atol=1e-6, rtol=0.0):
        return torch.pow(u, k)
    return torch.pow(torch.clamp(u, min=eps), k)


def flux_minus_divergence(
    u: torch.Tensor,
    k: torch.Tensor | float,
    h: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return interior -div_h(a(u) grad_h u) with shared face fluxes."""

    a = power_coefficient(u, k, eps=eps)

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
    return -div_h


def nonlinear_poisson_residual(
    u: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor | float,
    h: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Interior residual R = -div_h(a(u) grad_h u) - f.

    Interface coefficients are arithmetic averages, so each face has one
    conservative flux shared by adjacent control volumes.
    """

    return flux_minus_divergence(
        u,
        k,
        h,
        eps=eps,
    ) - f[:, :, 1:-1, 1:-1]


def compute_pde_residual(
    u: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor | float,
    h: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    return nonlinear_poisson_residual(
        u,
        f,
        k,
        h,
        eps=eps,
    )


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
    params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Deterministic manufactured solution."""

    return torch.sin(math.pi * x) * torch.cos(math.pi * y)


def continuous_rhs_from_params(
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
    params: dict[str, torch.Tensor] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute f = -div(a(u) grad u) with analytic derivatives of manufactured_u."""

    pi = math.pi
    u = manufactured_u(x, y, params)
    u_x = pi * torch.cos(pi * x) * torch.cos(pi * y)
    u_y = -pi * torch.sin(pi * x) * torch.sin(pi * y)
    u_xx = -pi * pi * u
    u_yy = -pi * pi * u

    grad_sq = u_x * u_x + u_y * u_y
    k4 = _k_to_broadcast(k, u)
    if torch.allclose(k4, k4.round(), atol=1e-6, rtol=0.0):
        a = torch.pow(u, k4)
        da_du = k4 * torch.pow(u, k4 - 1.0)
    else:
        base = torch.clamp(u, min=eps)
        dbase_du = (u > eps).to(dtype=u.dtype)
        a = torch.pow(base, k4)
        da_du = k4 * torch.pow(base, k4 - 1.0) * dbase_du
    return -(da_du * grad_sq + a * (u_xx + u_yy))


def sample_manufactured_problem(
    N: int,
    device: torch.device,
    k_range: tuple[float, float] = (2.0, 2.0),
    positive_only: bool = True,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | float]:
    """Return one manufactured problem instance.

    The data path is LeVeque-style manufactured forcing: use fixed analytic
    u_exact = sin(pi x) cos(pi y), fixed k = 2, and compute
    f = -div(u_exact^k grad u_exact) with analytic derivatives. u_exact is not
    used by the training loss.
    """

    x0, y0, boundary_mask, interior_mask, h = make_grid(N, device=device, dtype=dtype)
    x = x0.clone().detach()
    y = y0.clone().detach()

    params: dict[str, torch.Tensor] | None = None
    u_exact = manufactured_u(x, y, params)

    k = torch.full((1,), 2.0, device=device, dtype=dtype)
    f = continuous_rhs_from_params(x, y, k, params)

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
) -> dict[str, torch.Tensor | float]:
    problems = [
        sample_manufactured_problem(
            N=N,
            device=device,
            k_range=k_range,
            positive_only=True,
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
    k = data["k"]  # type: ignore[assignment]
    fi = f[:, :, 1:-1, 1:-1]
    return InputStats(
        f_mean=float(fi.mean()),
        f_std=float(fi.std().clamp_min(1e-8)),
        k_mean=float(k.mean()),
        k_std=float(k.std(unbiased=False).clamp_min(1e-8)),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean((R / rms(f))^2) and raw flux residual R."""

    res = compute_pde_residual(
        u,
        f,
        k,
        h,
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
    normalize_k: bool,
    positive_interior: bool,
    include_gradient: bool = True,
) -> dict[str, EvalMetrics]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model.eval()
    n = data["f"].shape[0]  # type: ignore[union-attr]
    device = data["f"].device  # type: ignore[union-attr]
    totals = {"cnn": _MetricSums()}

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
                include_gradient,
            )

    return {name: _finalize_metrics(sums) for name, sums in totals.items()}


def evaluate_predictions(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | float],
    include_gradient: bool = True,
) -> dict[str, EvalMetrics]:
    metrics: dict[str, EvalMetrics] = {}
    for name, u_pred in predictions.items():
        sums = _MetricSums()
        _accumulate_metrics(sums, u_pred, batch, include_gradient)
        metrics[name] = _finalize_metrics(sums)
    return metrics


def oracle_floor(
    data: dict[str, torch.Tensor | float],
) -> float:
    """Raw resRMSE of the exact solution on the validation set."""

    with torch.no_grad():
        res = compute_pde_residual(
            data["u_exact"],  # type: ignore[arg-type]
            data["f"],  # type: ignore[arg-type]
            data["k"],  # type: ignore[arg-type]
            data["h"],  # type: ignore[arg-type]
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
    if not hasattr(args, "positive_interior"):
        args.positive_interior = False


def train(args: argparse.Namespace) -> nn.Module:
    set_seed(args.seed)
    device = select_device(args.device)
    k_range = (args.k_min, args.k_max)
    eval_batch_size = args.eval_batch_size or args.batch_size
    if args.eval_batches is not None:
        args.val_samples = args.eval_batches * eval_batch_size
    if args.train_samples < 1 or args.val_samples < 1:
        raise ValueError("train_samples and val_samples must be positive")
    if args.batch_size < 1 or eval_batch_size < 1:
        raise ValueError("batch_size and eval_batch_size must be positive")
    if args.eval_every < 1:
        raise ValueError("eval_every must be positive")
    apply_positive_interior_default(args)

    t0 = time.time()
    print(
        f"generating train set: {args.train_samples} samples "
        f"(N={args.N}, continuous manufactured forcing)",
        flush=True,
    )
    train_data = build_dataset(
        args.N,
        args.train_samples,
        device,
        k_range,
    )
    print(f"generating val set: {args.val_samples} samples", flush=True)
    val_data = build_dataset(
        args.N,
        args.val_samples,
        device,
        k_range,
    )
    stats = compute_input_stats(train_data, k_range)
    floor = oracle_floor(val_data)
    floor_label = "oracle resRMSE floor (finite-difference truncation floor)"

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
        f"device={device} N={args.N} train={args.train_samples} val={args.val_samples} "
        f"gen_time={time.time() - t0:.1f}s"
    )
    print(f"f stats: mean={stats.f_mean:.3f} std={stats.f_std:.3f}")
    print(f"{floor_label}: {floor:.4f}")
    print("training objective: normalized flux-form PDE residual")
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    step = 0
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
            )
            loss = loss_pde

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
                normalize_k=args.normalize_k,
                positive_interior=args.positive_interior,
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
                f"loss_pde={float(loss_pde.detach()):.3e} train_resRMSE={train_res_rmse:.4f} "
                f"bmax={train_boundary:.2e} | {format_eval_metrics('val cnn', val_cnn)} "
                f"(oracle floor {floor:.4f})"
            )
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
        k_min = float(getattr(checkpoint_args, "k_min", 2.0))
        k_max = float(getattr(checkpoint_args, "k_max", 2.0))
        stats = InputStats(
            f_mean=0.0,
            f_std=1.0,
            k_mean=0.5 * (k_min + k_max),
            k_std=max((k_max - k_min) / math.sqrt(12.0), 1e-8),
        )
        print("checkpoint has no input_stats; using legacy identity f normalization")
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
        normalize_k=getattr(checkpoint_args, "normalize_k", True),
        positive_interior=getattr(checkpoint_args, "positive_interior", False),
    )
    apply_positive_interior_default(model_config)
    return model_config


def resolve_prediction_config(
    runtime_args: argparse.Namespace,
    checkpoint_args: argparse.Namespace,
) -> PredictionConfig:
    N = int(getattr(checkpoint_args, "N", 32))
    k_min = float(getattr(checkpoint_args, "k_min", 2.0))
    k_max = float(getattr(checkpoint_args, "k_max", 2.0))

    if getattr(runtime_args, "predict_N", None) is not None:
        N = int(runtime_args.predict_N)
    if getattr(runtime_args, "predict_k_min", None) is not None:
        k_min = float(runtime_args.predict_k_min)
    if getattr(runtime_args, "predict_k_max", None) is not None:
        k_max = float(runtime_args.predict_k_max)

    return PredictionConfig(
        N=N,
        k_min=k_min,
        k_max=k_max,
        test_seed=int(getattr(runtime_args, "test_seed", getattr(runtime_args, "seed", 1234))),
        samples=int(runtime_args.predict_samples),
    )


def _save_prediction_payload(
    path: str | Path,
    batch: dict[str, torch.Tensor | float],
    u_pred: torch.Tensor,
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
    if save_path.suffix.lower() == ".npz":
        import numpy as np

        np_payload = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in tensors.items()
            if torch.is_tensor(tensor)
        }
        np_payload["h"] = np.array(float(batch["h"]))  # type: ignore[arg-type]
        np_payload["prediction_config"] = np.array(json.dumps(prediction_metadata))
        np_payload["evaluation_only_metrics"] = np.array(json.dumps(sorted(EVALUATION_ONLY_METRIC_FIELDS)))
        np.savez(save_path, **np_payload)
    else:
        torch_payload = {
            name: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
            for name, tensor in tensors.items()
        }
        torch_payload["h"] = float(batch["h"])  # type: ignore[arg-type]
        torch_payload["metrics"] = {name: vars(value) for name, value in metrics.items()}
        torch_payload["evaluation_only_metrics"] = sorted(EVALUATION_ONLY_METRIC_FIELDS)
        torch_payload["prediction_config"] = prediction_metadata
        torch_payload["args"] = args_metadata
        torch.save(torch_payload, save_path)
    print(f"saved predictions: {save_path}")


PredictionMetricValue = float | int | str

PREDICTION_CSV_FIELDS = [
    "sample_id", "k", "relL2", "rmse", "mae", "linf",
    "boundary_max", "pred_flux_resRMSE", "exact_flux_resRMSE",
    "f_mean", "f_std", "f_min", "f_max", "f_rmse",
]

# u_exact-dependent metrics are evaluation-only reporting fields. The training
# objective remains the flux residual on u_pred.
EVALUATION_ONLY_METRIC_FIELDS = {"relL2", "rmse", "mae", "linf", "boundary_max", "exact_flux_resRMSE"}


def _per_sample_rmse(values: torch.Tensor) -> torch.Tensor:
    return values.pow(2).mean(dim=1).sqrt()


def _per_sample_rel_l2(diff: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.linalg.vector_norm(diff, dim=1)
    den = torch.linalg.vector_norm(ref, dim=1).clamp_min(eps)
    return num / den


def _write_tensor_csv(values: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = values.detach().cpu()
    if values.ndim != 2:
        raise ValueError("Only 2D tensors can be written as grid CSV files.")
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(values.tolist())


def _prediction_eval_rows(
    batch: dict[str, torch.Tensor | float],
    u_pred: torch.Tensor,
    pred_flux_residual: torch.Tensor,
    exact_flux_residual: torch.Tensor,
) -> list[dict[str, PredictionMetricValue]]:
    u_exact = batch["u_exact"]  # type: ignore[assignment]
    k = batch["k"]  # type: ignore[assignment]
    boundary_mask = batch["mask"]  # type: ignore[assignment]
    if boundary_mask.shape[0] == 1 and u_pred.shape[0] > 1:
        boundary_mask = boundary_mask.expand_as(u_pred)

    error = u_pred - u_exact
    interior_error = error[:, :, 1:-1, 1:-1].flatten(start_dim=1)
    interior_abs_error = interior_error.abs()
    interior_exact = u_exact[:, :, 1:-1, 1:-1].flatten(start_dim=1)
    boundary_error = (error.abs() * boundary_mask).flatten(start_dim=1)
    pred_res_flat = pred_flux_residual.flatten(start_dim=1)
    exact_res_flat = exact_flux_residual.flatten(start_dim=1)
    f_flat = batch["f"].flatten(start_dim=1)  # type: ignore[union-attr]

    rel_l2 = _per_sample_rel_l2(interior_error, interior_exact)
    rmse = _per_sample_rmse(interior_error)
    mae = interior_abs_error.mean(dim=1)
    linf = interior_abs_error.amax(dim=1)
    boundary_max = boundary_error.amax(dim=1)
    pred_res_rmse = _per_sample_rmse(pred_res_flat)
    exact_res_rmse = _per_sample_rmse(exact_res_flat)
    f_mean = f_flat.mean(dim=1)
    f_std = f_flat.std(dim=1, unbiased=False)
    f_min = f_flat.amin(dim=1)
    f_max = f_flat.amax(dim=1)
    f_rmse = _per_sample_rmse(f_flat)

    rows: list[dict[str, PredictionMetricValue]] = []
    for sample_id in range(u_pred.shape[0]):
        rows.append(
            {
                "sample_id": sample_id,
                "k": float(k[sample_id].detach().cpu()),
                "relL2": float(rel_l2[sample_id].detach().cpu()),
                "rmse": float(rmse[sample_id].detach().cpu()),
                "mae": float(mae[sample_id].detach().cpu()),
                "linf": float(linf[sample_id].detach().cpu()),
                "boundary_max": float(boundary_max[sample_id].detach().cpu()),
                "pred_flux_resRMSE": float(pred_res_rmse[sample_id].detach().cpu()),
                "exact_flux_resRMSE": float(exact_res_rmse[sample_id].detach().cpu()),
                "f_mean": float(f_mean[sample_id].detach().cpu()),
                "f_std": float(f_std[sample_id].detach().cpu()),
                "f_min": float(f_min[sample_id].detach().cpu()),
                "f_max": float(f_max[sample_id].detach().cpu()),
                "f_rmse": float(f_rmse[sample_id].detach().cpu()),
            }
        )
    return rows


def _write_prediction_csv(rows: list[dict[str, PredictionMetricValue]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_matplotlib_pyplot(show_plots: bool = False):
    cache_dir = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg-cache"))

    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _nondegenerate_limits(vmin: float, vmax: float) -> tuple[float, float]:
    if math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
        pad = max(abs(vmin), 1.0) * 1e-6
        return vmin - pad, vmax + pad
    return vmin, vmax


def _symmetric_limits(values: torch.Tensor) -> tuple[float, float]:
    vmax = max(float(values.detach().abs().max().cpu()), 1e-12)
    return -vmax, vmax


def _heatmap_values(values: torch.Tensor):
    return values.detach().cpu().numpy()


def _eval_figure_title(
    sample_id: int,
    N: int,
    row: dict[str, PredictionMetricValue],
) -> str:
    return (
        f"sample {sample_id} | N={N} k={float(row['k']):.4g} "
        f"f_rmse={float(row['f_rmse']):.3e} "
        f"relL2={float(row['relL2']):.3e} "
        f"RMSE={float(row['rmse']):.3e} "
        f"Linf={float(row['linf']):.3e} "
        f"flux_resRMSE={float(row['pred_flux_resRMSE']):.3e}"
    )


def _save_heatmap_figure(
    plt,
    values: torch.Tensor,
    path: Path,
    title: str,
    cmap: str,
    extent: list[float],
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(
        _heatmap_values(values),
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _add_panel_heatmap(
    fig,
    ax,
    values: torch.Tensor,
    title: str,
    cmap: str,
    extent: list[float],
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(
        _heatmap_values(values),
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _save_combined_eval_panel(
    plt,
    sample_dir: Path,
    figure_title: str,
    u_exact: torch.Tensor,
    u_pred: torch.Tensor,
    error: torch.Tensor,
    abs_error: torch.Tensor,
    pred_residual: torch.Tensor,
    exact_residual: torch.Tensor,
    full_extent: list[float],
    residual_extent: list[float],
    u_limits: tuple[float, float],
    error_limits: tuple[float, float],
    abs_error_max: float,
    pred_res_limits: tuple[float, float],
    exact_res_limits: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(figure_title)
    panels = [
        (axes[0][0], u_exact, "true u", "viridis", full_extent, u_limits[0], u_limits[1]),
        (axes[0][1], u_pred, "predicted u", "viridis", full_extent, u_limits[0], u_limits[1]),
        (axes[0][2], error, "u_pred - u_exact", "coolwarm", full_extent, error_limits[0], error_limits[1]),
        (axes[1][0], abs_error, "|u_pred - u_exact|", "magma", full_extent, 0.0, abs_error_max),
        (
            axes[1][1],
            pred_residual,
            "pred flux PDE residual",
            "coolwarm",
            residual_extent,
            pred_res_limits[0],
            pred_res_limits[1],
        ),
        (
            axes[1][2],
            exact_residual,
            "exact flux PDE residual",
            "coolwarm",
            residual_extent,
            exact_res_limits[0],
            exact_res_limits[1],
        ),
    ]
    for ax, values, title, cmap, extent, vmin, vmax in panels:
        _add_panel_heatmap(fig, ax, values, title, cmap, extent, vmin, vmax)
    fig.savefig(sample_dir / "combined_2x3_panel.png", dpi=180)
    plt.close(fig)


def _prediction_residual_tensors(
    batch: dict[str, torch.Tensor | float],
    u_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_exact = batch["u_exact"]  # type: ignore[assignment]
    pred_flux_residual = compute_pde_residual(
        u_pred,
        batch["f"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["h"],  # type: ignore[arg-type]
    )
    exact_flux_residual = compute_pde_residual(
        u_exact,
        batch["f"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["h"],  # type: ignore[arg-type]
    )
    return (
        pred_flux_residual,
        exact_flux_residual,
    )


def _prediction_metric_rows(
    batch: dict[str, torch.Tensor | float],
    u_pred: torch.Tensor,
) -> tuple[
    list[dict[str, PredictionMetricValue]],
    torch.Tensor,
    torch.Tensor,
]:
    residuals = _prediction_residual_tensors(batch, u_pred)
    rows = _prediction_eval_rows(batch, u_pred, *residuals)
    return rows, *residuals


def _flux_residual_panel_label() -> str:
    return "flux PDE residual"


@dataclass
class PredictionOutputBundle:
    rows: list[dict[str, PredictionMetricValue]]
    u_exact: torch.Tensor
    error: torch.Tensor
    abs_error: torch.Tensor
    pred_flux_residual: torch.Tensor
    exact_flux_residual: torch.Tensor


def _build_prediction_output_bundle(
    batch: dict[str, torch.Tensor | float],
    u_pred: torch.Tensor,
) -> PredictionOutputBundle:
    with torch.no_grad():
        u_exact = batch["u_exact"]  # type: ignore[assignment]
        error = u_pred - u_exact
        abs_error = error.abs()
        rows, pred_flux_residual, exact_flux_residual = _prediction_metric_rows(
            batch,
            u_pred,
        )
    return PredictionOutputBundle(
        rows=rows,
        u_exact=u_exact,
        error=error,
        abs_error=abs_error,
        pred_flux_residual=pred_flux_residual,
        exact_flux_residual=exact_flux_residual,
    )


def save_prediction_outputs(
    output_dir: str | Path,
    batch: dict[str, torch.Tensor | float],
    u_pred: torch.Tensor,
    *,
    output_type: str,
    prediction_config: PredictionConfig | None = None,
    show_plots: bool = False,
    output_bundle: PredictionOutputBundle | None = None,
) -> PredictionOutputBundle:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bundle = output_bundle or _build_prediction_output_bundle(batch, u_pred)
    is_evaluation = output_type == "evaluation"
    h = float(batch["h"])  # type: ignore[arg-type]
    f = batch["f"]  # type: ignore[assignment]
    full_extent = [-1.0, 1.0, -1.0, 1.0]
    residual_extent = [-1.0 + h, 1.0 - h, -1.0 + h, 1.0 - h]

    metrics_path = output_path / ("eval_metrics_per_sample.csv" if is_evaluation else "metrics.csv")
    _write_prediction_csv(bundle.rows, metrics_path)
    plt = _load_matplotlib_pyplot(show_plots=show_plots)

    for sample_index, row in enumerate(bundle.rows):
        sample_id = int(row["sample_id"])
        u_exact_i = bundle.u_exact[sample_index, 0].detach()
        u_pred_i = u_pred[sample_index, 0].detach()
        error_i = bundle.error[sample_index, 0].detach()
        abs_error_i = bundle.abs_error[sample_index, 0].detach()
        pred_residual_i = bundle.pred_flux_residual[sample_index, 0].detach()
        exact_residual_i = bundle.exact_flux_residual[sample_index, 0].detach()

        u_vmin = min(float(u_exact_i.min().cpu()), float(u_pred_i.min().cpu()))
        u_vmax = max(float(u_exact_i.max().cpu()), float(u_pred_i.max().cpu()))
        u_limits = _nondegenerate_limits(u_vmin, u_vmax)
        error_limits = _symmetric_limits(error_i)
        abs_error_max = max(float(abs_error_i.max().cpu()), 1e-12)

        if not is_evaluation:
            figure_path = output_path / f"sample_{sample_id:03d}_panel.png"
            _write_tensor_csv(f[sample_index, 0], output_path / f"sample_{sample_id:03d}_f.csv")
            _save_prediction_panel(
                plt,
                figure_path,
                _prediction_panel_title(row),
                u_exact_i,
                u_pred_i,
                error_i,
                abs_error_i,
                pred_residual_i,
                f[sample_index, 0].detach(),
                full_extent,
                residual_extent,
                show_plots,
            )
            continue

        sample_dir = output_path / f"sample_{sample_id:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write_tensor_csv(f[sample_index, 0], sample_dir / "forcing_f.csv")
        pred_res_limits = _symmetric_limits(pred_residual_i)
        exact_res_limits = _symmetric_limits(exact_residual_i)
        metrics_title = _eval_figure_title(sample_id, prediction_config.N, row)  # type: ignore[union-attr]

        _save_heatmap_figure(
            plt,
            u_exact_i,
            sample_dir / "true_u.png",
            f"true u (evaluation only)\n{metrics_title}",
            "viridis",
            full_extent,
            u_limits[0],
            u_limits[1],
        )
        _save_heatmap_figure(
            plt,
            u_pred_i,
            sample_dir / "pred_u.png",
            f"predicted u\n{metrics_title}",
            "viridis",
            full_extent,
            u_limits[0],
            u_limits[1],
        )
        _save_heatmap_figure(
            plt,
            error_i,
            sample_dir / "error_heatmap.png",
            f"evaluation-only error = u_pred - u_exact\n{metrics_title}",
            "coolwarm",
            full_extent,
            error_limits[0],
            error_limits[1],
        )
        _save_heatmap_figure(
            plt,
            abs_error_i,
            sample_dir / "abs_error_heatmap.png",
            f"evaluation-only absolute error\n{metrics_title}",
            "magma",
            full_extent,
            0.0,
            abs_error_max,
        )
        _save_heatmap_figure(
            plt,
            pred_residual_i,
            sample_dir / "pred_residual_heatmap.png",
            f"predicted {_flux_residual_panel_label()}\n{metrics_title}",
            "coolwarm",
            residual_extent,
            pred_res_limits[0],
            pred_res_limits[1],
        )
        _save_combined_eval_panel(
            plt,
            sample_dir,
            metrics_title,
            u_exact_i,
            u_pred_i,
            error_i,
            abs_error_i,
            pred_residual_i,
            exact_residual_i,
            full_extent,
            residual_extent,
            u_limits,
            error_limits,
            abs_error_max,
            pred_res_limits,
            exact_res_limits,
        )

    if is_evaluation:
        print(f"saved evaluation metrics: {metrics_path}")
        print(f"saved evaluation figures: {output_path}")
    else:
        print(f"saved prediction plot metrics: {metrics_path}")
        print(f"saved prediction panels: {output_path}")
    return bundle


def _save_prediction_panel(
    plt,
    path: Path,
    figure_title: str,
    u_exact: torch.Tensor,
    u_pred: torch.Tensor,
    error: torch.Tensor,
    abs_error: torch.Tensor,
    flux_residual: torch.Tensor,
    f_map: torch.Tensor,
    full_extent: list[float],
    residual_extent: list[float],
    show_plots: bool,
) -> None:
    u_vmin = min(float(u_exact.min().cpu()), float(u_pred.min().cpu()))
    u_vmax = max(float(u_exact.max().cpu()), float(u_pred.max().cpu()))
    u_limits = _nondegenerate_limits(u_vmin, u_vmax)
    error_limits = _symmetric_limits(error)
    abs_error_max = max(float(abs_error.max().cpu()), 1e-12)
    residual_limits = _symmetric_limits(flux_residual)
    f_limits = _symmetric_limits(f_map)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(figure_title)
    panels = [
        (axes[0][0], u_exact, "true solution u_exact", "viridis", full_extent, u_limits[0], u_limits[1]),
        (axes[0][1], u_pred, "CNN prediction u_pred", "viridis", full_extent, u_limits[0], u_limits[1]),
        (axes[0][2], error, "signed error u_pred - u_exact", "coolwarm", full_extent, error_limits[0], error_limits[1]),
        (axes[1][0], abs_error, "absolute error |u_pred - u_exact|", "magma", full_extent, 0.0, abs_error_max),
        (
            axes[1][1],
            flux_residual,
            _flux_residual_panel_label(),
            "coolwarm",
            residual_extent,
            residual_limits[0],
            residual_limits[1],
        ),
        (axes[1][2], f_map, "forcing f", "coolwarm", full_extent, f_limits[0], f_limits[1]),
    ]
    for ax, values, title, cmap, extent, vmin, vmax in panels:
        _add_panel_heatmap(fig, ax, values, title, cmap, extent, vmin, vmax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def _prediction_panel_title(row: dict[str, PredictionMetricValue]) -> str:
    return (
        f"sample {int(row['sample_id'])} | k={float(row['k']):.4g} "
        f"f_rmse={float(row['f_rmse']):.3e} "
        f"relL2={float(row['relL2']):.3e} "
        f"RMSE={float(row['rmse']):.3e} "
        f"MAE={float(row['mae']):.3e} "
        f"Linf={float(row['linf']):.3e} "
        f"flux_resRMSE={float(row['pred_flux_resRMSE']):.3e}"
    )


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
        f"N={prediction_config.N} continuous manufactured forcing "
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
    )
    u_pred = predict_on_batch(model, batch, stats, model_config)

    predictions = {"cnn": u_pred}
    metrics = evaluate_predictions(
        predictions,
        batch,
    )
    for name, metric in metrics.items():
        print(format_eval_metrics(f"predict {name}", metric))

    prediction_output_bundle = None
    eval_output_dir = getattr(args, "eval_output_dir", None)
    if eval_output_dir:
        prediction_output_bundle = save_prediction_outputs(
            eval_output_dir,
            batch,
            u_pred,
            output_type="evaluation",
            prediction_config=prediction_config,
            show_plots=getattr(args, "show_plots", False),
            output_bundle=prediction_output_bundle,
        )

    if getattr(args, "plot_predictions", False):
        prediction_output_bundle = save_prediction_outputs(
            getattr(args, "plot_dir", "results/figures"),
            batch,
            u_pred,
            output_type="plot",
            show_plots=getattr(args, "show_plots", False),
            output_bundle=prediction_output_bundle,
        )

    if args.predict_save:
        _save_prediction_payload(
            args.predict_save,
            batch,
            u_pred,
            metrics,
            prediction_config,
            model_config,
        )


def run_demo(args: argparse.Namespace) -> None:
    checkpoint_path = Path("checkpoints/demo.pt")
    best_checkpoint_path = add_path_suffix(checkpoint_path, "best")
    figure_dir = Path("results/demo_figures")

    print("running demo mode")
    print(f"demo checkpoint path: {checkpoint_path}")
    print(f"demo best checkpoint path: {best_checkpoint_path}")
    print(f"demo figure directory: {figure_dir}")

    train_args = argparse.Namespace(**vars(args))
    train_args.N = 32
    train_args.train_samples = 2048
    train_args.val_samples = 128
    train_args.epochs = 40
    train_args.batch_size = 16
    train_args.eval_batch_size = 0
    train_args.eval_batches = None
    train_args.eval_every = 1
    train_args.save_path = str(checkpoint_path)
    train_args.load_path = ""
    train_args.predict_manufactured = False
    train_args.predict_save = ""
    train_args.plot_predictions = False
    apply_positive_interior_default(train_args)
    train(train_args)

    prediction_checkpoint = best_checkpoint_path if best_checkpoint_path.exists() else checkpoint_path
    print(f"demo prediction checkpoint: {prediction_checkpoint}")

    predict_args = argparse.Namespace(**vars(args))
    predict_args.load_path = str(prediction_checkpoint)
    predict_args.predict_manufactured = True
    predict_args.predict_samples = 4
    predict_args.predict_save = ""
    predict_args.predict_N = None
    predict_args.predict_k_min = None
    predict_args.predict_k_max = None
    predict_args.eval_output_dir = ""
    predict_args.plot_predictions = True
    predict_args.plot_dir = str(figure_dir)
    run_prediction(predict_args)

    print(f"demo checkpoint saved: {checkpoint_path}")
    print(f"demo best checkpoint saved: {best_checkpoint_path}")
    print(f"demo figures saved: {figure_dir}")
    print(f"demo metrics saved: {figure_dir / 'metrics.csv'}")


def run_oracle_floor_check(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    k_range = (args.k_min, args.k_max)

    batch = build_dataset(
        args.N,
        args.batch_size,
        device,
        k_range,
    )
    with torch.no_grad():
        res = compute_pde_residual(
            batch["u_exact"],  # type: ignore[arg-type]
            batch["f"],  # type: ignore[arg-type]
            batch["k"],  # type: ignore[arg-type]
            batch["h"],  # type: ignore[arg-type]
        )
        rmse = torch.sqrt((res * res).mean())
        max_abs = res.abs().max()
        scaled_rmse = torch.sqrt(
            ((res * (batch["h"] ** 2)) ** 2).mean()  # type: ignore[operator]
        )
    print(
        "continuous manufactured oracle floor "
        "(flux residual on analytic RHS, finite-difference truncation floor): "
        f"rmse={float(rmse.cpu()):.4e}, "
        f"scaled_rmse={float(scaled_rmse.cpu()):.4e}, "
        f"max_abs={float(max_abs.cpu()):.4e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--N", type=int, default=32, help="Interior grid size. Full grid is (N+2)^2.")
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40, help="Real epochs over the fixed dataset.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Defaults to --batch-size.")
    parser.add_argument("--eval-batches", type=int, default=None, help="Compatibility alias for val samples.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--k-min", type=float, default=2.0)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-path", default="")
    parser.add_argument("--load-path", default="")
    parser.add_argument("--predict-manufactured", action="store_true")
    parser.add_argument("--predict-samples", type=int, default=8)
    parser.add_argument("--predict-save", default="")
    parser.add_argument("--plot-predictions", action="store_true")
    parser.add_argument("--plot-dir", default="results/figures")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--predict-N", "--predict-n", dest="predict_N", type=int, default=None)
    parser.add_argument("--predict-k-min", type=float, default=None)
    parser.add_argument("--predict-k-max", type=float, default=None)
    parser.add_argument(
        "--eval-output-dir",
        default="results/eval_figures",
        help="Directory for --predict-manufactured evaluation CSV and figures.",
    )
    parser.add_argument("--test-seed", type=int, default=1234)
    parser.add_argument(
        "--check-oracle-floor",
        dest="check_oracle_floor",
        action="store_true",
        help="Print the continuous manufactured oracle residual floor.",
    )
    parser.add_argument(
        "--check-residual",
        dest="check_oracle_floor",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--train-after-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(sys.argv) == 1:
        args.demo = True
    apply_positive_interior_default(args)
    if args.demo:
        run_demo(args)
        return
    if args.check_oracle_floor:
        run_oracle_floor_check(args)
        if not args.train_after_check:
            return
    if args.predict_manufactured:
        run_prediction(args)
        return
    if args.load_path:
        raise ValueError("--load-path is only used with --predict-manufactured.")
    train(args)


if __name__ == "__main__":
    main()
