"""CNN baseline for a nonlinear Poisson equation in flux form.

The model learns a grid-to-grid map for

    -div(a(u) grad u) = f,  a(u) = u**k,  u = g on boundary,

on Omega = [-1, 1]^2.  The loss is the conservative finite-difference
flux residual on the interior grid nodes, with Dirichlet boundary values
enforced exactly by an output projection.

Example:
    python Baseline.py --check-residual
    python Baseline.py --epochs 200 --N 32 --rhs-mode continuous
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
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


def power_coefficient(
    u: torch.Tensor,
    k: torch.Tensor | float,
    mode: str = "pow",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a(u).

    mode="pow" clamps only as a numerical guard. For non-integer k, keep the
    generated solution family positive if you want the mathematical PDE u**k.
    """

    k = _k_to_broadcast(k, u)
    if mode == "pow":
        return torch.pow(torch.clamp(u, min=eps), k)
    if mode == "abs_pow":
        return torch.pow(torch.clamp(u.abs(), min=eps), k)
    if mode == "softplus_pow":
        return torch.pow(F.softplus(u) + eps, k)
    raise ValueError(f"Unknown coeff_mode: {mode}")


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


def pde_loss(
    u: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor | float,
    h: float,
    scale_with_h: bool = True,
    coeff_mode: str = "pow",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    res = nonlinear_poisson_residual(u, f, k, h, coeff_mode=coeff_mode, eps=eps)
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


def manufactured_u(x: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """A positive solution family with nontrivial boundary values."""

    phi = (1.0 - x * x) * (1.0 - y * y)
    boundary_part = (
        params["c0"] + params["c1"] * x + params["c2"] * y + params["c3"] * x * y
    )
    interior_part = (
        params["a1"] * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        + params["a2"] * torch.cos(2.0 * math.pi * x) * torch.cos(math.pi * y)
    )
    return boundary_part + phi * interior_part


def continuous_rhs_from_u(
    u_exact: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Compute f = -div(u**k grad u) by differentiating the analytic family."""

    k4 = k.view(1, 1, 1, 1)
    a = torch.pow(u_exact, k4)

    du_dx = torch.autograd.grad(
        u_exact,
        x,
        grad_outputs=torch.ones_like(u_exact),
        create_graph=True,
        retain_graph=True,
    )[0]
    du_dy = torch.autograd.grad(
        u_exact,
        y,
        grad_outputs=torch.ones_like(u_exact),
        create_graph=True,
        retain_graph=True,
    )[0]

    qx = a * du_dx
    qy = a * du_dy

    dqx_dx = torch.autograd.grad(
        qx,
        x,
        grad_outputs=torch.ones_like(qx),
        create_graph=False,
        retain_graph=True,
    )[0]
    dqy_dy = torch.autograd.grad(
        qy,
        y,
        grad_outputs=torch.ones_like(qy),
        create_graph=False,
        retain_graph=False,
    )[0]
    return -(dqx_dx + dqy_dy).detach()


def discrete_rhs_from_u(
    u_exact: torch.Tensor,
    k: torch.Tensor,
    h: float,
    coeff_mode: str = "pow",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build f so u_exact satisfies the same discrete residual to roundoff."""

    f = torch.zeros_like(u_exact)
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
    }


def sample_manufactured_problem(
    N: int,
    device: torch.device,
    k_range: tuple[float, float] = (0.5, 2.5),
    rhs_mode: str = "continuous",
    positive_only: bool = True,
    coeff_mode: str = "pow",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | float]:
    """Return one manufactured problem instance."""

    x0, y0, boundary_mask, interior_mask, h = make_grid(N, device=device, dtype=dtype)
    needs_grad = rhs_mode == "continuous"
    x = x0.clone().detach().requires_grad_(needs_grad)
    y = y0.clone().detach().requires_grad_(needs_grad)

    params = _sample_params(device, dtype)
    u_exact = manufactured_u(x, y, params)
    if positive_only:
        margin = 0.2 - u_exact.detach().min()
        if margin.item() > 0:
            params["c0"] = params["c0"] + margin
            u_exact = manufactured_u(x, y, params)

    k = torch.empty((1,), device=device, dtype=dtype).uniform_(k_range[0], k_range[1])
    if rhs_mode == "continuous":
        f = continuous_rhs_from_u(u_exact, x, y, k)
    elif rhs_mode == "discrete":
        f = discrete_rhs_from_u(u_exact.detach(), k, h, coeff_mode=coeff_mode)
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
) -> dict[str, torch.Tensor | float]:
    problems = [
        sample_manufactured_problem(
            N=N,
            device=device,
            k_range=k_range,
            rhs_mode=rhs_mode,
            positive_only=True,
            coeff_mode=coeff_mode,
        )
        for _ in range(batch_size)
    ]
    return collate_problem_list(problems)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


@dataclass
class TrainStats:
    loss: float
    loss_sup: float
    loss_pde: float
    rel_l2: float
    res_rmse: float
    boundary_max: float


def forward_with_projection(
    model: nn.Module,
    batch: dict[str, torch.Tensor | float],
    normalize_k: bool,
    k_range: tuple[float, float],
    positive_interior: bool,
) -> torch.Tensor:
    k_mean = 0.5 * (k_range[0] + k_range[1]) if normalize_k else None
    k_std = (k_range[1] - k_range[0]) / math.sqrt(12.0) if normalize_k else None
    inp = build_input_tensor(
        batch["f"],  # type: ignore[arg-type]
        batch["g"],  # type: ignore[arg-type]
        batch["mask"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["x"],  # type: ignore[arg-type]
        batch["y"],  # type: ignore[arg-type]
        k_mean=k_mean,
        k_std=k_std,
    )
    u_raw = model(inp)
    return enforce_dirichlet(
        u_raw,
        batch["g"],  # type: ignore[arg-type]
        batch["mask"],  # type: ignore[arg-type]
        positive_interior=positive_interior,
    )


def compute_stats(
    u_pred: torch.Tensor,
    batch: dict[str, torch.Tensor | float],
    loss: torch.Tensor,
    loss_sup: torch.Tensor,
    loss_pde: torch.Tensor,
    res: torch.Tensor,
) -> TrainStats:
    rel_l2 = relative_l2_error(
        u_pred,
        batch["u_exact"],  # type: ignore[arg-type]
        mask=batch["interior_mask"],  # type: ignore[arg-type]
    )
    res_rmse = torch.sqrt((res * res).mean())
    bmax = boundary_max_error(
        u_pred,
        batch["g"],  # type: ignore[arg-type]
        batch["mask"],  # type: ignore[arg-type]
    )
    return TrainStats(
        loss=float(loss.detach().cpu()),
        loss_sup=float(loss_sup.detach().cpu()),
        loss_pde=float(loss_pde.detach().cpu()),
        rel_l2=float(rel_l2.detach().cpu()),
        res_rmse=float(res_rmse.detach().cpu()),
        boundary_max=float(bmax.detach().cpu()),
    )


def evaluate(
    model: nn.Module,
    N: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    k_range: tuple[float, float],
    rhs_mode: str,
    coeff_mode: str,
    normalize_k: bool,
    positive_interior: bool,
) -> TrainStats:
    model.eval()
    totals = torch.zeros(6, dtype=torch.float64)

    for _ in range(num_batches):
        batch = make_batch(N, batch_size, device, k_range, rhs_mode, coeff_mode)
        with torch.no_grad():
            u_pred = forward_with_projection(
                model,
                batch,
                normalize_k=normalize_k,
                k_range=k_range,
                positive_interior=positive_interior,
            )
            loss_pde, res = pde_loss(
                u_pred,
                batch["f"],  # type: ignore[arg-type]
                batch["k"],  # type: ignore[arg-type]
                batch["h"],  # type: ignore[arg-type]
                scale_with_h=True,
                coeff_mode=coeff_mode,
            )
            loss_sup = masked_mse(
                u_pred,
                batch["u_exact"],  # type: ignore[arg-type]
                batch["interior_mask"],  # type: ignore[arg-type]
            )
            loss = 0.1 * loss_sup + loss_pde
            stats = compute_stats(u_pred, batch, loss, loss_sup, loss_pde, res)
        totals += torch.tensor(
            [
                stats.loss,
                stats.loss_sup,
                stats.loss_pde,
                stats.rel_l2,
                stats.res_rmse,
                stats.boundary_max,
            ],
            dtype=torch.float64,
        )

    avg = totals / max(num_batches, 1)
    return TrainStats(*[float(x) for x in avg.tolist()])


def train(args: argparse.Namespace) -> nn.Module:
    set_seed(args.seed)
    device = select_device(args.device)
    k_range = (args.k_min, args.k_max)

    model = UNet2d(in_ch=6, out_ch=1, base_ch=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"device={device} N={args.N} rhs_mode={args.rhs_mode} "
        f"epochs={args.epochs} batch_size={args.batch_size} base_ch={args.base_ch}"
    )

    for epoch in range(args.epochs):
        model.train()
        batch = make_batch(args.N, args.batch_size, device, k_range, args.rhs_mode, args.coeff_mode)
        u_pred = forward_with_projection(
            model,
            batch,
            normalize_k=args.normalize_k,
            k_range=k_range,
            positive_interior=args.positive_interior,
        )

        loss_pde, res = pde_loss(
            u_pred,
            batch["f"],  # type: ignore[arg-type]
            batch["k"],  # type: ignore[arg-type]
            batch["h"],  # type: ignore[arg-type]
            scale_with_h=True,
            coeff_mode=args.coeff_mode,
        )
        loss_sup = masked_mse(
            u_pred,
            batch["u_exact"],  # type: ignore[arg-type]
            batch["interior_mask"],  # type: ignore[arg-type]
        )

        if args.pure_physics:
            loss = loss_pde
        elif epoch < args.pretrain_epochs:
            loss = loss_sup
        else:
            loss = args.supervised_weight * loss_sup + args.pde_weight * loss_pde

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                train_stats = compute_stats(u_pred, batch, loss, loss_sup, loss_pde, res)
            eval_stats = evaluate(
                model,
                N=args.N,
                batch_size=args.eval_batch_size,
                num_batches=args.eval_batches,
                device=device,
                k_range=k_range,
                rhs_mode=args.rhs_mode,
                coeff_mode=args.coeff_mode,
                normalize_k=args.normalize_k,
                positive_interior=args.positive_interior,
            )
            print(
                f"epoch={epoch:04d} "
                f"train loss={train_stats.loss:.4e} sup={train_stats.loss_sup:.4e} "
                f"pde={train_stats.loss_pde:.4e} relL2={train_stats.rel_l2:.4e} "
                f"resRMSE={train_stats.res_rmse:.4e} bmax={train_stats.boundary_max:.2e} | "
                f"eval relL2={eval_stats.rel_l2:.4e} resRMSE={eval_stats.res_rmse:.4e}"
            )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
            },
            save_path,
        )
        print(f"saved checkpoint: {save_path}")

    return model


def run_residual_check(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    k_range = (args.k_min, args.k_max)
    batch = make_batch(args.N, args.batch_size, device, k_range, "discrete", args.coeff_mode)
    with torch.no_grad():
        _, res = pde_loss(
            batch["u_exact"],  # type: ignore[arg-type]
            batch["f"],  # type: ignore[arg-type]
            batch["k"],  # type: ignore[arg-type]
            batch["h"],  # type: ignore[arg-type]
            scale_with_h=False,
            coeff_mode=args.coeff_mode,
        )
        rmse = torch.sqrt((res * res).mean())
        max_abs = res.abs().max()
        scaled_rmse = torch.sqrt(((res * (batch["h"] ** 2)) ** 2).mean())  # type: ignore[operator]
    print(
        "discrete RHS residual check: "
        f"rmse={float(rmse.cpu()):.4e}, "
        f"scaled_rmse={float(scaled_rmse.cpu()):.4e}, "
        f"max_abs={float(max_abs.cpu()):.4e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=32, help="Interior grid size. Full grid is (N+2)^2.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--rhs-mode", choices=["continuous", "discrete"], default="continuous")
    parser.add_argument("--coeff-mode", choices=["pow", "abs_pow", "softplus_pow"], default="pow")
    parser.add_argument("--k-min", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=0.1)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--pure-physics", action="store_true")
    parser.add_argument("--positive-interior", action="store_true")
    parser.add_argument("--normalize-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-path", default="")
    parser.add_argument("--check-residual", action="store_true")
    parser.add_argument("--train-after-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_residual:
        run_residual_check(args)
        if not args.train_after_check:
            return
    train(args)


if __name__ == "__main__":
    main()
