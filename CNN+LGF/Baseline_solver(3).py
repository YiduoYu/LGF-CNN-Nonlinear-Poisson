"""
方案 3：多 kappa 固定 LGF 核 + CNN + 残差展开（solver 数据）。

求解
    -div(a(x,y) grad u) = f,  (x,y) in [-1,1]^2,  u|boundary = g,
    a(x,y) = exp(sigma_coef * psi(x,y)) > 0.

solver 数据按随机 f/g/a -> 线性有限差分求解器 -> u_ref 正向生成。
模型一次 forward 内部展开 T 次残差修正；空间传播只使用由
LGF_python.lgf.LGF_FFT 生成的固定多 kappa 核，可学习空间卷积被完全
移除，只保留 1x1 通道混合、GroupNorm、GELU 和残差连接。
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _add_lgf_search_paths():
    """Allow LGF_python beside the script, in an ancestor, or in the cwd."""
    here = Path(__file__).resolve().parent
    candidates = [here, Path.cwd(), *here.parents]
    for root in candidates:
        if (root / "LGF_python").is_dir():
            sys.path.insert(0, str(root))
            return


_add_lgf_search_paths()
try:
    from LGF_python.lgf import LGF_FFT
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "找不到 LGF_python。请将解压后的 LGF_python/ 放在脚本同级目录、"
        "任一父目录，或通过 PYTHONPATH 指向其父目录。"
    ) from exc


DATA_KIND = "solver"
MODEL_TYPE = "scheme3_lgf_cnn_solver"
LGF_LOG_A_RANGE = (-1.5, 1.5)
LGF_CACHE_DIR = Path("checkpoints") / "scheme3_lgf_bank"


# ---------------------------------------------------------------------------
# Grid, PDE operator, and coefficient fields
# ---------------------------------------------------------------------------

def make_grid(N, device="cpu", dtype=torch.float32):
    """N interior points per direction and one Dirichlet layer on each side."""
    size = N + 2
    h = 2.0 / (N + 1)
    coord = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    x = xx[None, None]
    y = yy[None, None]
    boundary = torch.ones(1, 1, size, size, device=device, dtype=dtype)
    boundary[:, :, 1:-1, 1:-1] = 0.0
    return x, y, boundary, 1.0 - boundary, h


def enforce_dirichlet(u, g, boundary):
    """Hard projection: boundary is always exactly g."""
    return boundary * g + (1.0 - boundary) * u


def flux_divergence(u, a, h):
    """Flux-form five-point discretization of -div(a grad u)."""
    uc = u[:, :, 1:-1, 1:-1]
    ac = a[:, :, 1:-1, 1:-1]
    ae = 0.5 * (ac + a[:, :, 1:-1, 2:])
    aw = 0.5 * (ac + a[:, :, 1:-1, :-2])
    an = 0.5 * (ac + a[:, :, 2:, 1:-1])
    ass = 0.5 * (ac + a[:, :, :-2, 1:-1])
    fe = ae * (u[:, :, 1:-1, 2:] - uc) / h
    fw = aw * (uc - u[:, :, 1:-1, :-2]) / h
    fn = an * (u[:, :, 2:, 1:-1] - uc) / h
    fs = ass * (uc - u[:, :, :-2, 1:-1]) / h
    return -(fe - fw + fn - fs) / h


def pde_residual(u, f, a, h):
    """A_h(a)u - f on interior points."""
    return flux_divergence(u, a, h) - f[:, :, 1:-1, 1:-1]


def random_fourier_field(N, modes, rng, normalize=True):
    coord = np.linspace(-1.0, 1.0, N + 2)
    xx, yy = np.meshgrid(coord, coord)
    field = np.zeros_like(xx)
    for m in range(1, modes + 1):
        for n in range(1, modes + 1):
            sx, cx = np.sin(m * np.pi * xx), np.cos(m * np.pi * xx)
            sy, cy = np.sin(n * np.pi * yy), np.cos(n * np.pi * yy)
            field += (
                rng.uniform(-1.0, 1.0) * sx * sy
                + rng.uniform(-1.0, 1.0) * sx * cy
                + rng.uniform(-1.0, 1.0) * cx * sy
                + rng.uniform(-1.0, 1.0) * cx * cy
            )
    if normalize:
        field /= max(float(np.abs(field).max()), 1e-12)
    return field


def random_positive_field(N, sigma_coef, modes, rng):
    """a=exp(sigma_coef*psi), psi normalized to [-1,1]."""
    psi = random_fourier_field(N, modes, rng, normalize=True)
    return np.exp(sigma_coef * psi)


# ---------------------------------------------------------------------------
# Fixed multi-kappa LGF kernel banks
# ---------------------------------------------------------------------------

def _one_lgf_kernel(h, a_value, radius, n_fft):
    """K[p,q]=h^2 G(|p-R|,|q-R|), centered at zero and never normalized."""
    if n_fft % 2:
        raise ValueError("--lgf-fft-size must be even")
    if radius >= n_fft // 2:
        raise ValueError("--lgf-radius must be smaller than lgf_fft_size/2")
    quadrant = LGF_FFT(h, float(a_value), n_fft)
    kernel = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    for i, di in enumerate(range(-radius, radius + 1)):
        for j, dj in enumerate(range(-radius, radius + 1)):
            kernel[i, j] = quadrant[abs(di), abs(dj)]
    return (h * h) * kernel


def build_lgf_kernel_bank(
    h,
    M,
    radius,
    n_fft,
    log_a_range=LGF_LOG_A_RANGE,
    cache_dir=LGF_CACHE_DIR,
):
    """Create M log-equispaced physical kernels and cache them once."""
    if M < 2:
        raise ValueError("--lgf-M must be at least 2")
    lo, hi = log_a_range
    cache_dir = Path(cache_dir)
    tag = f"h{h:.10g}_M{M}_R{radius}_fft{n_fft}_r{lo:g}_{hi:g}"
    path = cache_dir / f"{tag}.npz"
    if path.exists():
        saved = np.load(path)
        return (
            torch.from_numpy(saved["kernels"]).float().unsqueeze(1),
            torch.from_numpy(saved["a_nodes"]).float(),
        )

    log_nodes = np.linspace(lo, hi, M)
    a_nodes = np.exp(log_nodes)
    kernels = np.stack(
        [_one_lgf_kernel(h, value, radius, n_fft) for value in a_nodes]
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, kernels=kernels, a_nodes=a_nodes)
    return torch.from_numpy(kernels).float().unsqueeze(1), torch.from_numpy(a_nodes).float()


def _apply_all_kernels(x, kernels):
    """Shared physical bank, depthwise over all hidden channels."""
    batch, channels, height, width = x.shape
    radius = kernels.shape[-1] // 2
    response = F.conv2d(
        x.reshape(batch * channels, 1, height, width),
        kernels,
        padding=radius,
    )
    return response.view(batch, channels, kernels.shape[0], height, width)


def _gather_kernel_response(response, indices):
    channels = response.shape[1]
    expanded = indices.unsqueeze(2).expand(-1, channels, -1, -1, -1)
    return torch.gather(response, 2, expanded).squeeze(2)


class FixedLGFKernelBank(nn.Module):
    """A fixed buffer plus pointwise single/nearest/log-linear selection."""

    def __init__(self, h, M, radius, n_fft):
        super().__init__()
        kernels, a_nodes = build_lgf_kernel_bank(h, M, radius, n_fft)
        self.register_buffer("kernels", kernels)
        self.register_buffer("a_nodes", a_nodes)
        self.register_buffer("log_a_nodes", a_nodes.log())
        self.h = float(h)
        self.radius = int(radius)
        self.n_fft = int(n_fft)

    def prepare(self, a, mode="linear"):
        """Per-point kernel selection; constant across all residual steps."""
        log_a = a.clamp(self.a_nodes[0], self.a_nodes[-1]).log()
        if mode == "single":
            index = int(torch.argmin(self.log_a_nodes.abs()).item())
            return ("single", index)
        if mode == "nearest":
            index = (log_a.unsqueeze(2) - self.log_a_nodes.view(1, 1, -1, 1, 1))
            return ("nearest", index.abs().argmin(dim=2))
        if mode != "linear":
            raise ValueError(f"unknown kernel mode: {mode}")
        upper = torch.searchsorted(
            self.log_a_nodes, log_a.contiguous()
        ).clamp(1, self.log_a_nodes.numel() - 1)
        lower = upper - 1
        lo = self.log_a_nodes[lower]
        hi = self.log_a_nodes[upper]
        weight = ((log_a - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
        return ("linear", lower, upper, weight)

    def forward(self, x, plan):
        mode = plan[0]
        if mode == "single":
            index = plan[1]
            return _apply_all_kernels(
                x, self.kernels[index : index + 1]
            ).squeeze(2)
        response = _apply_all_kernels(x, self.kernels)
        if mode == "nearest":
            return _gather_kernel_response(response, plan[1])
        _, lower, upper, weight = plan
        low_response = _gather_kernel_response(response, lower)
        high_response = _gather_kernel_response(response, upper)
        return (1.0 - weight) * low_response + weight * high_response


def _group_norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class MultiKappaLGFBlock(nn.Module):
    """Fixed LGF propagation -> learnable 1x1 mixing -> GN -> GELU -> residual."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels, channels, 1)
        self.norm = _group_norm(channels)
        self.activation = nn.GELU()

    def forward(self, x, bank, plan):
        propagated = bank(x, plan)
        return x + self.activation(self.norm(self.mix(propagated)))


class LGFCNNCorrection(nn.Module):
    """1x1 lift -> L fixed multi-kappa LGF blocks -> 1x1 correction head."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        total_blocks,
        h,
        M,
        radius,
        n_fft,
        kernel_mode,
    ):
        super().__init__()
        if total_blocks < 1:
            raise ValueError("--lgf-L must be positive")
        self.kernel_mode = kernel_mode
        self.bank = FixedLGFKernelBank(h, M, radius, n_fft)
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [MultiKappaLGFBlock(hidden_channels) for _ in range(total_blocks)]
        )
        self.head = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, plan):
        hidden = self.lift(features)
        for block in self.blocks:
            hidden = block(hidden, self.bank, plan)
        return self.head(hidden)


@dataclass
class NormalizationStats:
    f_mean: float
    f_std: float
    u_mean: float
    u_std: float
    q_mean: float
    q_std: float
    loga_mean: float
    loga_std: float


class Scheme3LGFNet(nn.Module):
    """One public forward call containing T residual-correction steps."""

    def __init__(self, args):
        super().__init__()
        self.T = args.lgf_T
        h = 2.0 / (args.N + 1)
        self.correction = LGFCNNCorrection(
            in_channels=9,
            hidden_channels=args.base_ch,
            total_blocks=args.lgf_L,
            h=h,
            M=args.lgf_M,
            radius=args.lgf_radius,
            n_fft=args.lgf_fft_size,
            kernel_mode=args.kernel_mode,
        )
        self.eta = nn.Parameter(torch.ones(self.T))

    def forward(self, batch, stats, collect_diagnostics=False):
        f, g, a = batch["f"], batch["g"], batch["a"]
        boundary, h = batch["mask"], batch["h"]
        batch_size = f.shape[0]
        boundary_b = _expand_batch(boundary, batch_size)
        x_b = _expand_batch(batch["x"], batch_size)
        y_b = _expand_batch(batch["y"], batch_size)
        loga_norm = (
            a.clamp_min(1e-12).log() - stats.loga_mean
        ) / stats.loga_std
        f_norm = (f - stats.f_mean) / stats.f_std * (1.0 - boundary_b)
        u = enforce_dirichlet(torch.zeros_like(g), g, boundary)
        plan = self.correction.bank.prepare(a, self.correction.kernel_mode)

        trace = [] if collect_diagnostics else None
        if collect_diagnostics:
            trace.append(pde_residual(u, f, a, h).pow(2).mean().sqrt().detach())

        for step in range(self.T):
            residual = pde_residual(u, f, a, h)
            r_over_a = torch.zeros_like(u)
            r_over_a[:, :, 1:-1, 1:-1] = (
                -residual / a[:, :, 1:-1, 1:-1].clamp_min(1e-12)
            )
            q_norm = (r_over_a - stats.q_mean) / stats.q_std
            q_norm = q_norm * (1.0 - boundary_b)
            u_norm = (u - stats.u_mean) / stats.u_std
            time_channel = torch.full_like(f, step / self.T)
            features = torch.cat(
                [
                    u_norm,
                    q_norm,
                    f_norm,
                    g,
                    boundary_b,
                    loga_norm,
                    x_b,
                    y_b,
                    time_channel,
                ],
                dim=1,
            )
            delta = (1.0 - boundary_b) * self.correction(features, plan)
            u = enforce_dirichlet(u + self.eta[step] * delta, g, boundary)
            if collect_diagnostics:
                trace.append(
                    pde_residual(u, f, a, h).pow(2).mean().sqrt().detach()
                )

        if collect_diagnostics:
            return u, torch.stack(trace)
        return u


# ---------------------------------------------------------------------------
# Forward-solver data
# ---------------------------------------------------------------------------

def random_forcing(N, scale, modes, rng):
    forcing = scale * random_fourier_field(N, modes, rng, normalize=True)
    forcing[0, :] = forcing[-1, :] = 0.0
    forcing[:, 0] = forcing[:, -1] = 0.0
    return forcing


def random_boundary(N, base, scale, modes, rng):
    raw = random_fourier_field(N, modes, rng, normalize=True)
    boundary = np.zeros((N + 2, N + 2), dtype=np.float64)
    boundary[0, :] = base + scale * raw[0, :]
    boundary[-1, :] = base + scale * raw[-1, :]
    boundary[:, 0] = base + scale * raw[:, 0]
    boundary[:, -1] = base + scale * raw[:, -1]
    return boundary


def assemble_linear_system(a, g, f, h):
    """Sparse system for the same flux-form operator used by the loss."""
    N = a.shape[0] - 2
    size = N * N
    inv_h2 = 1.0 / (h * h)
    ac = a[1 : N + 1, 1 : N + 1]
    ae = 0.5 * (ac + a[1 : N + 1, 2 : N + 2])
    aw = 0.5 * (ac + a[1 : N + 1, :N])
    an = 0.5 * (ac + a[2 : N + 2, 1 : N + 1])
    ass = 0.5 * (ac + a[:N, 1 : N + 1])

    rhs = f[1 : N + 1, 1 : N + 1].ravel().copy()
    rhs[N - 1 :: N] += ae[:, N - 1] * inv_h2 * g[1 : N + 1, N + 1]
    rhs[0::N] += aw[:, 0] * inv_h2 * g[1 : N + 1, 0]
    rhs[(N - 1) * N :] += an[N - 1, :] * inv_h2 * g[N + 1, 1 : N + 1]
    rhs[:N] += ass[0, :] * inv_h2 * g[0, 1 : N + 1]

    diagonal = (ae + aw + an + ass).ravel() * inv_h2
    index = np.arange(size)
    east = index % N != N - 1
    west = index % N != 0
    north = index < size - N
    south = index >= N
    rows = np.concatenate(
        [index, index[east], index[west], index[north], index[south]]
    )
    columns = np.concatenate(
        [
            index,
            index[east] + 1,
            index[west] - 1,
            index[north] + N,
            index[south] - N,
        ]
    )
    values = np.concatenate(
        [
            diagonal,
            -ae.ravel()[east] * inv_h2,
            -aw.ravel()[west] * inv_h2,
            -an.ravel()[north] * inv_h2,
            -ass.ravel()[south] * inv_h2,
        ]
    )
    matrix = sp.coo_matrix(
        (values, (rows, columns)), shape=(size, size)
    ).tocsr()
    return matrix, rhs


def solve_linear_fd(f, g, a, h):
    matrix, rhs = assemble_linear_system(a, g, f, h)
    interior = spla.spsolve(matrix, rhs)
    solution = g.copy()
    solution[1:-1, 1:-1] = interior.reshape(a.shape[0] - 2, a.shape[1] - 2)
    return solution


def make_dataset(N, n_samples, device, args, seed):
    """Random f/g/a followed by one linear finite-difference solve."""
    rng = np.random.RandomState(seed)
    x, y, boundary, interior, h = make_grid(N, device)
    fields = {key: [] for key in ("f", "g", "a", "sigma", "u_ref")}
    residual_sq = 0.0

    for _ in range(n_samples):
        sigma = (
            rng.uniform(args.sigma_min, args.sigma_max)
            if args.random_sigma
            else args.sigma
        )
        a_np = random_positive_field(N, sigma, args.fourier_m, rng)
        f_np = random_forcing(N, args.f_scale, args.fourier_m, rng)
        g_np = random_boundary(
            N, args.g_base, args.g_scale, args.fourier_m, rng
        )
        u_np = solve_linear_fd(f_np, g_np, a_np, h)
        a = torch.from_numpy(a_np)[None, None]
        f = torch.from_numpy(f_np)[None, None]
        g = torch.from_numpy(g_np)[None, None]
        u_ref = torch.from_numpy(u_np)[None, None]
        residual_sq += float(
            pde_residual(u_ref, f, a, h).pow(2).sum()
        )
        fields["f"].append(f.float())
        fields["g"].append(g.float())
        fields["a"].append(a.float())
        fields["sigma"].append(torch.tensor([sigma], dtype=torch.float32))
        fields["u_ref"].append(u_ref.float())

    return {
        **{key: torch.cat(value).to(device) for key, value in fields.items()},
        "mask": boundary,
        "interior_mask": interior,
        "x": x,
        "y": y,
        "h": h,
        "ref_res64": math.sqrt(residual_sq / (n_samples * N * N)),
    }


# ---------------------------------------------------------------------------
# Normalization, training, validation, prediction
# ---------------------------------------------------------------------------

def _mean_std(tensor):
    return tensor.mean().item(), tensor.std(unbiased=False).clamp_min(1e-8).item()


def compute_stats(data):
    f_int = data["f"][:, :, 1:-1, 1:-1]
    u_int = data["u_ref"][:, :, 1:-1, 1:-1]
    loga = data["a"].clamp_min(1e-12).log()
    u0 = enforce_dirichlet(torch.zeros_like(data["g"]), data["g"], data["mask"])
    r0 = -pde_residual(u0, data["f"], data["a"], data["h"])
    q0 = r0 / data["a"][:, :, 1:-1, 1:-1].clamp_min(1e-12)
    f_mean, f_std = _mean_std(f_int)
    u_mean, u_std = _mean_std(u_int)
    q_mean, q_std = _mean_std(q0)
    loga_mean, loga_std = _mean_std(loga)
    return NormalizationStats(
        f_mean, f_std, u_mean, u_std, q_mean, q_std, loga_mean, loga_std
    )


def _expand_batch(tensor, batch_size):
    return tensor.expand(batch_size, -1, -1, -1) if tensor.shape[0] == 1 else tensor


def take_batch(data, index):
    sample_keys = ("f", "g", "a", "sigma", "u_ref")
    shared_keys = ("mask", "interior_mask", "x", "y")
    return {
        "h": data["h"],
        **{key: data[key][index] for key in sample_keys},
        **{key: data[key][:1] for key in shared_keys},
    }


def compute_loss(u_pred, batch, args):
    if args.loss == "supervised":
        interior = batch["interior_mask"].expand_as(u_pred)
        return (((u_pred - batch["u_ref"]) * interior) ** 2).mean()
    return pde_residual(u_pred, batch["f"], batch["a"], batch["h"]).pow(2).mean()


def create_optimizer(model, args):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if name.endswith(".bias") or parameter.ndim <= 1 else decay
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )


def train_one_epoch(model, data, stats, optimizer, scheduler, args, device):
    model.train()
    order = torch.randperm(data["f"].shape[0], device=device)
    total, count = 0.0, 0
    for start in range(0, len(order), args.batch_size):
        batch = take_batch(data, order[start : start + args.batch_size])
        loss = compute_loss(model(batch, stats), batch, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        size = batch["f"].shape[0]
        total += loss.detach().item() * size
        count += size
    return total / count


def per_sample_metrics(u_pred, batch, start_id=0):
    u_ref = batch["u_ref"]
    interior = batch["interior_mask"].expand_as(u_pred)
    boundary = batch["mask"].expand_as(u_pred)
    u0 = enforce_dirichlet(torch.zeros_like(batch["g"]), batch["g"], batch["mask"])
    error = ((u_pred - u_ref) * interior)[:, :, 1:-1, 1:-1].flatten(1)
    initial_error = ((u0 - u_ref) * interior)[:, :, 1:-1, 1:-1].flatten(1)
    reference = u_ref[:, :, 1:-1, 1:-1].flatten(1)
    pred_res = pde_residual(u_pred, batch["f"], batch["a"], batch["h"]).flatten(1)
    ref_res = pde_residual(u_ref, batch["f"], batch["a"], batch["h"]).flatten(1)
    u0_res = pde_residual(u0, batch["f"], batch["a"], batch["h"]).flatten(1)
    h = float(batch["h"])

    def gradient(value):
        dx = (value[:, :, 1:-1, 2:] - value[:, :, 1:-1, :-2]) / (2.0 * h)
        dy = (value[:, :, 2:, 1:-1] - value[:, :, :-2, 1:-1]) / (2.0 * h)
        return torch.cat([dx.flatten(1), dy.flatten(1)], dim=1)

    grad_error = gradient(u_pred) - gradient(u_ref)
    grad_ref = gradient(u_ref)
    rows = []
    for i in range(u_pred.shape[0]):
        pred_rmse = pred_res[i].pow(2).mean().sqrt()
        initial_rmse = u0_res[i].pow(2).mean().sqrt()
        rows.append(
            {
                "sample_id": start_id + i,
                "sigma": batch["sigma"][i].item(),
                "relL2": (
                    error[i].norm() / reference[i].norm().clamp_min(1e-12)
                ).item(),
                "RMSE": error[i].pow(2).mean().sqrt().item(),
                "maxError": error[i].abs().max().item(),
                "predResRMSE": pred_rmse.item(),
                "refResRMSE": ref_res[i].pow(2).mean().sqrt().item(),
                "boundaryError": (
                    (u_pred[i : i + 1] - batch["g"][i : i + 1]).abs()
                    * boundary[i : i + 1]
                ).max().item(),
                "gradRelL2": (
                    grad_error[i].norm() / grad_ref[i].norm().clamp_min(1e-12)
                ).item(),
                "u0RelL2": (
                    initial_error[i].norm()
                    / reference[i].norm().clamp_min(1e-12)
                ).item(),
                "u0ResRMSE": initial_rmse.item(),
                "residualRatio": (
                    pred_rmse / initial_rmse.clamp_min(1e-12)
                ).item(),
            }
        )
    return rows


def _aggregate(rows):
    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    def maximum(key):
        return float(np.max([row[key] for row in rows]))

    return {
        "relL2": mean("relL2"),
        "resRMSE": math.sqrt(np.mean([row["predResRMSE"] ** 2 for row in rows])),
        "maxError": maximum("maxError"),
        "meanBoundaryError": mean("boundaryError"),
        "maxBoundaryError": maximum("boundaryError"),
        "gradRelL2": mean("gradRelL2"),
        "u0RelL2": mean("u0RelL2"),
        "u0ResRMSE": mean("u0ResRMSE"),
        "residualRatio": mean("residualRatio"),
        "beatsU0": mean("relL2") < mean("u0RelL2"),
        "residualDecreased": mean("residualRatio") < 1.0,
        "per_sample": rows,
    }


@torch.no_grad()
def evaluate(model, data, stats, batch_size):
    model.eval()
    rows, traces = [], []
    for start in range(0, data["f"].shape[0], batch_size):
        batch = take_batch(data, slice(start, start + batch_size))
        prediction, trace = model(batch, stats, collect_diagnostics=True)
        rows.extend(per_sample_metrics(prediction, batch, start))
        traces.append((trace.cpu().numpy(), prediction.shape[0]))
    total = sum(size for _, size in traces)
    mean_trace = np.sqrt(
        sum(size * trace ** 2 for trace, size in traces) / total
    ).tolist()
    metrics = _aggregate(rows)
    metrics["residualTrace"] = mean_trace
    return metrics


def _write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bank_snapshot(model):
    return model.correction.bank.kernels.detach().cpu().clone()


def _assert_bank_unchanged(model, snapshot):
    if not torch.equal(model.correction.bank.kernels.detach().cpu(), snapshot):
        raise AssertionError("fixed LGF kernel bank changed")


def train(args, run_dir):
    _set_seed(args.seed)
    device = _select_device(args.device)
    train_data = make_dataset(args.N, args.train_samples, device, args, args.seed)
    val_data = make_dataset(args.N, args.val_samples, device, args, args.seed + 1)
    stats = compute_stats(train_data)
    model = Scheme3LGFNet(args).to(device)
    snapshot = _bank_snapshot(model)
    optimizer = create_optimizer(model, args)
    steps_per_epoch = math.ceil(args.train_samples / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    best_metric, history = math.inf, []
    best_path = run_dir / "best.pt"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"data={DATA_KIND} device={device} parameters="
        f"{sum(p.numel() for p in model.parameters()):,} "
        f"ref_res64={val_data['ref_res64']:.3e}"
    )
    print(
        f"stats: f_std={stats.f_std:.3e} u_std={stats.u_std:.3e} "
        f"q_std={stats.q_std:.3e} loga_std={stats.loga_std:.3e}"
    )

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model, train_data, stats, optimizer, scheduler, args, device
        )
        if epoch % args.eval_every and epoch != args.epochs - 1:
            continue
        metrics = evaluate(
            model, val_data, stats, args.eval_batch_size or args.batch_size
        )
        selection = metrics["resRMSE"] if args.loss == "physics" else metrics["relL2"]
        if selection < best_metric:
            best_metric = selection
            torch.save(
                {
                    "model_type": MODEL_TYPE,
                    "model_state_dict": model.state_dict(),
                    "stats": astuple(stats),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
                best_path,
            )
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "val_relL2": metrics["relL2"],
                "val_resRMSE": metrics["resRMSE"],
                "val_maxError": metrics["maxError"],
                "val_boundaryError": metrics["maxBoundaryError"],
                "val_gradRelL2": metrics["gradRelL2"],
                "val_u0RelL2": metrics["u0RelL2"],
                "val_residualRatio": metrics["residualRatio"],
            }
        )
        trace_text = " ".join(f"{value:.3e}" for value in metrics["residualTrace"])
        print(
            f"epoch={epoch:04d} loss={loss:.3e} relL2={metrics['relL2']:.3e} "
            f"resRMSE={metrics['resRMSE']:.3e} "
            f"boundary={metrics['maxBoundaryError']:.2e}"
        )
        print(f"  residual u0..uT: {trace_text}")
        if not metrics["residualDecreased"] or not metrics["beatsU0"]:
            print("  WARNING: current validation uT has not yet improved over u0")

    _assert_bank_unchanged(model, snapshot)
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "model_state_dict": model.state_dict(),
            "stats": astuple(stats),
            "args": vars(args),
        },
        run_dir / "last.pt",
    )
    _write_csv(history, run_dir / "history.csv")
    return best_path


def run_prediction(args):
    if not args.predict_from:
        raise ValueError("--predict requires --predict-from")
    device = _select_device(args.device)
    checkpoint = torch.load(args.predict_from, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"checkpoint type must be {MODEL_TYPE!r}, got "
            f"{checkpoint.get('model_type')!r}"
        )
    saved_args = argparse.Namespace(**checkpoint["args"])
    model = Scheme3LGFNet(saved_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    stats = NormalizationStats(*checkpoint["stats"])
    data_args = argparse.Namespace(**vars(saved_args))
    data_args.random_sigma = False
    data_args.sigma = args.predict_sigma
    data = make_dataset(
        saved_args.N, args.predict_samples, device, data_args, args.test_seed
    )
    with torch.no_grad():
        prediction, trace = model(data, stats, collect_diagnostics=True)
    rows = per_sample_metrics(prediction, data)
    metrics = _aggregate(rows)
    metrics["residualTrace"] = trace.cpu().tolist()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "metrics.csv")
    with (output_dir / "summary.json").open("w") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    if args.predict_save:
        target = Path(args.predict_save)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "f": data["f"].cpu(),
                "g": data["g"].cpu(),
                "a": data["a"].cpu(),
                "u_ref": data["u_ref"].cpu(),
                "u_pred": prediction.cpu(),
                "metrics": metrics,
            },
            target,
        )
    print(
        f"prediction relL2={metrics['relL2']:.3e} "
        f"resRMSE={metrics['resRMSE']:.3e} "
        f"boundary={metrics['maxBoundaryError']:.2e} "
        f"residualRatio={metrics['residualRatio']:.3e}"
    )
    print(f"saved metrics -> {output_dir}")


# ---------------------------------------------------------------------------
# Scheme-3 validation checks
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sanity(args):
    _set_seed(args.seed)
    device = _select_device(args.device)
    data = make_dataset(args.N, args.train_samples, device, args, args.seed)
    stats = compute_stats(data)
    model = Scheme3LGFNet(args).to(device)
    snapshot = _bank_snapshot(model)
    bank = model.correction.bank
    gen = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(1, args.base_ch, args.N + 2, args.N + 2, generator=gen, device=device)
    middle = bank.a_nodes.numel() // 2
    a_node = bank.a_nodes[middle]
    a_field = torch.full(
        (1, 1, args.N + 2, args.N + 2), float(a_node), device=device
    )
    interpolated = bank(x, a_field, "linear")
    direct = _apply_all_kernels(x, bank.kernels[middle : middle + 1]).squeeze(2)
    node_error = float((interpolated - direct).abs().max())
    if node_error > 1e-5:
        raise AssertionError(f"kernel-node interpolation failed: {node_error:.3e}")

    lower = max(0, min(middle - 1, bank.a_nodes.numel() - 2))
    log_mid = 0.5 * (bank.log_a_nodes[lower] + bank.log_a_nodes[lower + 1])
    a_mid = float(log_mid.exp())
    mid_field = torch.full_like(a_field, a_mid)
    mid_interp = bank(x, mid_field, "linear")
    mid_kernel = torch.from_numpy(
        _one_lgf_kernel(bank.h, a_mid, bank.radius, bank.n_fft)
    ).float().to(device)[None, None]
    mid_direct = _apply_all_kernels(x, mid_kernel).squeeze(2)
    mid_rel = float(
        (mid_interp - mid_direct).norm() / mid_direct.norm().clamp_min(1e-12)
    )

    prediction, trace = model(data, stats, collect_diagnostics=True)
    boundary_error = float(
        ((prediction - data["g"]).abs() * data["mask"]).max()
    )
    _assert_bank_unchanged(model, snapshot)
    if boundary_error > 1e-7:
        raise AssertionError(f"hard boundary projection failed: {boundary_error:.3e}")
    if not torch.isfinite(prediction).all():
        raise AssertionError("non-finite model output")

    print("fixed kernel bank: finite and unchanged")
    print(f"exact node interpolation max error: {node_error:.3e}")
    print(f"midpoint interpolation vs direct LGF relative error: {mid_rel:.3e}")
    print(f"hard boundary error: {boundary_error:.3e}")
    print("untrained residual u0..uT:", " ".join(f"{v:.3e}" for v in trace.tolist()))
    print("sanity checks passed")


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


def experiment_key(args):
    sigma = (
        f"sig{args.sigma_min:g}-{args.sigma_max:g}"
        if args.random_sigma
        else f"sig{args.sigma:g}"
    )
    visible = (
        f"N{args.N}_{sigma}_T{args.lgf_T}_M{args.lgf_M}_"
        f"L{args.lgf_L}_R{args.lgf_radius}_"
        f"{args.kernel_mode}_{args.loss}"
    )
    affecting = (
        args.epochs,
        args.train_samples,
        args.val_samples,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.grad_clip,
        args.base_ch,
        args.lgf_fft_size,
        args.fourier_m,
        args.f_scale,
        args.g_base,
        args.g_scale,
        args.seed,
    )
    digest = hashlib.md5(repr(affecting).encode()).hexdigest()[:8]
    return f"{visible}_{digest}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-ch", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--lgf-T", type=int, default=12)
    parser.add_argument("--lgf-M", type=int, default=8)
    parser.add_argument(
        "--lgf-L",
        type=int,
        default=4,
        help="multi-kappa LGF blocks inside one correction CNN",
    )
    parser.add_argument("--lgf-radius", type=int, default=4)
    parser.add_argument("--lgf-fft-size", type=int, default=512)
    parser.add_argument(
        "--kernel-mode",
        choices=("single", "nearest", "linear"),
        default="linear",
        help="scheme-3 ablation; linear is the full model",
    )

    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument(
        "--random-sigma",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sigma-min", type=float, default=0.5)
    parser.add_argument("--sigma-max", type=float, default=1.5)
    parser.add_argument("--fourier-m", type=int, default=6)
    parser.add_argument("--f-scale", type=float, default=6.0)
    parser.add_argument("--g-base", type=float, default=0.0)
    parser.add_argument("--g-scale", type=float, default=0.6)

    parser.add_argument(
        "--loss",
        choices=("supervised", "physics"),
        default="physics",
    )
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict-from", default="")
    parser.add_argument("--predict-samples", type=int, default=8)
    parser.add_argument("--predict-sigma", type=float, default=0.5)
    parser.add_argument("--test-seed", type=int, default=9999)
    parser.add_argument("--predict-save", default="")
    parser.add_argument("--output-dir", default="results/scheme3_lgf_cnn_solver/predict")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.N < 2:
        raise ValueError("--N must be at least 2")
    if min(args.epochs, args.train_samples, args.val_samples, args.batch_size) < 1:
        raise ValueError("epochs and sample/batch counts must be positive")
    if args.eval_every < 1 or args.base_ch < 1:
        raise ValueError("--eval-every and --base-ch must be positive")
    if min(args.lgf_T, args.lgf_M, args.lgf_L) < 1:
        raise ValueError("T, M, and L must be positive")
    if args.lgf_M < 2:
        raise ValueError("--lgf-M must be at least 2")
    if args.lgf_radius < 0:
        raise ValueError("--lgf-radius must be non-negative")
    if args.lgf_fft_size % 2 or args.lgf_radius >= args.lgf_fft_size // 2:
        raise ValueError("LGF FFT size must be even and larger than 2*radius")
    if not 0.0 <= args.sigma <= 1.5:
        raise ValueError("scheme 3 requires --sigma in [0,1.5]")
    if not 0.0 <= args.sigma_min <= args.sigma_max <= 1.5:
        raise ValueError("scheme 3 requires sigma in [0,1.5] for the fixed bank range")
    if not 0.0 <= args.predict_sigma <= 1.5:
        raise ValueError("scheme 3 requires --predict-sigma in [0,1.5]")
    if args.sanity:
        return run_sanity(args)
    if args.predict:
        return run_prediction(args)

    run_dir = Path("checkpoints") / "scheme3_lgf_cnn_solver" / experiment_key(args)
    best_path = run_dir / "best.pt"
    if best_path.exists() and not args.retrain:
        print(f"found {best_path}; use --retrain to overwrite")
        return
    train(args, run_dir)
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()