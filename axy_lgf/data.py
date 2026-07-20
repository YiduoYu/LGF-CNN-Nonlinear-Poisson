"""Random manufactured batches for a(x,y) rectangle problems."""

from __future__ import annotations

import numpy as np
import torch

from .grid import SquareGrid
from .torch_ops import TorchSquareOps


def _mesh_np(grid: SquareGrid) -> tuple[np.ndarray, np.ndarray]:
    return grid.mesh()


def sample_coefficient(
    grid: SquareGrid,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample smooth positive a(x,y) = 1 + contrast * field, field in [-1,1]."""
    rng = np.random.default_rng() if rng is None else rng
    X, Y = _mesh_np(grid)
    a = np.empty((batch_size, grid.n, grid.n), dtype=np.float64)
    for b in range(batch_size):
        field = np.zeros_like(X)
        nmodes = int(rng.integers(1, 4))
        for _ in range(nmodes):
            px = int(rng.integers(1, 4))
            py = int(rng.integers(1, 4))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp = rng.uniform(0.3, 1.0)
            field += amp * np.sin(2.0 * np.pi * px * X + phase) * np.cos(
                2.0 * np.pi * py * Y
            )
        field = field / (np.max(np.abs(field)) + 1e-14)
        a[b] = 1.0 + contrast * field
        # keep strictly positive
        a[b] = np.clip(a[b], 1.0 - 0.95 * contrast, 1.0 + contrast)
    return a


def sample_solution(
    grid: SquareGrid,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample homogeneous-Dirichlet solutions as sine combinations."""
    rng = np.random.default_rng() if rng is None else rng
    X, Y = _mesh_np(grid)
    u = np.zeros((batch_size, grid.n, grid.n), dtype=np.float64)
    for b in range(batch_size):
        nmodes = int(rng.integers(1, 4))
        for _ in range(nmodes):
            px = int(rng.integers(1, 4))
            py = int(rng.integers(1, 4))
            amp = rng.normal(scale=1.0)
            u[b] += amp * np.sin(px * np.pi * X / grid.L) * np.sin(
                py * np.pi * Y / grid.L
            )
    return u


def make_batch(
    grid: SquareGrid,
    ops: TorchSquareOps,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a, f, u_exact as torch tensors (B, n, n).

    f is the discrete residual operator applied to u_exact, so a consistent
    discrete problem is solved exactly by u_exact.
    """
    rng = np.random.default_rng() if rng is None else rng
    a_np = sample_coefficient(grid, batch_size, contrast=contrast, rng=rng)
    u_np = sample_solution(grid, batch_size, rng=rng)

    a = torch.as_tensor(a_np, dtype=dtype, device=device)
    u = torch.as_tensor(u_np, dtype=dtype, device=device)
    f = ops.apply_div_a_grad(u, a)
    return a, f, u
