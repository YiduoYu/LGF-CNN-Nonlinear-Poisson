"""Random training batches for a(x,y) rectangle problems.

Two data modes, same random fields underneath:
  - "mms":    sample u_exact, set f = A_h u_exact
  - "solver": sample f, solve A_h u = f via PCG to get u_exact
"""

from __future__ import annotations

import numpy as np
import torch

from .grid import SquareGrid
from .solvers import solve_axy_pcg
from .torch_ops import TorchSquareOps

DATA_KINDS = ("mms", "solver")
COEFFICIENT_FLOOR_FACTOR = 0.95


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")


def sample_coefficient(
    grid: SquareGrid,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample smooth positive a(x,y) = 1 + contrast * field, field in [-1, 1]."""
    _validate_batch_size(batch_size)
    if not np.isfinite(contrast) or not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be finite and lie in [0, 1]")
    rng = np.random.default_rng() if rng is None else rng
    xi, yi = grid.unit_mesh()
    a = np.empty((batch_size, grid.n, grid.n), dtype=np.float64)
    lower_bound = 1.0 - COEFFICIENT_FLOOR_FACTOR * contrast
    upper_bound = 1.0 + contrast
    for b in range(batch_size):
        field = np.zeros_like(xi)
        nmodes = int(rng.integers(1, 4))
        for _ in range(nmodes):
            px = int(rng.integers(1, 4))
            py = int(rng.integers(1, 4))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp = rng.uniform(0.3, 1.0)
            field += (
                amp
                * np.sin(2.0 * np.pi * px * xi + phase)
                * np.cos(2.0 * np.pi * py * yi)
            )
        field = field / (np.max(np.abs(field)) + 1e-14)
        a[b] = 1.0 + contrast * field
        a[b] = np.clip(a[b], lower_bound, upper_bound)
    return a


def sample_solution(
    grid: SquareGrid,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample homogeneous-Dirichlet solutions as sine combinations."""
    _validate_batch_size(batch_size)
    rng = np.random.default_rng() if rng is None else rng
    xi, yi = grid.unit_mesh()
    u = np.zeros((batch_size, grid.n, grid.n), dtype=np.float64)
    for b in range(batch_size):
        nmodes = int(rng.integers(1, 4))
        for _ in range(nmodes):
            px = int(rng.integers(1, 4))
            py = int(rng.integers(1, 4))
            amp = rng.normal(scale=1.0)
            u[b] += amp * np.sin(px * np.pi * xi) * np.sin(py * np.pi * yi)
    return u


def make_batch(
    grid: SquareGrid,
    ops: TorchSquareOps,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    data_kind: str = "mms",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a, f, u_exact as torch tensors (B, n, n).

    Both modes draw the same random coefficient and sine field; they only
    differ in which one becomes u_exact and which becomes f. Same seed
    gives directly comparable mms and solver batches.
    """
    if data_kind not in DATA_KINDS:
        raise ValueError(f"data_kind must be one of {DATA_KINDS}, got {data_kind!r}")
    rng = np.random.default_rng() if rng is None else rng

    a_np = sample_coefficient(grid, batch_size, contrast=contrast, rng=rng)
    sine_field = sample_solution(grid, batch_size, rng=rng)
    a = torch.as_tensor(a_np, dtype=dtype, device=device)

    if data_kind == "mms":
        u = torch.as_tensor(sine_field, dtype=dtype, device=device)
        f = ops.apply_div_a_grad(u, a)
        return a, f, u

    # solver mode: sine_field is f, solve for u_exact via PCG
    labels = []
    for b in range(batch_size):
        solved = solve_axy_pcg(a_np[b], sine_field[b], grid, tol=1e-10, maxiter=500)
        if not solved.converged:
            raise RuntimeError(
                "PCG did not converge while generating a solver label "
                f"(sample {b}, rel_res={solved.residual_history[-1]:.3e})"
            )
        labels.append(solved.u)

    f = torch.as_tensor(sine_field, dtype=dtype, device=device)
    u = torch.as_tensor(np.stack(labels), dtype=dtype, device=device)
    return a, f, u
