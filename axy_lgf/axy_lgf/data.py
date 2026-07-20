"""Random training batches for a(x,y) rectangle problems.

Two clearly separated data kinds share the same random fields:
  - "mms":    sample u_exact, set f = A_h u_exact (manufactured solution).
  - "solver": sample f, solve A_h u = f by PCG for u_exact (forward solver).
"""

from __future__ import annotations

import numpy as np
import torch

from .grid import SquareGrid
from .solvers import solve_axy_pcg
from .torch_ops import TorchSquareOps

DATA_KINDS = ("mms", "solver")
MIN_COEFFICIENT = 0.05


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")


def _validate_contrast(contrast: float) -> None:
    if not np.isfinite(contrast) or not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be finite and lie in [0, 1]")


def sample_coefficient(
    grid: SquareGrid,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample smooth positive a(x,y) = 1 + contrast * field, field in [-1,1]."""
    _validate_batch_size(batch_size)
    _validate_contrast(contrast)
    rng = np.random.default_rng() if rng is None else rng
    X, Y = grid.mesh()
    a = np.empty((batch_size, grid.n, grid.n), dtype=np.float64)
    for b in range(batch_size):
        field = np.zeros_like(X)
        nmodes = int(rng.integers(1, 4))
        for _ in range(nmodes):
            px = int(rng.integers(1, 4))
            py = int(rng.integers(1, 4))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp = rng.uniform(0.3, 1.0)
            field += (
                amp
                * np.sin(2.0 * np.pi * px * X / grid.L + phase)
                * np.cos(2.0 * np.pi * py * Y / grid.L)
            )
        field = field / (np.max(np.abs(field)) + 1e-14)
        a[b] = np.maximum(1.0 + contrast * field, MIN_COEFFICIENT)
    return a


def sample_solution(
    grid: SquareGrid,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample homogeneous-Dirichlet solutions as sine combinations."""
    _validate_batch_size(batch_size)
    rng = np.random.default_rng() if rng is None else rng
    X, Y = grid.mesh()
    u = np.zeros((batch_size, grid.n, grid.n), dtype=np.float64)
    for b in range(batch_size):
        nmodes = int(rng.integers(1, 4))
        for _ in range(nmodes):
            px = int(rng.integers(1, 4))
            py = int(rng.integers(1, 4))
            amp = rng.normal(scale=1.0)
            u[b] += (
                amp * np.sin(px * np.pi * X / grid.L) * np.sin(py * np.pi * Y / grid.L)
            )
    return u


def make_batch_mms(
    grid: SquareGrid,
    ops: TorchSquareOps,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MMS batch: sample u_exact, then f = A_h u_exact.

    f is the discrete operator applied to u_exact, so a consistent discrete
    problem is solved exactly by u_exact.  No linear solve is involved.
    """
    rng = np.random.default_rng() if rng is None else rng
    a_np = sample_coefficient(grid, batch_size, contrast=contrast, rng=rng)
    u_np = sample_solution(grid, batch_size, rng=rng)

    a = torch.as_tensor(a_np, dtype=dtype, device=device)
    u = torch.as_tensor(u_np, dtype=dtype, device=device)
    f = ops.apply_div_a_grad(u, a)
    return a, f, u


def make_batch_solver(
    grid: SquareGrid,
    ops: TorchSquareOps,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solver batch: sample f, then solve A_h u = f by PCG for u_exact.

    The forcing f is drawn from the same sine family as sample_solution;
    u_exact then comes from a forward PCG solve (tol=1e-10), mirroring the
    Baseline_solver data generation.
    """
    rng = np.random.default_rng() if rng is None else rng
    a_np = sample_coefficient(grid, batch_size, contrast=contrast, rng=rng)
    f_np = sample_solution(grid, batch_size, rng=rng)
    solutions = []
    for b in range(batch_size):
        result = solve_axy_pcg(
            a_np[b],
            f_np[b],
            grid,
            tol=1e-10,
            maxiter=500,
        )
        if not result.converged:
            raise RuntimeError(
                "PCG failed while generating solver labels "
                f"(sample={b}, rel_res={result.residual_history[-1]:.3e})"
            )
        solutions.append(result.u)
    u_np = np.stack(solutions)

    a = torch.as_tensor(a_np, dtype=dtype, device=device)
    f = torch.as_tensor(f_np, dtype=dtype, device=device)
    u = torch.as_tensor(u_np, dtype=dtype, device=device)
    return a, f, u


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

    data_kind selects the generation mode: "mms" (sample u_exact, derive f)
    or "solver" (sample f, solve for u_exact).
    """
    if data_kind not in DATA_KINDS:
        raise ValueError(f"data_kind must be one of {DATA_KINDS}, got {data_kind!r}")
    maker = make_batch_mms if data_kind == "mms" else make_batch_solver
    return maker(
        grid,
        ops,
        batch_size,
        contrast=contrast,
        rng=rng,
        device=device,
        dtype=dtype,
    )
