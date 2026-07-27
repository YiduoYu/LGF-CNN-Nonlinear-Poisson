"""
Dirichlet lattice Green's function inverse on a rectangle via DST-I.

For the operator  -a0 * Delta_h + sigma * I  with homogeneous Dirichlet BC,
the inverse is applied exactly (up to floating point) by a 2D fast sine
transform. This is the rectangular Dirichlet LGF — no free-space truncation,
no BAE needed.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dstn, idstn

from .grid import SquareGrid


def _validate_inverse_parameters(a0: float, sigma: float) -> None:
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 must be finite and positive")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")


def _validate_rhs(rhs: np.ndarray, grid: SquareGrid) -> None:
    if rhs.shape != (grid.n, grid.n):
        raise ValueError("rhs shape mismatch")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("rhs must contain only finite values")


def _solve_in_dst_basis(rhs: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide by denominator in the DST-I basis.

    Broadcasting lets one rhs be solved against a stack of denominators,
    which is how the (J, n, n) screening bank works.
    """
    rhs_hat = dstn(rhs, type=1, norm="ortho", axes=(-2, -1))
    return idstn(rhs_hat / denominator, type=1, norm="ortho", axes=(-2, -1))


def _laplacian_eigenvalues(grid: SquareGrid) -> np.ndarray:
    """Eigenvalues of -Delta_h on the interior grid (positive).

    For mode (p, q), p,q = 1..n:
        mu_{pq} = (4/h^2) * (sin^2(p*pi*h/(2L)) + sin^2(q*pi*h/(2L)))
    with h = L/(n+1). Equivalent form:
        mu = (2/h^2) * (2 - cos(p*pi/(n+1)) - cos(q*pi/(n+1))).
    """
    modes = np.arange(1, grid.n + 1, dtype=np.float64)
    cosines = np.cos(modes * np.pi / (grid.n + 1))
    lam1d = (2.0 / grid.h**2) * (1.0 - cosines)
    return lam1d[:, None] + lam1d[None, :]


def apply_dirichlet_lgf(
    rhs: np.ndarray,
    grid: SquareGrid,
    a0: float = 1.0,
    sigma: float = 0.0,
) -> np.ndarray:
    """Solve (a0 * (-Delta_h) + sigma I) u = rhs, u=0 on boundary.

    a0 > 0 is the reference diffusivity, sigma >= 0 the screening
    coefficient. rhs and the returned u are both (n, n).
    """
    _validate_inverse_parameters(a0, sigma)
    _validate_rhs(rhs, grid)
    return _solve_in_dst_basis(rhs, a0 * _laplacian_eigenvalues(grid) + sigma)


class DirichletLGF:
    """Callable LGF inverse for -a0 Delta (+ optional sigma)."""

    def __init__(self, grid: SquareGrid, a0: float = 1.0, sigma: float = 0.0):
        _validate_inverse_parameters(a0, sigma)
        self.grid = grid
        self.a0 = float(a0)
        self.sigma = float(sigma)
        self._denom = self.a0 * _laplacian_eigenvalues(grid) + self.sigma

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        _validate_rhs(rhs, self.grid)
        return _solve_in_dst_basis(rhs, self._denom)
