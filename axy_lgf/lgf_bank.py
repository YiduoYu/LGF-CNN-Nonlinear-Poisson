"""Screened Dirichlet LGF filter bank on a rectangle (DST, no BAE).

LGF 先给出 J 个不同空间尺度的候选修正，CNN 再决定每个位置用哪几个。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import SquareGrid
from .lgf_inverse import _laplacian_eigenvalues, _solve_in_dst_basis


def default_screening_values(
    h: float, J: int = 6, include_poisson: bool = True
) -> np.ndarray:
    """Screening coefficients with length scales ~ {inf, 2h, 4h, ...}.

    lambda = 0 is the pure Dirichlet Poisson LGF (included by default; on a
    rectangle it's exact via DST and doesn't need truncation).
    """
    if J < 1:
        raise ValueError("J must be >= 1")

    poisson = [0.0] if include_poisson else []
    ell = h * 2.0 ** np.arange(1, J - len(poisson) + 1)
    return np.asarray(poisson + (1.0 / ell**2).tolist(), dtype=np.float64)


@dataclass
class ScreenedLGFBank:
    """Bank of preconditioners: apply (-a0 Delta + lambda_j I)^{-1} to r.

    Same denominator convention as ScreenedLGFBankTorch; the two must agree
    for solve_axy_lgf_bank to be an honest baseline for LGF-CNN.
    """

    grid: SquareGrid
    lambdas: np.ndarray
    a0: float = 1.0

    def __post_init__(self) -> None:
        self.lambdas = np.asarray(self.lambdas, dtype=np.float64).ravel()
        if self.lambdas.size == 0:
            raise ValueError("lambdas must not be empty")
        if not np.all(np.isfinite(self.lambdas)):
            raise ValueError("lambdas must be finite")
        if np.any(self.lambdas < 0):
            raise ValueError("lambdas must be nonnegative")
        if not np.isfinite(self.a0) or self.a0 <= 0.0:
            raise ValueError("a0 must be finite and positive")
        mu = _laplacian_eigenvalues(self.grid)
        self._denominators = self.a0 * mu[None, :, :] + self.lambdas[:, None, None]

    @classmethod
    def from_defaults(
        cls,
        grid: SquareGrid,
        J: int = 6,
        a0: float = 1.0,
        include_poisson: bool = True,
    ) -> "ScreenedLGFBank":
        return cls(
            grid=grid,
            lambdas=default_screening_values(
                grid.h, J=J, include_poisson=include_poisson
            ),
            a0=a0,
        )

    @property
    def J(self) -> int:
        return len(self.lambdas)

    def apply(self, r: np.ndarray) -> np.ndarray:
        """Return stack z[j] = G_{lambda_j} * r, shape (J, n, n)."""
        if r.shape != (self.grid.n, self.grid.n):
            raise ValueError("r must have shape (n, n)")
        if not np.all(np.isfinite(r)):
            raise ValueError("r must contain only finite values")
        return _solve_in_dst_basis(r, self._denominators)

    def mix(self, r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute sum_j alpha_j * G_{lambda_j} r.

        alpha can be (J,) for global weights or (J, n, n) for pointwise.
        """
        z = self.apply(r)
        weights = np.asarray(alpha, dtype=np.float64)
        if weights.shape == (self.J,):
            return np.tensordot(weights, z, axes=(0, 0))
        if weights.shape == z.shape:
            return np.sum(weights * z, axis=0)
        raise ValueError(f"alpha must have shape ({self.J},) or {z.shape}")
