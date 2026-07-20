"""Screened Dirichlet LGF filter bank on a rectangle (DST-based, no BAE)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dstn, idstn

from .grid import SquareGrid
from .lgf_inverse import _laplacian_eigenvalues


def default_screening_values(
    h: float, J: int = 6, include_poisson: bool = True
) -> np.ndarray:
    """Screening coefficients λ_j with length scales ~ {∞, 2h, 4h, ...}.

    λ = 0 is the pure Dirichlet Poisson LGF (included by default on a rectangle,
    where it is exact via DST and does not need truncation).
    """
    if J < 1:
        raise ValueError("J must be >= 1")

    sigmas: list[float] = []
    if include_poisson:
        sigmas.append(0.0)
        n_screened = J - 1
    else:
        n_screened = J

    if n_screened > 0:
        # length scales in units of h: 2, 4, 8, ...
        ell = h * (2.0 ** np.arange(1, n_screened + 1))
        sigmas.extend((1.0 / ell**2).tolist())

    return np.asarray(sigmas[:J], dtype=np.float64)


@dataclass
class ScreenedLGFBank:
    """Bank of preconditioners: apply (-a0 Delta + λ_j I)^{-1} to r for each λ_j."""

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
        """Return stack z[j] = G_{λ_j} * r with shape (J, n, n)."""
        if r.shape != (self.grid.n, self.grid.n):
            raise ValueError("r must have shape (n, n)")
        if not np.all(np.isfinite(r)):
            raise ValueError("r must contain only finite values")
        r_hat = dstn(r, type=1, norm="ortho")
        z_hat = r_hat[None, :, :] / self._denominators
        return idstn(z_hat, type=1, norm="ortho", axes=(-2, -1))

    def mix(self, r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute sum_j alpha_j * G_{λ_j} r.

        alpha: (J,) global weights, or (J, n, n) pointwise weights.
        """
        z = self.apply(r)
        a = np.asarray(alpha, dtype=np.float64)
        if a.ndim == 1:
            if a.shape[0] != self.J:
                raise ValueError("alpha shape mismatch")
            return np.tensordot(a, z, axes=(0, 0))
        if a.shape != z.shape:
            raise ValueError("alpha shape mismatch")
        return np.sum(a * z, axis=0)
