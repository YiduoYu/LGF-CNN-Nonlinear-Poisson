"""Uniform Cartesian grid on a square with homogeneous Dirichlet BC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DOMAIN_MIN = -1.0
DEFAULT_DOMAIN_MAX = 1.0


@dataclass(frozen=True)
class SquareGrid:
    """Interior grid for Omega = (x_min, x_max)^2.

    Interior nodes: x_i = x_min + i*h, i=1..n. The boundary at i=0 and
    i=n+1 carries u=0 (homogeneous Dirichlet).
    """

    n: int
    x_min: float = DEFAULT_DOMAIN_MIN
    x_max: float = DEFAULT_DOMAIN_MAX

    def __post_init__(self) -> None:
        if not isinstance(self.n, (int, np.integer)) or self.n < 2:
            raise ValueError("n must be an integer >= 2")
        if not np.isfinite(self.x_min) or not np.isfinite(self.x_max):
            raise ValueError("x_min and x_max must be finite")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

    @property
    def L(self) -> float:
        """Side length of the domain."""
        return self.x_max - self.x_min

    @property
    def h(self) -> float:
        """Grid spacing h = L / (n+1)."""
        return self.L / (self.n + 1)

    @property
    def x1d(self) -> np.ndarray:
        """Coordinates of the n interior nodes along one axis."""
        return self.x_min + np.arange(1, self.n + 1, dtype=np.float64) * self.h

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Cartesian coordinates of all interior nodes."""
        x = self.x1d
        return np.meshgrid(x, x, indexing="xy")

    def unit_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Interior coordinates mapped to [0, 1]^2."""
        x, y = self.mesh()
        return (x - self.x_min) / self.L, (y - self.x_min) / self.L

    def zeros(self) -> np.ndarray:
        """Zero field on the interior grid."""
        return np.zeros((self.n, self.n), dtype=np.float64)
