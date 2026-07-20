"""Uniform Cartesian grid on a configurable square with homogeneous Dirichlet BC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DOMAIN_MIN = -1.0
DEFAULT_DOMAIN_MAX = 1.0


@dataclass(frozen=True)
class SquareGrid:
    """Interior grid for Omega = (x_min, x_max)^2 with zero boundary values.

    Interior nodes: x_i = x_min + i*h, i=1..n.
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
        """Side length of the square domain."""
        return self.x_max - self.x_min

    @property
    def h(self) -> float:
        return self.L / (self.n + 1)

    @property
    def x1d(self) -> np.ndarray:
        return self.x_min + np.arange(1, self.n + 1, dtype=np.float64) * self.h

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x1d
        return np.meshgrid(x, x, indexing="xy")

    def zeros(self) -> np.ndarray:
        return np.zeros((self.n, self.n), dtype=np.float64)
