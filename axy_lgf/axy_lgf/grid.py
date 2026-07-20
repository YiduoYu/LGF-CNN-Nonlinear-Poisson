"""Uniform Cartesian grid on a square with homogeneous Dirichlet BC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SquareGrid:
    """Interior grid for Omega = (0, L)^2 with u = 0 on the boundary.

    Interior nodes: x_i = i * h, i = 1..n, with h = L / (n + 1).
    """

    n: int
    L: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.n, (int, np.integer)) or self.n < 2:
            raise ValueError("n must be an integer >= 2")
        if not np.isfinite(self.L) or self.L <= 0:
            raise ValueError("L must be finite and positive")

    @property
    def h(self) -> float:
        return self.L / (self.n + 1)

    @property
    def x1d(self) -> np.ndarray:
        return np.arange(1, self.n + 1, dtype=np.float64) * self.h

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x1d
        return np.meshgrid(x, x, indexing="xy")

    def zeros(self) -> np.ndarray:
        return np.zeros((self.n, self.n), dtype=np.float64)
