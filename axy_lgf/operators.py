"""Discrete operators: gradient, divergence, -div(a grad) on a square grid."""

from __future__ import annotations

import numpy as np

from .grid import SquareGrid


def apply_div_a_grad(u: np.ndarray, a: np.ndarray, grid: SquareGrid) -> np.ndarray:
    """Compute -div_h(a grad_h u) with homogeneous Dirichlet BC.

    Standard 5-point flux form. Face diffusivities are the arithmetic mean
    of a at the two adjacent nodes; boundary faces use the node's own a.
    """
    if u.shape != a.shape or u.shape != (grid.n, grid.n):
        raise ValueError("u and a must have shape (n, n)")

    h = grid.h
    padded_u = np.pad(u, pad_width=1, mode="constant", constant_values=0.0)

    ax = np.concatenate([a[:, :1], 0.5 * (a[:, :-1] + a[:, 1:]), a[:, -1:]], axis=1)
    ay = np.concatenate([a[:1, :], 0.5 * (a[:-1, :] + a[1:, :]), a[-1:, :]], axis=0)

    interior = padded_u[1:-1, 1:-1]
    flux_left = ax[:, :-1] * ((interior - padded_u[1:-1, :-2]) / h)
    flux_right = ax[:, 1:] * ((padded_u[1:-1, 2:] - interior) / h)
    flux_down = ay[:-1, :] * ((interior - padded_u[:-2, 1:-1]) / h)
    flux_up = ay[1:, :] * ((padded_u[2:, 1:-1] - interior) / h)

    return -((flux_right - flux_left) + (flux_up - flux_down)) / h


def apply_minus_laplacian(u: np.ndarray, grid: SquareGrid) -> np.ndarray:
    """Apply -Delta_h with homogeneous Dirichlet BC (5-point stencil)."""
    if u.shape != (grid.n, grid.n):
        raise ValueError("u must have shape (n, n)")
    padded_u = np.pad(u, pad_width=1, mode="constant", constant_values=0.0)
    return (
        4.0 * padded_u[1:-1, 1:-1]
        - padded_u[1:-1, :-2]
        - padded_u[1:-1, 2:]
        - padded_u[:-2, 1:-1]
        - padded_u[2:, 1:-1]
    ) / grid.h**2


def l2_norm(v: np.ndarray, grid: SquareGrid) -> float:
    """Discrete L2 norm on the interior grid."""
    return float(np.sqrt(np.sum(v * v) * grid.h**2))
