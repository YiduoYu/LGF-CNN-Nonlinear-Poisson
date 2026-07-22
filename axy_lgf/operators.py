"""Discrete gradient, divergence, and -div(a grad) on a square grid."""

from __future__ import annotations

import numpy as np

from .grid import SquareGrid


def apply_div_a_grad(u: np.ndarray, a: np.ndarray, grid: SquareGrid) -> np.ndarray:
    """Compute -div_h(a grad_h u) with homogeneous Dirichlet BC.

    Standard 5-point flux form: flux_x at faces between nodes, flux_y
    similarly. Face diffusivities are the arithmetic mean of a at the two
    adjacent nodes; boundary faces reuse the adjacent node's own a.
    """
    if u.shape != a.shape or u.shape != (grid.n, grid.n):
        raise ValueError("u and a must have shape (n, n)")

    h = grid.h
    padded_u = np.pad(u, pad_width=1, mode="constant", constant_values=0.0)

    n = a.shape[0]
    ax = np.zeros((n, n + 1), dtype=np.float64)
    ax[:, 0] = a[:, 0]
    ax[:, -1] = a[:, -1]
    ax[:, 1:-1] = 0.5 * (a[:, :-1] + a[:, 1:])

    ay = np.zeros((n + 1, n), dtype=np.float64)
    ay[0, :] = a[0, :]
    ay[-1, :] = a[-1, :]
    ay[1:-1, :] = 0.5 * (a[:-1, :] + a[1:, :])

    ux_left = (padded_u[1:-1, 1:-1] - padded_u[1:-1, :-2]) / h
    ux_right = (padded_u[1:-1, 2:] - padded_u[1:-1, 1:-1]) / h
    uy_down = (padded_u[1:-1, 1:-1] - padded_u[:-2, 1:-1]) / h
    uy_up = (padded_u[2:, 1:-1] - padded_u[1:-1, 1:-1]) / h

    flux_left = ax[:, :-1] * ux_left
    flux_right = ax[:, 1:] * ux_right
    flux_down = ay[:-1, :] * uy_down
    flux_up = ay[1:, :] * uy_up

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
