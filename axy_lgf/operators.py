"""Discrete gradient, divergence, and -div(a grad) on a square grid."""

from __future__ import annotations

import numpy as np

from .grid import SquareGrid


def pad_dirichlet(u: np.ndarray) -> np.ndarray:
    """Pad interior array with a one-cell zero Dirichlet halo."""
    return np.pad(u, pad_width=1, mode="constant", constant_values=0.0)


def face_a_x(a: np.ndarray) -> np.ndarray:
    """Harmonic-mean diffusivity on vertical faces (between x-neighbors).

    a is defined at cell centers / nodes of shape (n, n).
    Returns ax of shape (n, n+1) for faces between padded nodes.
    For interior faces between u[:, j] and u[:, j+1], use avg of a.
    """
    # Arithmetic mean is standard for smooth positive a; harmonic mean
    # is more robust for large jumps. Use arithmetic for the first test.
    n = a.shape[0]
    ax = np.zeros((n, n + 1), dtype=np.float64)
    ax[:, 0] = a[:, 0]
    ax[:, -1] = a[:, -1]
    ax[:, 1:-1] = 0.5 * (a[:, :-1] + a[:, 1:])
    return ax


def face_a_y(a: np.ndarray) -> np.ndarray:
    """Arithmetic-mean diffusivity on horizontal faces (between y-neighbors)."""
    n = a.shape[0]
    ay = np.zeros((n + 1, n), dtype=np.float64)
    ay[0, :] = a[0, :]
    ay[-1, :] = a[-1, :]
    ay[1:-1, :] = 0.5 * (a[:-1, :] + a[1:, :])
    return ay


def apply_div_a_grad(u: np.ndarray, a: np.ndarray, grid: SquareGrid) -> np.ndarray:
    """Compute -div_h(a grad_h u) with homogeneous Dirichlet BC.

    Uses a standard 5-point flux form:
        flux_x at faces between nodes, flux_y similarly.
    """
    if u.shape != a.shape or u.shape != (grid.n, grid.n):
        raise ValueError("u and a must have shape (n, n)")

    h = grid.h
    U = pad_dirichlet(u)
    ax = face_a_x(a)
    ay = face_a_y(a)

    # Interior node (i,j) corresponds to U[i+1, j+1]
    # x-fluxes into node from left/right faces
    # left face between U[i+1,j] and U[i+1,j+1]: index ax[i, j]
    # right face between U[i+1,j+1] and U[i+1,j+2]: index ax[i, j+1]
    ux_left = (U[1:-1, 1:-1] - U[1:-1, :-2]) / h
    ux_right = (U[1:-1, 2:] - U[1:-1, 1:-1]) / h
    uy_down = (U[1:-1, 1:-1] - U[:-2, 1:-1]) / h
    uy_up = (U[2:, 1:-1] - U[1:-1, 1:-1]) / h

    flux_left = ax[:, :-1] * ux_left
    flux_right = ax[:, 1:] * ux_right
    flux_down = ay[:-1, :] * uy_down
    flux_up = ay[1:, :] * uy_up

    # -div = -( (flux_right - flux_left)/h + (flux_up - flux_down)/h )
    return -((flux_right - flux_left) + (flux_up - flux_down)) / h


def apply_minus_laplacian(u: np.ndarray, grid: SquareGrid) -> np.ndarray:
    """Apply -Delta_h with homogeneous Dirichlet BC (5-point stencil)."""
    a = np.ones_like(u)
    return apply_div_a_grad(u, a, grid)


def l2_norm(v: np.ndarray, grid: SquareGrid) -> float:
    """Discrete L2 norm on the interior grid."""
    return float(np.sqrt(np.sum(v * v) * grid.h**2))
