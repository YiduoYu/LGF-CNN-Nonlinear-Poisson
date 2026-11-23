"""Discrete operators: gradient, divergence, -div(a grad) on a square grid."""

from __future__ import annotations

import numpy as np

from .grid import SquareGrid


def face_diffusivities(
    a: np.ndarray, grid: SquareGrid, *, a_nodes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic nodal averages at edge midpoints (x faces, y faces).

    Optional a_nodes includes boundary nodes, shape (n+2, n+2), with
    a_nodes[1:-1, 1:-1] == a. Omitting it preserves the legacy convention
    of copying the nearest interior coefficient to each boundary face.
    """
    if a.shape != (grid.n, grid.n):
        raise ValueError("a must have shape (n, n)")
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("a must be finite and strictly positive")
    if a_nodes is not None:
        if a_nodes.shape != (grid.n + 2, grid.n + 2):
            raise ValueError("a_nodes must have shape (n+2, n+2)")
        if not np.all(np.isfinite(a_nodes)) or np.any(a_nodes <= 0):
            raise ValueError("a_nodes must be finite and strictly positive")
        if not np.array_equal(a_nodes[1:-1, 1:-1], a):
            raise ValueError("a_nodes interior must equal a")
        return (
            0.5 * (a_nodes[1:-1, :-1] + a_nodes[1:-1, 1:]),
            0.5 * (a_nodes[:-1, 1:-1] + a_nodes[1:, 1:-1]),
        )
    ax = np.concatenate([a[:, :1], 0.5 * (a[:, :-1] + a[:, 1:]), a[:, -1:]], axis=1)
    ay = np.concatenate([a[:1, :], 0.5 * (a[:-1, :] + a[1:, :]), a[-1:, :]], axis=0)
    return ax, ay


def reaction_field(q: float | np.ndarray, grid: SquareGrid) -> np.ndarray:
    """Validate and broadcast a nonnegative nodal reaction coefficient."""
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 0 and values.shape != (grid.n, grid.n):
        raise ValueError("q must be scalar or have shape (n, n)")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("q must be finite and nonnegative")
    return np.broadcast_to(values, (grid.n, grid.n))


def apply_div_a_grad(
    u: np.ndarray, a: np.ndarray, grid: SquareGrid,
    *, a_nodes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute -div_h(a grad_h u) with homogeneous Dirichlet BC.

    Standard 5-point flux form. Face diffusivities are the arithmetic mean
    of a at the two adjacent nodes. Supply a_nodes to include boundary
    coefficient samples; by default boundary faces use the node's own a.
    """
    if u.shape != a.shape or u.shape != (grid.n, grid.n):
        raise ValueError("u and a must have shape (n, n)")

    h = grid.h
    padded_u = np.pad(u, pad_width=1, mode="constant", constant_values=0.0)

    ax, ay = face_diffusivities(a, grid, a_nodes=a_nodes)

    interior = padded_u[1:-1, 1:-1]
    flux_left = ax[:, :-1] * ((interior - padded_u[1:-1, :-2]) / h)
    flux_right = ax[:, 1:] * ((padded_u[1:-1, 2:] - interior) / h)
    flux_down = ay[:-1, :] * ((interior - padded_u[:-2, 1:-1]) / h)
    flux_up = ay[1:, :] * ((padded_u[2:, 1:-1] - interior) / h)

    return -((flux_right - flux_left) + (flux_up - flux_down)) / h


def apply_reaction_diffusion(
    u: np.ndarray, a: np.ndarray, grid: SquareGrid,
    q: float | np.ndarray = 0.0, *, a_nodes: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the linear flux operator -div_h(a grad_h u) + q*u."""
    qq = reaction_field(q, grid)
    return apply_div_a_grad(u, a, grid, a_nodes=a_nodes) + qq * u


def assemble_reaction_diffusion(
    a: np.ndarray, grid: SquareGrid, q: float | np.ndarray = 0.0,
    *, a_nodes: np.ndarray | None = None,
):
    """CSR matrix for exactly the same flux operator, C-order flattening.

    Boundary edges contribute to the diagonal, with zero prescribed u.
    Each interior edge contributes w * [[1, -1], [-1, 1]], w=a_face/h^2.
    """
    from scipy.sparse import coo_matrix

    ax, ay = face_diffusivities(a, grid, a_nodes=a_nodes)
    qq = reaction_field(q, grid)
    indices = np.arange(grid.n**2).reshape(grid.n, grid.n)
    diagonal = (ax[:, :-1] + ax[:, 1:] + ay[:-1, :] + ay[1:, :]) / grid.h**2 + qq
    left, right = indices[:, :-1].ravel(), indices[:, 1:].ravel()
    down, up = indices[:-1, :].ravel(), indices[1:, :].ravel()
    wx, wy = -ax[:, 1:-1].ravel() / grid.h**2, -ay[1:-1, :].ravel() / grid.h**2
    rows = np.concatenate([indices.ravel(), left, right, down, up])
    cols = np.concatenate([indices.ravel(), right, left, up, down])
    data = np.concatenate([diagonal.ravel(), wx, wx, wy, wy])
    return coo_matrix((data, (rows, cols)), shape=(grid.n**2, grid.n**2)).tocsr()


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
