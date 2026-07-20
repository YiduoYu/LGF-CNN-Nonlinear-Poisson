"""Iterative solvers for -div(a grad u) = f using Dirichlet LGF preconditioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import SquareGrid
from .lgf_bank import ScreenedLGFBank
from .lgf_inverse import DirichletLGF
from .operators import apply_div_a_grad, l2_norm


@dataclass
class SolverResult:
    u: np.ndarray
    iterations: int
    residual_history: list[float]
    converged: bool
    a0: float


def _default_a0(a: np.ndarray) -> float:
    """Reference diffusivity for the LGF preconditioner."""
    return float(np.mean(a))


def solve_axy_pcg(
    a: np.ndarray,
    f: np.ndarray,
    grid: SquareGrid,
    a0: float | None = None,
    tol: float = 1e-10,
    maxiter: int = 500,
    u0: np.ndarray | None = None,
) -> SolverResult:
    """PCG for -div(a grad u) = f with Dirichlet LGF preconditioner (-a0 Delta)^{-1}."""
    if a0 is None:
        a0 = _default_a0(a)
    if a0 <= 0:
        raise ValueError("a0 must be positive")

    Pinv = DirichletLGF(grid, a0=a0, sigma=0.0)
    u = grid.zeros() if u0 is None else u0.copy()

    r = f - apply_div_a_grad(u, a, grid)
    z = Pinv(r)
    p = z.copy()
    rz = float(np.sum(r * z))

    f_norm = l2_norm(f, grid)
    if f_norm == 0.0:
        f_norm = 1.0

    history: list[float] = [l2_norm(r, grid) / f_norm]
    converged = history[0] < tol

    it = 0
    while it < maxiter and not converged:
        Ap = apply_div_a_grad(p, a, grid)
        pAp = float(np.sum(p * Ap))
        if pAp <= 0:
            raise RuntimeError("PCG breakdown: operator not SPD (check a > 0)")
        alpha = rz / pAp
        u = u + alpha * p
        r = r - alpha * Ap
        rel = l2_norm(r, grid) / f_norm
        history.append(rel)
        converged = rel < tol
        it += 1
        if converged:
            break
        z = Pinv(r)
        rz_new = float(np.sum(r * z))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return SolverResult(
        u=u,
        iterations=it,
        residual_history=history,
        converged=converged,
        a0=a0,
    )


def solve_axy_richardson(
    a: np.ndarray,
    f: np.ndarray,
    grid: SquareGrid,
    a0: float | None = None,
    omega: float = 1.0,
    tol: float = 1e-10,
    maxiter: int = 2000,
    u0: np.ndarray | None = None,
) -> SolverResult:
    """Damped Richardson iteration with Dirichlet LGF preconditioner.

    u <- u + omega * (-a0 Delta)^{-1} (f - A u)

    This is the classical fixed-point form closest to an LGF-CNN block
    with constant mixing weights.
    """
    if a0 is None:
        a0 = _default_a0(a)
    Pinv = DirichletLGF(grid, a0=a0, sigma=0.0)
    u = grid.zeros() if u0 is None else u0.copy()

    f_norm = l2_norm(f, grid)
    if f_norm == 0.0:
        f_norm = 1.0

    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        r = f - apply_div_a_grad(u, a, grid)
        rel = l2_norm(r, grid) / f_norm
        history.append(rel)
        if rel < tol:
            converged = True
            break
        u = u + omega * Pinv(r)

    return SolverResult(
        u=u,
        iterations=it,
        residual_history=history,
        converged=converged,
        a0=a0,
    )


def solve_axy_lgf_bank(
    a: np.ndarray,
    f: np.ndarray,
    grid: SquareGrid,
    J: int = 6,
    a0: float | None = None,
    omega: float = 1.0,
    tol: float = 1e-10,
    maxiter: int = 2000,
    u0: np.ndarray | None = None,
) -> SolverResult:
    """Richardson with a screened LGF bank and local 1/a scaling.

    Uses equal bank weights as a classical baseline for LGF-CNN:
        δu = omega * (1/a) * mean_j G_{λ_j} r
    """
    if a0 is None:
        a0 = _default_a0(a)
    bank = ScreenedLGFBank.from_defaults(grid, J=J, a0=a0, include_poisson=True)
    alpha = np.full(bank.J, 1.0 / bank.J, dtype=np.float64)
    inv_a = 1.0 / np.clip(a, 1e-8, None)
    u = grid.zeros() if u0 is None else u0.copy()

    f_norm = l2_norm(f, grid)
    if f_norm == 0.0:
        f_norm = 1.0

    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        r = f - apply_div_a_grad(u, a, grid)
        rel = l2_norm(r, grid) / f_norm
        history.append(rel)
        if rel < tol:
            converged = True
            break
        du = inv_a * bank.mix(r, alpha)
        u = u + omega * du

    return SolverResult(
        u=u,
        iterations=it,
        residual_history=history,
        converged=converged,
        a0=a0,
    )
