"""Iterative solvers for -div(a grad u) = f using Dirichlet LGF preconditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


def _validate_problem(
    a: np.ndarray,
    f: np.ndarray,
    grid: SquareGrid,
    *,
    maxiter: int,
    u0: np.ndarray | None,
) -> None:
    shape = (grid.n, grid.n)
    if a.shape != shape or f.shape != shape:
        raise ValueError("a and f must have shape (n, n)")
    if u0 is not None and u0.shape != shape:
        raise ValueError("u0 must have shape (n, n)")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(f)):
        raise ValueError("a and f must contain only finite values")
    if u0 is not None and not np.all(np.isfinite(u0)):
        raise ValueError("u0 must contain only finite values")
    if np.any(a <= 0.0):
        raise ValueError("a must be strictly positive")
    if maxiter < 0:
        raise ValueError("maxiter must be nonnegative")


def _validate_iteration_parameters(
    *,
    a0: float,
    omega: float,
    tol: float,
) -> None:
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 must be finite and positive")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")


def _relative_residual(
    r: np.ndarray,
    f_norm: float,
    grid: SquareGrid,
) -> float:
    return l2_norm(r, grid) / f_norm


def _solve_fixed_point(
    a: np.ndarray,
    f: np.ndarray,
    grid: SquareGrid,
    correction: Callable[[np.ndarray], np.ndarray],
    *,
    a0: float,
    tol: float,
    maxiter: int,
    u0: np.ndarray | None,
) -> SolverResult:
    """Run a fixed-point iteration and record every returned iterate."""
    u = grid.zeros() if u0 is None else u0.copy()
    f_norm = max(l2_norm(f, grid), np.finfo(np.float64).tiny)
    r = f - apply_div_a_grad(u, a, grid)
    history = [_relative_residual(r, f_norm, grid)]
    converged = tol > 0.0 and history[-1] <= tol

    iterations = 0
    while iterations < maxiter and not converged:
        u = u + correction(r)
        iterations += 1
        r = f - apply_div_a_grad(u, a, grid)
        history.append(_relative_residual(r, f_norm, grid))
        converged = tol > 0.0 and history[-1] <= tol

    return SolverResult(
        u=u,
        iterations=iterations,
        residual_history=history,
        converged=converged,
        a0=a0,
    )


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
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if a0 is None:
        a0 = _default_a0(a)
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 must be finite and positive")

    Pinv = DirichletLGF(grid, a0=a0, sigma=0.0)
    u = grid.zeros() if u0 is None else u0.copy()

    r = f - apply_div_a_grad(u, a, grid)
    z = Pinv(r)
    p = z.copy()
    rz = float(np.sum(r * z))

    f_norm = max(l2_norm(f, grid), np.finfo(np.float64).tiny)

    history = [_relative_residual(r, f_norm, grid)]
    converged = history[0] <= tol

    it = 0
    while it < maxiter and not converged:
        Ap = apply_div_a_grad(p, a, grid)
        pAp = float(np.sum(p * Ap))
        if pAp <= 0:
            raise RuntimeError("PCG breakdown: operator not SPD (check a > 0)")
        alpha = rz / pAp
        u = u + alpha * p
        r = r - alpha * Ap
        rel = _relative_residual(r, f_norm, grid)
        history.append(rel)
        converged = rel <= tol
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
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    a0 = _default_a0(a) if a0 is None else float(a0)
    _validate_iteration_parameters(a0=a0, omega=omega, tol=tol)
    pinv = DirichletLGF(grid, a0=a0, sigma=0.0)
    return _solve_fixed_point(
        a,
        f,
        grid,
        lambda r: omega * pinv(r),
        a0=a0,
        tol=tol,
        maxiter=maxiter,
        u0=u0,
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
    """Richardson with a screened LGF bank and local a0/a scaling.

    Uses equal bank weights as a classical baseline for LGF-CNN:
        δu = omega * (a0/a) * mean_j G_{λ_j} r

    The factor a0/a preserves invariance under a uniform rescaling of a while
    retaining the local coefficient correction.
    """
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    a0 = _default_a0(a) if a0 is None else float(a0)
    _validate_iteration_parameters(a0=a0, omega=omega, tol=tol)
    bank = ScreenedLGFBank.from_defaults(grid, J=J, a0=a0, include_poisson=True)
    alpha = np.full(bank.J, 1.0 / bank.J, dtype=np.float64)
    local_scale = a0 / a
    return _solve_fixed_point(
        a,
        f,
        grid,
        lambda r: omega * local_scale * bank.mix(r, alpha),
        a0=a0,
        tol=tol,
        maxiter=maxiter,
        u0=u0,
    )
