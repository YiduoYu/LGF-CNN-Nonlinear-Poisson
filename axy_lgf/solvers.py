"""Linear flux diffusion/reaction solvers with Dirichlet LGF preconditioning."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .grid import SquareGrid
from .lgf_bank import ScreenedLGFBank, default_screening_values
from .lgf_inverse import DirichletLGF
from .operators import apply_reaction_diffusion, l2_norm, reaction_field


@dataclass
class SolverResult:
    """Solution and convergence diagnostics from one solver run."""

    u: np.ndarray
    iterations: int
    residual_history: list[float]
    converged: bool
    a0: float
    q0: float = 0.0
    setup_time: float = 0.0
    solve_time: float = 0.0


def _default_a0(a: np.ndarray) -> float:
    return float(np.mean(a))


def _validate_problem(
    a: np.ndarray, f: np.ndarray, grid: SquareGrid,
    *, maxiter: int, u0: np.ndarray | None,
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


def _validate_iteration_parameters(*, a0: float, omega: float, tol: float) -> None:
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 must be finite and positive")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")


def _relative_residual(r: np.ndarray, f_norm: float, grid: SquareGrid) -> float:
    return l2_norm(r, grid) / f_norm


def _solve_fixed_point(
    a: np.ndarray, f: np.ndarray, grid: SquareGrid,
    correction: Callable[[np.ndarray], np.ndarray],
    *, a0: float, tol: float, maxiter: int, u0: np.ndarray | None,
    q: float | np.ndarray = 0.0, a_nodes: np.ndarray | None = None,
) -> SolverResult:
    """Fixed-point iteration with residual tracking."""
    u = grid.zeros() if u0 is None else u0.copy()
    f_norm = max(l2_norm(f, grid), np.finfo(np.float64).tiny)
    r = f - apply_reaction_diffusion(u, a, grid, q, a_nodes=a_nodes)
    history = [_relative_residual(r, f_norm, grid)]
    converged = tol > 0.0 and history[-1] <= tol

    iterations = 0
    while iterations < maxiter and not converged:
        u = u + correction(r)
        iterations += 1
        r = f - apply_reaction_diffusion(u, a, grid, q, a_nodes=a_nodes)
        history.append(_relative_residual(r, f_norm, grid))
        converged = tol > 0.0 and history[-1] <= tol

    return SolverResult(
        u=u, iterations=iterations, residual_history=history,
        converged=converged, a0=a0, q0=float(np.mean(reaction_field(q, grid))),
    )


def solve_axy_pcg(
    a: np.ndarray, f: np.ndarray, grid: SquareGrid,
    a0: float | None = None, tol: float = 1e-10, maxiter: int = 500,
    u0: np.ndarray | None = None,
    *, q: float | np.ndarray = 0.0, q0: float | None = None,
    a_nodes: np.ndarray | None = None,
) -> SolverResult:
    """PCG for -div(a grad u)+q*u=f with (a0*(-Delta_h)+q0*I)^-1.

    q0 defaults to mean(q). The unchanged defaults solve the legacy
    pure diffusion problem, including its boundary coefficient convention.
    """
    started = perf_counter()
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if a0 is None:
        a0 = _default_a0(a)
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 must be finite and positive")

    qq = reaction_field(q, grid)
    q0 = float(np.mean(qq)) if q0 is None else float(q0)
    Pinv = DirichletLGF(grid, a0=a0, sigma=q0)
    action = lambda u: apply_reaction_diffusion(u, a, grid, qq, a_nodes=a_nodes)
    action(grid.zeros())  # Validate boundary samples during setup.
    setup_time = perf_counter() - started
    result = _pcg(action, Pinv, f, grid, tol=tol, maxiter=maxiter, u0=u0, a0=a0)
    result.q0 = q0
    result.setup_time = setup_time
    return result


def _pcg(
    action: Callable[[np.ndarray], np.ndarray],
    Pinv: Callable[[np.ndarray], np.ndarray],
    f: np.ndarray, grid: SquareGrid, *, tol: float, maxiter: int,
    u0: np.ndarray | None, a0: float,
) -> SolverResult:
    """Common PCG loop. Success requires a recomputed algebraic residual."""
    started = perf_counter()
    u = grid.zeros() if u0 is None else u0.copy()

    r = f - action(u)
    z = Pinv(r)
    p = z.copy()
    rz = float(np.sum(r * z))

    f_norm = max(l2_norm(f, grid), np.finfo(np.float64).tiny)
    history = [_relative_residual(r, f_norm, grid)]
    converged = history[0] <= tol

    it = 0
    while it < maxiter and not converged:
        Ap = action(p)
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
            r = f - action(u)
            history[-1] = _relative_residual(r, f_norm, grid)
            converged = history[-1] <= tol
            if converged:
                break
            # Residual replacement and restart if recurrence drifted.
            z = Pinv(r)
            p = z.copy()
            rz = float(np.sum(r * z))
            continue
        z = Pinv(r)
        rz_new = float(np.sum(r * z))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    history[-1] = _relative_residual(f - action(u), f_norm, grid)
    converged = history[-1] <= tol
    return SolverResult(
        u=u, iterations=it, residual_history=history,
        converged=converged, a0=a0, solve_time=perf_counter() - started,
    )


def solve_axy_richardson(
    a: np.ndarray, f: np.ndarray, grid: SquareGrid,
    a0: float | None = None, omega: float = 1.0,
    tol: float = 1e-10, maxiter: int = 2000,
    u0: np.ndarray | None = None,
    *, q: float | np.ndarray = 0.0, a_nodes: np.ndarray | None = None,
) -> SolverResult:
    """Damped Richardson with Dirichlet LGF preconditioner.

    u <- u + omega * (a0*(-Delta_h)+mean(q)*I)^{-1} (f - A u)

    This is the classical fixed-point form closest to an LGF-CNN block
    with constant mixing weights.
    """
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    a0 = _default_a0(a) if a0 is None else float(a0)
    _validate_iteration_parameters(a0=a0, omega=omega, tol=tol)
    q0 = float(np.mean(reaction_field(q, grid)))
    pinv = DirichletLGF(grid, a0=a0, sigma=q0)
    return _solve_fixed_point(
        a, f, grid, lambda r: omega * pinv(r),
        a0=a0, tol=tol, maxiter=maxiter, u0=u0,
        q=q, a_nodes=a_nodes,
    )


def solve_axy_lgf_bank(
    a: np.ndarray, f: np.ndarray, grid: SquareGrid,
    J: int = 6, a0: float | None = None, omega: float = 1.0,
    tol: float = 1e-10, maxiter: int = 2000,
    u0: np.ndarray | None = None,
    *, q: float | np.ndarray = 0.0, a_nodes: np.ndarray | None = None,
) -> SolverResult:
    """Richardson with a screened LGF bank and local a0/a scaling.

    The sample reference coefficient a0 also defines the bank, so retaining
    it in the local factor preserves the correction under a uniform scaling
    of the coefficient field:
        du = omega * (a0/a) * mean_j G_{lambda_j} r
    With reaction q, all bank screening values are shifted by mean(q).
    """
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    a0 = _default_a0(a) if a0 is None else float(a0)
    _validate_iteration_parameters(a0=a0, omega=omega, tol=tol)
    q0 = float(np.mean(reaction_field(q, grid)))
    bank = ScreenedLGFBank(
        grid, default_screening_values(grid.h, J=J, include_poisson=True) + q0, a0=a0,
    )
    alpha = np.full(bank.J, 1.0 / bank.J, dtype=np.float64)
    local_scale = a0 / a

    def correction(residual: np.ndarray) -> np.ndarray:
        filtered_residual = bank.mix(residual, alpha)
        return omega * local_scale * filtered_residual

    return _solve_fixed_point(
        a, f, grid, correction,
        a0=a0, tol=tol, maxiter=maxiter, u0=u0,
        q=q, a_nodes=a_nodes,
    )
