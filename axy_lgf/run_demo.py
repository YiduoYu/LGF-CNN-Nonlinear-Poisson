#!/usr/bin/env python3
"""Demo: solve -div(a(x,y) grad u) = f on a square with homogeneous Dirichlet BC.

Uses Dirichlet LGF (DST inverse of -a0 Delta) as a preconditioner. No BAE.

Usage:
    source ~/myenv/bin/activate
    python -m axy_lgf.run_demo
    python -m axy_lgf.run_demo --n 127 --problem smooth_sine --solver pcg
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .grid import SquareGrid
from .lgf_inverse import apply_dirichlet_lgf
from .manufactured import make_problem
from .operators import apply_minus_laplacian, l2_norm
from .solvers import solve_axy_pcg, solve_axy_richardson, solve_axy_lgf_bank


def check_constant_coefficient_lgf(n: int = 63) -> None:
    """Sanity check: LGF inverse recovers the Poisson solve for a ≡ 1."""
    grid = SquareGrid(n=n)
    X, Y = grid.mesh()
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = apply_minus_laplacian(u_exact, grid)
    u = apply_dirichlet_lgf(f, grid, a0=1.0)
    err = l2_norm(u - u_exact, grid)
    print(f"[check] constant-a Poisson LGF  n={n}  L2 error = {err:.3e}")
    if err > 1e-10:
        raise RuntimeError("Dirichlet LGF failed constant-coefficient check")


SOLVERS = {
    "pcg": solve_axy_pcg,
    "richardson": solve_axy_richardson,
    "lgf_bank": solve_axy_lgf_bank,
}


def _solve(prob, solver: str, tol: float):
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; choose from {list(SOLVERS)}")
    return SOLVERS[solver](prob.a, prob.f, prob.grid, tol=tol)


def run_once(
    problem: str, n: int, solver: str, tol: float, forcing: str | None
) -> None:
    prob = make_problem(name=problem, n=n, forcing=forcing)
    grid = prob.grid

    t0 = time.perf_counter()
    result = _solve(prob, solver, tol)
    elapsed = time.perf_counter() - t0

    err = l2_norm(result.u - prob.u_exact, grid)
    rel_res = result.residual_history[-1]
    a_min, a_max = float(prob.a.min()), float(prob.a.max())

    print(
        f"[solve] problem={problem}  n={n}  solver={solver}  "
        f"a0={result.a0:.4f}  a in [{a_min:.3f}, {a_max:.3f}]"
    )
    print(
        f"        iters={result.iterations}  rel_res={rel_res:.3e}  "
        f"L2_err={err:.3e}  time={elapsed:.3f}s  converged={result.converged}"
    )


def run_convergence(problem: str, solver: str, tol: float) -> None:
    """Grid refinement study.

    Only the analytic-forcing table measures discretization error.  With
    discrete forcing f = A_h u_exact holds exactly, so the solver reproduces
    u_exact down to its own tolerance and the resulting "error" is a solver
    floor, not an O(h^p) quantity -- a convergence rate from that column would
    be meaningless, so none is printed.
    """
    print(f"[conv] problem={problem}  solver={solver}")

    print(f"{'n':>6}  {'iters':>6}  {'L2_err':>12}   (discrete f: solver floor)")
    for n in (15, 31, 63, 127):
        prob = make_problem(name=problem, n=n, forcing="discrete")
        result = _solve(prob, solver, tol)
        err = l2_norm(result.u - prob.u_exact, prob.grid)
        print(f"{n:6d}  {result.iterations:6d}  {err:12.3e}")

    if problem != "smooth_sine":
        print("        (analytic convergence only implemented for smooth_sine)")
        return

    print(f"{'n':>6}  {'iters':>6}  {'L2_err':>12}  {'rate':>8}   (analytic f)")
    prev_err = None
    prev_h = None
    for n in (15, 31, 63, 127):
        prob = make_problem(name=problem, n=n, forcing="analytic")
        result = _solve(prob, solver, tol)
        err = l2_norm(result.u - prob.u_exact, prob.grid)
        rate_str = ""
        if prev_err is not None and err > 0:
            rate = np.log(prev_err / err) / np.log(prev_h / prob.grid.h)
            rate_str = f"{rate:8.2f}"
        print(f"{n:6d}  {result.iterations:6d}  {err:12.3e}  {rate_str:>8}")
        prev_err = err
        prev_h = prob.grid.h
        prev_h = prob.grid.h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=63, help="interior grid size")
    parser.add_argument(
        "--problem",
        default="smooth_sine",
        choices=["smooth_sine", "mild_sine", "smooth_sine2"],
    )
    parser.add_argument(
        "--solver",
        default="pcg",
        choices=["pcg", "richardson", "lgf_bank"],
    )
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument(
        "--forcing",
        default=None,
        choices=["analytic", "discrete"],
        help="override manufactured forcing mode (default: problem default)",
    )
    parser.add_argument(
        "--conv",
        action="store_true",
        help="run a small grid-convergence study",
    )
    args = parser.parse_args()

    check_constant_coefficient_lgf(n=min(args.n, 63))
    if args.conv:
        run_convergence(args.problem, args.solver, args.tol)
    else:
        run_once(args.problem, args.n, args.solver, args.tol, args.forcing)


if __name__ == "__main__":
    main()
