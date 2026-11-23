"""Geometric V-cycle and optional PyAMG baselines for the same flux system."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, kron

from .grid import SquareGrid
from .operators import apply_reaction_diffusion, assemble_reaction_diffusion
from .solvers import SolverResult, _pcg, _validate_problem


class AMGUnavailableError(ImportError):
    """The explicitly requested AMG dependency is unavailable."""


def geometric_interpolation(n: int):
    """Bilinear interpolation from ((n-1)/2)^2 to n^2 interior nodes.

    The omitted boundary values are zero. Restriction is P.T and coarse
    operators are P.T @ A @ P (variational scaling, not rediscretization).
    """
    if n < 3 or n % 2 != 1:
        raise ValueError("geometric interpolation needs an odd n >= 3")
    nc = (n - 1) // 2
    j = np.arange(nc)
    rows = np.concatenate([2*j, 2*j+1, 2*j+2])
    cols = np.tile(j, 3)
    weights = np.repeat([0.5, 1.0, 0.5], nc)
    p1 = coo_matrix((weights, (rows, cols)), shape=(n, nc)).tocsr()
    return kron(p1, p1, format="csr")


class GeometricVcycle:
    """Linear SPD V-cycle: geometric transfers, Galerkin coarse matrices.

    Two damped-Jacobi sweeps before and after coarse correction, starting
    every correction from zero. On each level omega=1/bound, where bound
    is the maximum absolute row sum of D^-1 A. Thus omega*lambda_max <= 1,
    and the symmetrized smoother is positive definite. P.T restriction,
    full-rank P, and an exact Cholesky coarse solve preserve SPD recursively.
    """

    def __init__(self, matrix, n: int):
        if n < 3 or (n + 1) & n:
            raise ValueError("GMG requires n=2^k-1 with k>=2 (e.g. 15,31,63,127)")
        self.matrices = [matrix.tocsr()]
        self.transfers = []
        self.smoothers = []
        self.sizes = [n]
        while n > 3:
            aa = self.matrices[-1]
            diagonal = aa.diagonal()
            bound = float(np.max(np.asarray(abs(aa).sum(axis=1)).ravel() / diagonal))
            self.smoothers.append((1.0 / bound) / diagonal)
            pp = geometric_interpolation(n)
            self.transfers.append(pp)
            self.matrices.append((pp.T @ aa @ pp).tocsr())
            n = (n - 1) // 2
            self.sizes.append(n)
        self.coarse_factor = cho_factor(self.matrices[-1].toarray(), lower=True)

    def _cycle(self, level: int, rhs: np.ndarray) -> np.ndarray:
        if level == len(self.matrices) - 1:
            return cho_solve(self.coarse_factor, rhs)
        aa, pp = self.matrices[level], self.transfers[level]
        scale = self.smoothers[level]
        u = np.zeros_like(rhs)
        for _ in range(2):
            u += scale * (rhs - aa @ u)
        u += pp @ self._cycle(level + 1, pp.T @ (rhs - aa @ u))
        for _ in range(2):
            u += scale * (rhs - aa @ u)
        return u

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        return self._cycle(0, rhs.ravel()).reshape(rhs.shape)


def amg_preconditioner(matrix):
    """Ruge-Stuben AMG with adjoint GS sweeps and Cholesky coarse solve.

    For this real SPD matrix PyAMG uses R=P.T and Galerkin coarse systems.
    Forward pre-GS and backward post-GS form an SPD-compatible V-cycle.
    No adaptive/inner Krylov solve is used, so the preconditioner is fixed.
    """
    try:
        import pyamg
    except ImportError as exc:
        raise AMGUnavailableError(
            "AMG requires optional PyAMG. Install into your active environment: "
            "python -m pip install pyamg"
        ) from exc
    hierarchy = pyamg.ruge_stuben_solver(
        matrix, presmoother=("gauss_seidel", {"sweep": "forward"}),
        postsmoother=("gauss_seidel", {"sweep": "backward"}),
        coarse_solver="cholesky", max_coarse=9,
    )
    inverse = hierarchy.aspreconditioner(cycle="V")
    inverse @ np.zeros(matrix.shape[0])  # Include lazy coarse factorization in setup.
    return inverse


def solve_multigrid_pcg(
    a: np.ndarray, f: np.ndarray, grid: SquareGrid, *, method: str,
    q: float | np.ndarray = 0.0, a_nodes: np.ndarray | None = None,
    tol: float = 1e-10, maxiter: int = 500, u0: np.ndarray | None = None,
) -> SolverResult:
    """GMG/AMG V-cycle-preconditioned CG, sharing the LGF-PCG loop.

    Matrix assembly and preconditioner construction belong to setup time.
    All outer actions and final residuals use the common matrix-free operator.
    """
    started = perf_counter()
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    aa = assemble_reaction_diffusion(a, grid, q, a_nodes=a_nodes)
    if method == "gmg":
        pinv = GeometricVcycle(aa, grid.n)
    elif method == "amg":
        inverse = amg_preconditioner(aa)
        pinv = lambda rhs: (inverse @ rhs.ravel()).reshape(rhs.shape)
    else:
        raise ValueError("method must be 'gmg' or 'amg'")
    setup_time = perf_counter() - started
    action = lambda u: apply_reaction_diffusion(u, a, grid, q, a_nodes=a_nodes)
    result = _pcg(action, pinv, f, grid, tol=tol, maxiter=maxiter,
                  u0=u0, a0=float(np.mean(a)))
    result.setup_time = setup_time
    return result
