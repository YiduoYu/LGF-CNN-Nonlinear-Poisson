"""Manufactured solutions for -div(a(x,y) grad u) = f with homogeneous Dirichlet BC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grid import SquareGrid
from .operators import apply_div_a_grad


ArrayFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class ManufacturedProblem:
    name: str
    grid: SquareGrid
    a: np.ndarray
    u_exact: np.ndarray
    f: np.ndarray
    a_fn: ArrayFn
    u_fn: ArrayFn


def _a_smooth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth positive diffusivity."""
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


def _a_mild(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Milder contrast for easy first tests."""
    return 1.0 + 0.2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def _u_sine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Homogeneous Dirichlet manufactured solution."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _u_sine2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)


def _forcing_analytic_smooth_sine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic f = -div(a grad u) for a=_a_smooth, u=_u_sine on unit square.

    u = sin(pi x) sin(pi y)
    a = 1 + 0.5 sin(2 pi x) cos(2 pi y)

    Computed from expanded divergence form:
      f = -a Delta u - grad a · grad u
    """
    pi = np.pi
    u = np.sin(pi * x) * np.sin(pi * y)
    a = _a_smooth(x, y)

    # Delta u = -2 pi^2 u
    lap_u = -2.0 * pi**2 * u

    ux = pi * np.cos(pi * x) * np.sin(pi * y)
    uy = pi * np.sin(pi * x) * np.cos(pi * y)

    ax = 0.5 * 2.0 * pi * np.cos(2.0 * pi * x) * np.cos(2.0 * pi * y)
    ay = 0.5 * np.sin(2.0 * pi * x) * (-2.0 * pi * np.sin(2.0 * pi * y))

    # -div(a grad u) = -a lap u - grad a · grad u
    return -a * lap_u - (ax * ux + ay * uy)


PROBLEMS = {
    "smooth_sine": (_a_smooth, _u_sine, "analytic"),
    "mild_sine": (_a_mild, _u_sine, "discrete"),
    "smooth_sine2": (_a_smooth, _u_sine2, "discrete"),
}


def make_problem(
    name: str = "smooth_sine",
    n: int = 63,
    L: float = 1.0,
    forcing: str | None = None,
) -> ManufacturedProblem:
    """Build a manufactured problem on an n x n interior grid.

    forcing:
      - 'analytic': use closed-form f when available
      - 'discrete': set f = A u_exact with the discrete operator (consistency test)
      - None: choose the problem default
    """
    if name not in PROBLEMS:
        raise KeyError(f"Unknown problem {name!r}. Choose from {list(PROBLEMS)}")

    a_fn, u_fn, default_forcing = PROBLEMS[name]
    mode = default_forcing if forcing is None else forcing

    grid = SquareGrid(n=n, L=L)
    X, Y = grid.mesh()
    a = a_fn(X, Y)
    u_exact = u_fn(X, Y)

    if mode == "analytic":
        if name != "smooth_sine":
            raise ValueError(f"No analytic forcing for problem {name!r}")
        f = _forcing_analytic_smooth_sine(X, Y)
    elif mode == "discrete":
        f = apply_div_a_grad(u_exact, a, grid)
    else:
        raise ValueError("forcing must be 'analytic' or 'discrete'")

    return ManufacturedProblem(
        name=name,
        grid=grid,
        a=a,
        u_exact=u_exact,
        f=f,
        a_fn=a_fn,
        u_fn=u_fn,
    )
