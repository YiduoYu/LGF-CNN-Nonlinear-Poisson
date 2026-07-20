"""Manufactured solutions for -div(a(x,y) grad u) = f with homogeneous Dirichlet BC."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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


def _a_smooth(x: np.ndarray, y: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Smooth positive diffusivity."""
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * x / L) * np.cos(2.0 * np.pi * y / L)


def _a_mild(x: np.ndarray, y: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Milder contrast for easy first tests."""
    return 1.0 + 0.2 * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)


def _u_sine(x: np.ndarray, y: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Homogeneous Dirichlet manufactured solution."""
    return np.sin(np.pi * x / L) * np.sin(np.pi * y / L)


def _u_sine2(x: np.ndarray, y: np.ndarray, L: float = 1.0) -> np.ndarray:
    return np.sin(2.0 * np.pi * x / L) * np.sin(np.pi * y / L)


def _forcing_analytic_smooth_sine(
    x: np.ndarray,
    y: np.ndarray,
    L: float = 1.0,
) -> np.ndarray:
    """Analytic f = -div(a grad u) for a=_a_smooth, u=_u_sine.

    u = sin(pi x/L) sin(pi y/L)
    a = 1 + 0.5 sin(2 pi x/L) cos(2 pi y/L)

    Computed from expanded divergence form:
      f = -a Delta u - grad a · grad u
    """
    k = np.pi / L
    u = np.sin(k * x) * np.sin(k * y)
    a = _a_smooth(x, y, L)

    # Delta u = -2 (pi/L)^2 u
    lap_u = -2.0 * k**2 * u

    ux = k * np.cos(k * x) * np.sin(k * y)
    uy = k * np.sin(k * x) * np.cos(k * y)

    ax = k * np.cos(2.0 * k * x) * np.cos(2.0 * k * y)
    ay = -k * np.sin(2.0 * k * x) * np.sin(2.0 * k * y)

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

    a_base, u_base, default_forcing = PROBLEMS[name]
    mode = default_forcing if forcing is None else forcing

    grid = SquareGrid(n=n, L=L)
    X, Y = grid.mesh()
    a_fn: ArrayFn = partial(a_base, L=L)
    u_fn: ArrayFn = partial(u_base, L=L)
    a = a_fn(X, Y)
    u_exact = u_fn(X, Y)

    if mode == "analytic":
        if name != "smooth_sine":
            raise ValueError(f"No analytic forcing for problem {name!r}")
        f = _forcing_analytic_smooth_sine(X, Y, L=L)
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
