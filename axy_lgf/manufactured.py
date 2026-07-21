"""Manufactured solutions for -div(a(x,y) grad u) = f with homogeneous Dirichlet BC."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

from .grid import DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN, SquareGrid
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


def _normalized(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    length = x_max - x_min
    return (x - x_min) / length, (y - x_min) / length, length


def _a_smooth(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = DEFAULT_DOMAIN_MIN,
    x_max: float = DEFAULT_DOMAIN_MAX,
) -> np.ndarray:
    """Smooth positive diffusivity."""
    xi, yi, _ = _normalized(x, y, x_min, x_max)
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * xi) * np.cos(2.0 * np.pi * yi)


def _a_mild(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = DEFAULT_DOMAIN_MIN,
    x_max: float = DEFAULT_DOMAIN_MAX,
) -> np.ndarray:
    """Milder contrast for easy first tests."""
    xi, yi, _ = _normalized(x, y, x_min, x_max)
    return 1.0 + 0.2 * np.sin(np.pi * xi) * np.sin(np.pi * yi)


def _u_sine(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = DEFAULT_DOMAIN_MIN,
    x_max: float = DEFAULT_DOMAIN_MAX,
) -> np.ndarray:
    """Homogeneous Dirichlet manufactured solution."""
    xi, yi, _ = _normalized(x, y, x_min, x_max)
    return np.sin(np.pi * xi) * np.sin(np.pi * yi)


def _u_sine2(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = DEFAULT_DOMAIN_MIN,
    x_max: float = DEFAULT_DOMAIN_MAX,
) -> np.ndarray:
    xi, yi, _ = _normalized(x, y, x_min, x_max)
    return np.sin(2.0 * np.pi * xi) * np.sin(np.pi * yi)


def _forcing_analytic_smooth_sine(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = DEFAULT_DOMAIN_MIN,
    x_max: float = DEFAULT_DOMAIN_MAX,
) -> np.ndarray:
    """Analytic f = -div(a grad u) for a=_a_smooth, u=_u_sine.

    u = sin(pi xi) sin(pi yi)
    a = 1 + 0.5 sin(2 pi xi) cos(2 pi yi)

    Computed from expanded divergence form:
      f = -a Delta u - grad a · grad u
    """
    xi, yi, length = _normalized(x, y, x_min, x_max)
    k = np.pi / length
    u = np.sin(np.pi * xi) * np.sin(np.pi * yi)
    a = _a_smooth(x, y, x_min, x_max)

    # Delta u = -2 (pi/domain_length)^2 u
    lap_u = -2.0 * k**2 * u

    ux = k * np.cos(np.pi * xi) * np.sin(np.pi * yi)
    uy = k * np.sin(np.pi * xi) * np.cos(np.pi * yi)

    ax = k * np.cos(2.0 * np.pi * xi) * np.cos(2.0 * np.pi * yi)
    ay = -k * np.sin(2.0 * np.pi * xi) * np.sin(2.0 * np.pi * yi)

    # -div(a grad u) = -a lap u - grad a · grad u
    return -a * lap_u - (ax * ux + ay * uy)


PROBLEMS = {
    "smooth_sine": (_a_smooth, _u_sine, "analytic"),
    "mild_sine": (_a_mild, _u_sine, "discrete"),
    "smooth_sine2": (_a_smooth, _u_sine2, "discrete"),
}

FIXED_CASES = ("u1", "u2", "u3", "u4")


def fixed_case_solutions(grid: SquareGrid) -> tuple[list[str], np.ndarray]:
    """The four fixed u1-u4 test solutions, stacked as (4, n, n).

    Every case vanishes on the configured square boundary, which keeps them
    consistent with the zero Dirichlet halo used by the discrete operator.
    Single definition shared by evaluation and plotting.
    """
    x, y = grid.mesh()
    xi, yi, _ = _normalized(x, y, grid.x_min, grid.x_max)
    fields = {
        "u1": np.sin(np.pi * xi) * np.sin(np.pi * yi),
        "u2": np.sin(2.0 * np.pi * xi) * np.sin(np.pi * yi),
        "u3": np.sin(np.pi * xi) * np.sin(2.0 * np.pi * yi)
        - 0.5 * np.sin(2.0 * np.pi * xi) * np.sin(2.0 * np.pi * yi),
        "u4": np.sin(3.0 * np.pi * xi) * np.sin(3.0 * np.pi * yi),
    }
    return list(fields), np.stack(list(fields.values()), axis=0)


def make_problem(
    name: str = "smooth_sine",
    n: int = 63,
    x_min: float = DEFAULT_DOMAIN_MIN,
    x_max: float = DEFAULT_DOMAIN_MAX,
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

    grid = SquareGrid(n=n, x_min=x_min, x_max=x_max)
    X, Y = grid.mesh()
    a_fn: ArrayFn = partial(a_base, x_min=x_min, x_max=x_max)
    u_fn: ArrayFn = partial(u_base, x_min=x_min, x_max=x_max)
    a = a_fn(X, Y)
    u_exact = u_fn(X, Y)

    if mode == "analytic":
        if name != "smooth_sine":
            raise ValueError(f"No analytic forcing for problem {name!r}")
        f = _forcing_analytic_smooth_sine(
            X,
            Y,
            x_min=x_min,
            x_max=x_max,
        )
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
