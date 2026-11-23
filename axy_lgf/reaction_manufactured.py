"""Continuous/discrete unit-square MMS, including exponential stratification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import SquareGrid
from .operators import apply_reaction_diffusion


@dataclass
class ReactionProblem:
    grid: SquareGrid
    a: np.ndarray
    q: np.ndarray
    a_nodes: np.ndarray
    u_exact: np.ndarray
    f: np.ndarray
    forcing: str
    coefficient: str = "normal"
    a_scale: float = 1.0
    chi: float | None = None


def diffusion_coefficient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 + 0.2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)


def continuous_fields(x: np.ndarray, y: np.ndarray):
    """Analytic samples (a, q, u, f); no discrete operator in this function.

    grad(a) = 0.4*pi*(cos(2*pi*x)*sin(2*pi*y),
                         sin(2*pi*x)*cos(2*pi*y)).
    f = 2*pi^2*a*u - grad(a).grad(u) + q*u.
    """
    pi = np.pi
    u = np.sin(pi*x) * np.sin(pi*y)
    a = diffusion_coefficient(x, y)
    q = 1.0 + 0.2 * np.cos(2*pi*x) * np.cos(2*pi*y)
    ux = pi * np.cos(pi*x) * np.sin(pi*y)
    uy = pi * np.sin(pi*x) * np.cos(pi*y)
    ax = 0.4*pi * np.cos(2*pi*x) * np.sin(2*pi*y)
    ay = 0.4*pi * np.sin(2*pi*x) * np.cos(2*pi*y)
    f = 2*pi**2*a*u - ax*ux - ay*uy + q*u
    return a, q, u, f


def validate_exponential_parameters(a_scale: float, chi: float) -> None:
    """Require positive, finite float64 coefficients over the unit square."""
    if not np.isfinite(a_scale) or a_scale <= 0:
        raise ValueError("a_scale must be finite and positive")
    if not np.isfinite(chi) or chi < 1:
        raise ValueError("chi must be finite and >= 1")
    if a_scale > np.finfo(np.float64).max / chi:
        raise ValueError("a_scale*chi must be representable in float64")


def exponential_diffusion_coefficient(
    x: np.ndarray, y: np.ndarray, *, a_scale: float, chi: float,
) -> np.ndarray:
    """a(x,y)=a_scale*chi**x; a_scale is the value at x=0, not LGF mean(a).

    On the closed unit square the range is [a_scale,a_scale*chi]. Both
    parameters are explicit so choosing a new coefficient does not silently
    select a contrast. All boundary samples use this same continuous function.
    """
    validate_exponential_parameters(a_scale, chi)
    xx, _ = np.broadcast_arrays(np.asarray(x, dtype=np.float64), y)
    return a_scale * np.power(chi, xx)


def exponential_continuous_fields(
    x: np.ndarray, y: np.ndarray, *, a_scale: float, chi: float,
):
    """Same u and q as the normal baseline, with analytically updated f.

    grad(a)=(log(chi)*a,0), so
    f=(2*pi^2*a+q)*u - pi*log(chi)*a*cos(pi*x)*sin(pi*y).
    No discrete operator is used to generate this continuous forcing.
    """
    a = exponential_diffusion_coefficient(x, y, a_scale=a_scale, chi=chi)
    pi = np.pi
    u = np.sin(pi*x) * np.sin(pi*y)
    q = 1.0 + 0.2 * np.cos(2*pi*x) * np.cos(2*pi*y)
    f = (2*pi**2*a + q)*u - pi*np.log(chi)*a*np.cos(pi*x)*np.sin(pi*y)
    return a, q, u, f


def make_reaction_problem(
    n: int = 63, forcing: str = "continuous", *, coefficient: str = "normal",
    a_scale: float = 1.0, chi: float | None = None,
) -> ReactionProblem:
    """Unit-square, float64 MMS; boundary coefficients sampled explicitly.

    continuous: analytic forcing for discretization-error convergence.
    discrete: A_h*u_exact for algebraic solver consistency only.
    coefficient='normal' preserves the original problem bit-for-bit.
    coefficient='exponential' uses a_scale*chi**x and requires explicit chi.
    """
    if coefficient not in ("normal", "exponential"):
        raise ValueError("coefficient must be 'normal' or 'exponential'")
    if coefficient == "exponential" and chi is None:
        raise ValueError("exponential coefficient requires explicit chi")
    if coefficient == "normal" and (chi is not None or a_scale != 1.0):
        raise ValueError("a_scale and chi apply only to the exponential coefficient")
    grid = SquareGrid(n, x_min=0.0, x_max=1.0)
    x, y = grid.mesh()
    nodes = np.arange(n + 2, dtype=np.float64) * grid.h
    xx, yy = np.meshgrid(nodes, nodes, indexing="xy")
    if coefficient == "exponential":
        a, q, u, f = exponential_continuous_fields(x, y, a_scale=a_scale, chi=chi)
        a_nodes = exponential_diffusion_coefficient(xx, yy, a_scale=a_scale, chi=chi)
    else:
        a, q, u, f = continuous_fields(x, y)
        a_nodes = diffusion_coefficient(xx, yy)
    if forcing == "discrete":
        f = apply_reaction_diffusion(u, a, grid, q, a_nodes=a_nodes)
    elif forcing != "continuous":
        raise ValueError("forcing must be 'continuous' or 'discrete'")
    return ReactionProblem(grid, a, q, a_nodes, u, f, forcing,
                           coefficient=coefficient, a_scale=a_scale, chi=chi)
