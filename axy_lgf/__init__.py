"""
axy_lgf: variable-coefficient elliptic solver on a rectangle via LGF/DST.

Solves
    -div(a(x,y) grad u) = f    in Omega = (x_min, x_max)^2
    u = 0                      on the boundary of Omega

using a discrete Dirichlet Green's function (DST) as the LGF inverse
of -a0 * Delta, with no BAE. Includes screened LGF banks and LGF-CNN.
"""

from .solvers import solve_axy_pcg, solve_axy_richardson, solve_axy_lgf_bank
from .manufactured import make_problem
from .lgf_bank import ScreenedLGFBank

__all__ = [
    "solve_axy_pcg",
    "solve_axy_richardson",
    "solve_axy_lgf_bank",
    "make_problem",
    "ScreenedLGFBank",
]
