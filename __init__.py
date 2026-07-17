"""
LGF_python: Lattice Green's Function / Boundary Algebraic Equation (LGF-BAE)
solver for 2D Poisson and modified Helmholtz equations on arbitrary domains.

Based on the Difference Potential Method (DPM) of Ryaben'kii, using
lattice Green's functions and Lagrange interpolation on Cartesian grids.

Supports:
  - Poisson equation (sigma=0) and modified Helmholtz (sigma != 0)
  - Arbitrary 2D domains defined by level set functions
  - Dirichlet, Neumann, and Robin boundary conditions
  - 2nd-order and 4th-order finite difference schemes
  - Single-layer and double-layer potential formulations
  - Method of manufactured solutions for convergence testing
"""

from .solver import solve_helmholtz_2d
from .operator import get_operator, get_eigen
from .fft_solver import fft_solve, fst2d
from .lgf import evaluate_LGF, LGF_FFT
from .boundary import get_intersect_ptr, get_box3_index_v5, get_extrap
from .lagrange import get_lagrange_1d, get_lagrange_2d
from .potentials import get_sl, get_dl, get_dp_fft
from .geometry import get_geometry, get_unit_normal_star
from .manufactured import manufactured_solution, manufactured_gradient, manufactured_forcing

__version__ = "1.0.0"
