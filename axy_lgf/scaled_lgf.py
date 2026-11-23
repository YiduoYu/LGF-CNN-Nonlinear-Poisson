"""Opt-in symmetric LGF scaling for a(x,y)=a_scale*chi**x on (0,1)^2.

The target matrix and the common PCG loop are unchanged. This preconditioner
is specialized to the exponential coefficient family, not to general jumps.
"""

from time import perf_counter

import numpy as np

from .lgf_inverse import DirichletLGF
from .operators import apply_reaction_diffusion, reaction_field
from .reaction_manufactured import validate_exponential_parameters
from .solvers import _pcg, _validate_problem


class ExponentialScaledLGF:
    """Apply S^-1 (-Delta_h + sigma I)^-1 S^-1, S=diag(sqrt(a)).

    With the project's arithmetic face averages and sampled boundary a,
    S^-1 A S^-1 = c*Lx + Ly + mu*I + diag(q/a), where
    c=cosh(h*log(chi)/2), mu=2*c*(c-1)/h^2.
    The reference replaces c by 1 and q/a by its interior mean. Its inverse
    reuses the existing DST-I LGF and is fixed, linear and SPD.
    """

    def __init__(self, a, grid, *, chi, a_scale=1.0, q=0.0, a_nodes):
        validate_exponential_parameters(a_scale, chi)
        if grid.x_min != 0.0 or grid.x_max != 1.0:
            raise ValueError("exponential scaling requires the unit square")
        nodes = np.arange(grid.n + 2, dtype=np.float64) * grid.h
        expected = np.broadcast_to(a_scale * np.power(chi, nodes), (grid.n+2, grid.n+2))
        if a_nodes is None or a_nodes.shape != expected.shape or not np.allclose(
            a_nodes, expected, rtol=1e-13, atol=0.0
        ):
            raise ValueError("a_nodes must sample a_scale*chi**x including the boundary")
        # Reuse the operator's validation of interior and boundary consistency.
        apply_reaction_diffusion(grid.zeros(), a, grid, q, a_nodes=a_nodes)
        self.inverse_sqrt_a = 1.0 / np.sqrt(a)
        z = grid.h * np.log(chi) / 2.0
        self.c = float(np.cosh(z))
        # c-1 = 2*sinh(z/2)^2 avoids cancellation as chi tends to 1.
        self.mu = float(4.0 * self.c * np.sinh(z/2.0)**2 / grid.h**2)
        self.sigma = self.mu + float(np.mean(reaction_field(q, grid) / a))
        self.inverse = DirichletLGF(grid, a0=1.0, sigma=self.sigma)

    def __call__(self, rhs):
        return self.inverse_sqrt_a * self.inverse(self.inverse_sqrt_a * rhs)


def solve_exponential_scaled_pcg(
    a, f, grid, *, chi, a_scale=1.0, q=0.0, a_nodes,
    tol=1e-10, maxiter=500, u0=None,
):
    """Solve the original flux system using the scaled LGF reference.

    Returned a0=1 and q0=sigma describe the *scaled* reference operator.
    Setup includes all validation/scaling/reference construction. Solve time
    is measured by the unchanged common PCG, with the original true residual.
    """
    started = perf_counter()
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    inverse = ExponentialScaledLGF(a, grid, chi=chi, a_scale=a_scale, q=q, a_nodes=a_nodes)
    setup_time = perf_counter() - started
    action = lambda u: apply_reaction_diffusion(u, a, grid, q, a_nodes=a_nodes)
    result = _pcg(action, inverse, f, grid, tol=tol, maxiter=maxiter, u0=u0, a0=1.0)
    result.setup_time = setup_time
    result.q0 = inverse.sigma
    return result
