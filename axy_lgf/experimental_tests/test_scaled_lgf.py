"""Independent matrix, SPD, reference and MMS checks for opt-in scaling."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

from axy_lgf.lgf_inverse import DirichletLGF
from axy_lgf.operators import assemble_reaction_diffusion, apply_reaction_diffusion
from axy_lgf.reaction_manufactured import make_reaction_problem
from axy_lgf.scaled_lgf import ExponentialScaledLGF, solve_exponential_scaled_pcg


@pytest.mark.parametrize("chi", [1.0, 1e2, 1e7])
def test_scaled_matrix_identity_including_boundary_rows(chi):
    p = make_reaction_problem(7, coefficient="exponential", a_scale=1.7, chi=chi)
    b = ExponentialScaledLGF(p.a, p.grid, chi=chi, a_scale=1.7, q=p.q, a_nodes=p.a_nodes)
    a = assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=p.a_nodes)
    n, h = p.grid.n, p.grid.h
    ll = diags((-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)), (-1, 0, 1))/h**2
    sinv = diags(1/np.sqrt(p.a.ravel()))
    expected = b.c*kron(eye(n), ll) + kron(ll, eye(n)) + b.mu*eye(n*n) + diags((p.q/p.a).ravel())
    assert_allclose((sinv @ a @ sinv).toarray(), expected.toarray(), rtol=3e-15, atol=3e-13)


def test_preconditioner_is_symmetric_positive_definite_and_linear():
    p = make_reaction_problem(7, coefficient="exponential", chi=1e7)
    b = ExponentialScaledLGF(p.a, p.grid, chi=1e7, q=p.q, a_nodes=p.a_nodes)
    basis = np.eye(49)
    dense = np.column_stack([b(v.reshape(7, 7)).ravel() for v in basis])
    assert_allclose(dense, dense.T, rtol=2e-13, atol=1e-17)
    assert np.linalg.eigvalsh(dense)[0] > 0
    rng = np.random.default_rng(27)
    x, y = rng.standard_normal((2, 7, 7))
    assert_allclose(b(2*x-3*y), 2*b(x)-3*b(y), rtol=2e-13, atol=1e-17)


def test_constant_diffusion_reduces_to_existing_screened_lgf():
    p = make_reaction_problem(15, coefficient="exponential", a_scale=2.3, chi=1)
    rhs = np.random.default_rng(1).standard_normal(p.a.shape)
    b = ExponentialScaledLGF(p.a, p.grid, chi=1, a_scale=2.3, q=0.4, a_nodes=p.a_nodes)
    assert_allclose(b(rhs), DirichletLGF(p.grid, a0=2.3, sigma=0.4)(rhs), rtol=3e-13, atol=1e-17)


@pytest.mark.parametrize("chi", [1e2, 1e7])
@pytest.mark.parametrize("q", [0.0, 1.1])
def test_independent_discrete_reference_and_true_residual(chi, q):
    p = make_reaction_problem(31, coefficient="exponential", chi=chi)
    x, y = p.grid.mesh()
    reference = np.sin(2*np.pi*x)*np.sin(3*np.pi*y) + 0.3*np.sin(5*np.pi*x)*np.sin(np.pi*y)
    rhs = apply_reaction_diffusion(reference, p.a, p.grid, q, a_nodes=p.a_nodes)
    result = solve_exponential_scaled_pcg(p.a, rhs, p.grid, chi=chi, q=q, a_nodes=p.a_nodes)
    residual = rhs-apply_reaction_diffusion(result.u, p.a, p.grid, q, a_nodes=p.a_nodes)
    assert result.converged
    assert np.linalg.norm(residual)/np.linalg.norm(rhs) <= 1e-10
    assert np.linalg.norm(result.u-reference)/np.linalg.norm(reference) <= 1e-8


def test_random_rhs_agrees_with_independent_sparse_direct_solution():
    p = make_reaction_problem(31, coefficient="exponential", chi=1e7)
    rhs = np.random.default_rng(207).standard_normal(p.a.shape)
    a = assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=p.a_nodes)
    reference = spsolve(a, rhs.ravel()).reshape(rhs.shape)
    result = solve_exponential_scaled_pcg(p.a, rhs, p.grid, chi=1e7, q=p.q, a_nodes=p.a_nodes)
    assert result.converged
    assert np.linalg.norm(rhs.ravel()-a@result.u.ravel())/np.linalg.norm(rhs) <= 1e-10
    assert np.linalg.norm(result.u-reference)/np.linalg.norm(reference) <= 1e-8


def test_continuous_mms_second_order_at_fixed_high_contrast():
    errors = []
    for n in (15, 31, 63, 127):
        p = make_reaction_problem(n, coefficient="exponential", chi=1e7)
        result = solve_exponential_scaled_pcg(p.a, p.f, p.grid, chi=1e7, q=p.q, a_nodes=p.a_nodes)
        assert result.converged
        errors.append(np.linalg.norm(result.u-p.u_exact)/np.linalg.norm(p.u_exact))
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert orders[-1] > 1.9


def test_refuse_coefficients_outside_specialized_family():
    p = make_reaction_problem(7)
    with pytest.raises(ValueError, match="a_nodes"):
        ExponentialScaledLGF(p.a, p.grid, chi=10, q=p.q, a_nodes=p.a_nodes)
