"""Optional native hypre tests, using the exact same flux system and PCG."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from axy_lgf import boomer_amg as boomer
from axy_lgf.operators import apply_reaction_diffusion, assemble_reaction_diffusion, l2_norm
from axy_lgf.reaction_manufactured import make_reaction_problem
from axy_lgf.solvers import solve_axy_pcg


@pytest.fixture
def native():
    try:
        metadata = boomer.boomer_metadata()
    except boomer.BoomerUnavailableError as exc:
        pytest.skip(str(exc))
    assert metadata["hypre_version"] == "3.1.0"
    assert metadata["settings"]["inner_tolerance"] == 0
    assert metadata["settings"]["inner_maxiter"] == 1


def test_missing_boomer_has_build_instructions(monkeypatch, tmp_path):
    boomer._library.cache_clear()
    monkeypatch.setattr(boomer, "BRIDGE", tmp_path / "missing.so")
    with pytest.raises(boomer.BoomerUnavailableError, match="python -m axy_lgf.build_boomer"):
        boomer._library()
    boomer._library.cache_clear()


@pytest.mark.parametrize("n", [7, 15])
def test_hypre_matrix_matches_common_operator(native, n):
    p = make_reaction_problem(n)
    matrix = assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=p.a_nodes)
    u = np.random.default_rng(314).normal(size=(n, n))
    expected = apply_reaction_diffusion(u, p.a, p.grid, p.q, a_nodes=p.a_nodes)
    with boomer.BoomerVcycle(matrix) as inverse:
        actual = inverse.apply_matrix(u)
    assert np.linalg.norm(actual-expected) / np.linalg.norm(expected) < 2e-15


def test_boomer_cycle_linear_symmetric_positive_definite(native):
    p = make_reaction_problem(7)
    matrix = assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=p.a_nodes)
    with boomer.BoomerVcycle(matrix) as inverse:
        dense = np.column_stack([inverse(column) for column in np.eye(49)])
        u, v = np.random.default_rng(42).normal(size=(2, 49))
        assert_allclose(inverse(2*u-3*v), 2*inverse(u)-3*inverse(v), atol=2e-15)
        # Check state independence after unrelated RHS and matrix applications.
        expected = inverse(u)
        inverse(v)
        inverse.apply_matrix(v)
        assert_allclose(inverse(u), expected, atol=1e-16)
    assert np.linalg.norm(dense-dense.T) / np.linalg.norm(dense) < 5e-13
    assert np.linalg.eigvalsh(dense).min() > 0
    with pytest.raises(RuntimeError, match="closed"):
        inverse(u)


@pytest.mark.parametrize("n", [31, 63, 127, 255])
def test_boomer_discrete_mms(native, n):
    p = make_reaction_problem(n, "discrete")
    result = boomer.solve_boomer_pcg(p.a, p.f, p.grid, q=p.q, a_nodes=p.a_nodes)
    residual = p.f - apply_reaction_diffusion(result.u, p.a, p.grid, p.q, a_nodes=p.a_nodes)
    assert result.converged
    assert l2_norm(residual, p.grid) / l2_norm(p.f, p.grid) <= 1e-10
    assert l2_norm(result.u-p.u_exact, p.grid) / l2_norm(p.u_exact, p.grid) <= 1e-10
    assert result.setup_time > 0 and result.solve_time > 0


@pytest.mark.parametrize("n", [31, 63])
def test_boomer_continuous_solution_agrees_with_lgf(native, n):
    p = make_reaction_problem(n)
    kwargs = dict(q=p.q, a_nodes=p.a_nodes, tol=1e-10)
    result = boomer.solve_boomer_pcg(p.a, p.f, p.grid, **kwargs)
    reference = solve_axy_pcg(p.a, p.f, p.grid, **kwargs)
    assert result.converged and reference.converged
    assert_allclose(result.u, reference.u, atol=1e-10)


def test_boomer_initial_guess_zero_rhs_and_nonconvergence(native):
    p = make_reaction_problem(15, "discrete")
    kwargs = dict(q=p.q, a_nodes=p.a_nodes)
    result = boomer.solve_boomer_pcg(p.a, p.f, p.grid, u0=p.u_exact, **kwargs)
    assert result.converged and result.iterations == 0
    result = boomer.solve_boomer_pcg(p.a, p.f, p.grid, maxiter=0, **kwargs)
    assert not result.converged and result.iterations == 0
    result = boomer.solve_boomer_pcg(p.a, p.grid.zeros(), p.grid, **kwargs)
    assert result.converged and result.iterations == 0


def test_boomer_q_zero(native):
    p = make_reaction_problem(15)
    f = apply_reaction_diffusion(p.u_exact, p.a, p.grid, 0.0, a_nodes=p.a_nodes)
    result = boomer.solve_boomer_pcg(p.a, f, p.grid, a_nodes=p.a_nodes)
    assert result.converged
    assert_allclose(result.u, p.u_exact, atol=1e-10)
