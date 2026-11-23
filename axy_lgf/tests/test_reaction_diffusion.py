import builtins
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse.linalg import spsolve

from axy_lgf.grid import SquareGrid
from axy_lgf.lgf_inverse import DirichletLGF
from axy_lgf.manufactured import make_problem
from axy_lgf.multigrid import (
    AMGUnavailableError, GeometricVcycle, amg_preconditioner,
    geometric_interpolation, solve_multigrid_pcg,
)
from axy_lgf.operators import (
    apply_div_a_grad, apply_minus_laplacian, apply_reaction_diffusion,
    assemble_reaction_diffusion, face_diffusivities, l2_norm,
)
from axy_lgf.reaction_manufactured import continuous_fields, make_reaction_problem
from axy_lgf.run_reaction_diffusion import run_experiment, save_results
from axy_lgf.solvers import solve_axy_lgf_bank, solve_axy_pcg, solve_axy_richardson


def legacy_operator(u, a, grid):
    """Independent frozen copy of the original stencil, for regression."""
    padded = np.pad(u, 1)
    ax = np.concatenate([a[:, :1], (a[:, :-1]+a[:, 1:])/2, a[:, -1:]], axis=1)
    ay = np.concatenate([a[:1], (a[:-1]+a[1:])/2, a[-1:]], axis=0)
    center = padded[1:-1, 1:-1]
    left = ax[:, :-1] * ((center-padded[1:-1, :-2])/grid.h)
    right = ax[:, 1:] * ((padded[1:-1, 2:]-center)/grid.h)
    down = ay[:-1] * ((center-padded[:-2, 1:-1])/grid.h)
    up = ay[1:] * ((padded[2:, 1:-1]-center)/grid.h)
    return -((right-left)+(up-down))/grid.h


def solve(problem, method, **kwargs):
    if method == "amg":
        pytest.importorskip("pyamg")
    common = dict(q=problem.q, a_nodes=problem.a_nodes, **kwargs)
    if method == "lgf":
        return solve_axy_pcg(problem.a, problem.f, problem.grid, **common)
    return solve_multigrid_pcg(problem.a, problem.f, problem.grid,
                              method=method, **common)


def test_q_zero_preserves_original_operator_exactly():
    grid = SquareGrid(9)
    rng = np.random.default_rng(7)
    a = 0.8+0.4*rng.random((9, 9))
    u = rng.normal(size=a.shape)
    expected = legacy_operator(u, a, grid)
    assert_array_equal(apply_div_a_grad(u, a, grid), expected)
    assert_array_equal(apply_reaction_diffusion(u, a, grid, 0), expected)


def test_reaction_sign_linearity_and_boundary_faces():
    p = make_reaction_problem(7)
    rng = np.random.default_rng(23)
    u, v = rng.normal(size=(2, 7, 7))
    def action(w):
        return apply_reaction_diffusion(w, p.a, p.grid, p.q, a_nodes=p.a_nodes)
    diffusion = apply_div_a_grad(u, p.a, p.grid, a_nodes=p.a_nodes)
    assert_allclose(action(u)-diffusion, p.q*u, atol=2e-13)
    assert_allclose(action(2*u-3*v), 2*action(u)-3*action(v), atol=1e-12)
    ax, ay = face_diffusivities(p.a, p.grid, a_nodes=p.a_nodes)
    assert_allclose(ax[:, 0], (p.a_nodes[1:-1, 0]+p.a[:, 0])/2)
    assert_allclose(ax[:, -1], (p.a_nodes[1:-1, -1]+p.a[:, -1])/2)
    assert_allclose(ay[0], (p.a_nodes[0, 1:-1]+p.a[0])/2)
    assert_allclose(ay[-1], (p.a_nodes[-1, 1:-1]+p.a[-1])/2)
    assert np.max(abs(ax[:, 0]-p.a[:, 0])) > 0.01


@pytest.mark.parametrize("with_boundary", [False, True])
@pytest.mark.parametrize("q", [0.0, 1.3, "variable"])
def test_sparse_action_symmetry_spd_and_energy(with_boundary, q):
    p = make_reaction_problem(5)
    q = p.q if isinstance(q, str) else q
    nodes = p.a_nodes if with_boundary else None
    matrix = assemble_reaction_diffusion(p.a, p.grid, q, a_nodes=nodes)
    u = np.random.default_rng(3).normal(size=p.a.shape)
    actual = apply_reaction_diffusion(u, p.a, p.grid, q, a_nodes=nodes)
    assert_allclose((matrix @ u.ravel()).reshape(u.shape), actual, atol=2e-13)
    dense = matrix.toarray()
    assert_array_equal(dense, dense.T)
    assert np.linalg.eigvalsh(dense).min() > 0
    ax, ay = face_diffusivities(p.a, p.grid, a_nodes=nodes)
    padded = np.pad(u, 1)
    dx = np.diff(padded[1:-1, :], axis=1)
    dy = np.diff(padded[:, 1:-1], axis=0)
    energy = (np.sum(ax*dx**2)+np.sum(ay*dy**2))/p.grid.h**2 + np.sum(q*u**2)
    assert_allclose(u.ravel() @ matrix @ u.ravel(), energy, rtol=1e-14)


@pytest.mark.parametrize("q", [-0.1, np.nan, np.inf, np.ones((3, 4))])
def test_invalid_reaction_rejected(q):
    p = make_reaction_problem(7)
    with pytest.raises(ValueError, match="q must"):
        solve_axy_pcg(p.a, p.f, p.grid, q=q)


def test_invalid_diffusion_and_inconsistent_boundary_rejected():
    p = make_reaction_problem(7)
    invalid = p.a_nodes.copy()
    invalid[0, 2] = -1
    with pytest.raises(ValueError, match="positive"):
        apply_reaction_diffusion(p.u_exact, p.a, p.grid, p.q, a_nodes=invalid)
    invalid = p.a_nodes.copy()
    invalid[2, 2] += 0.01
    with pytest.raises(ValueError, match="interior must equal"):
        assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=invalid)


@pytest.mark.parametrize("sigma", [0.0, 1.2])
def test_screened_lgf_is_exact_for_reference_operator(sigma):
    grid = SquareGrid(15, 0, 1)
    u = np.random.default_rng(4).normal(size=(15, 15))
    rhs = 1.17*apply_minus_laplacian(u, grid) + sigma*u
    assert_allclose(DirichletLGF(grid, a0=1.17, sigma=sigma)(rhs), u, atol=7e-15)


def test_continuous_forcing_analytic_values_and_distinct_discrete_forcing():
    x = np.array([0.5, 0.25])
    _, _, _, f = continuous_fields(x, x)
    assert_allclose(f, [2*np.pi**2+1.2, 1.2*np.pi**2+0.5], rtol=2e-15)
    p, discrete = make_reaction_problem(15), make_reaction_problem(15, "discrete")
    assert np.linalg.norm(p.f-discrete.f) > 0.01
    assert p.grid.x_min == 0 and p.grid.x_max == 1
    assert p.a.dtype == p.q.dtype == p.f.dtype == np.float64
    assert_allclose([p.a.min(), p.a.max()], [0.8, 1.2])
    assert p.q.min() >= 0.8 and p.q.max() <= 1.2


def test_geometric_hierarchy_and_interpolation():
    p = make_reaction_problem(15)
    aa = assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=p.a_nodes)
    mg = GeometricVcycle(aa, 15)
    assert mg.sizes == [15, 7, 3]
    pp = geometric_interpolation(15)
    coarse = np.random.default_rng(42).normal(size=(7, 7))
    fine = (pp @ coarse.ravel()).reshape(15, 15)
    assert_allclose(fine[1::2, 1::2], coarse)
    assert_allclose(fine[0, 1::2], 0.5*coarse[0])
    for i, transfer in enumerate(mg.transfers):
        expected = transfer.T @ mg.matrices[i] @ transfer
        assert_allclose(mg.matrices[i+1].toarray(), expected.toarray())
    with pytest.raises(ValueError, match=r"2\^k-1"):
        GeometricVcycle(aa, 14)


@pytest.mark.parametrize("method", ["gmg", "amg"])
def test_vcycle_preconditioner_is_linear_symmetric_positive_definite(method):
    p = make_reaction_problem(7)
    aa = assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=p.a_nodes)
    if method == "amg":
        pytest.importorskip("pyamg")
        preconditioner = amg_preconditioner(aa)
        action = lambda x: preconditioner @ x
    else:
        mg = GeometricVcycle(aa, 7)
        action = lambda x: mg(x.reshape(7, 7)).ravel()
    eye = np.eye(49)
    dense = np.column_stack([action(column) for column in eye])
    assert_allclose(dense, dense.T, atol=2e-17, rtol=2e-13)
    assert np.linalg.eigvalsh(dense).min() > 0
    u, v = np.random.default_rng(8).normal(size=(2, 49))
    assert_allclose(action(2*u-3*v), 2*action(u)-3*action(v), atol=1e-15)


@pytest.mark.parametrize("n", [15, 31, 63, 127])
@pytest.mark.parametrize("method", ["lgf", "gmg", "amg"])
def test_discrete_mms_solver_accuracy(n, method):
    p = make_reaction_problem(n, "discrete")
    result = solve(p, method, tol=1e-10)
    residual = p.f - apply_reaction_diffusion(result.u, p.a, p.grid, p.q, a_nodes=p.a_nodes)
    relres = l2_norm(residual, p.grid) / l2_norm(p.f, p.grid)
    assert result.converged and relres <= 1e-10
    assert l2_norm(result.u-p.u_exact, p.grid)/l2_norm(p.u_exact, p.grid) <= 1e-10
    assert result.residual_history[-1] == relres
    assert result.setup_time >= 0 and result.solve_time > 0


@pytest.mark.parametrize("method", ["lgf", "gmg", "amg"])
def test_continuous_mms_second_order(method):
    errors = []
    for n in (15, 31, 63, 127):
        p = make_reaction_problem(n)
        result = solve(p, method, tol=1e-10)
        assert result.converged
        errors.append(l2_norm(result.u-p.u_exact, p.grid))
    rates = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all((rates > 1.9) & (rates < 2.1)), rates


def test_boundary_convention_second_order_for_nonconstant_boundary_a():
    # Independent polynomial MMS with nonconstant a on every boundary.
    errors = []
    for n in (15, 31, 63):
        grid = SquareGrid(n, 0, 1)
        x, y = grid.mesh()
        xx, yy = np.meshgrid(np.arange(n+2)*grid.h, np.arange(n+2)*grid.h)
        a, nodes = 1+0.2*x**2+0.1*y**2, 1+0.2*xx**2+0.1*yy**2
        u = x*(1-x)*y*(1-y)
        f = 2*a*(x*(1-x)+y*(1-y)) - 0.4*x*(1-2*x)*y*(1-y) - 0.2*y*x*(1-x)*(1-2*y) + u
        result = solve_axy_pcg(a, f, grid, q=1, a_nodes=nodes, tol=1e-11)
        assert result.converged
        errors.append(l2_norm(result.u-u, grid))
    # Quadratic a makes the nodal face average differ from its midpoint
    # value by O(h^2), including boundary-adjacent rows.
    assert np.all(np.log2(np.array(errors[:-1])/errors[1:]) > 1.95)


@pytest.mark.parametrize("method", ["lgf", "gmg", "amg"])
def test_common_initial_guess_zero_rhs_and_failure_status(method):
    p = make_reaction_problem(15, "discrete")
    result = solve(p, method, tol=1e-10, u0=p.u_exact.copy())
    assert result.converged and result.iterations == 0
    result = solve(p, method, tol=1e-10, maxiter=0)
    assert not result.converged and result.iterations == 0
    assert result.residual_history[-1] == 1
    result = solve(p, method, tol=1e-10, u0=0.1*p.u_exact)
    assert result.converged
    p.f = p.grid.zeros()
    result = solve(p, method, tol=1e-10)
    assert result.converged and result.iterations == 0
    assert_array_equal(result.u, p.f)


@pytest.mark.parametrize("fn", [solve_axy_pcg, solve_axy_richardson, solve_axy_lgf_bank])
def test_legacy_solvers_and_extended_fixed_point_solvers(fn):
    old = make_problem("mild_sine", n=15, forcing="discrete")
    result = fn(old.a, old.f, old.grid, tol=1e-10)
    assert result.converged
    assert_allclose(result.u, old.u_exact, atol=1e-10)
    p = make_reaction_problem(15, "discrete")
    result = fn(p.a, p.f, p.grid, q=p.q, a_nodes=p.a_nodes, tol=1e-10)
    assert result.converged
    assert_allclose(result.u, p.u_exact, atol=1e-10)


def test_small_system_agrees_with_independent_direct_solve():
    p = make_reaction_problem(7)
    aa = assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=p.a_nodes)
    reference = spsolve(aa, p.f.ravel()).reshape(7, 7)
    result = solve(p, "lgf", tol=1e-12)
    assert_allclose(result.u, reference, atol=2e-13)


def test_missing_amg_is_explicit_and_saved_results_never_overwritten(monkeypatch, tmp_path):
    original_import = builtins.__import__
    def without_pyamg(name, *args, **kwargs):
        if name == "pyamg":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_pyamg)
    rows = run_experiment([7], ["lgf", "amg"], maxiter=0)
    assert rows[0]["status"] == "not_converged"
    assert rows[1]["status"] == "unavailable" and not rows[1]["converged"]
    assert "python -m pip install pyamg" in rows[1]["message"]
    first = save_results(rows, tmp_path, plot=False)
    snapshot = (first / "results.json").read_bytes()
    second = save_results(rows, tmp_path, plot=False)
    assert first != second and (first / "results.json").read_bytes() == snapshot
    saved = json.loads(snapshot)
    assert saved["status"] == "incomplete"
    assert saved["results"][0]["observed_order"] is None


def test_cli_returns_failure_and_saves_unconverged_rows(monkeypatch, tmp_path):
    from axy_lgf.run_reaction_diffusion import main
    monkeypatch.setattr(sys, "argv", ["run_reaction_diffusion", "--n", "7",
                        "--solvers", "lgf", "--maxiter", "0", "--no-plot",
                        "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    saved = json.loads(next(tmp_path.rglob("results.json")).read_text())
    assert saved["results"][0]["status"] == "not_converged"


def test_classical_import_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import axy_lgf; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
