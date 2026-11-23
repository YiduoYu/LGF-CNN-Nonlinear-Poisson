"""Coefficient/MMS verification at chi=1 or 2, without a contrast sweep."""

import json
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from axy_lgf import run_grid_scaling as scaling
from axy_lgf.multigrid import solve_multigrid_pcg
from axy_lgf.operators import (
    apply_reaction_diffusion, assemble_reaction_diffusion, face_diffusivities, l2_norm,
)
from axy_lgf.reaction_manufactured import (
    exponential_continuous_fields, make_reaction_problem,
)
from axy_lgf.solvers import solve_axy_pcg


def test_endpoint_values_stratification_and_sampled_contrast():
    p = make_reaction_problem(15, coefficient="exponential", a_scale=2.0, chi=2.0)
    assert_allclose(p.a_nodes[:, 0], 2.0)
    assert_allclose(p.a_nodes[:, -1], 4.0)
    assert_array_equal(p.a_nodes[1:-1, 1:-1], p.a)
    assert_array_equal(p.a, np.broadcast_to(p.a[0], p.a.shape))
    assert p.a_nodes.max() / p.a_nodes.min() == 2.0
    assert_allclose(p.a.max() / p.a.min(), 2.0**(1-2*p.grid.h))
    ax, ay = face_diffusivities(p.a, p.grid, a_nodes=p.a_nodes)
    assert_allclose(ax[:, 0], 0.5 * (2.0 + 2.0*2.0**p.grid.h))
    assert_allclose(ax[:, -1], 0.5 * (4.0 + 2.0*2.0**(1-p.grid.h)))
    assert_array_equal(ay[0], p.a[0])
    old = make_reaction_problem(15)
    assert_array_equal(p.q, old.q)
    assert_array_equal(p.u_exact, old.u_exact)


def test_chi_one_constant_diffusion_forcing():
    p = make_reaction_problem(7, coefficient="exponential", a_scale=1.7, chi=1.0)
    assert_array_equal(p.a, np.full((7, 7), 1.7))
    assert_allclose(p.f, (2*np.pi**2*1.7+p.q)*p.u_exact, rtol=2e-15)


def test_analytic_forcing_at_independent_hand_computable_points():
    x, y = np.array([0.5, 0.25]), np.array([0.5, 0.5])
    _, _, _, f = exponential_continuous_fields(x, y, a_scale=1.3, chi=2.0)
    a_mid, a_quarter = 1.3*np.sqrt(2.0), 1.3*2.0**0.25
    expected = [2*np.pi**2*a_mid+1.2,
                np.sqrt(0.5)*(2*np.pi**2*a_quarter+1-np.pi*np.log(2.0)*a_quarter)]
    assert_allclose(f, expected, rtol=2e-15)


def test_discrete_forcing_remains_separate_from_continuous_forcing():
    p = make_reaction_problem(15, coefficient="exponential", chi=2.0)
    d = make_reaction_problem(15, "discrete", coefficient="exponential", chi=2.0)
    assert_array_equal(d.f, apply_reaction_diffusion(d.u_exact, d.a, d.grid, d.q, a_nodes=d.a_nodes))
    assert np.linalg.norm(p.f-d.f) > 1e-3


def test_exponential_requires_explicit_chi():
    with pytest.raises(ValueError, match="explicit chi"):
        make_reaction_problem(7, coefficient="exponential")


def test_normal_default_preserved_and_extra_parameters_not_ignored():
    p = make_reaction_problem(7)
    x, y = p.grid.mesh()
    assert_array_equal(p.a, 1.0+0.2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))
    assert p.coefficient == "normal" and p.chi is None
    with pytest.raises(ValueError, match="only to the exponential"):
        make_reaction_problem(7, chi=2.0)


@pytest.mark.parametrize("a_scale,chi", [
    (0, 2), (-1, 2), (np.nan, 2), (np.inf, 2),
    (1, 0.5), (1, 0), (1, -2), (1, np.nan), (1, np.inf), (np.finfo(float).max, 2),
])
def test_invalid_exponential_parameters(a_scale, chi):
    with pytest.raises(ValueError):
        make_reaction_problem(7, coefficient="exponential", a_scale=a_scale, chi=chi)


@pytest.mark.parametrize("method", ["lgf", "gmg", "amg", "boomer"])
def test_discrete_mms_with_each_existing_solver(method):
    p = make_reaction_problem(15, "discrete", coefficient="exponential", chi=2.0)
    kwargs = dict(q=p.q, a_nodes=p.a_nodes)
    if method == "lgf":
        result = solve_axy_pcg(p.a, p.f, p.grid, **kwargs)
    elif method == "boomer":
        from axy_lgf.boomer_amg import BoomerUnavailableError, solve_boomer_pcg
        try:
            result = solve_boomer_pcg(p.a, p.f, p.grid, **kwargs)
        except BoomerUnavailableError as exc:
            pytest.skip(str(exc))
    else:
        if method == "amg":
            pytest.importorskip("pyamg")
        result = solve_multigrid_pcg(p.a, p.f, p.grid, method=method, **kwargs)
    residual = p.f-apply_reaction_diffusion(result.u, p.a, p.grid, p.q, a_nodes=p.a_nodes)
    assert result.converged and l2_norm(residual, p.grid)/l2_norm(p.f, p.grid) <= 1e-10
    assert l2_norm(result.u-p.u_exact, p.grid)/l2_norm(p.u_exact, p.grid) <= 1e-10


def test_continuous_mms_second_order_at_chi_two():
    errors = []
    for n in (15, 31, 63, 127):
        p = make_reaction_problem(n, coefficient="exponential", chi=2.0)
        result = solve_axy_pcg(p.a, p.f, p.grid, q=p.q, a_nodes=p.a_nodes)
        assert result.converged
        errors.append(l2_norm(result.u-p.u_exact, p.grid))
    rates = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all((rates > 1.9) & (rates < 2.1)), rates


def test_sparse_matrix_action_and_spd_for_stratification():
    p = make_reaction_problem(7, coefficient="exponential", a_scale=0.7, chi=2.0)
    matrix = assemble_reaction_diffusion(p.a, p.grid, p.q, a_nodes=p.a_nodes)
    u = np.random.default_rng(71).normal(size=(7, 7))
    assert_allclose((matrix@u.ravel()).reshape(u.shape),
                    apply_reaction_diffusion(u, p.a, p.grid, p.q, a_nodes=p.a_nodes), atol=3e-13)
    assert_array_equal(matrix.toarray(), matrix.toarray().T)
    assert np.linalg.eigvalsh(matrix.toarray()).min() > 0


def test_single_mms_cli_records_coefficient_parameters(monkeypatch, tmp_path):
    from axy_lgf.run_reaction_diffusion import main
    monkeypatch.setattr(sys, "argv", ["run_reaction_diffusion", "--coefficient", "exponential",
                        "--chi", "2", "--n", "7", "--solvers", "lgf", "--no-plot",
                        "--output-dir", str(tmp_path)])
    main()
    saved = json.loads(next(tmp_path.rglob("results.json")).read_text())
    assert saved["coefficient"] == {"family": "exponential", "formula": "a_scale*chi**x",
                                    "a_scale": 1.0, "chi": 2.0}
    row = saved["results"][0]
    assert row["converged"] and row["a_nodes_contrast"] == 2.0
    assert row["a_contrast"] < row["chi"]


def test_grid_scaling_routes_selected_coefficient_without_benchmarking(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(scaling, "SIZES", (7,))
    monkeypatch.setattr(scaling, "METHODS", ("lgf",))
    monkeypatch.setattr(scaling, "run_preflight", lambda output: {"successful": True})
    monkeypatch.setattr(scaling, "save_plots", lambda *args, **kwargs: None)
    def fake_run(problem, method, phase, repetition):
        assert problem.coefficient == "exponential"
        assert_allclose(problem.a_nodes[:, -1], 3.0)
        observed.append(phase)
        return dict(phase=phase, repetition=repetition, converged=True, iterations=1,
                    relative_residual=1e-12, setup_time_ms=1, solve_time_ms=2, total_time_ms=3,
                    coefficient=problem.coefficient, a_scale=problem.a_scale, chi=problem.chi,
                    a_contrast=float(problem.a.max()/problem.a.min()), a_nodes_contrast=2.0)
    monkeypatch.setattr(scaling, "run_once", fake_run)
    output = tmp_path / "scaling"
    monkeypatch.setattr(sys, "argv", ["run_grid_scaling", "--coefficient", "exponential",
                        "--chi", "2", "--a-scale", "1.5", "--output-dir", str(output)])
    scaling.main()
    saved = json.loads((output / "grid_scaling_summary.json").read_text())
    assert saved["coefficient"]["chi"] == 2 and saved["coefficient"]["a_scale"] == 1.5
    assert saved["grids"][0]["a_nodes_contrast"] == 2.0
    assert saved["results"][0]["chi"] == 2.0
    assert observed == ["warmup"] + ["timed"]*5


@pytest.mark.parametrize("module", ["run_reaction_diffusion", "run_grid_scaling"])
def test_cli_requires_chi_before_creating_output(module, monkeypatch, tmp_path):
    from importlib import import_module
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(sys, "argv", [module, "--coefficient", "exponential", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as exc:
        import_module("axy_lgf."+module).main()
    assert exc.value.code == 2
    assert not output.exists()
