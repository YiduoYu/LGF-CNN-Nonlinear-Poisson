"""Aggregation and failure reporting checks for the benchmark wrapper."""

from types import SimpleNamespace

import pytest

from axy_lgf import run_grid_scaling as scaling
from axy_lgf.reaction_manufactured import make_reaction_problem


def timed_records():
    return [dict(phase="timed", iterations=11, relative_residual=(i+1)*1e-11,
                 setup_time_ms=setup, solve_time_ms=solve,
                 total_time_ms=setup+solve, converged=True)
            for i, (setup, solve) in enumerate(zip([9, 1, 3, 5, 7], [1, 9, 4, 2, 6]))]


def test_warmup_excluded_and_median_totals_computed_per_run():
    records = timed_records()
    warmup = dict(records[0], phase="warmup", setup_time_ms=1000,
                  solve_time_ms=1000, total_time_ms=2000, iterations=100)
    row = scaling.summarize_case(31, "lgf", [warmup, *records])
    assert row["setup_time_ms"] == 5
    assert row["solve_time_ms"] == 4
    assert row["total_time_ms"] == 10  # Not the sum of marginal medians (9).
    assert row["iterations"] == 11
    assert row["relative_residual"] == pytest.approx(5e-11, rel=1e-15, abs=0)
    assert row["converged"] and row["successful_repetitions"] == 5


def test_failed_repetition_is_not_removed_from_statistics():
    records = timed_records()
    records[0].update(converged=False, setup_time_ms=None, solve_time_ms=None,
                      total_time_ms=None, iterations=None, relative_residual=None)
    row = scaling.summarize_case(31, "gmg", records)
    assert not row["converged"] and row["status"] == "failed"
    assert row["failed_repetitions"] == 1 and row["successful_repetitions"] == 4
    assert row["setup_time_ms"] is row["solve_time_ms"] is row["total_time_ms"] is None


def test_exception_and_false_success_are_explicit(monkeypatch):
    problem = make_reaction_problem(7)
    def broken(*args, **kwargs):
        raise RuntimeError("test breakdown")
    monkeypatch.setattr(scaling, "solve_axy_pcg", broken)
    record = scaling.run_once(problem, "lgf", "timed", 1)
    assert record["status"] == "error" and not record["converged"]
    assert "test breakdown" in record["message"] and record["total_time_ms"] is None

    def false_success(a, f, grid, **kwargs):
        assert a is problem.a and f is problem.f
        assert kwargs["q"] is problem.q and kwargs["a_nodes"] is problem.a_nodes
        assert kwargs["tol"] == 1e-10 and not kwargs["u0"].any()
        return SimpleNamespace(u=grid.zeros(), iterations=1, converged=True,
                               setup_time=0.001, solve_time=0.002)
    monkeypatch.setattr(scaling, "solve_axy_pcg", false_success)
    record = scaling.run_once(problem, "lgf", "timed", 2)
    assert record["status"] == "not_converged" and not record["converged"]
    assert record["relative_residual"] == 1
    assert record["total_time_ms"] == 3
