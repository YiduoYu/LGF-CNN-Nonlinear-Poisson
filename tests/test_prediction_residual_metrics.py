import csv
from pathlib import Path

import pytest
import torch

import Baseline


REQUIRED_PREDICTION_COLUMNS = {
    "sample_id",
    "k",
    "relL2",
    "rmse",
    "mae",
    "linf",
    "boundary_max",
    "pred_flux_resRMSE",
    "exact_flux_resRMSE",
}

REMOVED_PREDICTION_COLUMNS = {
    "selected_discretization",
    "pred_selected_resRMSE",
    "exact_selected_resRMSE",
    "pred_kirchhoff_resRMSE",
    "exact_kirchhoff_resRMSE",
}


def _batch_and_prediction() -> tuple[dict[str, torch.Tensor | float], torch.Tensor]:
    torch.manual_seed(2026)
    batch = Baseline.build_dataset(
        N=6,
        num_samples=2,
        device=torch.device("cpu"),
        k_range=(0.8, 1.4),
        rhs_mode="discrete",
        coeff_mode="pow",
        solution_mode="base",
    )
    x = batch["x"].expand_as(batch["u_exact"])  # type: ignore[union-attr]
    y = batch["y"].expand_as(batch["u_exact"])  # type: ignore[union-attr]
    interior = batch["interior_mask"].expand_as(batch["u_exact"])  # type: ignore[union-attr]
    perturbation = 0.03 * interior * (1.0 - x * x) * (1.0 - y * y) * torch.sin(x + 2.0 * y)
    return batch, batch["u_exact"] + perturbation  # type: ignore[operator]


def _per_sample_rmse(values: torch.Tensor) -> torch.Tensor:
    return values.flatten(start_dim=1).pow(2).mean(dim=1).sqrt()


def test_prediction_flux_residual_columns_match_flux_operator() -> None:
    batch, u_pred = _batch_and_prediction()

    rows, pred_flux, exact_flux = Baseline._prediction_metric_rows(batch, u_pred, "pow")

    expected_pred_flux = Baseline.nonlinear_poisson_residual(
        u_pred,
        batch["f"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["h"],  # type: ignore[arg-type]
        coeff_mode="pow",
    )
    expected_exact_flux = Baseline.nonlinear_poisson_residual(
        batch["u_exact"],  # type: ignore[arg-type]
        batch["f"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["h"],  # type: ignore[arg-type]
        coeff_mode="pow",
    )

    assert torch.allclose(pred_flux, expected_pred_flux)
    assert torch.allclose(exact_flux, expected_exact_flux)

    pred_flux_rmse = _per_sample_rmse(expected_pred_flux)
    exact_flux_rmse = _per_sample_rmse(expected_exact_flux)

    for sample_id, row in enumerate(rows):
        assert set(row).isdisjoint(REMOVED_PREDICTION_COLUMNS)
        assert row["pred_flux_resRMSE"] == pytest.approx(float(pred_flux_rmse[sample_id]))
        assert row["exact_flux_resRMSE"] == pytest.approx(float(exact_flux_rmse[sample_id]))


def test_flux_residual_panel_label_has_no_removed_terms() -> None:
    label = Baseline._flux_residual_panel_label()

    assert label == "flux PDE residual"
    assert "selected" not in label.lower()
    assert "kirchhoff" not in label.lower()


def test_prediction_csv_contains_only_flux_residual_columns(tmp_path: Path) -> None:
    batch, u_pred = _batch_and_prediction()
    rows, *_ = Baseline._prediction_metric_rows(batch, u_pred, "pow")
    path = tmp_path / "metrics.csv"

    Baseline._write_prediction_plot_csv(rows, path)

    with path.open(newline="") as f:
        header = set(next(csv.reader(f)))
    assert REQUIRED_PREDICTION_COLUMNS.issubset(header)
    assert header.isdisjoint(REMOVED_PREDICTION_COLUMNS)
