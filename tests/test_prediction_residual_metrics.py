import csv
from pathlib import Path

import pytest
import torch

import Baseline


def _batch_and_prediction() -> tuple[dict[str, torch.Tensor | float], torch.Tensor]:
    torch.manual_seed(2026)
    batch = Baseline.build_dataset(
        N=6,
        num_samples=2,
        device=torch.device("cpu"),
        k_range=(0.8, 1.4),
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

    rows, pred_flux, exact_flux = Baseline._prediction_metric_rows(batch, u_pred)

    expected_pred_flux = Baseline.nonlinear_poisson_residual(
        u_pred,
        batch["f"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["h"],  # type: ignore[arg-type]
    )
    expected_exact_flux = Baseline.nonlinear_poisson_residual(
        batch["u_exact"],  # type: ignore[arg-type]
        batch["f"],  # type: ignore[arg-type]
        batch["k"],  # type: ignore[arg-type]
        batch["h"],  # type: ignore[arg-type]
    )

    assert torch.allclose(pred_flux, expected_pred_flux)
    assert torch.allclose(exact_flux, expected_exact_flux)

    pred_flux_rmse = _per_sample_rmse(expected_pred_flux)
    exact_flux_rmse = _per_sample_rmse(expected_exact_flux)

    for sample_id, row in enumerate(rows):
        assert row["pred_flux_resRMSE"] == pytest.approx(float(pred_flux_rmse[sample_id]))
        assert row["exact_flux_resRMSE"] == pytest.approx(float(exact_flux_rmse[sample_id]))


def test_flux_residual_panel_label_has_no_removed_terms() -> None:
    label = Baseline._flux_residual_panel_label()

    assert label == "flux PDE residual"
    assert "selected" not in label.lower()


def test_prediction_output_pipeline_writes_requested_demo_columns(tmp_path: Path) -> None:
    batch, u_pred = _batch_and_prediction()
    Baseline.save_prediction_outputs(
        tmp_path / "prediction_outputs",
        batch,
        u_pred,
        output_type="plot",
    )

    with (tmp_path / "prediction_outputs" / "metrics.csv").open(newline="") as f:
        header = next(csv.reader(f))
    assert header == Baseline.PREDICTION_CSV_FIELDS
    assert "f_rmse" in header
    assert "exact_flux_resRMSE" in Baseline.EVALUATION_ONLY_METRIC_FIELDS
    assert (tmp_path / "prediction_outputs" / "sample_000_panel.png").exists()
    assert (tmp_path / "prediction_outputs" / "sample_000_f.csv").exists()


def test_prediction_panel_title_places_f_rmse_after_k() -> None:
    row = {
        "sample_id": 0,
        "k": 2.0,
        "f_rmse": 0.12345,
        "relL2": 0.01,
        "rmse": 0.02,
        "mae": 0.03,
        "linf": 0.04,
        "pred_flux_resRMSE": 0.05,
    }

    title = Baseline._prediction_panel_title(row)

    assert "k=2 f_rmse=1.235e-01" in title


def test_oracle_floor_flag_keeps_residual_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Baseline.sys, "argv", ["Baseline.py", "--check-oracle-floor"])
    assert Baseline.parse_args().check_oracle_floor is True

    monkeypatch.setattr(Baseline.sys, "argv", ["Baseline.py", "--check-residual"])
    assert Baseline.parse_args().check_oracle_floor is True
