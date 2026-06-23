import argparse
from pathlib import Path

import torch

import Baseline


def _write_checkpoint(path: Path) -> None:
    model = Baseline.UNet2d(base_ch=1)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_stats": {
                "f_mean": 0.0,
                "f_std": 1.0,
                "k_mean": 1.5,
                "k_std": 0.5,
            },
            "args": {
                "N": 8,
                "base_ch": 1,
                "rhs_mode": "continuous",
                "coeff_mode": "pow",
                "discretization": "kirchhoff",
                "solution_mode": "base",
                "k_min": 0.5,
                "k_max": 2.5,
                "normalize_k": True,
                "positive_interior": True,
            },
        },
        path,
    )


def _prediction_args(
    checkpoint_path: Path,
    save_path: Path,
    compare_transform_solver: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        load_path=str(checkpoint_path),
        predict_manufactured=True,
        predict_samples=1,
        predict_save=str(save_path),
        compare_transform_solver=compare_transform_solver,
        cg_tol=None,
        cg_max_iter=None,
        device="cpu",
        seed=1234,
        check_residual=False,
        train_after_check=False,
        grid_convergence=False,
        grid_Ns=[8],
        rhs_mode="continuous",
        coeff_mode="pow",
        solution_mode="base",
        k_min=0.5,
        k_max=2.5,
        N=8,
    )


def test_predict_save_without_compare_does_not_call_transform_solver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_path = tmp_path / "prediction.pt"
    _write_checkpoint(checkpoint_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("solve_transformed_poisson should not be called")

    monkeypatch.setattr(Baseline, "solve_transformed_poisson", fail_if_called)

    Baseline.run_prediction(_prediction_args(checkpoint_path, save_path))

    payload = torch.load(save_path, map_location="cpu")
    assert "u_pred" in payload
    assert "u_transform" not in payload
    assert set(payload["metrics"]) == {"cnn"}


def test_predict_save_with_compare_includes_transform_solver_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_path = tmp_path / "prediction.pt"
    _write_checkpoint(checkpoint_path)

    def fake_transform(f, g_map, k, mask, h, **kwargs):
        return torch.ones_like(g_map), Baseline.PoissonCGDiagnostics(
            iterations=1,
            final_max_relative_residual=0.0,
            converged=True,
        )

    monkeypatch.setattr(Baseline, "solve_transformed_poisson", fake_transform)

    Baseline.run_prediction(
        _prediction_args(
            checkpoint_path,
            save_path,
            compare_transform_solver=True,
        )
    )

    payload = torch.load(save_path, map_location="cpu")
    assert "u_pred" in payload
    assert "u_transform" in payload
    assert torch.equal(payload["u_transform"], torch.ones_like(payload["u_transform"]))
    assert set(payload["metrics"]) == {"cnn", "transform"}
