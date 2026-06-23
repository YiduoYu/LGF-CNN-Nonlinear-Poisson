import argparse
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import Baseline


def _runtime_args(**overrides) -> argparse.Namespace:
    values = {
        "N": 99,
        "rhs_mode": "continuous",
        "solution_mode": "base",
        "k_min": 10.0,
        "k_max": 20.0,
        "predict_N": None,
        "predict_rhs_mode": None,
        "predict_solution_mode": None,
        "predict_k_min": None,
        "predict_k_max": None,
        "predict_samples": 2,
        "test_seed": 777,
        "seed": 1234,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _checkpoint_args(**overrides) -> argparse.Namespace:
    values = {
        "N": 12,
        "rhs_mode": "discrete",
        "solution_mode": "mixed",
        "k_min": 0.2,
        "k_max": 0.8,
        "base_ch": 5,
        "coeff_mode": "abs_pow",
        "normalize_k": False,
        "positive_interior": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_checkpoint(path: Path, checkpoint_args: dict[str, object]) -> None:
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
            "args": checkpoint_args,
        },
        path,
    )


def _run_prediction_payload(
    tmp_path: Path,
    test_seed: int,
    checkpoint_args: dict[str, object] | None = None,
) -> dict[str, object]:
    checkpoint_path = tmp_path / f"checkpoint_{test_seed}.pt"
    save_path = tmp_path / f"prediction_{test_seed}.pt"
    args = {
        "N": 8,
        "base_ch": 1,
        "rhs_mode": "continuous",
        "coeff_mode": "pow",
        "solution_mode": "base",
        "k_min": 0.5,
        "k_max": 2.5,
        "normalize_k": True,
    }
    if checkpoint_args is not None:
        args = checkpoint_args
    _write_checkpoint(checkpoint_path, args)

    runtime_args = argparse.Namespace(
        load_path=str(checkpoint_path),
        predict_manufactured=True,
        predict_samples=2,
        predict_save=str(save_path),
        predict_N=None,
        predict_rhs_mode=None,
        predict_solution_mode=None,
        predict_k_min=None,
        predict_k_max=None,
        test_seed=test_seed,
        compare_transform_solver=False,
        cg_tol=None,
        cg_max_iter=None,
        device="cpu",
        seed=1234,
    )
    Baseline.run_prediction(runtime_args)
    return torch.load(save_path, map_location="cpu")


def test_prediction_config_uses_checkpoint_values_without_overrides() -> None:
    config = Baseline.resolve_prediction_config(_runtime_args(), _checkpoint_args())

    assert config == Baseline.PredictionConfig(
        N=12,
        rhs_mode="discrete",
        solution_mode="mixed",
        k_min=0.2,
        k_max=0.8,
        test_seed=777,
        samples=2,
    )


@pytest.mark.parametrize(
    ("override_name", "override_value", "field_name", "expected"),
    [
        ("predict_N", 16, "N", 16),
        ("predict_rhs_mode", "continuous", "rhs_mode", "continuous"),
        ("predict_solution_mode", "gaussian", "solution_mode", "gaussian"),
        ("predict_k_min", 1.1, "k_min", 1.1),
        ("predict_k_max", 3.3, "k_max", 3.3),
    ],
)
def test_prediction_overrides_change_only_intended_field(
    override_name: str,
    override_value: object,
    field_name: str,
    expected: object,
) -> None:
    baseline = Baseline.resolve_prediction_config(_runtime_args(), _checkpoint_args())
    overridden = Baseline.resolve_prediction_config(
        _runtime_args(**{override_name: override_value}),
        _checkpoint_args(),
    )

    baseline_dict = asdict(baseline)
    overridden_dict = asdict(overridden)
    assert overridden_dict[field_name] == expected
    for name, value in baseline_dict.items():
        if name != field_name:
            assert overridden_dict[name] == value


def test_model_config_comes_from_checkpoint() -> None:
    config = Baseline.resolve_checkpoint_model_config(_checkpoint_args())

    assert config.base_ch == 5
    assert config.coeff_mode == "abs_pow"
    assert config.normalize_k is False
    assert config.positive_interior is False


def test_test_seed_reproduces_prediction_dataset(tmp_path: Path) -> None:
    first = _run_prediction_payload(tmp_path, test_seed=444)
    second = _run_prediction_payload(tmp_path, test_seed=444)

    for key in ["f", "g", "k", "u_exact"]:
        assert torch.equal(first[key], second[key])
    assert first["prediction_config"]["test_seed"] == 444


def test_different_test_seeds_change_prediction_dataset(tmp_path: Path) -> None:
    first = _run_prediction_payload(tmp_path, test_seed=444)
    second = _run_prediction_payload(tmp_path, test_seed=445)

    assert not torch.equal(first["k"], second["k"]) or not torch.equal(
        first["u_exact"],
        second["u_exact"],
    )


def test_legacy_checkpoint_missing_newer_prediction_fields_loads(tmp_path: Path) -> None:
    payload = _run_prediction_payload(
        tmp_path,
        test_seed=123,
        checkpoint_args={
            "N": 8,
            "base_ch": 1,
            "rhs_mode": "continuous",
            "coeff_mode": "pow",
            "k_min": 0.5,
            "k_max": 2.5,
            "normalize_k": True,
        },
    )

    assert payload["prediction_config"]["solution_mode"] == "base"
    assert payload["args"]["discretization"] == "flux"
    assert payload["prediction_config"]["N"] == 8
    assert "u_pred" in payload
