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
                "k_min": 0.5,
                "k_max": 2.5,
                "normalize_k": True,
            },
        },
        path,
    )


def _prediction_args(checkpoint_path: Path, save_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        load_path=str(checkpoint_path),
        predict_manufactured=True,
        predict_samples=1,
        predict_save=str(save_path),
        device="cpu",
        seed=1234,
        predict_N=None,
        predict_k_min=None,
        predict_k_max=None,
        eval_output_dir="",
        plot_predictions=False,
        show_plots=False,
        test_seed=1234,
    )


def test_predict_save_writes_minimal_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_path = tmp_path / "prediction.pt"
    _write_checkpoint(checkpoint_path)

    Baseline.run_prediction(_prediction_args(checkpoint_path, save_path))

    payload = torch.load(save_path, map_location="cpu")
    assert "u_pred" in payload
    assert "u_transform" not in payload
    assert set(payload["metrics"]) == {"cnn"}
