import argparse
import tempfile
import unittest
from pathlib import Path

import torch

import Baseline


class CheckpointCompatibilityTest(unittest.TestCase):
    def _write_checkpoint(self, payload: dict[str, object]) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "checkpoint.pt"
        torch.save(payload, path)
        return path

    def _state_dict(self) -> dict[str, torch.Tensor]:
        return Baseline.UNet2d(base_ch=1).state_dict()

    def test_loads_legacy_inputstats_object_checkpoint(self) -> None:
        path = self._write_checkpoint(
            {
                "model_state_dict": self._state_dict(),
                "input_stats": Baseline.InputStats(
                    f_mean=1.0,
                    f_std=2.0,
                    k_mean=3.0,
                    k_std=4.0,
                ),
                "args": argparse.Namespace(
                    base_ch=1,
                    normalize_k=True,
                ),
            }
        )

        model, stats, args = Baseline.load_model_from_checkpoint(path, torch.device("cpu"))

        self.assertFalse(model.training)
        self.assertEqual(stats, Baseline.InputStats(1.0, 2.0, 3.0, 4.0))
        self.assertFalse(args.positive_interior)

    def test_loads_dict_stats_checkpoint(self) -> None:
        path = self._write_checkpoint(
            {
                "model_state_dict": self._state_dict(),
                "input_stats": {
                    "f_mean": 0.5,
                    "f_std": 1.5,
                    "k_mean": 2.5,
                    "k_std": 3.5,
                },
                "args": {
                    "base_ch": 1,
                    "normalize_k": False,
                },
            }
        )

        _, stats, args = Baseline.load_model_from_checkpoint(path, torch.device("cpu"))

        self.assertEqual(stats, Baseline.InputStats(0.5, 1.5, 2.5, 3.5))
        self.assertFalse(args.positive_interior)


if __name__ == "__main__":
    unittest.main()
