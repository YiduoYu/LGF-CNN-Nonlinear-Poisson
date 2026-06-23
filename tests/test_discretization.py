import sys
from pathlib import Path

import pytest
import torch

import Baseline


def _exact_discrete_residual(discretization: str) -> torch.Tensor:
    problem = Baseline.sample_manufactured_problem(
        N=10,
        device=torch.device("cpu"),
        rhs_mode="discrete",
        coeff_mode="pow",
        discretization=discretization,
        solution_mode="mixed",
        dtype=torch.float64,
    )
    return Baseline.compute_pde_residual(
        problem["u_exact"],  # type: ignore[arg-type]
        problem["f"],  # type: ignore[arg-type]
        problem["k"],  # type: ignore[arg-type]
        problem["h"],  # type: ignore[arg-type]
        discretization=discretization,
        coeff_mode="pow",
    )


def test_kirchhoff_discrete_rhs_matches_exact_solution_to_roundoff() -> None:
    res = _exact_discrete_residual("kirchhoff")

    assert float(res.abs().max()) < 1e-10


def test_flux_discrete_rhs_matches_exact_solution_to_roundoff() -> None:
    res = _exact_discrete_residual("flux")

    assert float(res.abs().max()) < 1e-10


def test_flux_and_kirchhoff_residuals_agree_when_k_is_zero() -> None:
    torch.manual_seed(123)
    u = torch.rand((2, 1, 7, 7), dtype=torch.float64) + 1.0
    f = torch.randn((2, 1, 7, 7), dtype=torch.float64)
    k = torch.zeros((2,), dtype=torch.float64)
    h = 0.25

    flux = Baseline.compute_pde_residual(
        u,
        f,
        k,
        h,
        discretization="flux",
        coeff_mode="pow",
    )
    kirchhoff = Baseline.compute_pde_residual(
        u,
        f,
        k,
        h,
        discretization="kirchhoff",
        coeff_mode="pow",
    )

    assert torch.allclose(flux, kirchhoff, atol=1e-12, rtol=1e-12)


def test_kirchhoff_rejects_non_pow_coefficient_mode() -> None:
    u = torch.ones((1, 1, 5, 5), dtype=torch.float64)
    f = torch.zeros_like(u)
    k = torch.ones((1,), dtype=torch.float64)

    with pytest.raises(ValueError, match="kirchhoff.*coeff_mode='pow'"):
        Baseline.compute_pde_residual(
            u,
            f,
            k,
            0.5,
            discretization="kirchhoff",
            coeff_mode="abs_pow",
        )


def test_transform_solver_comparison_rejects_flux_discretization() -> None:
    model = Baseline.UNet2d(base_ch=1)
    stats = Baseline.InputStats(f_mean=0.0, f_std=1.0, k_mean=1.5, k_std=0.5)
    data = {
        "f": torch.zeros((1, 1, 10, 10)),
        "g": torch.ones((1, 1, 10, 10)),
        "u_exact": torch.ones((1, 1, 10, 10)),
        "k": torch.ones((1,)),
        "mask": torch.zeros((1, 1, 10, 10)),
        "interior_mask": torch.ones((1, 1, 10, 10)),
        "x": torch.zeros((1, 1, 10, 10)),
        "y": torch.zeros((1, 1, 10, 10)),
        "h": 2.0 / 9.0,
    }

    with pytest.raises(ValueError, match="different discrete equations"):
        Baseline.evaluate_on(
            model,
            data,
            stats,
            batch_size=1,
            coeff_mode="pow",
            discretization="flux",
            normalize_k=True,
            positive_interior=True,
            compare_transform_solver=True,
        )


def test_legacy_checkpoint_without_discretization_resolves_to_flux(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "legacy.pt"
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
                "base_ch": 1,
                "coeff_mode": "pow",
                "normalize_k": True,
            },
        },
        checkpoint_path,
    )

    _, _, checkpoint_args = Baseline.load_model_from_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )
    model_config = Baseline.resolve_checkpoint_model_config(checkpoint_args)

    assert checkpoint_args.discretization == "flux"
    assert model_config.discretization == "flux"


def test_new_checkpoint_stores_discretization(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "new.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "Baseline.py",
            "--N",
            "8",
            "--train-samples",
            "1",
            "--val-samples",
            "1",
            "--epochs",
            "0",
            "--batch-size",
            "1",
            "--eval-batch-size",
            "1",
            "--base-ch",
            "1",
            "--device",
            "cpu",
            "--save-path",
            str(checkpoint_path),
            "--discretization",
            "kirchhoff",
        ],
    )
    args = Baseline.parse_args()
    Baseline.apply_positive_interior_default(args)

    Baseline.train(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint["args"]["discretization"] == "kirchhoff"


def test_kirchhoff_residual_backprop_has_finite_gradients() -> None:
    u = (torch.rand((1, 1, 6, 6), dtype=torch.float64) + 1.0).requires_grad_()
    f = torch.randn((1, 1, 6, 6), dtype=torch.float64)
    k = torch.tensor([1.4], dtype=torch.float64)

    res = Baseline.kirchhoff_residual_from_u(u, f, k, h=0.4)
    loss = res.pow(2).mean()
    loss.backward()

    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
