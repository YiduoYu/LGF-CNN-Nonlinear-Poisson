from pathlib import Path

import torch

import Baseline


def test_discrete_rhs_matches_flux_residual_to_roundoff() -> None:
    problem = Baseline.sample_manufactured_problem(
        N=10,
        device=torch.device("cpu"),
        rhs_mode="discrete",
        coeff_mode="pow",
        solution_mode="mixed",
        dtype=torch.float64,
    )

    res = Baseline.compute_pde_residual(
        problem["u_exact"],  # type: ignore[arg-type]
        problem["f"],  # type: ignore[arg-type]
        problem["k"],  # type: ignore[arg-type]
        problem["h"],  # type: ignore[arg-type]
        coeff_mode="pow",
    )

    assert float(res.abs().max()) < 1e-10


def test_compute_pde_residual_is_flux_wrapper() -> None:
    torch.manual_seed(123)
    u = torch.rand((2, 1, 7, 7), dtype=torch.float64) + 1.0
    f = torch.randn((2, 1, 7, 7), dtype=torch.float64)
    k = torch.tensor([0.5, 1.7], dtype=torch.float64)
    h = 0.25

    wrapped = Baseline.compute_pde_residual(u, f, k, h, coeff_mode="pow")
    direct = Baseline.nonlinear_poisson_residual(u, f, k, h, coeff_mode="pow")

    assert torch.allclose(wrapped, direct, atol=0.0, rtol=0.0)


def test_legacy_checkpoint_discretization_field_is_ignored(tmp_path: Path) -> None:
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
                "discretization": "kirchhoff",
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

    assert not hasattr(checkpoint_args, "discretization")
    assert not hasattr(model_config, "discretization")
