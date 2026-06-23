import torch

import Baseline


def test_manufactured_problem_uses_continuous_rhs() -> None:
    problem = Baseline.sample_manufactured_problem(
        N=10,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    res = Baseline.compute_pde_residual(
        problem["u_exact"],  # type: ignore[arg-type]
        problem["f"],  # type: ignore[arg-type]
        problem["k"],  # type: ignore[arg-type]
        problem["h"],  # type: ignore[arg-type]
    )

    assert torch.isfinite(res).all()
    assert float(res.pow(2).mean().sqrt()) > 0.0


def test_compute_pde_residual_is_flux_wrapper() -> None:
    torch.manual_seed(123)
    u = torch.rand((2, 1, 7, 7), dtype=torch.float64) + 1.0
    f = torch.randn((2, 1, 7, 7), dtype=torch.float64)
    k = torch.tensor([0.5, 1.7], dtype=torch.float64)
    h = 0.25

    wrapped = Baseline.compute_pde_residual(u, f, k, h)
    direct = Baseline.nonlinear_poisson_residual(u, f, k, h)

    assert torch.allclose(wrapped, direct, atol=0.0, rtol=0.0)


def test_flux_minus_divergence_is_residual_plus_rhs() -> None:
    torch.manual_seed(456)
    u = torch.rand((2, 1, 7, 7), dtype=torch.float64) + 1.0
    f = torch.randn((2, 1, 7, 7), dtype=torch.float64)
    k = torch.tensor([0.8, 1.2], dtype=torch.float64)
    h = 0.25

    minus_div = Baseline.flux_minus_divergence(u, k, h)
    residual = Baseline.compute_pde_residual(u, f, k, h)

    assert torch.allclose(residual, minus_div - f[:, :, 1:-1, 1:-1])
