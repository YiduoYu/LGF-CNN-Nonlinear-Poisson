"""Tests for the core PDE discretization (grid, manufactured solution, flux residual)."""

import torch

import Baseline


def test_u_exact_formula() -> None:
    """u_exact(x,y) = sin(pi*x) * cos(pi*y)."""
    x, y, *_ = Baseline.make_grid(4, dtype=torch.float64)
    expected = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    assert torch.allclose(Baseline.u_exact(x, y), expected)


def test_pde_residual_on_exact_solution() -> None:
    """PDE residual of the exact solution ≈ truncation error (small but >0)."""
    data = Baseline.generate_dataset(N=10, n_samples=1,
                                     device=torch.device("cpu"))
    res = Baseline.pde_residual(
        data["u_exact"], data["f"], data["k"], data["h"],
    )
    assert torch.isfinite(res).all()
    rmse = float(res.pow(2).mean().sqrt())
    assert 0.0 < rmse < 2.0, f"unexpected truncation error: {rmse:.2e}"


def test_flux_divergence_equals_residual_plus_forcing() -> None:
    """By definition: pde_residual = flux_divergence - f_interior."""
    torch.manual_seed(456)
    u = torch.rand(2, 1, 7, 7, dtype=torch.float64) + 1.0
    f = torch.randn(2, 1, 7, 7, dtype=torch.float64)
    k = torch.tensor([[0.8], [1.2]], dtype=torch.float64)
    h = 0.25

    div = Baseline.flux_divergence(u, k, h)
    res = Baseline.pde_residual(u, f, k, h)

    assert torch.allclose(res, div - f[:, :, 1:-1, 1:-1])


def test_dirichlet_enforcement() -> None:
    """Boundary values are exactly enforced after enforce_dirichlet."""
    x, y, boundary, interior, h = Baseline.make_grid(8)
    u_raw = torch.rand(1, 1, 10, 10)  # random prediction
    g = boundary * torch.sin(torch.pi * x)  # arbitrary boundary values

    u_out = Baseline.enforce_dirichlet(u_raw, g, boundary)
    # boundary error should be exactly zero
    bmax = float(((u_out - g) * boundary).abs().max())
    assert bmax == 0.0, f"boundary not exactly enforced: {bmax:.2e}"


def test_power_coefficient_integer_k() -> None:
    """a(u) = u^2 for k=2 (no clamp needed for positive u)."""
    u = torch.tensor([[[[0.5, 2.0]]]], dtype=torch.float64)
    k = torch.tensor([[2.0]], dtype=torch.float64)
    a = Baseline.power_coefficient(u, k)
    expected = u ** 2
    assert torch.allclose(a, expected)
