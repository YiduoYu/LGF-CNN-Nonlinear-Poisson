import argparse

import pytest
import torch

import Baseline


def _random_poisson_problem(N: int, batch_size: int = 2, dtype: torch.dtype = torch.float64):
    x, y, mask, _, h = Baseline.make_grid(N, device="cpu", dtype=dtype)
    del x, y
    full_shape = (batch_size, 1, N + 2, N + 2)
    f = torch.randn(full_shape, dtype=dtype)
    boundary_values = torch.randn(full_shape, dtype=dtype)
    mask_b = mask.expand(batch_size, -1, -1, -1)
    w_boundary = mask_b * boundary_values
    return f, w_boundary, h, mask


def _dimensionless_relative_residual(w: torch.Tensor, f: torch.Tensor, w_boundary: torch.Tensor, h: float) -> float:
    lhs = Baseline._negative_laplacian_interior_matvec(w[:, :, 1:-1, 1:-1])
    rhs = Baseline._poisson_dirichlet_rhs(f, w_boundary, h)
    num = torch.linalg.vector_norm((lhs - rhs).flatten(start_dim=1), dim=1)
    den = torch.linalg.vector_norm(rhs.flatten(start_dim=1), dim=1).clamp_min(1.0)
    return float((num / den).max())


@pytest.mark.parametrize("N", [4, 8])
def test_cg_matches_dense_reference_for_small_float64_problems(N: int) -> None:
    torch.manual_seed(100 + N)
    f, w_boundary, h, mask = _random_poisson_problem(N)

    dense = Baseline._solve_poisson_dirichlet_dense_reference(f, w_boundary, h, mask)
    cg, diagnostics = Baseline.solve_poisson_dirichlet_cg(
        f,
        w_boundary,
        h,
        mask,
        cg_tol=1e-12,
        cg_max_iter=500,
    )

    rel_l2 = torch.linalg.vector_norm((cg - dense).flatten()) / torch.linalg.vector_norm(
        dense.flatten()
    ).clamp_min(1.0)
    assert diagnostics.converged
    assert rel_l2 < 1e-10


def test_cg_reconstructs_exact_boundary_values() -> None:
    torch.manual_seed(123)
    f, w_boundary, h, mask = _random_poisson_problem(8)

    w, _ = Baseline.solve_poisson_dirichlet_cg(f, w_boundary, h, mask, cg_tol=1e-12)

    expanded_mask = mask.expand_as(w_boundary)
    assert torch.equal(w * expanded_mask, w_boundary * expanded_mask)


def test_cg_five_point_equation_residual_meets_tolerance() -> None:
    torch.manual_seed(124)
    f, w_boundary, h, mask = _random_poisson_problem(8)
    tol = 1e-11

    w, diagnostics = Baseline.solve_poisson_dirichlet_cg(
        f,
        w_boundary,
        h,
        mask,
        cg_tol=tol,
    )

    assert diagnostics.final_max_relative_residual <= tol
    assert _dimensionless_relative_residual(w, f, w_boundary, h) <= 2.0 * tol


def test_cg_supports_batched_inputs_with_different_values() -> None:
    torch.manual_seed(125)
    f, w_boundary, h, mask = _random_poisson_problem(6, batch_size=3)
    f[1] = 2.0 * f[1] + 0.25
    w_boundary[2] = 3.0 * w_boundary[2] - 0.5

    w, diagnostics = Baseline.solve_poisson_dirichlet_cg(
        f,
        w_boundary,
        h,
        mask,
        cg_tol=1e-11,
    )

    assert diagnostics.converged
    assert w.shape == f.shape
    assert not torch.equal(w[0], w[1])
    assert not torch.equal(w[1], w[2])


def test_zero_rhs_and_zero_boundary_gives_zero_solution() -> None:
    _, _, mask, _, h = Baseline.make_grid(8, device="cpu", dtype=torch.float64)
    f = torch.zeros((2, 1, 10, 10), dtype=torch.float64)
    w_boundary = torch.zeros_like(f)

    w, diagnostics = Baseline.solve_poisson_dirichlet_cg(f, w_boundary, h, mask)

    assert diagnostics.iterations == 0
    assert diagnostics.converged
    assert torch.equal(w, torch.zeros_like(w))


def test_n64_cg_does_not_allocate_dense_system_matrix(monkeypatch) -> None:
    torch.manual_seed(126)
    N = 64
    f, w_boundary, h, mask = _random_poisson_problem(N, batch_size=1, dtype=torch.float32)
    original_zeros = torch.zeros

    def fail_on_dense_matrix_shape(*args, **kwargs):
        shape = args[0] if args else kwargs.get("size")
        if shape == (N * N, N * N):
            raise AssertionError("dense N^2 by N^2 matrix allocation is not allowed")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(Baseline.torch, "zeros", fail_on_dense_matrix_shape)

    w, diagnostics = Baseline.solve_poisson_dirichlet_cg(
        f,
        w_boundary,
        h,
        mask,
        cg_tol=1e-5,
    )

    assert diagnostics.converged
    assert w.shape == f.shape


def test_cg_non_convergence_raises_clear_error() -> None:
    torch.manual_seed(127)
    f, w_boundary, h, mask = _random_poisson_problem(8)

    with pytest.raises(RuntimeError, match="Poisson CG failed to converge"):
        Baseline.solve_poisson_dirichlet_cg(
            f,
            w_boundary,
            h,
            mask,
            cg_tol=1e-14,
            cg_max_iter=1,
        )


def test_grid_convergence_smoke_uses_transformed_cg(capsys) -> None:
    args = argparse.Namespace(
        predict_samples=1,
        coeff_mode="pow",
        discretization="kirchhoff",
        seed=1234,
        device="cpu",
        k_min=0.5,
        k_max=2.5,
        rhs_mode="continuous",
        solution_mode="base",
        grid_Ns=[4, 8],
        cg_tol=1e-8,
        cg_max_iter=200,
    )

    Baseline.run_grid_convergence(args)

    captured = capsys.readouterr()
    assert "cg_iters=" in captured.out
