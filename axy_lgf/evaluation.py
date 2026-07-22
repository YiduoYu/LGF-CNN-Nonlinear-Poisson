"""Shared numerical evaluation for LGF-CNN predictions."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

from .data import sample_coefficient
from .grid import SquareGrid
from .lgf_cnn import LGFCNN
from .lgf_inverse import DirichletLGF
from .operators import apply_div_a_grad, l2_norm
from .solvers import solve_axy_pcg, solve_axy_richardson

METRIC_KEYS = (
    "relL2",
    "RMSE",
    "MAE",
    "maxError",
    "predResRMSE",
    "refResRMSE",
    "relRes",
    "gradRelL2",
    "gradLinf",
    "constBaselineRelL2",
    "linearBaselineRelL2",
    "f_RMSE",
)

# Training samples p, q in {1, 2, 3}; everything beyond 3 is extrapolation.
OOD_FREQUENCIES = (1, 2, 3, 4, 5, 6, 8)
# Training solutions carry N(0, 1) modal amplitudes, i.e. |f| ~ O(1).
OOD_FORCING_SCALES = (1e-2, 1e-1, 1.0, 1e1, 1e2)
OOD_CONTRASTS = (0.2, 0.5, 0.8, 0.95, 1.0)

COMPARISON_METRIC_KEYS = (
    "relL2",
    "RMSE",
    "MAE",
    "maxError",
    "relRes",
    "resRMSE",
    "time_ms",
    "iterations",
)


def summarize_rows(rows: list[dict], metric_keys: Sequence[str]) -> dict:
    """Return mean/std/min/max for each finite numeric metric."""
    summary = {}
    for key in metric_keys:
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if row.get(key) is not None and np.isfinite(float(row[key]))
            ],
            dtype=np.float64,
        )
        if values.size:
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def _synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work so elapsed-time measurements are valid."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _centered_gradient(u: np.ndarray, h: float) -> np.ndarray:
    """Centered gradient on interior nodes using the zero Dirichlet halo."""
    full = np.pad(u, 1, mode="constant", constant_values=0.0)
    du_dx = (full[1:-1, 2:] - full[1:-1, :-2]) / (2.0 * h)
    du_dy = (full[2:, 1:-1] - full[:-2, 1:-1]) / (2.0 * h)
    return np.stack((du_dx, du_dy), axis=0)


def _relative_l2(error: np.ndarray, reference: np.ndarray, grid: SquareGrid) -> float:
    """Return a discrete relative L2 error with a safe zero denominator."""
    denominator = max(l2_norm(reference, grid), np.finfo(np.float64).tiny)
    return l2_norm(error, grid) / denominator


def _solution_metrics(
    u: np.ndarray,
    u_exact: np.ndarray,
    a: np.ndarray,
    f: np.ndarray,
    grid: SquareGrid,
) -> dict[str, float]:
    """Compute solution-error and PDE-residual metrics for one method."""
    error = u - u_exact
    residual = f - apply_div_a_grad(u, a, grid)
    f_norm = max(l2_norm(f, grid), np.finfo(np.float64).tiny)
    return {
        "relL2": _relative_l2(error, u_exact, grid),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "maxError": float(np.max(np.abs(error))),
        "relRes": l2_norm(residual, grid) / f_norm,
        "resRMSE": float(np.sqrt(np.mean(residual**2))),
    }


def _full_metrics(
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    a: np.ndarray,
    f: np.ndarray,
    linear_baseline: np.ndarray,
    grid: SquareGrid,
) -> dict[str, float]:
    """Compute the complete CNN metric set, including gradient baselines."""
    metrics = _solution_metrics(u_pred, u_exact, a, f, grid)
    ref_residual = f - apply_div_a_grad(u_exact, a, grid)
    grad_pred = _centered_gradient(u_pred, grid.h)
    grad_exact = _centered_gradient(u_exact, grid.h)
    grad_error = grad_pred - grad_exact
    constant_baseline = np.full_like(u_exact, float(u_exact.mean()))
    grad_denominator = max(
        float(np.linalg.norm(grad_exact.ravel())),
        np.finfo(np.float64).tiny,
    )
    # No bndErr here: both fields carry an explicit zero Dirichlet halo, so a
    # boundary error metric is identically zero and measures nothing.
    metrics.update(
        {
            "predResRMSE": metrics.pop("resRMSE"),
            "refResRMSE": float(np.sqrt(np.mean(ref_residual**2))),
            "gradRelL2": float(np.linalg.norm(grad_error.ravel()))
            / grad_denominator,
            "gradLinf": float(np.max(np.abs(grad_error))),
            "constBaselineRelL2": _relative_l2(
                constant_baseline - u_exact,
                u_exact,
                grid,
            ),
            "linearBaselineRelL2": _relative_l2(
                linear_baseline - u_exact,
                u_exact,
                grid,
            ),
            "f_RMSE": float(np.sqrt(np.mean(f**2))),
        }
    )
    return metrics


@torch.no_grad()
def run_ood_sweeps(
    model: LGFCNN,
    contrast: float = 0.8,
    seed: int = 123,
) -> list[dict]:
    """Probe generalisation along one axis of the training distribution at a time.

    The fixed u1-u4 cases and the held-out batch both live inside the training
    family (1-3 separable sin/cos modes with p, q in {1,2,3}, N(0,1) amplitudes,
    one contrast), so neither of them actually tests extrapolation.  Each sweep
    here steps outside exactly one axis and reports the same-budget Richardson
    result alongside, so a degradation can be attributed to the network rather
    than to the problem simply getting harder.
    """
    model.eval()
    device = model.bank.lambdas.device
    grid = model.grid
    x, y = grid.mesh()
    xi = (x - grid.x_min) / grid.L
    yi = (y - grid.x_min) / grid.L

    def as_batch(field: np.ndarray) -> torch.Tensor:
        """Convert one NumPy field to a one-sample tensor on the model device."""
        return torch.as_tensor(field[None], dtype=torch.float64, device=device)

    def probe(sweep: str, setting: str, in_distribution: bool, a_np, u_np) -> dict:
        """Evaluate one OOD setting against the same-budget Richardson solver."""
        a_t = as_batch(a_np)
        f_t = model.ops.apply_div_a_grad(as_batch(u_np), a_t)
        f_np = f_t.cpu().numpy()[0]
        pred = model(a_t, f_t).cpu().numpy()[0]
        rich = solve_axy_richardson(
            a_np, f_np, grid, omega=1.0, tol=0.0, maxiter=model.K
        )
        residual = f_np - apply_div_a_grad(pred, a_np, grid)
        return {
            "sweep": sweep,
            "setting": setting,
            "in_distribution": in_distribution,
            "relL2": _relative_l2(pred - u_np, u_np, grid),
            "relRes": l2_norm(residual, grid)
            / max(l2_norm(f_np, grid), np.finfo(np.float64).tiny),
            "richardsonRelL2": _relative_l2(rich.u - u_np, u_np, grid),
        }

    a_ref = sample_coefficient(
        grid, 1, contrast=contrast, rng=np.random.default_rng(seed)
    )[0]
    rows = []

    # 1. Frequency: the LGF filters are global spectral operators and the mixer
    #    is pointwise, so there is no fixed receptive field to limit the model.
    for mode in OOD_FREQUENCIES:
        u_np = np.sin(mode * np.pi * xi) * np.sin(mode * np.pi * yi)
        rows.append(probe("frequency", f"p=q={mode}", mode <= 3, a_ref, u_np))

    # 2. Forcing amplitude: exact by construction now that the mixer inputs are
    #    normalised, so this sweep is a regression guard on that property.
    u_unit = np.sin(np.pi * xi) * np.sin(np.pi * yi)
    for scale in OOD_FORCING_SCALES:
        rows.append(
            probe("amplitude", f"|f| x {scale:g}", scale == 1.0, a_ref, scale * u_unit)
        )

    # 3. Contrast: genuinely harder problems, and beyond ~0.95 the coefficient
    #    floor in sample_coefficient starts to flatten a(x, y).
    for value in OOD_CONTRASTS:
        a_np = sample_coefficient(
            grid, 1, contrast=value, rng=np.random.default_rng(seed)
        )[0]
        rows.append(
            probe("contrast", f"c={value:g}", value == contrast, a_np, u_unit)
        )

    return rows


@torch.no_grad()
def evaluate_batch(
    model: LGFCNN,
    names: Sequence[str],
    a: torch.Tensor,
    f: torch.Tensor,
    u_exact: torch.Tensor,
    *,
    richardson_steps: int | None = None,
    pcg_tol: float = 1e-10,
    pcg_maxiter: int = 500,
) -> dict:
    """Evaluate LGF-CNN and classical solvers on one shared batch."""
    if not (len(names) == a.shape[0] == f.shape[0] == u_exact.shape[0]):
        raise ValueError("names, a, f, and u_exact must have the same batch size")
    if not names:
        raise ValueError("evaluation batch must not be empty")

    model.eval()
    device = model.bank.lambdas.device
    grid = model.grid
    steps = model.K if richardson_steps is None else richardson_steps
    if steps < 0:
        raise ValueError("richardson_steps must be nonnegative")

    # Time a plain per-sample forward pass, matching how Richardson and PCG are
    # timed below. Batching and the diagnostic tensors would otherwise make the
    # LGF-CNN column optimistic and not comparable across methods.
    cnn_ms = []
    for index in range(len(names)):
        sample_a = a[index : index + 1]
        sample_f = f[index : index + 1]
        _synchronize(device)
        started = time.perf_counter()
        model(sample_a, sample_f)
        _synchronize(device)
        cnn_ms.append(1000.0 * (time.perf_counter() - started))

    u_pred, diagnostics = model.forward_with_diagnostics(a, f)

    a_np = a.detach().cpu().numpy()
    f_np = f.detach().cpu().numpy()
    exact_np = u_exact.detach().cpu().numpy()
    pred_np = u_pred.detach().cpu().numpy()

    poisson_inverse = DirichletLGF(grid, a0=1.0, sigma=0.0)
    linear_np = np.stack([poisson_inverse(ff) for ff in f_np])
    richardson_np = np.empty_like(exact_np)
    pcg_np = np.empty_like(exact_np)

    metric_rows: list[dict] = []
    comparison_rows: list[dict] = []
    residual_rows: list[dict] = []

    for index, name in enumerate(names):
        aa = a_np[index]
        ff = f_np[index]
        ue = exact_np[index]
        up = pred_np[index]

        full = _full_metrics(up, ue, aa, ff, linear_np[index], grid)
        metric_rows.append(
            {
                "sample_id": index,
                "case": name,
                **full,
                "cnn_ms_per_sample": cnn_ms[index],
            }
        )

        rich_started = time.perf_counter()
        rich = solve_axy_richardson(
            aa,
            ff,
            grid,
            omega=1.0,
            tol=0.0,
            maxiter=steps,
        )
        rich_ms = 1000.0 * (time.perf_counter() - rich_started)
        richardson_np[index] = rich.u

        pcg_started = time.perf_counter()
        pcg = solve_axy_pcg(
            aa,
            ff,
            grid,
            tol=pcg_tol,
            maxiter=pcg_maxiter,
        )
        pcg_ms = 1000.0 * (time.perf_counter() - pcg_started)
        pcg_np[index] = pcg.u

        method_results = (
            (
                "LGF-CNN",
                up,
                cnn_ms[index],
                model.K,
                full["relRes"] <= pcg_tol,
            ),
            ("Richardson", rich.u, rich_ms, rich.iterations, rich.converged),
            ("PCG", pcg.u, pcg_ms, pcg.iterations, pcg.converged),
        )
        for method, solution, elapsed_ms, iterations, converged in method_results:
            comparison_rows.append(
                {
                    "sample_id": index,
                    "case": name,
                    "method": method,
                    **_solution_metrics(solution, ue, aa, ff, grid),
                    "time_ms": elapsed_ms,
                    "iterations": iterations,
                    "converged": converged,
                }
            )

        f_norm = max(l2_norm(ff, grid), np.finfo(np.float64).tiny)
        cnn_history = [1.0]
        for diagnostic in diagnostics:
            iterate = diagnostic["u"][index].detach().cpu().numpy()
            residual = ff - apply_div_a_grad(iterate, aa, grid)
            cnn_history.append(l2_norm(residual, grid) / f_norm)
        max_length = max(
            len(cnn_history),
            len(rich.residual_history),
            len(pcg.residual_history),
        )
        for iteration in range(max_length):
            residual_rows.append(
                {
                    "sample_id": index,
                    "case": name,
                    "iteration": iteration,
                    "lgf_cnn_rel_res": (
                        cnn_history[iteration]
                        if iteration < len(cnn_history)
                        else None
                    ),
                    "richardson_rel_res": (
                        rich.residual_history[iteration]
                        if iteration < len(rich.residual_history)
                        else None
                    ),
                    "pcg_rel_res": (
                        pcg.residual_history[iteration]
                        if iteration < len(pcg.residual_history)
                        else None
                    ),
                }
            )

    method_summary = {}
    for method in ("LGF-CNN", "Richardson", "PCG"):
        method_rows = [row for row in comparison_rows if row["method"] == method]
        method_summary[method] = summarize_rows(
            method_rows,
            COMPARISON_METRIC_KEYS,
        )

    return {
        "names": list(names),
        "a": a_np,
        "f": f_np,
        "u_exact": exact_np,
        "u_pred": pred_np,
        "u_linear_baseline": linear_np,
        "u_richardson": richardson_np,
        "u_pcg": pcg_np,
        "metrics": metric_rows,
        "metric_summary": summarize_rows(metric_rows, METRIC_KEYS),
        "solver_comparison": comparison_rows,
        "solver_summary": method_summary,
        "residual_history": residual_rows,
    }
