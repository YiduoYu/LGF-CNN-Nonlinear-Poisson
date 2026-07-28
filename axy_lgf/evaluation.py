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
    "relL2", "RMSE", "MAE", "maxError",
    "predResRMSE", "refResRMSE", "relRes",
    "gradRelL2", "gradLinf",
    "constBaselineRelL2", "linearBaselineRelL2",
    "f_RMSE",
)

# Training uses p, q in {1, 2, 3}; larger frequencies are extrapolation.
GENERALIZATION_FREQUENCIES = (1, 2, 3, 4, 5, 6, 8)
# Amplitude changes are scale-regression checks, not strict OOD samples.
GENERALIZATION_FORCING_SCALES = (1e-2, 1e-1, 1.0, 1e1, 1e2)
GENERALIZATION_CONTRASTS = (0.2, 0.5, 0.8, 0.95, 1.0)

COMPARISON_METRIC_KEYS = (
    "relL2", "RMSE", "MAE", "maxError",
    "relRes", "resRMSE", "time_ms", "iterations",
)
REFERENCE_SOURCES = ("manufactured", "pcg")
DEFAULT_PCG_SOLVER_TOL = 1e-10
DEFAULT_TARGET_REL_RES = 1e-10


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
    """Wait for queued accelerator work so timing is valid."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _centered_gradient(u: np.ndarray, h: float) -> np.ndarray:
    """Centred gradient on interior nodes, using the zero Dirichlet halo."""
    full = np.pad(u, 1, mode="constant", constant_values=0.0)
    du_dx = (full[1:-1, 2:] - full[1:-1, :-2]) / (2.0 * h)
    du_dy = (full[2:, 1:-1] - full[:-2, 1:-1]) / (2.0 * h)
    return np.stack((du_dx, du_dy), axis=0)


def _safe_norm(v: np.ndarray, grid: SquareGrid) -> float:
    """Discrete L2 norm, floored away from zero for safe denominators."""
    return max(l2_norm(v, grid), np.finfo(np.float64).tiny)


def _relative_l2(error: np.ndarray, reference: np.ndarray, grid: SquareGrid) -> float:
    return l2_norm(error, grid) / _safe_norm(reference, grid)


def _solution_metrics(
    u: np.ndarray, u_exact: np.ndarray,
    a: np.ndarray, f: np.ndarray, grid: SquareGrid,
    residual: np.ndarray | None = None,
) -> dict[str, float]:
    """Error and residual metrics for one solution."""
    error = u - u_exact
    if residual is None:
        residual = f - apply_div_a_grad(u, a, grid)
    f_norm = _safe_norm(f, grid)
    return {
        "relL2": _relative_l2(error, u_exact, grid),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "maxError": float(np.max(np.abs(error))),
        "relRes": l2_norm(residual, grid) / f_norm,
        "resRMSE": float(np.sqrt(np.mean(residual**2))),
    }


def _full_metrics(
    u_pred: np.ndarray, u_exact: np.ndarray,
    a: np.ndarray, f: np.ndarray,
    linear_baseline: np.ndarray, grid: SquareGrid,
    pred_residual: np.ndarray, reference_residual: np.ndarray,
) -> dict[str, float]:
    """Complete CNN metric set including gradient and baseline comparisons."""
    metrics = _solution_metrics(
        u_pred, u_exact, a, f, grid, residual=pred_residual,
    )
    grad_exact = _centered_gradient(u_exact, grid.h)
    grad_error = _centered_gradient(u_pred, grid.h) - grad_exact
    grad_denominator = max(
        float(np.linalg.norm(grad_exact.ravel())), np.finfo(np.float64).tiny,
    )
    constant_baseline = np.full_like(u_exact, float(u_exact.mean()))

    # no bndErr: both fields carry a zero Dirichlet halo, so it's
    # identically zero and measures nothing
    metrics["predResRMSE"] = metrics.pop("resRMSE")
    metrics["refResRMSE"] = float(np.sqrt(np.mean(reference_residual**2)))
    metrics["gradRelL2"] = (
        float(np.linalg.norm(grad_error.ravel())) / grad_denominator
    )
    metrics["gradLinf"] = float(np.max(np.abs(grad_error)))
    metrics["constBaselineRelL2"] = _relative_l2(
        constant_baseline - u_exact, u_exact, grid,
    )
    metrics["linearBaselineRelL2"] = _relative_l2(
        linear_baseline - u_exact, u_exact, grid,
    )
    metrics["f_RMSE"] = float(np.sqrt(np.mean(f**2)))
    return metrics


@torch.no_grad()
def run_generalization_sweeps(
    model: LGFCNN, contrast: float = 0.8, seed: int = 123,
) -> list[dict]:
    """Probe one generalisation axis at a time.

    Frequency includes both training support and extrapolation. Amplitude
    checks exact scale equivariance. Contrast compares the reference training
    value with coefficient shifts. Same-budget Richardson is reported beside
    every network result.
    """
    model.eval()
    device = model.bank.lambdas.device
    grid = model.grid
    xi, yi = grid.unit_mesh()

    def as_batch(field: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(field[None], dtype=torch.float64, device=device)

    def probe(sweep: str, setting: str, regime: str,
              a_np: np.ndarray, u_np: np.ndarray) -> dict:
        a_t = as_batch(a_np)
        f_t = model.ops.apply_div_a_grad(as_batch(u_np), a_t)
        f_np = f_t.cpu().numpy()[0]
        pred = model(a_t, f_t).cpu().numpy()[0]
        rich = solve_axy_richardson(
            a_np, f_np, grid, omega=1.0, tol=0.0, maxiter=model.K,
        )
        residual = f_np - apply_div_a_grad(pred, a_np, grid)
        return {
            "sweep": sweep, "setting": setting,
            "regime": regime,
            "relL2": _relative_l2(pred - u_np, u_np, grid),
            "relRes": l2_norm(residual, grid) / _safe_norm(f_np, grid),
            "richardsonRelL2": _relative_l2(rich.u - u_np, u_np, grid),
        }

    a_ref = sample_coefficient(
        grid, 1, contrast=contrast, rng=np.random.default_rng(seed),
    )[0]
    rows = []

    # frequency sweep: LGF filters are global, no fixed receptive field
    for mode in GENERALIZATION_FREQUENCIES:
        u_np = np.sin(mode * np.pi * xi) * np.sin(mode * np.pi * yi)
        regime = "training_support" if mode <= 3 else "extrapolation"
        rows.append(probe("frequency", f"p=q={mode}", regime, a_ref, u_np))

    # amplitude sweep: exact by construction (mixer inputs normalised),
    # so this is a regression guard
    u_unit = np.sin(np.pi * xi) * np.sin(np.pi * yi)
    for scale in GENERALIZATION_FORCING_SCALES:
        regime = "reference" if scale == 1.0 else "scale_stress"
        rows.append(probe(
            "amplitude", f"|f| x {scale:g}", regime, a_ref, scale * u_unit,
        ))

    # contrast sweep: genuinely harder; beyond ~0.95 the coefficient floor
    # starts to flatten a(x, y)
    for value in GENERALIZATION_CONTRASTS:
        a_np = sample_coefficient(
            grid, 1, contrast=value, rng=np.random.default_rng(seed),
        )[0]
        regime = "reference" if value == contrast else "coefficient_shift"
        rows.append(probe("contrast", f"c={value:g}", regime, a_np, u_unit))

    return rows


@torch.no_grad()
def evaluate_batch(
    model: LGFCNN, names: Sequence[str],
    a: torch.Tensor, f: torch.Tensor, u_exact: torch.Tensor,
    *, richardson_steps: int | None = None,
    pcg_solver_tol: float = DEFAULT_PCG_SOLVER_TOL,
    target_rel_res: float = DEFAULT_TARGET_REL_RES,
    pcg_maxiter: int = 500,
    reference_source: str = "manufactured",
) -> dict:
    """Evaluate LGF-CNN and valid classical baselines on one shared batch.

    ``reference_source="pcg"`` means PCG generated ``u_exact``. In that
    case PCG is omitted as a comparison method because rerunning the same
    solver would only compare the labels with themselves.
    """
    if not (len(names) == a.shape[0] == f.shape[0] == u_exact.shape[0]):
        raise ValueError("names, a, f, and u_exact must have the same batch size")
    if not names:
        raise ValueError("evaluation batch must not be empty")
    if reference_source not in REFERENCE_SOURCES:
        choices = ", ".join(REFERENCE_SOURCES)
        raise ValueError(f"reference_source must be one of: {choices}")
    if not np.isfinite(pcg_solver_tol) or pcg_solver_tol <= 0.0:
        raise ValueError("pcg_solver_tol must be finite and positive")
    if not np.isfinite(target_rel_res) or target_rel_res <= 0.0:
        raise ValueError("target_rel_res must be finite and positive")
    if pcg_maxiter < 0:
        raise ValueError("pcg_maxiter must be nonnegative")

    model.eval()
    device = model.bank.lambdas.device
    grid = model.grid
    steps = model.K if richardson_steps is None else richardson_steps
    if steps < 0:
        raise ValueError("richardson_steps must be nonnegative")

    # time per-sample forward pass, same as Richardson/PCG below
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
    include_pcg = reference_source != "pcg"
    pcg_np = np.empty_like(exact_np) if include_pcg else None
    pred_residual_np = np.empty_like(exact_np)
    reference_residual_np = np.empty_like(exact_np)

    metric_rows: list[dict] = []
    comparison_rows: list[dict] = []
    residual_rows: list[dict] = []

    for index, name in enumerate(names):
        aa = a_np[index]
        ff = f_np[index]
        ue = exact_np[index]
        up = pred_np[index]

        pred_residual = ff - apply_div_a_grad(up, aa, grid)
        reference_residual = ff - apply_div_a_grad(ue, aa, grid)
        pred_residual_np[index] = pred_residual
        reference_residual_np[index] = reference_residual
        full = _full_metrics(
            up, ue, aa, ff, linear_np[index], grid,
            pred_residual, reference_residual,
        )
        metric_rows.append({
            "sample_id": index, "case": name, **full,
            "cnn_ms_per_sample": cnn_ms[index],
        })

        rich_started = time.perf_counter()
        rich = solve_axy_richardson(
            aa, ff, grid, omega=1.0, tol=0.0, maxiter=steps,
        )
        rich_ms = 1000.0 * (time.perf_counter() - rich_started)
        richardson_np[index] = rich.u

        cnn_solution_metrics = {
            "relL2": full["relL2"],
            "RMSE": full["RMSE"],
            "MAE": full["MAE"],
            "maxError": full["maxError"],
            "relRes": full["relRes"],
            "resRMSE": full["predResRMSE"],
        }
        rich_residual = ff - apply_div_a_grad(rich.u, aa, grid)
        rich_solution_metrics = _solution_metrics(
            rich.u, ue, aa, ff, grid, residual=rich_residual,
        )
        method_results = [
            (
                "LGF-CNN", cnn_solution_metrics, cnn_ms[index], model.K,
                "fixed_budget", None,
            ),
            (
                "Richardson", rich_solution_metrics, rich_ms, rich.iterations,
                "fixed_budget", None,
            ),
        ]

        pcg = None
        if include_pcg:
            pcg_started = time.perf_counter()
            pcg = solve_axy_pcg(
                aa, ff, grid, tol=pcg_solver_tol, maxiter=pcg_maxiter,
            )
            pcg_ms = 1000.0 * (time.perf_counter() - pcg_started)
            pcg_np[index] = pcg.u
            pcg_residual = ff - apply_div_a_grad(pcg.u, aa, grid)
            pcg_solution_metrics = _solution_metrics(
                pcg.u, ue, aa, ff, grid, residual=pcg_residual,
            )
            method_results.append(
                (
                    "PCG", pcg_solution_metrics, pcg_ms, pcg.iterations,
                    "tolerance_driven", pcg.converged,
                ),
            )

        for (
            method, solution_metrics, elapsed_ms, iterations,
            execution_mode, solver_converged,
        ) in method_results:
            comparison_rows.append({
                "sample_id": index, "case": name, "method": method,
                **solution_metrics,
                "time_ms": elapsed_ms, "iterations": iterations,
                "execution_mode": execution_mode,
                "meets_target": solution_metrics["relRes"] <= target_rel_res,
                "solver_converged": solver_converged,
            })

        f_norm = _safe_norm(ff, grid)
        cnn_history = [1.0]
        for diagnostic in diagnostics:
            iterate = diagnostic["u"][index].detach().cpu().numpy()
            residual = ff - apply_div_a_grad(iterate, aa, grid)
            cnn_history.append(l2_norm(residual, grid) / f_norm)
        history_lengths = [len(cnn_history), len(rich.residual_history)]
        if pcg is not None:
            history_lengths.append(len(pcg.residual_history))
        max_length = max(history_lengths)
        for iteration in range(max_length):
            residual_rows.append({
                "sample_id": index, "case": name,
                "iteration": iteration,
                "lgf_cnn_rel_res": (
                    cnn_history[iteration]
                    if iteration < len(cnn_history) else None
                ),
                "richardson_rel_res": (
                    rich.residual_history[iteration]
                    if iteration < len(rich.residual_history) else None
                ),
                "pcg_rel_res": (
                    pcg.residual_history[iteration]
                    if pcg is not None and iteration < len(pcg.residual_history)
                    else None
                ),
            })

    method_summary = {}
    method_names = ("LGF-CNN", "Richardson")
    if include_pcg:
        method_names += ("PCG",)
    for method in method_names:
        method_rows = [row for row in comparison_rows if row["method"] == method]
        method_summary[method] = summarize_rows(method_rows, COMPARISON_METRIC_KEYS)

    return {
        "reference_source": reference_source,
        "names": list(names),
        "a": a_np, "f": f_np, "u_exact": exact_np, "u_pred": pred_np,
        "u_pred_residual": pred_residual_np,
        "u_exact_residual": reference_residual_np,
        "u_linear_baseline": linear_np,
        "u_richardson": richardson_np, "u_pcg": pcg_np,
        "metrics": metric_rows,
        "metric_summary": summarize_rows(metric_rows, METRIC_KEYS),
        "solver_comparison": comparison_rows,
        "solver_summary": method_summary,
        "residual_history": residual_rows,
    }
