"""training, panel, and centerline figures for LGF-CNN."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .lgf_cnn import LGFCNN
from .manufactured import FIXED_CASES


def _get_plt():
    """Load headless pyplot with a writable, process-independent cache."""
    cache = Path(tempfile.gettempdir()) / "axy_lgf_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _with_zero_boundary(field: np.ndarray) -> np.ndarray:
    """Pad an interior field with its homogeneous Dirichlet boundary."""
    return np.pad(field, 1, mode="constant", constant_values=0.0)


def _sym(values: np.ndarray) -> tuple[float, float]:
    """Return nondegenerate color limits symmetric about zero."""
    limit = max(float(np.max(np.abs(values))), 1e-12)
    return -limit, limit


def save_panel(
    u_true: np.ndarray,
    u_pred: np.ndarray,
    error: np.ndarray,
    f_map: np.ndarray,
    res_pred: np.ndarray,
    res_exact: np.ndarray,
    path: Path,
    title: str,
    full_extent: list[float],
    residual_extent: list[float],
    dpi: int = 200,
) -> None:
    """Match the PINN 2x3 panel using the LGF-CNN discrete operator."""
    plt = _get_plt()
    u_min = min(float(u_true.min()), float(u_pred.min()))
    u_max = max(float(u_true.max()), float(u_pred.max()))
    error_limits = _sym(error)
    residual_limit = max(
        float(np.max(np.abs(res_pred))),
        float(np.max(np.abs(res_exact))),
        1e-12,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)
    panels = [
        (0, 0, u_true, "true u", "viridis", full_extent, u_min, u_max),
        (
            0,
            1,
            u_pred,
            "LGF-CNN prediction",
            "viridis",
            full_extent,
            u_min,
            u_max,
        ),
        (
            0,
            2,
            error,
            "signed error",
            "coolwarm",
            full_extent,
            *error_limits,
        ),
        (
            1,
            0,
            f_map,
            "forcing f",
            "coolwarm",
            # f does not vanish on the boundary, so it is drawn on the
            # interior nodes rather than padded with a fabricated zero ring.
            residual_extent,
            *_sym(f_map),
        ),
        (
            1,
            1,
            res_pred,
            "pred FD residual",
            "coolwarm",
            residual_extent,
            -residual_limit,
            residual_limit,
        ),
        (
            1,
            2,
            res_exact,
            "ref FD residual (floor)",
            "coolwarm",
            residual_extent,
            -residual_limit,
            residual_limit,
        ),
    ]
    for row, col, values, label, cmap, extent, vmin, vmax in panels:
        ax = axes[row][col]
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_centerline(
    u_true: np.ndarray,
    u_pred: np.ndarray,
    path: Path,
    title: str,
    x_min: float,
    x_max: float,
    dpi: int = 150,
) -> None:
    """Match the PINN horizontal centerline figure."""
    plt = _get_plt()
    height, width = u_true.shape
    row = height // 2
    xs = np.linspace(x_min, x_max, width)
    y_value = np.linspace(x_min, x_max, height)[row]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(xs, u_true[row], "b-", label="u_ref")
    ax.plot(xs, u_pred[row], "r--", label="u_pred")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"{title}  (centerline y={y_value:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _read_history(path: Path) -> list[dict[str, str]]:
    """Read a training-history CSV as dictionaries keyed by column name."""
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _present(value: str | None) -> bool:
    """Return whether a serialized optional metric contains a real value."""
    return value not in (None, "", "None", "nan")


def save_training_curve(
    history_path: Path,
    path: Path,
    title: str,
    loss_mode: str,
    dpi: int = 150,
) -> None:
    """PINN-style two-panel curve adapted to LGF-CNN's single training loss."""
    if not history_path.is_file():
        print(f"[plot] skipped training curve: history not found at {history_path}")
        return
    history = _read_history(history_path)
    if not history:
        return

    plt = _get_plt()
    epochs = np.asarray([int(row["epoch"]) for row in history])
    tiny = np.finfo(np.float64).tiny
    losses = np.maximum(
        np.asarray([float(row["loss"]) for row in history]),
        tiny,
    )
    monitored = [
        row
        for row in history
        if _present(row.get("val_rel_u"))
        and _present(row.get("val_fd_res_rmse"))
    ]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )
    fig.suptitle(title)

    loss_label = {
        "supervised": "supervised loss",
        "physics": "physics loss",
        "legacy_mixed": "legacy mixed loss",
    }.get(loss_mode, f"{loss_mode} loss")
    ax1.plot(epochs, losses, "b-", alpha=0.7, label=loss_label)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.set_title("Training loss")
    ax1.legend(fontsize="small")
    ax1.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    if monitored:
        val_epochs = [int(row["epoch"]) for row in monitored]
        rel_l2 = [float(row["val_rel_u"]) for row in monitored]
        fd_res_rmse = [float(row["val_fd_res_rmse"]) for row in monitored]
        line1, = ax2.plot(val_epochs, rel_l2, "g-", label="relL2")
        line2, = ax2b.plot(
            val_epochs,
            fd_res_rmse,
            "m--",
            label="fdResRMSE",
        )
        ax2.legend(handles=[line1, line2], loc="upper right", fontsize="small")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("relL2")
    ax2b.set_ylabel("fdResRMSE")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Grid metrics (monitoring only)")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def generate_pinn_style_figures(
    model: LGFCNN,
    history_path: Path,
    figure_dir: Path,
    experiment_name: str,
    case: str,
    contrast: float,
    loss_mode: str,
    evaluation: dict,
) -> None:
    """Create one training curve and panel/centerline figures for selected cases."""
    selected_cases = FIXED_CASES if case == "all" else (case,)
    invalid_cases = [
        selected_case
        for selected_case in selected_cases
        if selected_case not in FIXED_CASES
    ]
    if invalid_cases:
        raise ValueError(f"unknown plot case: {invalid_cases[0]}")

    model.eval()
    grid = model.grid
    device = model.bank.lambdas.device
    evaluation_names = list(evaluation["names"])
    missing = [
        selected_case
        for selected_case in selected_cases
        if selected_case not in evaluation_names
    ]
    if missing:
        raise ValueError(f"evaluation is missing plot case: {missing[0]}")
    full_extent = [grid.x_min, grid.x_max, grid.x_min, grid.x_max]
    residual_extent = [
        grid.x_min + grid.h,
        grid.x_max - grid.h,
        grid.x_min + grid.h,
        grid.x_max - grid.h,
    ]
    figure_dir.mkdir(parents=True, exist_ok=True)

    save_training_curve(
        history_path,
        figure_dir / "training_curve.png",
        f"LGF-CNN training — {experiment_name}",
        loss_mode,
    )

    for selected_case in selected_cases:
        index = evaluation_names.index(selected_case)
        a = torch.as_tensor(
            evaluation["a"][index][None],
            dtype=torch.float64,
            device=device,
        )
        u_true = torch.as_tensor(
            evaluation["u_exact"][index][None],
            dtype=torch.float64,
            device=device,
        )
        u_pred = torch.as_tensor(
            evaluation["u_pred"][index][None],
            dtype=torch.float64,
            device=device,
        )
        f = model.ops.apply_div_a_grad(u_true, a)
        res_pred = model.residual(u_pred, a, f)
        res_exact = model.residual(u_true, a, f)

        u_true_np = u_true.cpu().numpy()[0]
        u_pred_np = u_pred.cpu().numpy()[0]
        f_np = f.cpu().numpy()[0]
        res_pred_np = res_pred.cpu().numpy()[0]
        res_exact_np = res_exact.cpu().numpy()[0]
        error_np = u_pred_np - u_true_np

        rel_l2 = float(
            np.linalg.norm(error_np.ravel())
            / max(np.linalg.norm(u_true_np.ravel()), 1e-14)
        )
        max_error = float(np.max(np.abs(error_np)))
        fd_res_rmse = float(np.sqrt(np.mean(res_pred_np**2)))
        fd_ref_res_rmse = float(np.sqrt(np.mean(res_exact_np**2)))

        title = (
            f"LGF-CNN {selected_case} contrast={contrast:g} N={grid.n} | "
            f"relL2={rel_l2:.3e}  maxErr={max_error:.3e}  "
            f"fdResRMSE={fd_res_rmse:.3e}  "
            f"fdRefResRMSE={fd_ref_res_rmse:.3e}"
        )
        save_panel(
            _with_zero_boundary(u_true_np),
            _with_zero_boundary(u_pred_np),
            _with_zero_boundary(error_np),
            f_np,
            res_pred_np,
            res_exact_np,
            figure_dir / f"{selected_case}_panel.png",
            title,
            full_extent,
            residual_extent,
        )
        save_centerline(
            _with_zero_boundary(u_true_np),
            _with_zero_boundary(u_pred_np),
            figure_dir / f"{selected_case}_centerline.png",
            title,
            grid.x_min,
            grid.x_max,
        )
