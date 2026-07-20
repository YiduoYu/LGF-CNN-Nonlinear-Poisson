#!/usr/bin/env python3
"""Plot LGF-CNN predictions, errors, residuals, and mixing fields.

Usage:
    python -m axy_lgf.plot_lgf_cnn --ckpt path/to/best.pt
    python -m axy_lgf.plot_lgf_cnn \
        --ckpt path/to/best.pt --outdir results/figures
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    cache_root = Path(
        os.environ.get(
            "XDG_CACHE_HOME",
            Path(tempfile.gettempdir()) / "axy_lgf_cache",
        )
    )
    mpl_config = cache_root / "matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ["MPLCONFIGDIR"] = str(mpl_config)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import make_batch
from .grid import SquareGrid
from .lgf_cnn import LGFCNN
from .manufactured import make_problem
from .operators import apply_div_a_grad, l2_norm
from .solvers import solve_axy_richardson
from .train_lgf_cnn import load_model


@torch.no_grad()
def run_with_intermediates(
    model: LGFCNN,
    a: torch.Tensor,
    f: torch.Tensor,
) -> dict:
    """Forward pass collecting per-block fields for plotting."""
    u_initial = torch.zeros_like(f)
    u, diagnostics = model.forward_with_diagnostics(a, f, u0=u_initial)
    hist_u = [u_initial.cpu().numpy()[0].copy()]
    hist_u.extend(step["u"].cpu().numpy()[0].copy() for step in diagnostics)
    hist_res = [step["residual"].cpu().numpy()[0].copy() for step in diagnostics]
    alphas = [step["alpha"].cpu().numpy()[0].copy() for step in diagnostics]
    etas = [step["eta"].cpu().numpy()[0, 0].copy() for step in diagnostics]

    r_final = (f - model.ops.apply_div_a_grad(u, a)).cpu().numpy()[0]
    return {
        "u": u.cpu().numpy()[0],
        "hist_u": hist_u,
        "hist_res": hist_res,
        "r_final": r_final,
        "alphas": alphas,
        "etas": etas,
        "lambdas": model.bank.lambdas.cpu().numpy(),
    }


def _extent(grid: SquareGrid) -> list[float]:
    h = grid.h
    return [
        grid.x_min + 0.5 * h,
        grid.x_max - 0.5 * h,
        grid.x_min + 0.5 * h,
        grid.x_max - 0.5 * h,
    ]


def _imshow(ax, field, extent, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(
        field,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_solution_panel(
    out: Path,
    grid: SquareGrid,
    a: np.ndarray,
    u_exact: np.ndarray,
    u_nn: np.ndarray,
    u_rich: np.ndarray,
    r_nn: np.ndarray,
    title: str,
) -> None:
    extent = _extent(grid)
    err_nn = u_nn - u_exact
    err_rich = u_rich - u_exact
    emax = max(np.max(np.abs(err_nn)), np.max(np.abs(err_rich)), 1e-16)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    fig.suptitle(title, fontsize=12)

    _imshow(axes[0, 0], a, extent, r"$a(x,y)$", cmap="plasma")
    _imshow(axes[0, 1], u_exact, extent, r"$u_{\mathrm{exact}}$", cmap="RdBu_r")
    _imshow(axes[0, 2], u_nn, extent, r"$u$ (LGF-CNN)", cmap="RdBu_r")

    _imshow(
        axes[1, 0],
        err_nn,
        extent,
        r"$u_{\mathrm{CNN}}-u_{\mathrm{exact}}$"
        f"\nrel$_u$={l2_norm(err_nn, grid) / l2_norm(u_exact, grid):.2e}",
        cmap="coolwarm",
        vmin=-emax,
        vmax=emax,
    )
    _imshow(
        axes[1, 1],
        err_rich,
        extent,
        r"$u_{\mathrm{Rich}}-u_{\mathrm{exact}}$"
        f"\nrel$_u$={l2_norm(err_rich, grid) / l2_norm(u_exact, grid):.2e}",
        cmap="coolwarm",
        vmin=-emax,
        vmax=emax,
    )
    _imshow(
        axes[1, 2], r_nn, extent, r"residual $f-A u_{\mathrm{CNN}}$", cmap="coolwarm"
    )

    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_residual_history(
    out: Path,
    grid: SquareGrid,
    a: np.ndarray,
    f: np.ndarray,
    model: LGFCNN,
    title: str,
) -> None:
    device = model.bank.lambdas.device
    a_t = torch.as_tensor(a[None], dtype=torch.float64, device=device)
    f_t = torch.as_tensor(f[None], dtype=torch.float64, device=device)
    info = run_with_intermediates(model, a_t, f_t)

    f_norm = l2_norm(f, grid)
    cnn_res = []
    for u in info["hist_u"]:
        r = f - apply_div_a_grad(u, a, grid)
        cnn_res.append(l2_norm(r, grid) / f_norm)

    rich_res = solve_axy_richardson(
        a,
        f,
        grid,
        omega=1.0,
        tol=0.0,
        maxiter=model.K,
    ).residual_history

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ks = np.arange(0, model.K + 1)
    ax.semilogy(ks, cnn_res, "o-", label="LGF-CNN", color="#1f77b4")
    ax.semilogy(ks, rich_res, "s--", label="Richardson (LGF)", color="#ff7f0e")
    ax.set_xlabel("iteration / block $k$")
    ax.set_ylabel(r"relative residual $\|r\|_2/\|f\|_2$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_mixer_fields(
    out: Path,
    grid: SquareGrid,
    info: dict,
    a: np.ndarray,
    block: int = 0,
) -> None:
    extent = _extent(grid)
    alpha = info["alphas"][block]  # (J, n, n)
    eta = info["etas"][block]
    lambdas = info["lambdas"]
    J = alpha.shape[0]

    ncols = min(max(J, 2), 3)
    nrows = int(np.ceil(J / ncols)) + 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), constrained_layout=True
    )
    axes = np.asarray(axes, dtype=object).reshape(nrows, ncols)

    _imshow(axes[0, 0], a, extent, r"$a(x,y)$", cmap="plasma")
    _imshow(
        axes[0, 1],
        eta,
        extent,
        rf"$\eta(x)$ (block {block})",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    for c in range(2, ncols):
        axes[0, c].axis("off")

    for j in range(J):
        r = 1 + j // ncols
        c = j % ncols
        lam = lambdas[j]
        label = r"$\lambda=0$ (Poisson)" if lam == 0 else rf"$\lambda={lam:.3g}$"
        _imshow(
            axes[r, c],
            alpha[j],
            extent,
            rf"$\alpha_{{{j}}}$  {label}",
            cmap="magma",
            vmin=0,
            vmax=1,
        )

    for j in range(J, (nrows - 1) * ncols):
        r = 1 + j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    fig.suptitle(f"LGF-CNN mixing fields at block {block}", fontsize=12)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_alpha_mean_bars(out: Path, info: dict) -> None:
    lambdas = info["lambdas"]
    means = [alpha.mean(axis=(1, 2)) for alpha in info["alphas"]]  # list of (J,)
    means = np.stack(means, axis=0)  # (K, J)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    K = means.shape[0]
    x = np.arange(K)
    width = 0.8 / means.shape[1]
    for j in range(means.shape[1]):
        lam = lambdas[j]
        label = "Poisson" if lam == 0 else f"λ={lam:.3g}"
        ax.bar(x + j * width, means[:, j], width=width, label=label)
    ax.set_xlabel("block k")
    ax.set_ylabel(r"domain-mean $\alpha_j$")
    ax.set_title("Average mixing weights per block")
    ax.set_xticks(x + 0.4 * (means.shape[1] - 1) * width)
    ax.set_xticklabels([str(k) for k in range(K)])
    ax.legend(fontsize=8, ncols=2)
    ax.set_ylim(0, 1)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="checkpoint to plot; pass an explicit best.pt or last.pt",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/lgf_cnn_axy/figures"),
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, device=args.device)
    grid = model.grid
    contrast = cfg.get("contrast", 0.8)
    data_kind = cfg.get("data_kind", "mms")
    device = torch.device(args.device)

    # --- Held-out random sample ---
    rng = np.random.default_rng(args.seed)
    a_t, f_t, u_t = make_batch(
        grid,
        model.ops,
        batch_size=1,
        contrast=contrast,
        rng=rng,
        device=device,
        dtype=torch.float64,
        data_kind=data_kind,
    )
    info = run_with_intermediates(model, a_t, f_t)
    a = a_t.cpu().numpy()[0]
    f = f_t.cpu().numpy()[0]
    u_exact = u_t.cpu().numpy()[0]
    rich = solve_axy_richardson(a, f, grid, omega=1.0, tol=0.0, maxiter=model.K)

    plot_solution_panel(
        args.outdir / "heldout_solution.png",
        grid,
        a,
        u_exact,
        info["u"],
        rich.u,
        info["r_final"],
        title=f"Held-out random sample (contrast={contrast}, K={model.K}, n={grid.n})",
    )
    plot_residual_history(
        args.outdir / "heldout_residual_history.png",
        grid,
        a,
        f,
        model,
        title="Held-out: residual vs iteration",
    )
    plot_mixer_fields(
        args.outdir / "heldout_mixer_block0.png",
        grid,
        info,
        a,
        block=0,
    )
    plot_mixer_fields(
        args.outdir / "heldout_mixer_block_last.png",
        grid,
        info,
        a,
        block=model.K - 1,
    )
    plot_alpha_mean_bars(args.outdir / "heldout_alpha_means.png", info)

    # --- Manufactured smooth_sine ---
    prob = make_problem(
        "smooth_sine",
        n=grid.n,
        x_min=grid.x_min,
        x_max=grid.x_max,
        forcing="discrete",
    )
    a_t = torch.as_tensor(prob.a[None], dtype=torch.float64, device=device)
    f_t = torch.as_tensor(prob.f[None], dtype=torch.float64, device=device)
    info = run_with_intermediates(model, a_t, f_t)
    rich = solve_axy_richardson(
        prob.a, prob.f, grid, omega=1.0, tol=0.0, maxiter=model.K
    )
    plot_solution_panel(
        args.outdir / "smooth_sine_solution.png",
        grid,
        prob.a,
        prob.u_exact,
        info["u"],
        rich.u,
        info["r_final"],
        title=f"Manufactured smooth_sine (discrete f, K={model.K})",
    )
    plot_residual_history(
        args.outdir / "smooth_sine_residual_history.png",
        grid,
        prob.a,
        prob.f,
        model,
        title="smooth_sine: residual vs iteration",
    )

    print(f"\nAll figures in {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
