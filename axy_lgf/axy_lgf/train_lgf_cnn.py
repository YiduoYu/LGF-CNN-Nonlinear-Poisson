#!/usr/bin/env python3
"""Train / evaluate LGF-CNN on rectangle a(x,y) problems.

Usage:
    python -m axy_lgf.train_lgf_cnn
    python -m axy_lgf.train_lgf_cnn --epochs 100 --n 31 --K 6
    python -m axy_lgf.train_lgf_cnn --data-kind solver
    python -m axy_lgf.train_lgf_cnn --predict \
        --predict-from axy_lgf/checkpoints/lgf_cnn_axy/<key>/best.pt

Data kinds (see data.py):
    mms    sample u_exact, derive f = A_h u_exact   (default)
    solver sample f, solve A_h u = f by PCG for u_exact

Checkpoint / result bookkeeping:
  - run directory  <checkpoint-root>/<experiment_key>/
  - best.pt selected on a fixed held-out validation batch, last.pt at the end
  - history.csv; --retrain guards an existing best.pt
  - --predict writes per-case u1-u4 metrics.csv + summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import make_batch, sample_coefficient
from .grid import SquareGrid
from .lgf_cnn import LGFCNN, LGFCNNTrainConfig
from .operators import apply_div_a_grad, l2_norm
from .manufactured import make_problem
from .solvers import solve_axy_pcg, solve_axy_richardson

MODEL_TYPE = "lgf_cnn_axy"


def make_model(cfg: LGFCNNTrainConfig) -> LGFCNN:
    device = torch.device(cfg.device)
    model = LGFCNN(
        SquareGrid(n=cfg.n),
        J=cfg.J,
        K=cfg.K,
        hidden=cfg.hidden,
        eta_max=cfg.eta_max,
        share_weights=True,
        dtype=torch.float64,
        device=device,
    )
    return model.to(device)


def relative_metrics(
    model: LGFCNN,
    a: torch.Tensor,
    f: torch.Tensor,
    u_exact: torch.Tensor,
    u_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample relative residual and relative solution error, (B,) each."""
    res = model.residual(u_pred, a, f)
    rel_r = model.ops.l2_norm(res) / model.ops.l2_norm(f).clamp_min(1e-14)
    rel_u = model.ops.l2_norm(u_pred - u_exact) / model.ops.l2_norm(u_exact).clamp_min(
        1e-14
    )
    return rel_r, rel_u


def experiment_key(cfg: LGFCNNTrainConfig) -> str:
    """Readable run tag plus a digest of every other training-affecting arg."""
    visible = (
        f"{cfg.data_kind}_n{cfg.n}_J{cfg.J}_K{cfg.K}_h{cfg.hidden}_c{cfg.contrast:g}"
    )
    affecting = (
        cfg.epochs,
        cfg.batch_size,
        cfg.samples_per_epoch,
        cfg.lr,
        cfg.eta_max,
        cfg.lambda_u,
        cfg.lambda_r,
        cfg.eval_every,
        cfg.val_samples,
        cfg.seed,
    )
    digest = hashlib.md5(repr(affecting).encode()).hexdigest()[:8]
    return f"{visible}_{digest}"


def save_checkpoint(
    path: Path,
    model: LGFCNN,
    cfg: LGFCNNTrainConfig,
    epoch: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_type": MODEL_TYPE,
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_model(ckpt_path: Path, device: str) -> tuple[LGFCNN, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_type = ckpt.get("model_type")
    # Checkpoints written before MODEL_TYPE existed carry no tag; accept them.
    if model_type is not None and model_type != MODEL_TYPE:
        raise ValueError(f"checkpoint type must be {MODEL_TYPE!r}, got {model_type!r}")
    cfg = LGFCNNTrainConfig(**{**ckpt["config"], "device": device})
    model = make_model(cfg)
    state_dict = dict(ckpt["state_dict"])
    # Legacy checkpoints registered the same TorchSquareOps module twice and
    # stored a dense DST matrix. Neither is needed by the FFT implementation.
    for legacy_key in ("ops.S", "ops.mu", "bank.ops.S"):
        state_dict.pop(legacy_key, None)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt["config"]


def _validate_train_config(cfg: LGFCNNTrainConfig) -> None:
    positive_ints = {
        "n": cfg.n,
        "J": cfg.J,
        "K": cfg.K,
        "hidden": cfg.hidden,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "samples_per_epoch": cfg.samples_per_epoch,
        "eval_every": cfg.eval_every,
        "val_samples": cfg.val_samples,
    }
    invalid = [name for name, value in positive_ints.items() if value < 1]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be positive")
    finite_nonnegative = {
        "contrast": cfg.contrast,
        "lambda_u": cfg.lambda_u,
        "lambda_r": cfg.lambda_r,
    }
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if cfg.contrast > 1.0:
        raise ValueError("contrast must lie in [0, 1]")
    if not math.isfinite(cfg.lr) or cfg.lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if cfg.lambda_u == 0.0 and cfg.lambda_r == 0.0:
        raise ValueError("at least one loss weight must be positive")


def _batch_sizes(total: int, batch_size: int) -> list[int]:
    """Split an epoch into exact batch sizes, including a final short batch."""
    if total < 1 or batch_size < 1:
        raise ValueError("total and batch_size must be positive")
    full, remainder = divmod(total, batch_size)
    return [batch_size] * full + ([remainder] if remainder else [])


def _relative_l2(
    error: np.ndarray,
    reference: np.ndarray,
    grid: SquareGrid,
) -> float:
    denominator = max(l2_norm(reference, grid), np.finfo(np.float64).tiny)
    return l2_norm(error, grid) / denominator


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            [
                {key: ("" if value is None else value) for key, value in row.items()}
                for row in rows
            ]
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _validate(
    model: LGFCNN,
    val_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    model.eval()
    a, f, u_exact = val_batch
    rel_r, rel_u = relative_metrics(model, a, f, u_exact, model(a, f))
    return float(rel_r.mean()), float(rel_u.mean())


def train(
    cfg: LGFCNNTrainConfig,
    run_dir: Path,
) -> LGFCNN:
    _validate_train_config(cfg)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    grid = SquareGrid(n=cfg.n)
    model = make_model(cfg)

    # Fixed validation batch for best.pt selection. The final test evaluation
    # uses a separate user-configurable seed.
    val_batch = make_batch(
        grid,
        model.ops,
        batch_size=cfg.val_samples,
        contrast=cfg.contrast,
        rng=np.random.default_rng(cfg.seed + 1),
        device=device,
        dtype=torch.float64,
        data_kind=cfg.data_kind,
    )

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    print(
        f"[train] data={cfg.data_kind}  n={cfg.n}  J={cfg.J}  K={cfg.K}  "
        f"epochs={cfg.epochs}  batch={cfg.batch_size}  device={cfg.device}"
    )
    print(
        f"        trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(f"        run dir: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_metric = float("inf")
    history: list[dict] = []

    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_res = 0.0
        epoch_uerr = 0.0

        for current_batch_size in _batch_sizes(
            cfg.samples_per_epoch,
            cfg.batch_size,
        ):
            a, f, u_exact = make_batch(
                grid,
                model.ops,
                batch_size=current_batch_size,
                contrast=cfg.contrast,
                rng=rng,
                device=device,
                dtype=torch.float64,
                data_kind=cfg.data_kind,
            )
            u_pred = model(a, f)
            rel_r, rel_u = relative_metrics(model, a, f, u_exact, u_pred)

            loss = cfg.lambda_r * torch.mean(rel_r**2) + cfg.lambda_u * torch.mean(
                rel_u**2
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach()) * current_batch_size
            epoch_res += float(rel_r.sum().detach())
            epoch_uerr += float(rel_u.sum().detach())

        epoch_loss /= cfg.samples_per_epoch
        epoch_res /= cfg.samples_per_epoch
        epoch_uerr /= cfg.samples_per_epoch

        val_rel_res = None
        val_rel_u = None
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val_rel_res, val_rel_u = _validate(model, val_batch)
            val_metric = cfg.lambda_r * val_rel_res**2 + cfg.lambda_u * val_rel_u**2
            if val_metric < best_metric:
                best_metric = val_metric
                save_checkpoint(best_path, model, cfg, epoch, best_metric)
            print(
                f"  epoch {epoch:4d}  loss={epoch_loss:.3e}  "
                f"rel_res={epoch_res:.3e}  rel_u={epoch_uerr:.3e}  "
                f"val_rel_res={val_rel_res:.3e}  val_rel_u={val_rel_u:.3e}  "
                f"val_loss={val_metric:.3e}"
            )

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "rel_res": epoch_res,
                "rel_u": epoch_uerr,
                "val_rel_res": val_rel_res,
                "val_rel_u": val_rel_u,
                "val_loss": val_metric if val_rel_u is not None else None,
            }
        )

    elapsed = time.perf_counter() - t0
    print(f"[train] done in {elapsed:.1f}s  best val_loss={best_metric:.3e}")

    save_checkpoint(run_dir / "last.pt", model, cfg, cfg.epochs, best_metric)
    _write_csv(history, run_dir / "history.csv")
    print(f"[train] saved {best_path} / last.pt / history.csv")

    # Continue with the best validation checkpoint.
    model, _ = load_model(best_path, cfg.device)
    return model


@torch.no_grad()
def evaluate(
    model: LGFCNN,
    n_eval: int = 16,
    contrast: float = 0.5,
    seed: int = 123,
    richardson_steps: int | None = None,
    data_kind: str = "mms",
) -> None:
    """Compare LGF-CNN vs Richardson (same K) and PCG on held-out batches."""
    if n_eval < 1:
        raise ValueError("n_eval must be positive")
    device = model.bank.lambdas.device
    grid = model.grid
    rng = np.random.default_rng(seed)
    K = model.K
    if richardson_steps is None:
        richardson_steps = K
    if richardson_steps < 0:
        raise ValueError("richardson_steps must be nonnegative")

    model.eval()
    a, f, u_exact = make_batch(
        grid,
        model.ops,
        batch_size=n_eval,
        contrast=contrast,
        rng=rng,
        device=device,
        dtype=torch.float64,
        data_kind=data_kind,
    )
    rel_r, rel_u = relative_metrics(model, a, f, u_exact, model(a, f))
    rel_r_nn = rel_r.mean()
    rel_u_nn = rel_u.mean()

    # Classical baselines on the same batch (numpy path)
    rel_r_rich = []
    rel_u_rich = []
    rel_r_pcg = []
    rel_u_pcg = []
    iters_pcg = []
    for b in range(n_eval):
        aa = a[b].cpu().numpy()
        ff = f[b].cpu().numpy()
        ue = u_exact[b].cpu().numpy()

        rich = solve_axy_richardson(
            aa, ff, grid, omega=1.0, tol=0.0, maxiter=richardson_steps
        )
        rel_r_rich.append(rich.residual_history[-1])
        rel_u_rich.append(_relative_l2(rich.u - ue, ue, grid))

        if data_kind == "mms":
            pcg = solve_axy_pcg(aa, ff, grid, tol=1e-10, maxiter=500)
            rel_r_pcg.append(pcg.residual_history[-1])
            rel_u_pcg.append(_relative_l2(pcg.u - ue, ue, grid))
            iters_pcg.append(pcg.iterations)
        else:
            # In solver mode ue is already the PCG-generated reference. Do not
            # rerun the same algorithm and report the tautological zero error.
            reference_residual = ff - apply_div_a_grad(ue, aa, grid)
            rel_r_pcg.append(_relative_l2(reference_residual, ff, grid))

    print(f"[eval] held-out random batch (data={data_kind})")
    print(f"  LGF-CNN (K={K}):          rel_res={rel_r_nn:.3e}  rel_u={rel_u_nn:.3e}")
    print(
        f"  Richardson (K={richardson_steps}):  rel_res={np.mean(rel_r_rich):.3e}  "
        f"rel_u={np.mean(rel_u_rich):.3e}"
    )
    if data_kind == "mms":
        print(
            f"  PCG (to 1e-10):         rel_res={np.mean(rel_r_pcg):.3e}  "
            f"rel_u={np.mean(rel_u_pcg):.3e}  "
            f"mean_iters={np.mean(iters_pcg):.1f}"
        )
    else:
        print(f"  PCG reference labels:   rel_res={np.mean(rel_r_pcg):.3e}  rel_u=n/a")

    # Fixed manufactured problem
    prob = make_problem("smooth_sine", n=grid.n, forcing="discrete")
    a_t = torch.as_tensor(prob.a[None], dtype=torch.float64, device=device)
    f_t = torch.as_tensor(prob.f[None], dtype=torch.float64, device=device)
    u_t = torch.as_tensor(prob.u_exact[None], dtype=torch.float64, device=device)
    rel_r_t, rel_u_t = relative_metrics(model, a_t, f_t, u_t, model(a_t, f_t))
    rich = solve_axy_richardson(prob.a, prob.f, grid, omega=1.0, tol=0.0, maxiter=K)
    r_rich = prob.f - apply_div_a_grad(rich.u, prob.a, grid)
    print("[eval] manufactured smooth_sine (discrete f)")
    print(
        f"  LGF-CNN:     rel_res={float(rel_r_t.mean()):.3e}  rel_u={float(rel_u_t.mean()):.3e}"
    )
    print(
        f"  Richardson:  rel_res={l2_norm(r_rich, grid) / l2_norm(prob.f, grid):.3e}  "
        f"rel_u={l2_norm(rich.u - prob.u_exact, grid) / l2_norm(prob.u_exact, grid):.3e}"
    )


def u1_u4_cases(grid: SquareGrid) -> tuple[list[str], np.ndarray]:
    """Four fixed test solutions that vanish on the boundary of (0, L)^2.

    This makes every case consistent with the zero Dirichlet halo used by the
    discrete operator.
    """
    x, y = grid.mesh()
    L = grid.L

    u1 = np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    u2 = np.sin(2.0 * np.pi * x / L) * np.sin(np.pi * y / L)
    u3 = np.sin(np.pi * x / L) * np.sin(2.0 * np.pi * y / L) - 0.5 * np.sin(
        2.0 * np.pi * x / L
    ) * np.sin(2.0 * np.pi * y / L)
    u4 = np.sin(3.0 * np.pi * x / L) * np.sin(3.0 * np.pi * y / L)

    cases = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}
    return list(cases), np.stack(list(cases.values()), axis=0)


@torch.no_grad()
def run_prediction(args: argparse.Namespace) -> None:
    """Predict fixed MMS u1-u4 with a trained model and save their metrics.

    The u1-u4 cases are fixed manufactured solutions (f = A_h u by
    definition), independent of the data kind the model was trained on.
    """
    if not args.predict_from:
        raise ValueError("--predict requires --predict-from")
    print("[predict] using fixed MMS u1-u4 cases for cross-model comparison")
    model, saved = load_model(args.predict_from, args.device)
    grid = model.grid
    device = model.bank.lambdas.device
    contrast = saved.get("contrast", 0.5)

    names, u_exact_np = u1_u4_cases(grid)
    # Use one fixed coefficient field for all four solutions so the comparison
    # isolates solution complexity rather than coefficient variation.
    a_single = sample_coefficient(
        grid,
        batch_size=1,
        contrast=contrast,
        rng=np.random.default_rng(args.test_seed),
    )
    a_np = np.repeat(a_single, len(names), axis=0)
    a = torch.as_tensor(a_np, dtype=torch.float64, device=device)
    u_exact = torch.as_tensor(u_exact_np, dtype=torch.float64, device=device)
    f = model.ops.apply_div_a_grad(u_exact, a)
    u_pred = model(a, f)
    rel_r, rel_u = relative_metrics(model, a, f, u_exact, u_pred)

    rows = [
        {
            "case": name,
            "rel_res": float(rel_r[i]),
            "rel_u": float(rel_u[i]),
        }
        for i, name in enumerate(names)
    ]
    for row in rows:
        print(
            f"[predict] {row['case']}: rel_res={row['rel_res']:.3e} "
            f"rel_u={row['rel_u']:.3e}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "metrics.csv")
    summary = {
        "model_type": MODEL_TYPE,
        "train_data_kind": saved.get("data_kind", "mms"),
        "prediction_data_kind": "mms",
        "coefficient_policy": "same_field_for_all_cases",
        "n": grid.n,
        "contrast": contrast,
        "test_seed": args.test_seed,
        "checkpoint_sha256": _file_sha256(Path(args.predict_from)),
        "relResMean": float(rel_r.mean()),
        "relResMax": float(rel_r.max()),
        "relUMean": float(rel_u.mean()),
        "relUMax": float(rel_u.max()),
        "cases": {
            row["case"]: {"rel_res": row["rel_res"], "rel_u": row["rel_u"]}
            for row in rows
        },
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    if args.predict_save:
        target = Path(args.predict_save)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "names": names,
                "a": a.cpu(),
                "f": f.cpu(),
                "u_exact": u_exact.cpu(),
                "u_pred": u_pred.cpu(),
                "metrics": summary,
            },
            target,
        )
    print(f"saved metrics to {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--J", type=int, default=6)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples-per-epoch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--contrast", type=float, default=0.8)
    p.add_argument(
        "--data-kind",
        choices=("mms", "solver"),
        default="mms",
        help=(
            "training/evaluation data: mms derives f from u_exact; "
            "solver PCG-solves u_exact from f. Prediction always uses MMS u1-u4"
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("axy_lgf/checkpoints/lgf_cnn_axy"),
        help="run directories are created under this root",
    )
    p.add_argument("--retrain", action="store_true")
    p.add_argument(
        "--eval-only", type=Path, default=None, help="load checkpoint and eval"
    )
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-from", type=Path, default=None)
    p.add_argument(
        "--test-seed",
        type=int,
        default=123,
        help="independent test/prediction seed (validation always uses seed+1)",
    )
    p.add_argument("--predict-save", default="")
    p.add_argument("--output-dir", default="results/lgf_cnn_axy/predict")
    args = p.parse_args()

    if args.predict:
        return run_prediction(args)

    if args.eval_only is not None:
        model, saved = load_model(args.eval_only, args.device)
        evaluate(
            model,
            contrast=saved.get("contrast", 0.5),
            seed=args.test_seed,
            data_kind=saved.get("data_kind", "mms"),
        )
        return

    cfg = LGFCNNTrainConfig(
        n=args.n,
        J=args.J,
        K=args.K,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        eval_every=args.eval_every,
        val_samples=args.val_samples,
        lr=args.lr,
        contrast=args.contrast,
        seed=args.seed,
        device=args.device,
        data_kind=args.data_kind,
    )
    _validate_train_config(cfg)
    run_dir = args.checkpoint_root / experiment_key(cfg)
    if (run_dir / "best.pt").exists() and not args.retrain:
        print(f"found {run_dir / 'best.pt'}; use --retrain to overwrite")
        return
    model = train(cfg, run_dir)
    evaluate(
        model,
        contrast=cfg.contrast,
        seed=args.test_seed,
        data_kind=cfg.data_kind,
    )
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
