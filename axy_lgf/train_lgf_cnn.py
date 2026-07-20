#!/usr/bin/env python3
"""Train / evaluate LGF-CNN on rectangle a(x,y) problems.

Usage:
    python -m axy_lgf.train_lgf_cnn
    python -m axy_lgf.train_lgf_cnn --epochs 100 --n 31 --K 6
    python -m axy_lgf.train_lgf_cnn --data-kind solver
    python -m axy_lgf.train_lgf_cnn --loss-mode supervised
    python -m axy_lgf.train_lgf_cnn --loss-mode physics
    python -m axy_lgf.train_lgf_cnn --predict \
        --predict-from axy_lgf/checkpoints/lgf_cnn_axy/<key>/best.pt

Data kinds (see data.py):
    mms    sample u_exact, derive f = A_h u_exact   (default)
    solver sample f, solve A_h u = f by PCG for u_exact

Checkpoint / result bookkeeping:
  - run directory  <checkpoint-root>/<experiment_key>/
  - best.pt selected on a fixed held-out validation batch, last.pt at the end
  - history.csv; --retrain guards an existing best.pt
  - result/<experiment_key>_s<test-seed>/{data,figures}/
  - complete u1-u4 and held-out metrics, solver comparisons, histories, tensors
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from .data import make_batch, sample_coefficient
from .evaluation import evaluate_batch
from .grid import DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN, SquareGrid
from .lgf_cnn import LGFCNN, LGFCNNTrainConfig
from .operators import apply_div_a_grad, l2_norm
from .manufactured import make_problem
from .solvers import solve_axy_richardson

MODEL_TYPE = "lgf_cnn_axy"
CHECKPOINT_FORMAT_VERSION = 3
EXPERIMENT_VERSION = 3
PROJECT_DIR = Path(__file__).resolve().parent
RUN_ARTIFACTS = ("best.pt", "last.pt", "history.csv")
RUN_MANIFEST = "completed.json"
MAX_TORCH_SEED = 2**64 - 1
DEFAULT_PREDICT_OUTPUT_DIR = None


def make_model(cfg: LGFCNNTrainConfig) -> LGFCNN:
    device = torch.device(cfg.device)
    model = LGFCNN(
        SquareGrid(n=cfg.n, x_min=cfg.domain_min, x_max=cfg.domain_max),
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


def training_loss(
    model: LGFCNN,
    a: torch.Tensor,
    f: torch.Tensor,
    u_exact: torch.Tensor,
    u_pred: torch.Tensor,
    loss_mode: str,
) -> torch.Tensor:
    """Single training objective: squared discrete L2 solution error or residual."""
    if loss_mode == "supervised":
        error = u_pred - u_exact
    elif loss_mode == "physics":
        error = model.residual(u_pred, a, f)
    else:
        raise ValueError("loss_mode must be 'supervised' or 'physics'")
    return torch.mean(model.ops.l2_norm(error) ** 2)


def _training_config_payload(cfg: LGFCNNTrainConfig) -> dict:
    """Canonical result-affecting config used for run identity checks."""
    payload = asdict(cfg)
    # Runtime placement does not define the trained model. The lambda fields
    # remain only for loading legacy checkpoints and are not used by training.
    payload.pop("device", None)
    payload.pop("lambda_u", None)
    payload.pop("lambda_r", None)
    for name in (
        "n",
        "J",
        "K",
        "hidden",
        "epochs",
        "batch_size",
        "samples_per_epoch",
        "eval_every",
        "val_samples",
        "seed",
    ):
        payload[name] = int(payload[name])
    for name in (
        "eta_max",
        "lr",
        "contrast",
        "domain_min",
        "domain_max",
    ):
        payload[name] = float(payload[name])
    payload["experiment_version"] = EXPERIMENT_VERSION
    return payload


def _payload_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def experiment_key(cfg: LGFCNNTrainConfig) -> str:
    """Readable run tag plus a digest of every training-affecting argument."""
    visible = (
        f"{cfg.loss_mode}_{cfg.data_kind}_n{cfg.n}_J{cfg.J}_K{cfg.K}_"
        f"h{cfg.hidden}_c{cfg.contrast:g}_"
        f"dom{cfg.domain_min:g}to{cfg.domain_max:g}"
    )
    digest = _payload_sha256(_training_config_payload(cfg))[:12]
    return f"{visible}_{digest}"


def save_checkpoint(
    path: Path,
    model: LGFCNN,
    cfg: LGFCNNTrainConfig,
    epoch: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "experiment_version": EXPERIMENT_VERSION,
                "model_type": MODEL_TYPE,
                "config": asdict(cfg),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(ckpt_path: Path, device: str) -> tuple[LGFCNN, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    format_version = ckpt.get("format_version", 1)
    if (
        type(format_version) is not int
        or format_version < 1
        or format_version > CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            f"unsupported checkpoint format_version={format_version!r}; "
            f"maximum supported version is {CHECKPOINT_FORMAT_VERSION}"
        )
    checkpoint_experiment_version = ckpt.get("experiment_version")
    invalid_experiment_version = checkpoint_experiment_version is not None and (
        type(checkpoint_experiment_version) is not int
        or checkpoint_experiment_version < 1
        or checkpoint_experiment_version > EXPERIMENT_VERSION
    )
    if (format_version >= 3 and checkpoint_experiment_version is None) or (
        invalid_experiment_version
    ):
        raise ValueError(
            "unsupported checkpoint experiment_version="
            f"{checkpoint_experiment_version!r}; maximum supported version "
            f"is {EXPERIMENT_VERSION}"
        )
    model_type = ckpt.get("model_type")
    # Checkpoints written before MODEL_TYPE existed carry no tag; accept them.
    if model_type is not None and model_type != MODEL_TYPE:
        raise ValueError(f"checkpoint type must be {MODEL_TYPE!r}, got {model_type!r}")
    saved_config = dict(ckpt["config"])
    # Checkpoints created before configurable domains used Omega=(0, 1)^2.
    saved_config.setdefault("domain_min", 0.0)
    saved_config.setdefault("domain_max", 1.0)
    # Before pure loss modes were introduced, training used a mixed objective.
    saved_config.setdefault("loss_mode", "legacy_mixed")
    cfg = LGFCNNTrainConfig(**{**saved_config, "device": device})
    model = make_model(cfg)
    state_dict = dict(ckpt["state_dict"])
    # Legacy checkpoints registered the same TorchSquareOps module twice and
    # stored a dense DST matrix. Neither is needed by the FFT implementation.
    for legacy_key in ("ops.S", "ops.mu", "bank.ops.S"):
        state_dict.pop(legacy_key, None)
    model.load_state_dict(state_dict)
    model.eval()
    return model, saved_config


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
    wrong_types = [
        name
        for name, value in positive_ints.items()
        if not isinstance(value, Integral) or isinstance(value, bool)
    ]
    if wrong_types:
        raise ValueError(f"{', '.join(wrong_types)} must be integers")
    invalid = [name for name, value in positive_ints.items() if value < 1]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be positive")
    if cfg.n < 2:
        raise ValueError("n must be at least 2")
    if (
        not isinstance(cfg.seed, Integral)
        or isinstance(cfg.seed, bool)
        or cfg.seed < 0
        or cfg.seed > MAX_TORCH_SEED
    ):
        raise ValueError(f"seed must be an integer in [0, {MAX_TORCH_SEED}]")
    finite_nonnegative = {
        "contrast": cfg.contrast,
    }
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if cfg.contrast > 1.0:
        raise ValueError("contrast must lie in [0, 1]")
    if not math.isfinite(cfg.lr) or cfg.lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(cfg.eta_max) or cfg.eta_max <= 0.0:
        raise ValueError("eta_max must be finite and positive")
    if not math.isfinite(cfg.domain_min) or not math.isfinite(cfg.domain_max):
        raise ValueError("domain bounds must be finite")
    if cfg.domain_max <= cfg.domain_min:
        raise ValueError("domain_max must be greater than domain_min")
    if not math.isfinite(cfg.domain_max - cfg.domain_min):
        raise ValueError("domain length must be finite")
    if cfg.loss_mode not in ("supervised", "physics"):
        raise ValueError("loss_mode must be 'supervised' or 'physics'")
    if cfg.data_kind not in ("mms", "solver"):
        raise ValueError("data_kind must be 'mms' or 'solver'")


def _batch_sizes(total: int, batch_size: int) -> list[int]:
    """Split an epoch into exact batch sizes, including a final short batch."""
    if total < 1 or batch_size < 1:
        raise ValueError("total and batch_size must be positive")
    full, remainder = divmod(total, batch_size)
    return [batch_size] * full + ([remainder] if remainder else [])


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                [
                    {
                        key: ("" if value is None else value)
                        for key, value in row.items()
                    }
                    for row in rows
                ]
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as file:
            json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_prediction_tensor(bundle: dict, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "names": bundle["names"],
                "a": torch.from_numpy(bundle["a"]),
                "f": torch.from_numpy(bundle["f"]),
                "u_exact": torch.from_numpy(bundle["u_exact"]),
                "u_pred": torch.from_numpy(bundle["u_pred"]),
                "u_linear_baseline": torch.from_numpy(
                    bundle["u_linear_baseline"]
                ),
                "u_richardson": torch.from_numpy(bundle["u_richardson"]),
                "u_pcg": torch.from_numpy(bundle["u_pcg"]),
                "metrics": bundle["metrics"],
                "metadata": metadata,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_evaluation_outputs(
    bundle: dict,
    data_dir: Path,
    prefix: str,
    metadata: dict,
) -> Path:
    """Save one evaluation set as CSV, JSON, and a tensor archive."""
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(bundle["metrics"], data_dir / f"{prefix}_metrics.csv")
    _write_csv(
        bundle["solver_comparison"],
        data_dir / f"{prefix}_solver_comparison.csv",
    )
    _write_csv(
        bundle["residual_history"],
        data_dir / f"{prefix}_residual_history.csv",
    )
    _write_json_atomic(
        {
            **metadata,
            "metrics": bundle["metric_summary"],
            "solvers": bundle["solver_summary"],
        },
        data_dir / f"{prefix}_summary.json",
    )
    predictions_path = data_dir / f"{prefix}_predictions.pt"
    _save_prediction_tensor(bundle, predictions_path, metadata)
    return predictions_path


def _copy_training_history(history_path: Path, data_dir: Path) -> None:
    if not history_path.is_file():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "training_history.csv"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(history_path, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _completion_payload(cfg: LGFCNNTrainConfig, run_dir: Path) -> dict:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "config_sha256": _payload_sha256(_training_config_payload(cfg)),
        "artifacts": {
            name: _file_sha256(run_dir / name) for name in RUN_ARTIFACTS
        },
    }


def _run_completion_issue(run_dir: Path, cfg: LGFCNNTrainConfig) -> str | None:
    manifest_path = run_dir / RUN_MANIFEST
    if not manifest_path.is_file():
        return f"{RUN_MANIFEST} is missing"
    try:
        with manifest_path.open() as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        return f"{RUN_MANIFEST} cannot be read: {exc}"
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        return "identity: experiment version does not match the current code"
    expected_config = _payload_sha256(_training_config_payload(cfg))
    if manifest.get("config_sha256") != expected_config:
        return "identity: training configuration digest does not match"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return "artifact hashes are missing from completed.json"
    for name in RUN_ARTIFACTS:
        path = run_dir / name
        expected_hash = artifacts.get(name)
        if not path.is_file():
            return f"{name} is missing"
        if not isinstance(expected_hash, str) or _file_sha256(path) != expected_hash:
            return f"{name} does not match completed.json"
    return None


def _promote_attempt(
    attempt_dir: Path,
    run_dir: Path,
    cfg: LGFCNNTrainConfig,
) -> None:
    missing = [name for name in RUN_ARTIFACTS if not (attempt_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"training attempt is missing: {', '.join(missing)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # The marker is committed last. Any interruption during promotion therefore
    # leaves a run that normal execution will refuse to reuse.
    (run_dir / RUN_MANIFEST).unlink(missing_ok=True)
    for name in RUN_ARTIFACTS:
        os.replace(attempt_dir / name, run_dir / name)
    _write_json_atomic(_completion_payload(cfg, run_dir), run_dir / RUN_MANIFEST)


@contextmanager
def _run_lock(checkpoint_root: Path, run_key: str):
    """Serialize writers for one experiment; the OS releases the lock on exit."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_root / f".{run_key}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cleanup_stale_attempts(checkpoint_root: Path, run_key: str) -> None:
    """Remove orphan attempts left by a process killed without normal cleanup."""
    prefix = f".{run_key}.attempt-"
    for path in checkpoint_root.iterdir():
        if path.name.startswith(prefix) and path.is_dir():
            shutil.rmtree(path)


@torch.no_grad()
def _validate(
    model: LGFCNN,
    val_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    loss_mode: str,
) -> tuple[float, float, float, float]:
    model.eval()
    a, f, u_exact = val_batch
    u_pred = model(a, f)
    rel_r, rel_u = relative_metrics(model, a, f, u_exact, u_pred)
    loss = training_loss(model, a, f, u_exact, u_pred, loss_mode)
    residual = model.residual(u_pred, a, f)
    fd_res_rmse = torch.sqrt(torch.mean(residual**2))
    return (
        float(loss),
        float(rel_r.mean()),
        float(rel_u.mean()),
        float(fd_res_rmse),
    )


def train(
    cfg: LGFCNNTrainConfig,
    run_dir: Path,
) -> LGFCNN:
    _validate_train_config(cfg)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    grid = SquareGrid(n=cfg.n, x_min=cfg.domain_min, x_max=cfg.domain_max)
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
        f"[train] loss={cfg.loss_mode}  data={cfg.data_kind}  "
        f"n={cfg.n}  J={cfg.J}  K={cfg.K}  "
        f"domain=[{cfg.domain_min:g}, {cfg.domain_max:g}]  "
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

            loss = training_loss(model, a, f, u_exact, u_pred, cfg.loss_mode)
            if not bool(torch.isfinite(loss).detach()):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}: {float(loss.detach())}"
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
        val_fd_res_rmse = None
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val_metric, val_rel_res, val_rel_u, val_fd_res_rmse = _validate(
                model, val_batch, cfg.loss_mode
            )
            if not math.isfinite(val_metric):
                raise FloatingPointError(
                    f"non-finite validation loss at epoch {epoch}: {val_metric}"
                )
            if val_metric < best_metric:
                best_metric = val_metric
                save_checkpoint(best_path, model, cfg, epoch, best_metric)
            print(
                f"  epoch {epoch:4d}  loss={epoch_loss:.3e}  "
                f"rel_res={epoch_res:.3e}  rel_u={epoch_uerr:.3e}  "
                f"val_rel_res={val_rel_res:.3e}  val_rel_u={val_rel_u:.3e}  "
                f"val_fdResRMSE={val_fd_res_rmse:.3e}  "
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
                "val_fd_res_rmse": val_fd_res_rmse,
                "val_loss": val_metric if val_rel_u is not None else None,
            }
        )

    elapsed = time.perf_counter() - t0
    print(f"[train] done in {elapsed:.1f}s  best val_loss={best_metric:.3e}")

    save_checkpoint(run_dir / "last.pt", model, cfg, cfg.epochs, best_metric)
    _write_csv(history, run_dir / "history.csv")
    print(f"[train] saved {best_path} / last.pt / history.csv")

    # Continue with the best validation checkpoint.
    if not best_path.is_file():
        raise RuntimeError("training finished without producing best.pt")
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
    output_dir: Path | None = None,
    checkpoint_path: Path | None = None,
) -> dict:
    """Compare LGF-CNN vs Richardson (same K) and PCG on held-out batches."""
    if n_eval < 1:
        raise ValueError("n_eval must be positive")
    device = model.bank.lambdas.device
    grid = model.grid
    rng = np.random.default_rng(seed)
    if richardson_steps is None:
        richardson_steps = model.K
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
    names = [f"sample_{index:03d}" for index in range(n_eval)]
    bundle = evaluate_batch(
        model,
        names,
        a,
        f,
        u_exact,
        richardson_steps=richardson_steps,
    )
    print(f"[eval] held-out random batch (data={data_kind})")
    for method in ("LGF-CNN", "Richardson", "PCG"):
        summary = bundle["solver_summary"][method]
        print(
            f"  {method:12s}  rel_res={summary['relRes']['mean']:.3e}  "
            f"rel_u={summary['relL2']['mean']:.3e}  "
            f"time={summary['time_ms']['mean']:.3f} ms/sample"
        )

    if output_dir is not None:
        metadata = {
            "model_type": MODEL_TYPE,
            "evaluation": "heldout_random",
            "data_kind": data_kind,
            "n": grid.n,
            "domain": [grid.x_min, grid.x_max],
            "h": grid.h,
            "contrast": contrast,
            "test_seed": seed,
            "samples": n_eval,
            "richardson_steps": richardson_steps,
            "pcg_tolerance": 1e-10,
        }
        if checkpoint_path is not None:
            metadata["checkpoint_sha256"] = _file_sha256(checkpoint_path)
        _save_evaluation_outputs(
            bundle,
            output_dir,
            "heldout",
            metadata,
        )
        print(f"[eval] data saved to {output_dir}")

    # Fixed manufactured problem
    prob = make_problem(
        "smooth_sine",
        n=grid.n,
        x_min=grid.x_min,
        x_max=grid.x_max,
        forcing="discrete",
    )
    a_t = torch.as_tensor(prob.a[None], dtype=torch.float64, device=device)
    f_t = torch.as_tensor(prob.f[None], dtype=torch.float64, device=device)
    u_t = torch.as_tensor(prob.u_exact[None], dtype=torch.float64, device=device)
    rel_r_t, rel_u_t = relative_metrics(model, a_t, f_t, u_t, model(a_t, f_t))
    rich = solve_axy_richardson(
        prob.a,
        prob.f,
        grid,
        omega=1.0,
        tol=0.0,
        maxiter=model.K,
    )
    r_rich = prob.f - apply_div_a_grad(rich.u, prob.a, grid)
    print("[eval] manufactured smooth_sine (discrete f)")
    print(
        f"  LGF-CNN:     rel_res={float(rel_r_t.mean()):.3e}  rel_u={float(rel_u_t.mean()):.3e}"
    )
    print(
        f"  Richardson:  rel_res={l2_norm(r_rich, grid) / l2_norm(prob.f, grid):.3e}  "
        f"rel_u={l2_norm(rich.u - prob.u_exact, grid) / l2_norm(prob.u_exact, grid):.3e}"
    )
    return bundle


def u1_u4_cases(grid: SquareGrid) -> tuple[list[str], np.ndarray]:
    """Four fixed test solutions that vanish on the configured square boundary.

    This makes every case consistent with the zero Dirichlet halo used by the
    discrete operator.
    """
    x, y = grid.mesh()
    L = grid.L
    xi = (x - grid.x_min) / L
    yi = (y - grid.x_min) / L

    u1 = np.sin(np.pi * xi) * np.sin(np.pi * yi)
    u2 = np.sin(2.0 * np.pi * xi) * np.sin(np.pi * yi)
    u3 = np.sin(np.pi * xi) * np.sin(2.0 * np.pi * yi) - 0.5 * np.sin(
        2.0 * np.pi * xi
    ) * np.sin(2.0 * np.pi * yi)
    u4 = np.sin(3.0 * np.pi * xi) * np.sin(3.0 * np.pi * yi)

    cases = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}
    return list(cases), np.stack(list(cases.values()), axis=0)


def _result_dir(result_root: Path, run_name: str, test_seed: int) -> Path:
    """One compact, seed-isolated result directory per trained model."""
    return result_root / f"{run_name}_s{test_seed}"


def _checkpoint_run_name(checkpoint_path: Path) -> str:
    parent = checkpoint_path.parent
    return parent.parent.name if parent.name == "checkpoints" else parent.name


@torch.no_grad()
def _evaluate_u1_u4(
    model: LGFCNN,
    contrast: float,
    seed: int,
) -> dict:
    grid = model.grid
    device = model.bank.lambdas.device
    names, u_exact_np = u1_u4_cases(grid)
    a_single = sample_coefficient(
        grid,
        batch_size=1,
        contrast=contrast,
        rng=np.random.default_rng(seed),
    )
    a_np = np.repeat(a_single, len(names), axis=0)
    a = torch.as_tensor(a_np, dtype=torch.float64, device=device)
    u_exact = torch.as_tensor(u_exact_np, dtype=torch.float64, device=device)
    f = model.ops.apply_div_a_grad(u_exact, a)
    return evaluate_batch(model, names, a, f, u_exact)


@torch.no_grad()
def run_prediction(args: argparse.Namespace) -> None:
    """Predict fixed MMS u1-u4 and save complete data plus matching figures.

    The u1-u4 cases are fixed manufactured solutions (f = A_h u by
    definition), independent of the data kind the model was trained on.
    """
    if not args.predict_from:
        raise ValueError("--predict requires --predict-from")
    print("[predict] using fixed MMS u1-u4 cases for cross-model comparison")
    model, saved = load_model(args.predict_from, args.device)
    grid = model.grid
    contrast = saved.get("contrast", 0.5)
    bundle = _evaluate_u1_u4(model, contrast, args.test_seed)
    for row in bundle["metrics"]:
        print(
            f"[predict] {row['case']}: rel_res={row['relRes']:.3e} "
            f"rel_u={row['relL2']:.3e}  RMSE={row['RMSE']:.3e}  "
            f"maxErr={row['maxError']:.3e}"
        )

    run_name = _checkpoint_run_name(args.predict_from)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else _result_dir(args.result_root, run_name, args.test_seed)
    )
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    metadata = {
        "model_type": MODEL_TYPE,
        "evaluation": "fixed_u1_u4",
        "train_data_kind": saved.get("data_kind", "mms"),
        "prediction_data_kind": "mms",
        "coefficient_policy": "same_field_for_all_cases",
        "n": grid.n,
        "domain": [grid.x_min, grid.x_max],
        "h": grid.h,
        "contrast": contrast,
        "test_seed": args.test_seed,
        "checkpoint_sha256": _file_sha256(Path(args.predict_from)),
    }
    predictions_path = _save_evaluation_outputs(
        bundle,
        data_dir,
        "u1_u4",
        metadata,
    )
    _copy_training_history(
        args.predict_from.parent / "history.csv",
        data_dir,
    )

    if args.predict_save:
        _save_prediction_tensor(bundle, Path(args.predict_save), metadata)

    if not args.no_plot:
        from .plots import generate_pinn_style_figures

        generate_pinn_style_figures(
            model=model,
            history_path=args.predict_from.parent / "history.csv",
            figure_dir=figure_dir,
            experiment_name=run_name,
            case=args.plot_case,
            contrast=contrast,
            seed=args.test_seed,
            loss_mode=saved.get("loss_mode", "supervised"),
            evaluation=bundle,
        )
    print(f"[predict] data: {data_dir}")
    if not args.no_plot:
        print(f"[predict] figures: {figure_dir}")
    print(f"[predict] predictions: {predictions_path}")


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
    p.add_argument("--domain-min", type=float, default=DEFAULT_DOMAIN_MIN)
    p.add_argument("--domain-max", type=float, default=DEFAULT_DOMAIN_MAX)
    p.add_argument(
        "--data-kind",
        choices=("mms", "solver"),
        default="mms",
        help=(
            "training/evaluation data: mms derives f from u_exact; "
            "solver PCG-solves u_exact from f. Prediction always uses MMS u1-u4"
        ),
    )
    p.add_argument(
        "--loss-mode",
        choices=("supervised", "physics"),
        default="supervised",
        help=(
            "single training objective: supervised uses squared solution L2 error; "
            "physics uses squared PDE-residual L2 error"
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "lgf_cnn_axy",
        help="run directories are created under this root",
    )
    p.add_argument("--retrain", action="store_true")
    run_mode = p.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--eval-only", type=Path, default=None, help="load checkpoint and eval"
    )
    run_mode.add_argument("--predict", action="store_true")
    p.add_argument("--predict-from", type=Path, default=None)
    p.add_argument(
        "--test-seed",
        type=int,
        default=123,
        help="independent test/prediction seed (validation always uses seed+1)",
    )
    p.add_argument("--predict-save", default="")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PREDICT_OUTPUT_DIR,
        help=(
            "prediction result directory; default derives a parameter-safe "
            "directory below --result-root"
        ),
    )
    p.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_DIR / "result",
        help="PINN-style figures are saved below this directory",
    )
    p.add_argument(
        "--plot-case",
        choices=("all", "u1", "u2", "u3", "u4"),
        default="all",
        help="MMS case(s) used for figures; default generates u1-u4",
    )
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if args.predict and args.predict_from is None:
        p.error("--predict requires --predict-from")
    if not args.predict and args.predict_from is not None:
        p.error("--predict-from requires --predict")
    if not args.predict and (args.predict_save or args.output_dir is not None):
        p.error("--predict-save and --output-dir require --predict")
    if args.retrain and (args.predict or args.eval_only is not None):
        p.error("--retrain cannot be combined with --predict or --eval-only")
    if args.test_seed < 0:
        p.error("--test-seed must be nonnegative")

    if args.predict:
        return run_prediction(args)

    if args.eval_only is not None:
        model, saved = load_model(args.eval_only, args.device)
        checkpoint_name = _checkpoint_run_name(args.eval_only)
        result_dir = _result_dir(
            args.result_root,
            checkpoint_name,
            args.test_seed,
        )
        data_dir = result_dir / "data"
        figure_dir = result_dir / "figures"
        fixed_bundle = _evaluate_u1_u4(
            model,
            saved.get("contrast", 0.5),
            args.test_seed,
        )
        fixed_metadata = {
            "model_type": MODEL_TYPE,
            "evaluation": "fixed_u1_u4",
            "train_data_kind": saved.get("data_kind", "mms"),
            "prediction_data_kind": "mms",
            "coefficient_policy": "same_field_for_all_cases",
            "n": model.grid.n,
            "domain": [model.grid.x_min, model.grid.x_max],
            "h": model.grid.h,
            "contrast": saved.get("contrast", 0.5),
            "test_seed": args.test_seed,
            "checkpoint_sha256": _file_sha256(args.eval_only),
        }
        _save_evaluation_outputs(
            fixed_bundle,
            data_dir,
            "u1_u4",
            fixed_metadata,
        )
        _copy_training_history(args.eval_only.parent / "history.csv", data_dir)
        if not args.no_plot:
            from .plots import generate_pinn_style_figures

            generate_pinn_style_figures(
                model=model,
                history_path=args.eval_only.parent / "history.csv",
                figure_dir=figure_dir,
                experiment_name=checkpoint_name,
                case=args.plot_case,
                contrast=saved.get("contrast", 0.5),
                seed=args.test_seed,
                loss_mode=saved.get("loss_mode", "supervised"),
                evaluation=fixed_bundle,
            )
            print(f"[plot] PINN-style figures saved to {figure_dir}")
        evaluate(
            model,
            contrast=saved.get("contrast", 0.5),
            seed=args.test_seed,
            data_kind=saved.get("data_kind", "mms"),
            output_dir=data_dir,
            checkpoint_path=args.eval_only,
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
        domain_min=args.domain_min,
        domain_max=args.domain_max,
        seed=args.seed,
        device=args.device,
        loss_mode=args.loss_mode,
        data_kind=args.data_kind,
    )
    _validate_train_config(cfg)
    run_key = experiment_key(cfg)
    run_dir = args.checkpoint_root / run_key
    best_path = run_dir / "best.pt"
    with _run_lock(args.checkpoint_root, run_key):
        _cleanup_stale_attempts(args.checkpoint_root, run_key)
        completion_issue = _run_completion_issue(run_dir, cfg)
        if (
            not args.retrain
            and completion_issue is not None
            and completion_issue.startswith("identity: ")
        ):
            raise ValueError(
                f"refusing to overwrite {run_dir}: "
                f"{completion_issue.removeprefix('identity: ')}"
            )
        if completion_issue is None and not args.retrain:
            print(f"found complete run {best_path}; loading it without retraining")
            model, saved = load_model(best_path, cfg.device)
            saved_cfg = LGFCNNTrainConfig(**{**saved, "device": cfg.device})
            if _training_config_payload(saved_cfg) != _training_config_payload(cfg):
                raise ValueError(
                    "existing checkpoint config does not match the requested "
                    "training config; choose another --checkpoint-root or use "
                    "--retrain to replace it"
                )
        else:
            if run_dir.exists() and not args.retrain:
                print(
                    f"[train] existing run is incomplete ({completion_issue}); "
                    "starting a clean training attempt"
                )
            attempt_dir = args.checkpoint_root / (
                f".{run_key}.attempt-{os.getpid()}-{time.time_ns()}"
            )
            try:
                model = train(cfg, attempt_dir)
                _promote_attempt(attempt_dir, run_dir, cfg)
                model, _ = load_model(best_path, cfg.device)
                print(f"[train] committed complete run to {run_dir}")
            finally:
                shutil.rmtree(attempt_dir, ignore_errors=True)
        # Keep the run lock while reading history so a concurrent --retrain
        # cannot pair an older in-memory model with a newly promoted curve.
        result_dir = _result_dir(args.result_root, run_key, args.test_seed)
        data_dir = result_dir / "data"
        figure_dir = result_dir / "figures"
        fixed_bundle = _evaluate_u1_u4(model, cfg.contrast, args.test_seed)
        _save_evaluation_outputs(
            fixed_bundle,
            data_dir,
            "u1_u4",
            {
                "model_type": MODEL_TYPE,
                "evaluation": "fixed_u1_u4",
                "train_data_kind": cfg.data_kind,
                "prediction_data_kind": "mms",
                "coefficient_policy": "same_field_for_all_cases",
                "n": model.grid.n,
                "domain": [model.grid.x_min, model.grid.x_max],
                "h": model.grid.h,
                "contrast": cfg.contrast,
                "test_seed": args.test_seed,
                "checkpoint_sha256": _file_sha256(best_path),
            },
        )
        _copy_training_history(run_dir / "history.csv", data_dir)
        if not args.no_plot:
            from .plots import generate_pinn_style_figures

            generate_pinn_style_figures(
                model=model,
                history_path=run_dir / "history.csv",
                figure_dir=figure_dir,
                experiment_name=run_key,
                case=args.plot_case,
                contrast=cfg.contrast,
                seed=args.test_seed,
                loss_mode=cfg.loss_mode,
                evaluation=fixed_bundle,
            )
            print(f"[plot] PINN-style figures saved to {figure_dir}")
    evaluate(
        model,
        contrast=cfg.contrast,
        seed=args.test_seed,
        data_kind=cfg.data_kind,
        output_dir=data_dir,
        checkpoint_path=best_path,
    )
    print(f"[result] data: {data_dir}")
    if not args.no_plot:
        print(f"[result] figures: {figure_dir}")
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
