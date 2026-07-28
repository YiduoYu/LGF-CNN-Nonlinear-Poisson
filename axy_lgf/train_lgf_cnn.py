#!/usr/bin/env python3
"""Train / evaluate LGF-CNN on rectangle a(x,y) problems.

Usage:
    python -m axy_lgf.train_lgf_cnn
    python -m axy_lgf.train_lgf_cnn --epochs 100 --n 31 --K 6
    python -m axy_lgf.train_lgf_cnn --data-kind solver
    python -m axy_lgf.train_lgf_cnn --loss-mode supervised
    python -m axy_lgf.train_lgf_cnn --loss-mode physics
    python -m axy_lgf.train_lgf_cnn --predict \\
        --predict-from axy_lgf/checkpoints/lgf_cnn_axy/<key>/best.pt

Data modes (see data.py):
    mms     sample u_exact, derive f = A_h u_exact   (default)
    solver  sample f, solve A_h u = f via PCG

Training artifacts go to
<checkpoint-root>/<experiment_key>/runs/<train_id>/, and evaluations land
in result/<experiment_key>/evaluations/<eval_id>/{data,figures}/.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import IO
from uuid import uuid4

import numpy as np
import torch

from .data import make_batch, sample_coefficient
from .evaluation import (
    DEFAULT_PCG_SOLVER_TOL,
    DEFAULT_TARGET_REL_RES,
    evaluate_batch,
    run_generalization_sweeps,
)
from .grid import DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN, SquareGrid
from .lgf_cnn import LGFCNN, LGFCNNTrainConfig
from .manufactured import fixed_case_solutions, make_problem
from .operators import apply_div_a_grad, l2_norm
from .plot_lgf_cnn import generate_pinn_style_figures, save_training_curve
from .solvers import solve_axy_richardson

MODEL_TYPE = "lgf_cnn_axy"
CHECKPOINT_FORMAT_VERSION = 8
EXPERIMENT_VERSION = 8
TRAINING_RUN_FORMAT_VERSION = 1
EVALUATION_FORMAT_VERSION = 3
PROJECT_DIR = Path(__file__).resolve().parent
CORE_RUN_ARTIFACTS = ("best.pt", "last.pt", "history.csv")
NEW_RUN_ARTIFACTS = (*CORE_RUN_ARTIFACTS, "figures/training_curve.png")
RUN_MANIFEST = "completed.json"
ATTEMPT_LOCK = ".attempt.lock"
MAX_TORCH_SEED = 2**64 - 1


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def make_model(cfg: LGFCNNTrainConfig) -> LGFCNN:
    device = torch.device(cfg.device)
    model = LGFCNN(
        SquareGrid(n=cfg.n, x_min=cfg.domain_min, x_max=cfg.domain_max),
        J=cfg.J, K=cfg.K, hidden=cfg.hidden,
        eta_max=cfg.eta_max, share_weights=True,
        dtype=torch.float64, device=device,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# metrics & loss
# ---------------------------------------------------------------------------

def relative_metrics(
    model: LGFCNN,
    a: torch.Tensor, f: torch.Tensor,
    u_exact: torch.Tensor, u_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample relative residual and relative solution error, (B,) each."""
    res = model.residual(u_pred, a, f)
    rel_r = model.ops.l2_norm(res) / model.ops.l2_norm(f).clamp_min(1e-14)
    rel_u = (
        model.ops.l2_norm(u_pred - u_exact)
        / model.ops.l2_norm(u_exact).clamp_min(1e-14)
    )
    return rel_r, rel_u


def training_loss(
    model: LGFCNN,
    a: torch.Tensor, f: torch.Tensor,
    u_exact: torch.Tensor, u_pred: torch.Tensor,
    loss_mode: str,
) -> torch.Tensor:
    """Squared discrete L2 solution or residual error.

    The physics objective minimises ||G_0 r|| rather than ||r||. Applying
    G_0 = (a0 (-Delta_h))^{-1} undoes the two orders of the elliptic
    operator, so the Gauss-Newton Hessian drops from A^T A (condition
    number O(N^4)) to roughly the identity. Costs one extra DST and is
    what closes most of the gap to the supervised loss.
    """
    if loss_mode == "supervised":
        error = u_pred - u_exact
    elif loss_mode == "physics":
        residual = model.residual(u_pred, a, f)
        error = model.ops.apply_lgf(residual, a0=a.mean(dim=(1, 2)))
    else:
        raise ValueError("loss_mode must be 'supervised' or 'physics'")
    return torch.mean(model.ops.l2_norm(error) ** 2)


# ---------------------------------------------------------------------------
# config hashing & experiment identity
# ---------------------------------------------------------------------------

def _training_config_payload(cfg: LGFCNNTrainConfig) -> dict:
    """Canonical result-affecting config used for run identity."""
    payload = asdict(cfg)
    payload.pop("device", None)  # runtime placement, not part of identity
    for name in (
        "n", "J", "K", "hidden", "epochs", "batch_size",
        "samples_per_epoch", "eval_every", "val_samples", "seed",
    ):
        payload[name] = int(payload[name])
    for name in ("eta_max", "lr", "contrast", "domain_min", "domain_max"):
        payload[name] = float(payload[name])
    payload["experiment_version"] = EXPERIMENT_VERSION
    return payload


def _payload_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _git_provenance() -> dict:
    """Best-effort source revision without making Git a runtime dependency."""

    def run_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(PROJECT_DIR), *arguments],
                check=False, capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    revision = run_git("rev-parse", "HEAD")
    if revision is None:
        return {"available": False}
    status = run_git("status", "--porcelain")
    return {
        "available": True,
        "revision": revision,
        "dirty": bool(status),
    }


def _runtime_provenance() -> dict:
    """Code and environment metadata needed to interpret a saved run."""
    return {
        "argv": list(sys.argv),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source": _git_provenance(),
    }


def experiment_key(cfg: LGFCNNTrainConfig) -> str:
    """Readable run tag + short config digest."""
    visible = (
        f"{cfg.loss_mode}_{cfg.data_kind}_n{cfg.n}_J{cfg.J}_K{cfg.K}_"
        f"h{cfg.hidden}_c{cfg.contrast:g}_"
        f"dom{cfg.domain_min:g}to{cfg.domain_max:g}"
    )
    digest = _payload_sha256(_training_config_payload(cfg))[:12]
    return f"{visible}_{digest}"


# ---------------------------------------------------------------------------
# atomic file I/O
# ---------------------------------------------------------------------------

@contextmanager
def _atomic_write(path: Path):
    """Write to a temp file, then atomically rename over the target.

    If the process is killed mid-write, the previous file stays intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with _atomic_write(path) as temporary, temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([
            {key: ("" if value is None else value) for key, value in row.items()}
            for row in rows
        ])


def _write_json_atomic(payload: dict, path: Path) -> None:
    with _atomic_write(path) as temporary, temporary.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)


def _file_sha256(path: Path) -> str:
    """SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# immutable evaluation runs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingReference:
    experiment_key: str
    training_run_id: str
    run_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    history_path: Path
    training_curve_path: Path | None
    checkpoint_artifact_uri: str | None
    completed_at_utc: str
    legacy_layout: bool = False


@dataclass(frozen=True)
class EvaluationRun:
    evaluation_id: str
    evaluation_kind: str
    training_experiment: str
    training_run_id: str
    experiment_dir: Path
    attempt_dir: Path
    final_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_artifact_uri: str | None
    training_artifacts: dict
    started_at_utc: str
    attempt_lock: IO[bytes]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(moment: datetime) -> str:
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _evaluation_id(
    moment: datetime, test_seed: int, checkpoint_sha256: str,
) -> str:
    timestamp = moment.strftime("%Y%m%dT%H%M%S%fZ")
    return (
        f"{timestamp}_s{test_seed}_{checkpoint_sha256[:8]}_"
        f"{uuid4().hex[:8]}"
    )


def _begin_evaluation_run(
    experiment_dir: Path,
    training_experiment: str,
    training_run_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_artifact_uri: str | None,
    training_artifacts: dict,
    test_seed: int,
    evaluation_kind: str,
) -> EvaluationRun:
    """Create a private attempt directory for one unique evaluation."""
    started = _utc_now()
    evaluation_id = _evaluation_id(started, test_seed, checkpoint_sha256)
    evaluations_dir = experiment_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_abandoned_evaluation_attempts(evaluations_dir)
    _repair_evaluation_latest(experiment_dir)
    attempt_dir = evaluations_dir / f".attempt-{evaluation_id}"
    final_dir = evaluations_dir / evaluation_id
    attempt_dir.mkdir(exist_ok=False)
    lock_file = (attempt_dir / ATTEMPT_LOCK).open("a+b")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return EvaluationRun(
        evaluation_id=evaluation_id,
        evaluation_kind=evaluation_kind,
        training_experiment=training_experiment,
        training_run_id=training_run_id,
        experiment_dir=experiment_dir,
        attempt_dir=attempt_dir,
        final_dir=final_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_artifact_uri=checkpoint_artifact_uri,
        training_artifacts=training_artifacts,
        started_at_utc=_iso_utc(started),
        attempt_lock=lock_file,
    )


def _artifact_inventory(root: Path) -> list[dict]:
    artifacts = []
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {ATTEMPT_LOCK, ".DS_Store"}
    )
    for path in files:
        artifacts.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        })
    return artifacts


@contextmanager
def _evaluation_latest_lock(experiment_dir: Path):
    experiment_dir.mkdir(parents=True, exist_ok=True)
    lock_path = experiment_dir / ".latest.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict | None:
    try:
        with path.open() as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _evaluation_latest_payload(
    manifest: dict, final_dir: Path, experiment_dir: Path,
) -> dict:
    return {
        "evaluation_id": manifest["evaluation_id"],
        "evaluation_kind": manifest["evaluation_kind"],
        "status": "completed",
        "completed_at_utc": manifest["completed_at_utc"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "manifest_sha256": _file_sha256(final_dir / "manifest.json"),
        "path": final_dir.relative_to(experiment_dir).as_posix(),
    }


def _latest_order(payload: dict | None) -> tuple[str, str]:
    if not payload:
        return "", ""
    return (
        str(payload.get("completed_at_utc", "")),
        str(payload.get("evaluation_id", "")),
    )


def _newest_completed_evaluation(experiment_dir: Path) -> dict | None:
    evaluations_dir = experiment_dir / "evaluations"
    if not evaluations_dir.is_dir():
        return None
    candidates = []
    for final_dir in evaluations_dir.iterdir():
        if not final_dir.is_dir() or final_dir.name.startswith("."):
            continue
        manifest = _read_json(final_dir / "manifest.json")
        if (
            manifest is None
            or manifest.get("status") != "completed"
            or manifest.get("evaluation_id") != final_dir.name
            or not isinstance(manifest.get("checkpoint"), dict)
        ):
            continue
        try:
            candidates.append(
                _evaluation_latest_payload(manifest, final_dir, experiment_dir)
            )
        except (KeyError, OSError, ValueError):
            continue
    return max(candidates, key=_latest_order, default=None)


def _repair_evaluation_latest(experiment_dir: Path) -> None:
    """Recover a missing/stale pointer from committed evaluation manifests."""
    with _evaluation_latest_lock(experiment_dir):
        newest = _newest_completed_evaluation(experiment_dir)
        if newest is None:
            return
        latest_path = experiment_dir / "latest.json"
        current = _read_json(latest_path)
        current_valid = False
        if current is not None:
            try:
                current_target = (
                    experiment_dir / str(current["path"])
                ).resolve()
                current_target.relative_to(
                    (experiment_dir / "evaluations").resolve()
                )
                current_manifest = _read_json(
                    current_target / "manifest.json"
                )
                canonical = (
                    _evaluation_latest_payload(
                        current_manifest, current_target,
                        experiment_dir.resolve(),
                    )
                    if current_manifest is not None else None
                )
                current_valid = (
                    canonical is not None
                    and all(current.get(key) == value
                            for key, value in canonical.items())
                )
            except (KeyError, OSError, ValueError):
                current_valid = False
        if not current_valid or _latest_order(newest) > _latest_order(current):
            _write_json_atomic(newest, latest_path)


def _update_evaluation_latest(experiment_dir: Path, payload: dict) -> None:
    """Advance latest monotonically even when evaluations finish concurrently."""
    with _evaluation_latest_lock(experiment_dir):
        latest_path = experiment_dir / "latest.json"
        current = _read_json(latest_path)
        if _latest_order(payload) >= _latest_order(current):
            _write_json_atomic(payload, latest_path)


def _cleanup_abandoned_evaluation_attempts(evaluations_dir: Path) -> None:
    """Remove attempts whose owning process no longer holds its file lock."""
    for attempt_dir in evaluations_dir.glob(".attempt-*"):
        if not attempt_dir.is_dir():
            continue
        lock_path = attempt_dir / ATTEMPT_LOCK
        if not lock_path.exists():
            age = time.time() - attempt_dir.stat().st_mtime
            if age > 24 * 60 * 60:
                shutil.rmtree(attempt_dir)
            continue
        with lock_path.open("a+b") as lock_file:
            try:
                fcntl.flock(
                    lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                continue
            try:
                shutil.rmtree(attempt_dir)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _release_evaluation_attempt_lock(run: EvaluationRun) -> None:
    if not run.attempt_lock.closed:
        fcntl.flock(run.attempt_lock.fileno(), fcntl.LOCK_UN)
        run.attempt_lock.close()


def _clear_macos_hidden_flags(path: Path) -> None:
    """Make atomically renamed artifacts visible in macOS Finder.

    Finder marks dot-prefixed staging directories as hidden at the filesystem
    level. ``os.replace`` preserves that flag even when the final name has no
    leading dot. File Provider directories can propagate it to descendants,
    so the completed tree is repaired recursively. Other platforms do not
    expose ``chflags``/``UF_HIDDEN`` and simply skip this compatibility step.
    """
    change_flags = getattr(os, "chflags", None)
    hidden_flag = getattr(stat, "UF_HIDDEN", 0)
    if change_flags is None or hidden_flag == 0:
        return

    def make_visible(candidate: Path) -> None:
        try:
            current_flags = getattr(candidate.stat(), "st_flags", 0)
            if current_flags & hidden_flag:
                change_flags(candidate, current_flags & ~hidden_flag)
        except OSError as exc:
            print(
                f"[eval-run] could not make {candidate} visible in Finder: "
                f"{exc}"
            )

    make_visible(path)
    if path.is_dir():
        for descendant in path.rglob("*"):
            make_visible(descendant)


def _complete_evaluation_run(
    run: EvaluationRun, parameters: dict,
) -> Path:
    """Write the manifest last, then atomically expose the completed run."""
    if run.final_dir.exists():
        raise FileExistsError(f"evaluation already exists: {run.final_dir}")
    if (
        not run.checkpoint_path.is_file()
        or _file_sha256(run.checkpoint_path) != run.checkpoint_sha256
    ):
        raise RuntimeError(
            "checkpoint changed while the evaluation was running; "
            "the incomplete evaluation was not committed"
        )
    artifacts = _artifact_inventory(run.attempt_dir)
    completed_at = _iso_utc(_utc_now())
    manifest = {
        "evaluation_format_version": EVALUATION_FORMAT_VERSION,
        "evaluation_id": run.evaluation_id,
        "evaluation_kind": run.evaluation_kind,
        "status": "completed",
        "training_experiment": run.training_experiment,
        "training_run": {
            "experiment_key": run.training_experiment,
            "training_run_id": run.training_run_id,
            "artifacts": run.training_artifacts,
        },
        "started_at_utc": run.started_at_utc,
        "completed_at_utc": completed_at,
        "checkpoint": {
            "path": str(run.checkpoint_path.resolve()),
            "sha256": run.checkpoint_sha256,
            "artifact_uri": run.checkpoint_artifact_uri,
        },
        "parameters": parameters,
        "runtime": _runtime_provenance(),
        "immutability": "append_only_by_application",
        "artifacts": artifacts,
    }
    _write_json_atomic(manifest, run.attempt_dir / "manifest.json")
    os.replace(run.attempt_dir, run.final_dir)
    _release_evaluation_attempt_lock(run)
    try:
        (run.final_dir / ATTEMPT_LOCK).unlink(missing_ok=True)
    except OSError as exc:
        print(f"[eval-run] could not remove internal attempt lock: {exc}")
    _clear_macos_hidden_flags(run.final_dir)
    latest_payload = _evaluation_latest_payload(
        manifest, run.final_dir, run.experiment_dir,
    )
    try:
        _update_evaluation_latest(run.experiment_dir, latest_payload)
    except OSError as exc:
        print(
            "[eval-run] run committed, but latest.json could not be updated; "
            f"it will be repaired on the next run: {exc}"
        )
    return run.final_dir


def _discard_evaluation_attempt(run: EvaluationRun) -> None:
    _release_evaluation_attempt_lock(run)
    if run.attempt_dir.exists():
        shutil.rmtree(run.attempt_dir)


# ---------------------------------------------------------------------------
# checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path, model: LGFCNN, cfg: LGFCNNTrainConfig,
    epoch: int, best_metric: float,
) -> None:
    with _atomic_write(path) as temporary:
        torch.save({
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "model_type": MODEL_TYPE,
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        }, temporary)


def _load_model_snapshot(
    ckpt_path: Path, device: str,
) -> tuple[LGFCNN, dict, str]:
    """Load and hash the same checkpoint byte snapshot."""
    checkpoint_bytes = ckpt_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    ckpt = torch.load(
        io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True,
    )
    for field, expected in (
        ("format_version", CHECKPOINT_FORMAT_VERSION),
        ("experiment_version", EXPERIMENT_VERSION),
        ("model_type", MODEL_TYPE),
    ):
        found = ckpt.get(field)
        if found != expected:
            raise ValueError(
                f"{ckpt_path}: {field}={found!r}, expected {expected!r}. "
                "Checkpoints from an earlier architecture cannot be reused; "
                "retrain with the current code."
            )
    saved_config = dict(ckpt["config"])
    cfg = LGFCNNTrainConfig(**{**saved_config, "device": device})
    model = make_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, saved_config, checkpoint_sha256


def load_model(ckpt_path: Path, device: str) -> tuple[LGFCNN, dict]:
    """Load a checkpoint for inference, validating version compatibility."""
    model, saved_config, _ = _load_model_snapshot(ckpt_path, device)
    return model, saved_config


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------

def _validate_train_config(cfg: LGFCNNTrainConfig) -> None:
    positive_ints = {
        "n": cfg.n, "J": cfg.J, "K": cfg.K, "hidden": cfg.hidden,
        "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "samples_per_epoch": cfg.samples_per_epoch,
        "eval_every": cfg.eval_every, "val_samples": cfg.val_samples,
    }
    wrong_types = [
        name for name, value in positive_ints.items()
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
        not isinstance(cfg.seed, Integral) or isinstance(cfg.seed, bool)
        or cfg.seed < 0 or cfg.seed > MAX_TORCH_SEED
    ):
        raise ValueError(f"seed must be an integer in [0, {MAX_TORCH_SEED}]")
    if not math.isfinite(cfg.contrast) or not 0.0 <= cfg.contrast <= 1.0:
        raise ValueError("contrast must be finite and lie in [0, 1]")
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


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------

def _batch_sizes(total: int, batch_size: int) -> list[int]:
    """Split an epoch into exact batch sizes, including a final short batch."""
    if total < 1 or batch_size < 1:
        raise ValueError("total and batch_size must be positive")
    full, remainder = divmod(total, batch_size)
    return [batch_size] * full + ([remainder] if remainder else [])


def _evaluation_metadata(bundle: dict, metadata: dict) -> dict:
    return {
        **metadata,
        "reference_source": bundle["reference_source"],
        "pcg_baseline_included": bundle["u_pcg"] is not None,
    }


def _save_prediction_tensor(bundle: dict, path: Path, metadata: dict) -> None:
    def tensor_or_none(array: np.ndarray | None) -> torch.Tensor | None:
        return None if array is None else torch.from_numpy(array)

    complete_metadata = _evaluation_metadata(bundle, metadata)
    with _atomic_write(path) as temporary:
        torch.save({
            "names": bundle["names"],
            "a": torch.from_numpy(bundle["a"]),
            "f": torch.from_numpy(bundle["f"]),
            "u_exact": torch.from_numpy(bundle["u_exact"]),
            "u_pred": torch.from_numpy(bundle["u_pred"]),
            "u_pred_residual": torch.from_numpy(bundle["u_pred_residual"]),
            "u_exact_residual": torch.from_numpy(bundle["u_exact_residual"]),
            "u_linear_baseline": torch.from_numpy(bundle["u_linear_baseline"]),
            "u_richardson": torch.from_numpy(bundle["u_richardson"]),
            "u_pcg": tensor_or_none(bundle["u_pcg"]),
            "metrics": bundle["metrics"],
            "metadata": complete_metadata,
        }, temporary)


def _save_evaluation_outputs(
    bundle: dict, data_dir: Path, prefix: str, metadata: dict,
) -> Path:
    """Save one evaluation set as CSV, JSON, and a tensor archive."""
    data_dir.mkdir(parents=True, exist_ok=True)
    complete_metadata = _evaluation_metadata(bundle, metadata)
    _write_csv(bundle["metrics"], data_dir / f"{prefix}_metrics.csv")
    _write_csv(
        bundle["solver_comparison"], data_dir / f"{prefix}_solver_comparison.csv",
    )
    _write_csv(
        bundle["residual_history"], data_dir / f"{prefix}_residual_history.csv",
    )
    _write_json_atomic(
        {**complete_metadata, "metrics": bundle["metric_summary"],
         "solvers": bundle["solver_summary"]},
        data_dir / f"{prefix}_summary.json",
    )
    predictions_path = data_dir / f"{prefix}_predictions.pt"
    _save_prediction_tensor(bundle, predictions_path, complete_metadata)
    return predictions_path


# ---------------------------------------------------------------------------
# run completion & promotion
# ---------------------------------------------------------------------------

def _training_id(moment: datetime, seed: int) -> str:
    timestamp = moment.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_s{seed}_{uuid4().hex[:8]}"


def _completion_payload(
    cfg: LGFCNNTrainConfig, run_dir: Path, *,
    run_key: str, training_run_id: str,
    started_at_utc: str, completed_at_utc: str,
) -> dict:
    return {
        "training_run_format_version": TRAINING_RUN_FORMAT_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "status": "completed",
        "experiment_key": run_key,
        "training_run_id": training_run_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "config_sha256": _payload_sha256(_training_config_payload(cfg)),
        "config": _training_config_payload(cfg),
        "runtime": _runtime_provenance(),
        "artifacts": {
            name: _file_sha256(run_dir / name) for name in NEW_RUN_ARTIFACTS
        },
    }


def _run_completion_issue(run_dir: Path, cfg: LGFCNNTrainConfig) -> str | None:
    """Return why a run is incomplete or incompatible, or None if valid."""
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
    run_format = manifest.get("training_run_format_version")
    if run_format not in (None, TRAINING_RUN_FORMAT_VERSION):
        return "identity: training run format is not supported"
    expected_config = _payload_sha256(_training_config_payload(cfg))
    if manifest.get("config_sha256") != expected_config:
        return "identity: training configuration digest does not match"
    if (
        "training_run_format_version" in manifest
        and manifest.get("status") != "completed"
    ):
        return "training run status is not completed"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return "artifact hashes are missing from completed.json"
    required = (
        NEW_RUN_ARTIFACTS
        if run_format == TRAINING_RUN_FORMAT_VERSION
        else CORE_RUN_ARTIFACTS
    )
    for name in required:
        path = run_dir / name
        expected_hash = artifacts.get(name)
        if not path.is_file():
            return f"{name} is missing"
        if not isinstance(expected_hash, str) or _file_sha256(path) != expected_hash:
            return f"{name} does not match completed.json"
    return None


def _promote_attempt(
    attempt_dir: Path, run_dir: Path, cfg: LGFCNNTrainConfig, *,
    run_key: str, training_run_id: str, started_at_utc: str,
) -> None:
    """Write the manifest inside the attempt, then rename the whole directory."""
    missing = [
        name for name in NEW_RUN_ARTIFACTS
        if not (attempt_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"training attempt is missing: {', '.join(missing)}")
    if run_dir.exists():
        raise FileExistsError(f"training run already exists: {run_dir}")
    completed_at = _iso_utc(_utc_now())
    _write_json_atomic(
        _completion_payload(
            cfg, attempt_dir, run_key=run_key,
            training_run_id=training_run_id,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at,
        ),
        attempt_dir / RUN_MANIFEST,
    )
    os.replace(attempt_dir, run_dir)
    _clear_macos_hidden_flags(run_dir)


@contextmanager
def _run_lock(checkpoint_root: Path, run_key: str):
    """Serialize writers for one experiment; OS releases lock on exit."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_root / f".{run_key}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cleanup_stale_training_attempts(runs_dir: Path) -> None:
    if not runs_dir.is_dir():
        return
    for path in runs_dir.glob(".attempt-*"):
        if path.is_dir():
            shutil.rmtree(path)


def _training_reference(
    run_dir: Path, run_key: str, checkpoint_sha256: str | None = None,
) -> TrainingReference:
    checkpoint_path = run_dir / "best.pt"
    if checkpoint_sha256 is None:
        checkpoint_sha256 = _file_sha256(checkpoint_path)
    manifest = _read_json(run_dir / RUN_MANIFEST) or {}
    is_new_layout = run_dir.parent.name == "runs"
    training_run_id = str(
        manifest.get("training_run_id")
        or f"legacy-{checkpoint_sha256[:12]}"
    )
    completed_at = manifest.get("completed_at_utc")
    if not isinstance(completed_at, str):
        completed_at = _iso_utc(
            datetime.fromtimestamp(
                (run_dir / RUN_MANIFEST).stat().st_mtime, timezone.utc,
            )
        )
    artifact_uri = (
        f"runs/{run_dir.name}/best.pt" if is_new_layout else "best.pt"
    )
    training_curve = run_dir / "figures" / "training_curve.png"
    return TrainingReference(
        experiment_key=run_key,
        training_run_id=training_run_id,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        history_path=run_dir / "history.csv",
        training_curve_path=training_curve if training_curve.is_file() else None,
        checkpoint_artifact_uri=artifact_uri,
        completed_at_utc=completed_at,
        legacy_layout=not is_new_layout,
    )


def _training_latest_payload(
    reference: TrainingReference, experiment_dir: Path,
) -> dict:
    return {
        "training_run_id": reference.training_run_id,
        "status": "completed",
        "completed_at_utc": reference.completed_at_utc,
        "checkpoint_sha256": reference.checkpoint_sha256,
        "path": reference.run_dir.relative_to(experiment_dir).as_posix() or ".",
        "checkpoint_artifact_uri": reference.checkpoint_artifact_uri,
    }


def _resolve_training_run(
    checkpoint_root: Path, run_key: str, cfg: LGFCNNTrainConfig,
) -> TrainingReference | None:
    """Find the newest valid immutable run, including the legacy flat layout."""
    experiment_dir = checkpoint_root / run_key
    if not experiment_dir.is_dir():
        return None

    candidates: list[TrainingReference] = []
    runs_dir = experiment_dir / "runs"
    if runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            if _run_completion_issue(run_dir, cfg) is None:
                candidates.append(_training_reference(run_dir, run_key))

    legacy_manifest = experiment_dir / RUN_MANIFEST
    if legacy_manifest.is_file():
        issue = _run_completion_issue(experiment_dir, cfg)
        if issue is None:
            candidates.append(_training_reference(experiment_dir, run_key))
        elif issue.startswith("identity: "):
            raise ValueError(
                f"refusing to reuse {experiment_dir}: "
                f"{issue.removeprefix('identity: ')}"
            )

    if not candidates:
        return None
    newest = max(
        candidates,
        key=lambda item: (item.completed_at_utc, item.training_run_id),
    )
    try:
        _write_json_atomic(
            _training_latest_payload(newest, experiment_dir),
            experiment_dir / "latest.json",
        )
    except OSError as exc:
        print(
            "[train] latest.json could not be repaired; the immutable run "
            f"remains usable: {exc}"
        )
    return newest


# ---------------------------------------------------------------------------
# validation (held-out, used for best.pt selection)
# ---------------------------------------------------------------------------

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
        float(loss), float(rel_r.mean()),
        float(rel_u.mean()), float(fd_res_rmse),
    )


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def train(cfg: LGFCNNTrainConfig, run_dir: Path) -> LGFCNN:
    """Train one model, save artifacts, return the best checkpoint."""
    _validate_train_config(cfg)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    grid = SquareGrid(n=cfg.n, x_min=cfg.domain_min, x_max=cfg.domain_max)
    model = make_model(cfg)

    # fixed validation batch for best.pt selection; test uses separate seed
    val_batch = make_batch(
        grid, model.ops, batch_size=cfg.val_samples,
        contrast=cfg.contrast,
        rng=np.random.default_rng(cfg.seed + 1),
        device=device, dtype=torch.float64, data_kind=cfg.data_kind,
    )

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
    )

    print(
        f"[train] loss={cfg.loss_mode}  data={cfg.data_kind}  "
        f"n={cfg.n}  J={cfg.J}  K={cfg.K}  "
        f"domain=[{cfg.domain_min:g}, {cfg.domain_max:g}]  "
        f"epochs={cfg.epochs}  batch={cfg.batch_size}  device={cfg.device}"
    )
    print(
        "        trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
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
            cfg.samples_per_epoch, cfg.batch_size,
        ):
            a, f, u_exact = make_batch(
                grid, model.ops, batch_size=current_batch_size,
                contrast=cfg.contrast, rng=rng,
                device=device, dtype=torch.float64, data_kind=cfg.data_kind,
            )
            u_pred = model(a, f)
            rel_r, rel_u = relative_metrics(model, a, f, u_exact, u_pred)

            loss = training_loss(model, a, f, u_exact, u_pred, cfg.loss_mode)
            if not bool(torch.isfinite(loss).detach()):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}: "
                    f"{float(loss.detach())}"
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

        val_metric = None
        val_rel_res = None
        val_rel_u = None
        val_fd_res_rmse = None
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val_metric, val_rel_res, val_rel_u, val_fd_res_rmse = _validate(
                model, val_batch, cfg.loss_mode,
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

        history.append({
            "epoch": epoch, "loss": epoch_loss,
            "rel_res": epoch_res, "rel_u": epoch_uerr,
            "val_rel_res": val_rel_res, "val_rel_u": val_rel_u,
            "val_fd_res_rmse": val_fd_res_rmse, "val_loss": val_metric,
        })

    elapsed = time.perf_counter() - t0
    print(f"[train] done in {elapsed:.1f}s  best val_loss={best_metric:.3e}")

    save_checkpoint(run_dir / "last.pt", model, cfg, cfg.epochs, best_metric)
    _write_csv(history, run_dir / "history.csv")
    print(f"[train] saved {best_path} / last.pt / history.csv")

    if not best_path.is_file():
        raise RuntimeError("training finished without producing best.pt")
    model, _ = load_model(best_path, cfg.device)
    return model


# ---------------------------------------------------------------------------
# evaluation (held-out + fixed cases + generalization sweeps)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: LGFCNN, n_eval: int = 16, contrast: float = 0.5,
    seed: int = 123, richardson_steps: int | None = None,
    data_kind: str = "mms", output_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    evaluation_id: str | None = None,
    announce_output: bool = True,
) -> dict:
    """Evaluate a held-out batch and report only independent baselines."""
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
        grid, model.ops, batch_size=n_eval, contrast=contrast,
        rng=rng, device=device, dtype=torch.float64, data_kind=data_kind,
    )
    names = [f"sample_{index:03d}" for index in range(n_eval)]
    bundle = evaluate_batch(
        model, names, a, f, u_exact, richardson_steps=richardson_steps,
        pcg_solver_tol=DEFAULT_PCG_SOLVER_TOL,
        target_rel_res=DEFAULT_TARGET_REL_RES,
        reference_source="pcg" if data_kind == "solver" else "manufactured",
    )
    print(f"[eval] held-out random batch (data={data_kind})")
    for method, summary in bundle["solver_summary"].items():
        print(
            f"  {method:12s}  rel_res={summary['relRes']['mean']:.3e}  "
            f"rel_u={summary['relL2']['mean']:.3e}  "
            f"time={summary['time_ms']['mean']:.3f} ms/sample"
        )
    if data_kind == "solver":
        print(
            "  PCG omitted: it generated u_exact and is not an "
            "independent baseline"
        )

    if output_dir is not None:
        metadata = {
            "model_type": MODEL_TYPE, "evaluation": "heldout_random",
            "data_kind": data_kind, "n": grid.n,
            "domain": [grid.x_min, grid.x_max], "h": grid.h,
            "contrast": contrast, "test_seed": seed, "samples": n_eval,
            "richardson_steps": richardson_steps,
            "target_relative_residual": DEFAULT_TARGET_REL_RES,
        }
        if evaluation_id is not None:
            metadata["evaluation_id"] = evaluation_id
        if data_kind != "solver":
            metadata["pcg_solver_tolerance"] = DEFAULT_PCG_SOLVER_TOL
        if checkpoint_sha256 is not None:
            metadata["checkpoint_sha256"] = checkpoint_sha256
        elif checkpoint_path is not None:
            metadata["checkpoint_sha256"] = _file_sha256(checkpoint_path)
        _save_evaluation_outputs(bundle, output_dir, "heldout", metadata)
        if announce_output:
            print(f"[eval] data saved to {output_dir}")

    sweep_rows = run_generalization_sweeps(
        model, contrast=contrast, seed=seed,
    )
    print("[eval] generalization sweeps (relL2, Richardson at the same K)")
    for sweep in ("frequency", "amplitude", "contrast"):
        for row in (r for r in sweep_rows if r["sweep"] == sweep):
            print(
                f"  [{row['regime']:17s}] {sweep:10s} {row['setting']:12s} "
                f"relL2={row['relL2']:.3e}  rich={row['richardsonRelL2']:.3e}"
            )
    if output_dir is not None:
        _write_csv(sweep_rows, output_dir / "generalization_sweeps.csv")

    # fixed manufactured problem
    prob = make_problem(
        "smooth_sine", n=grid.n, x_min=grid.x_min, x_max=grid.x_max,
        forcing="discrete",
    )
    a_t = torch.as_tensor(prob.a[None], dtype=torch.float64, device=device)
    f_t = torch.as_tensor(prob.f[None], dtype=torch.float64, device=device)
    u_t = torch.as_tensor(prob.u_exact[None], dtype=torch.float64, device=device)
    rel_r_t, rel_u_t = relative_metrics(model, a_t, f_t, u_t, model(a_t, f_t))
    rich = solve_axy_richardson(
        prob.a, prob.f, grid, omega=1.0, tol=0.0, maxiter=model.K,
    )
    r_rich = prob.f - apply_div_a_grad(rich.u, prob.a, grid)
    print("[eval] manufactured smooth_sine (discrete f)")
    print(
        f"  LGF-CNN:     rel_res={float(rel_r_t.mean()):.3e}  "
        f"rel_u={float(rel_u_t.mean()):.3e}"
    )
    print(
        f"  Richardson:  rel_res="
        f"{l2_norm(r_rich, grid) / l2_norm(prob.f, grid):.3e}  "
        f"rel_u="
        f"{l2_norm(rich.u - prob.u_exact, grid) / l2_norm(prob.u_exact, grid):.3e}"
    )
    return bundle


# ---------------------------------------------------------------------------
# fixed u1-u4 evaluation
# ---------------------------------------------------------------------------

def _experiment_result_dir(result_root: Path, run_name: str) -> Path:
    return result_root / run_name


def _checkpoint_run_name(checkpoint_path: Path) -> str:
    parent = checkpoint_path.parent
    if parent.parent.name == "runs":
        return parent.parent.parent.name
    return parent.parent.name if parent.name == "checkpoints" else parent.name


def _checkpoint_reference(
    checkpoint_path: Path, checkpoint_sha256: str,
) -> TrainingReference:
    run_dir = checkpoint_path.parent
    run_key = _checkpoint_run_name(checkpoint_path)
    manifest = _read_json(run_dir / RUN_MANIFEST) or {}
    is_new_layout = run_dir.parent.name == "runs"
    training_run_id = str(
        manifest.get("training_run_id")
        or (
            run_dir.name if is_new_layout
            else (
                f"legacy-{checkpoint_sha256[:12]}"
                if manifest else f"external-{checkpoint_sha256[:12]}"
            )
        )
    )
    completed_at = manifest.get("completed_at_utc")
    if not isinstance(completed_at, str):
        completed_at = _iso_utc(
            datetime.fromtimestamp(checkpoint_path.stat().st_mtime, timezone.utc)
        )
    if is_new_layout:
        artifact_uri = f"runs/{run_dir.name}/{checkpoint_path.name}"
    elif (run_dir / RUN_MANIFEST).is_file():
        artifact_uri = checkpoint_path.name
    else:
        artifact_uri = None
    training_curve = run_dir / "figures" / "training_curve.png"
    return TrainingReference(
        experiment_key=run_key,
        training_run_id=training_run_id,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        history_path=run_dir / "history.csv",
        training_curve_path=training_curve if training_curve.is_file() else None,
        checkpoint_artifact_uri=artifact_uri,
        completed_at_utc=completed_at,
        legacy_layout=not is_new_layout,
    )


def _training_artifact_links(reference: TrainingReference) -> dict:
    links = {
        "checkpoint": {
            "uri": reference.checkpoint_artifact_uri,
            "sha256": reference.checkpoint_sha256,
        },
    }
    if (
        reference.checkpoint_artifact_uri is not None
        and reference.history_path.is_file()
    ):
        artifact_parent = Path(reference.checkpoint_artifact_uri).parent
        history_uri = (artifact_parent / "history.csv").as_posix()
        links["history"] = {
            "uri": history_uri,
            "sha256": _file_sha256(reference.history_path),
        }
        if reference.training_curve_path is not None:
            links["training_curve"] = {
                "uri": (artifact_parent / "figures/training_curve.png").as_posix(),
                "sha256": _file_sha256(reference.training_curve_path),
            }
    return links


@torch.no_grad()
def _evaluate_u1_u4(
    model: LGFCNN, contrast: float, seed: int,
) -> dict:
    """Evaluate the four fixed manufactured solutions with one coefficient."""
    grid = model.grid
    device = model.bank.lambdas.device
    names, u_exact_np = fixed_case_solutions(grid)
    a_single = sample_coefficient(
        grid, batch_size=1, contrast=contrast,
        rng=np.random.default_rng(seed),
    )
    a_np = np.repeat(a_single, len(names), axis=0)
    a = torch.as_tensor(a_np, dtype=torch.float64, device=device)
    u_exact = torch.as_tensor(u_exact_np, dtype=torch.float64, device=device)
    f = model.ops.apply_div_a_grad(u_exact, a)
    return evaluate_batch(model, names, a, f, u_exact)


def _save_fixed_evaluation(
    model: LGFCNN, bundle: dict, *,
    checkpoint_path: Path, checkpoint_sha256: str,
    history_path: Path, output_dir: Path,
    experiment_name: str, contrast: float,
    data_kind: str, loss_mode: str, test_seed: int,
    plot_case: str, no_plot: bool, evaluation_id: str | None = None,
) -> tuple[Path, Path, Path, dict]:
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    metadata = {
        "model_type": MODEL_TYPE, "evaluation": "fixed_u1_u4",
        "train_data_kind": data_kind, "prediction_data_kind": "mms",
        "coefficient_policy": "same_field_for_all_cases",
        "n": model.grid.n,
        "domain": [model.grid.x_min, model.grid.x_max],
        "h": model.grid.h,
        "contrast": contrast, "test_seed": test_seed,
        "target_relative_residual": DEFAULT_TARGET_REL_RES,
        "pcg_solver_tolerance": DEFAULT_PCG_SOLVER_TOL,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if evaluation_id is not None:
        metadata["evaluation_id"] = evaluation_id
    predictions_path = _save_evaluation_outputs(
        bundle, data_dir, "u1_u4", metadata,
    )
    if not no_plot:
        generate_pinn_style_figures(
            model=model, history_path=history_path,
            figure_dir=figure_dir, experiment_name=experiment_name,
            case=plot_case, contrast=contrast,
            loss_mode=loss_mode, evaluation=bundle,
            include_training_curve=False,
        )
    return data_dir, figure_dir, predictions_path, metadata


def _evaluate_and_save_fixed_cases(
    model: LGFCNN, args: argparse.Namespace, *,
    checkpoint_path: Path, checkpoint_sha256: str,
    history_path: Path, result_dir: Path,
    experiment_name: str, contrast: float,
    data_kind: str, loss_mode: str, evaluation_id: str,
) -> tuple[Path, Path]:
    bundle = _evaluate_u1_u4(model, contrast, args.test_seed)
    data_dir, figure_dir, _, _ = _save_fixed_evaluation(
        model, bundle, checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        history_path=history_path, output_dir=result_dir,
        experiment_name=experiment_name, contrast=contrast,
        data_kind=data_kind, loss_mode=loss_mode,
        test_seed=args.test_seed, plot_case=args.plot_case,
        no_plot=args.no_plot, evaluation_id=evaluation_id,
    )
    return data_dir, figure_dir


def _evaluation_parameters(
    model: LGFCNN, args: argparse.Namespace, *,
    contrast: float, data_kind: str, loss_mode: str,
) -> dict:
    return {
        "test_seed": int(args.test_seed),
        "plot_case": args.plot_case,
        "plots_enabled": not args.no_plot,
        "contrast": float(contrast),
        "data_kind": data_kind,
        "loss_mode": loss_mode,
        "n": model.grid.n,
        "domain": [model.grid.x_min, model.grid.x_max],
        "device": str(model.bank.lambdas.device),
        "target_relative_residual": DEFAULT_TARGET_REL_RES,
        "pcg_solver_tolerance": DEFAULT_PCG_SOLVER_TOL,
    }


def _run_full_evaluation(
    model: LGFCNN, args: argparse.Namespace, *,
    training: TrainingReference, contrast: float,
    data_kind: str, loss_mode: str,
) -> Path:
    experiment_dir = _experiment_result_dir(
        args.result_root, training.experiment_key,
    )
    run = _begin_evaluation_run(
        experiment_dir=experiment_dir,
        training_experiment=training.experiment_key,
        training_run_id=training.training_run_id,
        checkpoint_path=training.checkpoint_path,
        checkpoint_sha256=training.checkpoint_sha256,
        checkpoint_artifact_uri=training.checkpoint_artifact_uri,
        training_artifacts=_training_artifact_links(training),
        test_seed=args.test_seed,
        evaluation_kind="full_evaluation",
    )
    try:
        data_dir, _ = _evaluate_and_save_fixed_cases(
            model, args, checkpoint_path=training.checkpoint_path,
            checkpoint_sha256=training.checkpoint_sha256,
            history_path=training.history_path, result_dir=run.attempt_dir,
            experiment_name=training.experiment_key, contrast=contrast,
            data_kind=data_kind, loss_mode=loss_mode,
            evaluation_id=run.evaluation_id,
        )
        evaluate(
            model, contrast=contrast, seed=args.test_seed,
            data_kind=data_kind, output_dir=data_dir,
            checkpoint_path=training.checkpoint_path,
            checkpoint_sha256=training.checkpoint_sha256,
            evaluation_id=run.evaluation_id,
            announce_output=False,
        )
        final_dir = _complete_evaluation_run(
            run,
            _evaluation_parameters(
                model, args, contrast=contrast,
                data_kind=data_kind, loss_mode=loss_mode,
            ),
        )
    except Exception:
        _discard_evaluation_attempt(run)
        raise

    print(f"[eval-run] committed immutable run: {final_dir}")
    print(f"[result] data: {final_dir / 'data'}")
    if not args.no_plot:
        print(f"[result] figures: {final_dir / 'figures'}")
    return final_dir


# ---------------------------------------------------------------------------
# prediction mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_prediction(args: argparse.Namespace) -> None:
    """Predict fixed MMS u1-u4 and save data + figures.

    u1-u4 are fixed manufactured solutions (f = A_h u by definition),
    independent of the data kind the model was trained on.
    """
    if not args.predict_from:
        raise ValueError("--predict requires --predict-from")
    print("[predict] using fixed MMS u1-u4 cases for cross-model comparison")
    model, saved, checkpoint_sha256 = _load_model_snapshot(
        args.predict_from, args.device,
    )
    training = _checkpoint_reference(args.predict_from, checkpoint_sha256)
    contrast = saved.get("contrast", 0.5)
    run_name = training.experiment_key
    experiment_dir = (
        args.output_dir
        if args.output_dir is not None
        else _experiment_result_dir(args.result_root, run_name)
    )
    run = _begin_evaluation_run(
        experiment_dir=experiment_dir,
        training_experiment=run_name,
        training_run_id=training.training_run_id,
        checkpoint_path=training.checkpoint_path,
        checkpoint_sha256=training.checkpoint_sha256,
        checkpoint_artifact_uri=training.checkpoint_artifact_uri,
        training_artifacts=_training_artifact_links(training),
        test_seed=args.test_seed,
        evaluation_kind="fixed_prediction",
    )
    data_kind = saved.get("data_kind", "mms")
    loss_mode = saved.get("loss_mode", "supervised")
    try:
        bundle = _evaluate_u1_u4(model, contrast, args.test_seed)
        for row in bundle["metrics"]:
            print(
                f"[predict] {row['case']}: rel_res={row['relRes']:.3e} "
                f"rel_u={row['relL2']:.3e}  RMSE={row['RMSE']:.3e}  "
                f"maxErr={row['maxError']:.3e}"
            )

        _, _, attempt_predictions, metadata = _save_fixed_evaluation(
            model, bundle, checkpoint_path=args.predict_from,
            checkpoint_sha256=training.checkpoint_sha256,
            history_path=training.history_path,
            output_dir=run.attempt_dir, experiment_name=run_name,
            contrast=contrast, data_kind=data_kind, loss_mode=loss_mode,
            test_seed=args.test_seed, plot_case=args.plot_case,
            no_plot=args.no_plot, evaluation_id=run.evaluation_id,
        )
        prediction_relative_path = attempt_predictions.relative_to(
            run.attempt_dir,
        )
        final_dir = _complete_evaluation_run(
            run,
            _evaluation_parameters(
                model, args, contrast=contrast,
                data_kind=data_kind, loss_mode=loss_mode,
            ),
        )
    except Exception:
        _discard_evaluation_attempt(run)
        raise

    if args.predict_save:
        _save_prediction_tensor(bundle, Path(args.predict_save), metadata)
    predictions_path = final_dir / prediction_relative_path
    print(f"[eval-run] committed immutable run: {final_dir}")
    print(f"[predict] data: {final_dir / 'data'}")
    if not args.no_plot:
        print(f"[predict] figures: {final_dir / 'figures'}")
    print(f"[predict] predictions: {predictions_path}")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=31)
    parser.add_argument("--J", type=int, default=6)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--contrast", type=float, default=0.8)
    parser.add_argument("--domain-min", type=float, default=DEFAULT_DOMAIN_MIN)
    parser.add_argument("--domain-max", type=float, default=DEFAULT_DOMAIN_MAX)
    parser.add_argument(
        "--data-kind", choices=("mms", "solver"), default="mms",
        help="training/eval data: mms derives f from u; solver PCG-solves u from f",
    )
    parser.add_argument(
        "--loss-mode", choices=("supervised", "physics"), default="supervised",
        help="training objective: supervised uses L2 solution error; "
             "physics minimizes the preconditioned residual "
             "||G0(f - A_h u)||",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=PROJECT_DIR / "checkpoints" / "lgf_cnn_axy",
        help="immutable training runs are created under this root",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="create a new immutable training run even if the config exists",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--eval-only", type=Path, default=None, help="load checkpoint and eval",
    )
    run_mode.add_argument("--predict", action="store_true")
    parser.add_argument("--predict-from", type=Path, default=None)
    parser.add_argument(
        "--test-seed", type=int, default=123,
        help="independent test seed (validation always uses seed+1)",
    )
    parser.add_argument("--predict-save", default="")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="prediction experiment root; a unique run is created below "
             "<output-dir>/evaluations/",
    )
    parser.add_argument(
        "--result-root", type=Path,
        default=PROJECT_DIR / "result",
        help="immutable evaluation runs are saved below this directory",
    )
    parser.add_argument(
        "--plot-case", choices=("all", "u1", "u2", "u3", "u4"), default="all",
        help="MMS case(s) for figures; default: all",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.predict and args.predict_from is None:
        parser.error("--predict requires --predict-from")
    if not args.predict and args.predict_from is not None:
        parser.error("--predict-from requires --predict")
    if not args.predict and (args.predict_save or args.output_dir is not None):
        parser.error("--predict-save and --output-dir require --predict")
    if args.retrain and (args.predict or args.eval_only is not None):
        parser.error("--retrain cannot be combined with --predict or --eval-only")
    if args.test_seed < 0:
        parser.error("--test-seed must be nonnegative")
    return args


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run prediction, evaluation-only, or train-and-evaluate mode."""
    args = _parse_args()

    if args.predict:
        return run_prediction(args)

    if args.eval_only is not None:
        model, saved, checkpoint_sha256 = _load_model_snapshot(
            args.eval_only, args.device,
        )
        training = _checkpoint_reference(args.eval_only, checkpoint_sha256)
        contrast = saved.get("contrast", 0.5)
        data_kind = saved.get("data_kind", "mms")
        _run_full_evaluation(
            model, args, training=training, contrast=contrast,
            data_kind=data_kind,
            loss_mode=saved.get("loss_mode", "supervised"),
        )
        return

    cfg = LGFCNNTrainConfig(
        n=args.n, J=args.J, K=args.K, hidden=args.hidden,
        epochs=args.epochs, batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        eval_every=args.eval_every, val_samples=args.val_samples,
        lr=args.lr, contrast=args.contrast,
        domain_min=args.domain_min, domain_max=args.domain_max,
        seed=args.seed, device=args.device,
        loss_mode=args.loss_mode, data_kind=args.data_kind,
    )
    _validate_train_config(cfg)
    run_key = experiment_key(cfg)
    experiment_dir = args.checkpoint_root / run_key
    runs_dir = experiment_dir / "runs"
    training: TrainingReference
    with _run_lock(args.checkpoint_root, run_key):
        _cleanup_stale_training_attempts(runs_dir)
        existing = _resolve_training_run(args.checkpoint_root, run_key, cfg)
        if existing is not None and not args.retrain:
            print(
                "found complete immutable training run "
                f"{existing.training_run_id}; loading it without retraining"
            )
            model, saved, loaded_sha256 = _load_model_snapshot(
                existing.checkpoint_path, cfg.device,
            )
            if loaded_sha256 != existing.checkpoint_sha256:
                raise RuntimeError(
                    "checkpoint changed while it was being loaded"
                )
            training = existing
            saved_cfg = LGFCNNTrainConfig(**{**saved, "device": cfg.device})
            if _training_config_payload(saved_cfg) != _training_config_payload(cfg):
                raise ValueError(
                    "existing checkpoint config does not match the requested "
                    "training config"
                )
        else:
            if experiment_dir.exists() and existing is None:
                print(
                    "[train] no reusable completed run was found; "
                    "starting a new immutable training run"
                )
            started = _utc_now()
            training_run_id = _training_id(started, cfg.seed)
            runs_dir.mkdir(parents=True, exist_ok=True)
            attempt_dir = runs_dir / f".attempt-{training_run_id}"
            final_run_dir = runs_dir / training_run_id
            try:
                model = train(cfg, attempt_dir)
                save_training_curve(
                    attempt_dir / "history.csv",
                    attempt_dir / "figures" / "training_curve.png",
                    f"LGF-CNN training — {run_key}",
                    cfg.loss_mode,
                )
                _promote_attempt(
                    attempt_dir, final_run_dir, cfg,
                    run_key=run_key,
                    training_run_id=training_run_id,
                    started_at_utc=_iso_utc(started),
                )
                model, _, checkpoint_sha256 = _load_model_snapshot(
                    final_run_dir / "best.pt", cfg.device,
                )
                training = _training_reference(
                    final_run_dir, run_key, checkpoint_sha256,
                )
                try:
                    _write_json_atomic(
                        _training_latest_payload(training, experiment_dir),
                        experiment_dir / "latest.json",
                    )
                except OSError as exc:
                    print(
                        "[train] run committed, but latest.json could not be "
                        f"updated; it will be repaired on the next run: {exc}"
                    )
                print(
                    "[train] committed immutable run to "
                    f"{final_run_dir}"
                )
            finally:
                shutil.rmtree(attempt_dir, ignore_errors=True)
    _run_full_evaluation(
        model, args, training=training, contrast=cfg.contrast,
        data_kind=cfg.data_kind, loss_mode=cfg.loss_mode,
    )
    print(f"done: {training.run_dir}")


if __name__ == "__main__":
    main()
