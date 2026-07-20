#!/usr/bin/env python3
"""
Standalone LGF-CNN: MMS training and u1-u4 prediction for -div(a(x,y) grad u)=f.

This file is self-contained and imports nothing beyond numpy / torch /
matplotlib.  The numerical method is a faithful inline copy of the ``axy_lgf``
package: interior grid with a homogeneous Dirichlet halo, flux-form five-point
operator, DST-I screened Dirichlet LGF bank, pointwise-mixed unrolled LGF-CNN,
random manufactured batches, and the relative residual + relative solution
error loss.

The four test solutions u1-u4 are sine combinations vanishing on the boundary,
i.e. the same homogeneous-Dirichlet family the model is trained on.

Checkpoint / result bookkeeping follows Baseline_mms.py:
  - run directory  checkpoints/lgf_cnn_mms_axy/<experiment_key>/
  - best.pt selected on a fixed held-out validation batch, last.pt at the end
  - history.csv, metrics.csv, summary.json, predictions.pt
  - existing best.pt is not overwritten unless --retrain

Example
-------
    python Baseline_mms_axy_lgf.py --epochs 150
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE = "lgf_cnn_mms_axy"


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SquareGrid:
    """Interior grid for Omega = (0, L)^2 with u = 0 on the boundary.

    Interior nodes: x_i = i * h, i = 1..n, with h = L / (n + 1).
    """

    n: int
    L: float = 1.0

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n must be >= 2")
        if self.L <= 0:
            raise ValueError("L must be positive")

    @property
    def h(self) -> float:
        return self.L / (self.n + 1)

    @property
    def x1d(self) -> np.ndarray:
        return np.arange(1, self.n + 1, dtype=np.float64) * self.h

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x1d
        return np.meshgrid(x, x, indexing="xy")


def laplacian_eigenvalues(grid: SquareGrid) -> np.ndarray:
    """Eigenvalues of -Delta_h on the interior grid (positive).

    For mode (p, q), p,q = 1..n:
        mu_{pq} = (2/h^2) * (2 - cos(p*pi/(n+1)) - cos(q*pi/(n+1)))
    """
    n = grid.n
    h = grid.h
    p = np.arange(1, n + 1, dtype=np.float64)
    c = np.cos(p * np.pi / (n + 1))
    lam1d = (2.0 / h**2) * (1.0 - c)
    return lam1d[:, None] + lam1d[None, :]


def default_screening_values(
    h: float,
    J: int = 6,
    include_poisson: bool = True,
) -> np.ndarray:
    """Screening coefficients λ_j with length scales ~ {∞, 2h, 4h, ...}.

    λ = 0 is the pure Dirichlet Poisson LGF (included by default on a
    rectangle, where it is exact via DST and does not need truncation).
    """
    if J < 1:
        raise ValueError("J must be >= 1")

    sigmas: list[float] = []
    if include_poisson:
        sigmas.append(0.0)
        n_screened = J - 1
    else:
        n_screened = J

    if n_screened > 0:
        # length scales in units of h: 2, 4, 8, ...
        ell = h * (2.0 ** np.arange(1, n_screened + 1))
        sigmas.extend((1.0 / ell**2).tolist())

    return np.asarray(sigmas[:J], dtype=np.float64)


# ---------------------------------------------------------------------------
# Discrete operators and the DST-I screened Dirichlet LGF bank
# ---------------------------------------------------------------------------

def dst_matrix(n: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """Orthonormal 1D DST-I matrix S, S_{pq} = sqrt(2/(n+1)) sin(p q π/(n+1))."""
    p = torch.arange(1, n + 1, dtype=dtype, device=device)
    S = torch.sin(np.pi * p[:, None] * p[None, :] / (n + 1))
    S = S * np.sqrt(2.0 / (n + 1))
    return S


class TorchSquareOps(nn.Module):
    """Grid operators for batches of fields shaped (B, n, n)."""

    def __init__(self, grid: SquareGrid, dtype=torch.float64, device=None):
        super().__init__()
        self.n = grid.n
        self.L = grid.L
        self.h = grid.h
        self.register_buffer(
            "S", dst_matrix(grid.n, dtype=dtype, device=device), persistent=True
        )
        mu = laplacian_eigenvalues(grid)
        self.register_buffer(
            "mu",
            torch.as_tensor(mu, dtype=dtype, device=device),
            persistent=True,
        )

    def apply_div_a_grad(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Batched -div(a grad u) with homogeneous Dirichlet BC.

        u, a: (B, n, n)
        """
        h = self.h
        # Pad Dirichlet halo: (B, n+2, n+2)
        U = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)

        # Face diffusivities (arithmetic mean), shapes (B, n, n+1) and (B, n+1, n)
        ax_inner = 0.5 * (a[:, :, :-1] + a[:, :, 1:])
        ax = torch.cat([a[:, :, :1], ax_inner, a[:, :, -1:]], dim=2)

        ay_inner = 0.5 * (a[:, :-1, :] + a[:, 1:, :])
        ay = torch.cat([a[:, :1, :], ay_inner, a[:, -1:, :]], dim=1)

        ux_left = (U[:, 1:-1, 1:-1] - U[:, 1:-1, :-2]) / h
        ux_right = (U[:, 1:-1, 2:] - U[:, 1:-1, 1:-1]) / h
        uy_down = (U[:, 1:-1, 1:-1] - U[:, :-2, 1:-1]) / h
        uy_up = (U[:, 2:, 1:-1] - U[:, 1:-1, 1:-1]) / h

        flux_left = ax[:, :, :-1] * ux_left
        flux_right = ax[:, :, 1:] * ux_right
        flux_down = ay[:, :-1, :] * uy_down
        flux_up = ay[:, 1:, :] * uy_up

        return -((flux_right - flux_left) + (flux_up - flux_down)) / h

    def apply_lgf(
        self,
        rhs: torch.Tensor,
        a0: float = 1.0,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        """Apply (a0 (-Delta) + sigma I)^{-1} via DST. rhs: (B, n, n)."""
        rhs_hat = torch.einsum("ij,bjk,kl->bil", self.S, rhs, self.S)
        u_hat = rhs_hat / (float(a0) * self.mu + sigma)
        return torch.einsum("ij,bjk,kl->bil", self.S, u_hat, self.S)

    def l2_norm(self, v: torch.Tensor) -> torch.Tensor:
        """Per-batch discrete L2 norm. v: (B, n, n) -> (B,)."""
        return torch.sqrt(torch.sum(v * v, dim=(1, 2)) * self.h**2)


class ScreenedLGFBankTorch(nn.Module):
    """Fixed screened Dirichlet LGF bank."""

    def __init__(
        self,
        grid: SquareGrid,
        J: int = 6,
        a0: float = 1.0,
        include_poisson: bool = True,
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()
        self.ops = TorchSquareOps(grid, dtype=dtype, device=device)
        lambdas = default_screening_values(
            grid.h, J=J, include_poisson=include_poisson
        )
        self.register_buffer(
            "lambdas",
            torch.as_tensor(lambdas, dtype=dtype, device=device),
            persistent=True,
        )
        self.a0 = float(a0)

    @property
    def J(self) -> int:
        return int(self.lambdas.numel())

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """r: (B, n, n) -> z: (B, J, n, n)."""
        zs = [
            self.ops.apply_lgf(r, a0=self.a0, sigma=float(lam))
            for lam in self.lambdas.tolist()
        ]
        return torch.stack(zs, dim=1)


# ---------------------------------------------------------------------------
# LGF-CNN
# ---------------------------------------------------------------------------

class PointwiseMixer(nn.Module):
    """1x1 MLP producing mixing weights (J) and damping (1).

    Mixing is a residual refinement of a Poisson-favoring default:
        alpha = softmax(bias_prior + MLP)
    so the untrained network behaves like damped (1/a) G_0.
    """

    def __init__(self, in_ch: int, J: int, hidden: int = 32):
        super().__init__()
        self.J = J
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, J + 1, kernel_size=1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            # Softmax ≈ one-hot on λ=0 (Poisson) at init.
            last.bias[0] = 5.0
            # sigmoid(3) ≈ 0.95 damping at init.
            last.bias[J] = 3.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, n, n) -> alpha (B, J, n, n), eta (B, 1, n, n)."""
        s = self.net(x)
        alpha = torch.softmax(s[:, : self.J], dim=1)
        eta = torch.sigmoid(s[:, self.J : self.J + 1])
        return alpha, eta


class LGFCNN(nn.Module):
    """Unrolled LGF-CNN solver."""

    def __init__(
        self,
        grid: SquareGrid,
        J: int = 6,
        K: int = 6,
        hidden: int = 32,
        a0: float = 1.0,
        eta_max: float = 1.0,
        share_weights: bool = True,
        include_poisson: bool = True,
        use_inv_a_scale: bool = True,
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()
        self.grid = grid
        self.K = K
        self.eta_max = float(eta_max)
        self.use_inv_a_scale = use_inv_a_scale
        self.bank = ScreenedLGFBankTorch(
            grid,
            J=J,
            a0=a0,
            include_poisson=include_poisson,
            dtype=dtype,
            device=device,
        )
        self.ops: TorchSquareOps = self.bank.ops

        # Channels: a, 1/a, r, z_1..z_J
        in_ch = 3 + self.bank.J
        # Mixers stay in float32; block() casts features to the mixer dtype.
        if share_weights:
            self.mixers = nn.ModuleList(
                [PointwiseMixer(in_ch, self.bank.J, hidden)]
            )
        else:
            self.mixers = nn.ModuleList(
                [PointwiseMixer(in_ch, self.bank.J, hidden) for _ in range(K)]
            )

    def _mixer(self, k: int) -> PointwiseMixer:
        return self.mixers[0 if len(self.mixers) == 1 else k]

    def block(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        f: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """One LGF-CNN correction block. u,a,f: (B, n, n) float64."""
        r = f - self.ops.apply_div_a_grad(u, a)
        z = self.bank.apply(r)  # (B, J, n, n)

        inv_a = 1.0 / a.clamp_min(1e-8)
        feats = torch.stack([a, inv_a, r], dim=1)  # (B, 3, n, n)
        feats = torch.cat([feats, z], dim=1)

        mixer = self._mixer(k)
        mdtype = next(mixer.parameters()).dtype
        alpha, eta = mixer(feats.to(dtype=mdtype))
        alpha = alpha.to(dtype=u.dtype)
        eta = eta.to(dtype=u.dtype)

        mixed = torch.sum(alpha * z, dim=1)  # (B, n, n)
        if self.use_inv_a_scale:
            mixed = inv_a * mixed
        return u + self.eta_max * eta.squeeze(1) * mixed

    def forward(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        u0: torch.Tensor | None = None,
        K: int | None = None,
    ) -> torch.Tensor:
        """a, f: (B, n, n) -> u_K: (B, n, n)."""
        steps = self.K if K is None else K
        u = torch.zeros_like(f) if u0 is None else u0
        for k in range(steps):
            u = self.block(u, a, f, k)
        return u

    def residual(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        return f - self.ops.apply_div_a_grad(u, a)


@dataclass
class LGFCNNTrainConfig:
    n: int = 31
    J: int = 6
    K: int = 6
    hidden: int = 32
    eta_max: float = 1.0
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 8
    samples_per_epoch: int = 64
    seed: int = 0
    device: str = "cpu"
    contrast: float = 0.5
    lambda_u: float = 1.0
    lambda_r: float = 1.0


# ---------------------------------------------------------------------------
# Random manufactured data
# ---------------------------------------------------------------------------

def sample_coefficient(
    grid: SquareGrid,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample smooth positive a(x,y) = 1 + contrast * field, field in [-1,1]."""
    rng = np.random.default_rng() if rng is None else rng
    X, Y = grid.mesh()
    a = np.empty((batch_size, grid.n, grid.n), dtype=np.float64)
    for b in range(batch_size):
        field = np.zeros_like(X)
        nmodes = int(rng.integers(1, 4))
        for _ in range(nmodes):
            px = int(rng.integers(1, 4))
            py = int(rng.integers(1, 4))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp = rng.uniform(0.3, 1.0)
            field += amp * np.sin(2.0 * np.pi * px * X + phase) * np.cos(
                2.0 * np.pi * py * Y
            )
        field = field / (np.max(np.abs(field)) + 1e-14)
        a[b] = 1.0 + contrast * field
        # keep strictly positive
        a[b] = np.clip(a[b], 1.0 - 0.95 * contrast, 1.0 + contrast)
    return a


def sample_solution(
    grid: SquareGrid,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample homogeneous-Dirichlet solutions as sine combinations."""
    rng = np.random.default_rng() if rng is None else rng
    X, Y = grid.mesh()
    u = np.zeros((batch_size, grid.n, grid.n), dtype=np.float64)
    for b in range(batch_size):
        nmodes = int(rng.integers(1, 4))
        for _ in range(nmodes):
            px = int(rng.integers(1, 4))
            py = int(rng.integers(1, 4))
            amp = rng.normal(scale=1.0)
            u[b] += amp * np.sin(px * np.pi * X / grid.L) * np.sin(
                py * np.pi * Y / grid.L
            )
    return u


def make_batch(
    grid: SquareGrid,
    ops: TorchSquareOps,
    batch_size: int,
    contrast: float = 0.5,
    rng: np.random.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a, f, u_exact as torch tensors (B, n, n).

    f is the discrete operator applied to u_exact, so a consistent discrete
    problem is solved exactly by u_exact.
    """
    rng = np.random.default_rng() if rng is None else rng
    a_np = sample_coefficient(grid, batch_size, contrast=contrast, rng=rng)
    u_np = sample_solution(grid, batch_size, rng=rng)

    a = torch.as_tensor(a_np, dtype=dtype, device=device)
    u = torch.as_tensor(u_np, dtype=dtype, device=device)
    f = ops.apply_div_a_grad(u, a)
    return a, f, u


# ---------------------------------------------------------------------------
# Training, checkpoints, validation
# ---------------------------------------------------------------------------

def select_device(name: str) -> torch.device:
    """Match the float64 implementation with a supported device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # This path intentionally uses float64; CPU is the safe fallback because
    # MPS does not provide complete float64 support.
    return torch.device("cpu")


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
    model.mixers.to(device)
    model.bank.to(device)
    return model


def experiment_key(cfg: LGFCNNTrainConfig) -> str:
    """Readable run tag plus a digest of every other training-affecting arg."""
    visible = f"n{cfg.n}_J{cfg.J}_K{cfg.K}_h{cfg.hidden}_c{cfg.contrast:g}"
    affecting = (
        cfg.epochs,
        cfg.batch_size,
        cfg.samples_per_epoch,
        cfg.lr,
        cfg.eta_max,
        cfg.lambda_u,
        cfg.lambda_r,
        cfg.seed,
    )
    digest = hashlib.md5(repr(affecting).encode()).hexdigest()[:8]
    return f"{visible}_{digest}"


def batch_metrics(
    model: LGFCNN,
    a: torch.Tensor,
    f: torch.Tensor,
    u_exact: torch.Tensor,
    u_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample relative residual and relative solution error."""
    residual = model.residual(u_pred, a, f)
    f_norm = model.ops.l2_norm(f).clamp_min(1e-14)
    u_norm = model.ops.l2_norm(u_exact).clamp_min(1e-14)
    relative_residual = model.ops.l2_norm(residual) / f_norm
    relative_solution_error = model.ops.l2_norm(u_pred - u_exact) / u_norm
    return relative_residual, relative_solution_error


@torch.no_grad()
def validate(
    model: LGFCNN,
    val_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    model.eval()
    a, f, u_exact = val_batch
    u_pred = model(a, f)
    rel_r, rel_u = batch_metrics(model, a, f, u_exact, u_pred)
    return float(rel_r.mean()), float(rel_u.mean())


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
            "model_type": MODEL_TYPE,
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "lambdas": model.bank.lambdas.detach().cpu().numpy(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[LGFCNN, dict]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"checkpoint type must be {MODEL_TYPE!r}, got "
            f"{checkpoint.get('model_type')!r}"
        )
    saved = checkpoint["config"]
    cfg = LGFCNNTrainConfig(**{**saved, "device": str(device)})
    model = make_model(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, saved


def write_csv(rows: list[dict], path: Path) -> None:
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


def read_history_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open() as file:
        for raw in csv.DictReader(file):
            rows.append(
                {
                    key: (None if value == "" else float(value))
                    for key, value in raw.items()
                }
            )
    return rows


def train(
    cfg: LGFCNNTrainConfig,
    run_dir: Path,
    eval_every: int,
    val_samples: int,
) -> tuple[LGFCNN, list[dict]]:
    """Online-batch training with best/last checkpoint selection on a fixed
    held-out validation batch."""
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    grid = SquareGrid(n=cfg.n)
    model = make_model(cfg)

    val_batch = make_batch(
        grid,
        model.ops,
        batch_size=val_samples,
        contrast=cfg.contrast,
        rng=np.random.default_rng(cfg.seed + 1),
        device=device,
        dtype=torch.float64,
    )

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.lr,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"[train] n={cfg.n} J={cfg.J} K={cfg.K} epochs={cfg.epochs} "
        f"batch={cfg.batch_size} device={cfg.device}"
    )
    print(f"        trainable params: {parameter_count}")
    print(f"        run dir: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_metric = float("inf")
    history: list[dict] = []
    start_time = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_residual = 0.0
        epoch_solution_error = 0.0
        n_batches = max(1, cfg.samples_per_epoch // cfg.batch_size)

        for _ in range(n_batches):
            a, f, u_exact = make_batch(
                grid,
                model.ops,
                batch_size=cfg.batch_size,
                contrast=cfg.contrast,
                rng=rng,
                device=device,
                dtype=torch.float64,
            )

            u_pred = model(a, f)
            relative_residual, relative_solution_error = batch_metrics(
                model, a, f, u_exact, u_pred
            )
            loss = (
                cfg.lambda_r * torch.mean(relative_residual**2)
                + cfg.lambda_u * torch.mean(relative_solution_error**2)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_residual += float(relative_residual.mean().detach())
            epoch_solution_error += float(relative_solution_error.mean().detach())

        epoch_loss /= n_batches
        epoch_residual /= n_batches
        epoch_solution_error /= n_batches

        val_rel_res = None
        val_rel_u = None
        if epoch == 1 or epoch % eval_every == 0 or epoch == cfg.epochs:
            val_rel_res, val_rel_u = validate(model, val_batch)
            if val_rel_u < best_metric:
                best_metric = val_rel_u
                save_checkpoint(best_path, model, cfg, epoch, best_metric)
            print(
                f"  epoch {epoch:4d} loss={epoch_loss:.3e} "
                f"rel_res={epoch_residual:.3e} rel_u={epoch_solution_error:.3e} "
                f"val_rel_res={val_rel_res:.3e} val_rel_u={val_rel_u:.3e}"
            )

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "rel_res": epoch_residual,
                "rel_u": epoch_solution_error,
                "val_rel_res": val_rel_res,
                "val_rel_u": val_rel_u,
            }
        )

    elapsed = time.perf_counter() - start_time
    print(f"[train] done in {elapsed:.1f}s  best val_rel_u={best_metric:.3e}")

    save_checkpoint(run_dir / "last.pt", model, cfg, cfg.epochs, best_metric)
    write_csv(history, run_dir / "history.csv")
    print(f"[train] saved {best_path} / last.pt / history.csv")

    # Predict with the best checkpoint, as Baseline_mms does.
    model, _ = load_model(best_path, device)
    return model, history


# ---------------------------------------------------------------------------
# u1-u4 prediction and result files
# ---------------------------------------------------------------------------

def test_cases_u1_u4(grid: SquareGrid) -> tuple[list[str], np.ndarray]:
    """Four fixed test solutions, in the spirit of the legacy Baseline_mms
    u1-u4 but rewritten to vanish on the boundary of (0, L)^2 (the legacy
    formulas were defined on [-1,1]^2 and mostly did not vanish there, which
    conflicts with the zero Dirichlet halo axy_lgf's operator assumes).
    """
    x, y = grid.mesh()
    L = grid.L

    u1 = np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    u2 = np.sin(2.0 * np.pi * x / L) * np.sin(np.pi * y / L)
    u3 = np.sin(np.pi * x / L) * np.sin(2.0 * np.pi * y / L) \
        - 0.5 * np.sin(2.0 * np.pi * x / L) * np.sin(2.0 * np.pi * y / L)
    u4 = np.sin(3.0 * np.pi * x / L) * np.sin(3.0 * np.pi * y / L)

    cases = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}
    names = list(cases)
    values = np.stack([cases[name] for name in names], axis=0)
    return names, values


@torch.no_grad() 
def predict_u1_u4(
    model: LGFCNN,
    contrast: float,
    seed: int,
) -> dict:
    """Build f=A_h u for u1-u4 and predict with the trained LGF-CNN."""
    model.eval()
    device = model.bank.lambdas.device
    grid = model.grid
    names, u_exact_np = test_cases_u1_u4(grid)

    a_np = sample_coefficient(
        grid,
        batch_size=len(names),
        contrast=contrast,
        rng=np.random.default_rng(seed),
    )
    a = torch.as_tensor(a_np, dtype=torch.float64, device=device)
    u_exact = torch.as_tensor(u_exact_np, dtype=torch.float64, device=device)
    f = model.ops.apply_div_a_grad(u_exact, a)
    u_pred = model(a, f)
    residual = model.residual(u_pred, a, f)
    relative_residual, relative_solution_error = batch_metrics(
        model, a, f, u_exact, u_pred
    )

    for index, name in enumerate(names):
        print(
            f"[predict] {name}: rel_res={relative_residual[index]:.3e} "
            f"rel_u={relative_solution_error[index]:.3e}"
        )

    return {
        "names": names,
        "a": a.cpu().numpy(),
        "f": f.cpu().numpy(),
        "u_exact": u_exact.cpu().numpy(),
        "u_pred": u_pred.cpu().numpy(),
        "residual": residual.cpu().numpy(),
        "rel_res": relative_residual.cpu().numpy(),
        "rel_u": relative_solution_error.cpu().numpy(),
    }


def save_prediction_results(
    prediction: dict,
    output_dir: Path,
    test_seed: int,
) -> None:
    """Per-case metrics.csv + aggregated summary.json + raw predictions.pt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = prediction["names"]
    rows = [
        {
            "case": name,
            "rel_res": float(prediction["rel_res"][index]),
            "rel_u": float(prediction["rel_u"][index]),
        }
        for index, name in enumerate(names)
    ]
    write_csv(rows, output_dir / "metrics.csv")

    summary = {
        "model_type": MODEL_TYPE,
        "test_seed": test_seed,
        "relResMean": float(np.mean(prediction["rel_res"])),
        "relResMax": float(np.max(prediction["rel_res"])),
        "relUMean": float(np.mean(prediction["rel_u"])),
        "relUMax": float(np.max(prediction["rel_u"])),
        "cases": {
            row["case"]: {"rel_res": row["rel_res"], "rel_u": row["rel_u"]}
            for row in rows
        },
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    torch.save(
        {
            "names": names,
            **{
                key: torch.from_numpy(prediction[key])
                for key in ("a", "f", "u_exact", "u_pred", "residual", "rel_res", "rel_u")
            },
        },
        output_dir / "predictions.pt",
    )
    print(
        f"[predict] saved metrics.csv / summary.json / predictions.pt -> {output_dir}"
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def setup_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "baseline_mms_axy_mpl"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training(history: list[dict], output_path: Path) -> None:
    if not history:
        print("[plot] no training history available; training.png skipped")
        return

    plt = setup_matplotlib()
    epochs = [row["epoch"] for row in history]
    val_points = [
        (row["epoch"], row["val_rel_res"], row["val_rel_u"])
        for row in history
        if row.get("val_rel_u") not in (None, "")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].semilogy(epochs, [row["loss"] for row in history], color="#1f77b4")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(
        epochs,
        [row["rel_res"] for row in history],
        label="train rel_res",
        color="#d62728",
    )
    axes[1].semilogy(
        epochs,
        [row["rel_u"] for row in history],
        label="train rel_u",
        color="#2ca02c",
    )
    if val_points:
        val_epochs = [point[0] for point in val_points]
        axes[1].semilogy(
            val_epochs,
            [point[1] for point in val_points],
            "o--",
            markersize=4,
            label="val rel_res",
            color="#ff7f0e",
        )
        axes[1].semilogy(
            val_epochs,
            [point[2] for point in val_points],
            "s--",
            markersize=4,
            label="val rel_u",
            color="#9467bd",
        )
    axes[1].set_title("Training metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("relative error")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}")


def plot_u1_u4_predictions(prediction: dict, output_path: Path, L: float) -> None:
    plt = setup_matplotlib()
    names = prediction["names"]
    u_exact = prediction["u_exact"]
    u_pred = prediction["u_pred"]
    rel_u = prediction["rel_u"]
    rel_res = prediction["rel_res"]

    fig, axes = plt.subplots(4, 3, figsize=(11, 14), constrained_layout=True)
    extent = [0.0, L, 0.0, L]

    for row, name in enumerate(names):
        error = u_pred[row] - u_exact[row]
        value_min = min(float(u_exact[row].min()), float(u_pred[row].min()))
        value_max = max(float(u_exact[row].max()), float(u_pred[row].max()))
        error_max = max(float(np.abs(error).max()), 1e-14)

        panels = (
            (u_exact[row], f"{name} exact", "RdBu_r", value_min, value_max),
            (
                u_pred[row],
                f"{name} prediction\nrel_u={rel_u[row]:.2e}",
                "RdBu_r",
                value_min,
                value_max,
            ),
            (
                error,
                f"{name} error\nrel_res={rel_res[row]:.2e}",
                "coolwarm",
                -error_max,
                error_max,
            ),
        )
        for column, (field, title, cmap, lower, upper) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=31)
    parser.add_argument("--J", type=int, default=6)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--contrast", type=float, default=0.8)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--lambda-r", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints") / MODEL_TYPE,
        help="run directories are created under this root",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="overwrite an existing best.pt for the same experiment key",
    )
    parser.add_argument(
        "--eval-only",
        type=Path,
        default=None,
        help="load a checkpoint instead of training",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n < 2:
        raise ValueError("--n must be at least 2")
    if min(args.J, args.K, args.hidden) < 1:
        raise ValueError("--J, --K, and --hidden must be positive")
    if args.eval_only is None:
        if min(args.epochs, args.batch_size, args.samples_per_epoch) < 1:
            raise ValueError(
                "--epochs, --batch-size, and --samples-per-epoch must be positive"
            )
        if args.lr <= 0.0:
            raise ValueError("--lr must be positive")
        if args.eval_every < 1 or args.val_samples < 1:
            raise ValueError("--eval-every and --val-samples must be positive")
        if min(args.lambda_u, args.lambda_r) < 0.0:
            raise ValueError("--lambda-u and --lambda-r must be non-negative")
        if max(args.lambda_u, args.lambda_r) <= 0.0:
            raise ValueError("at least one loss weight must be positive")
    if not 0.0 <= args.contrast < 1.0:
        raise ValueError("--contrast must be in [0, 1)")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = select_device(args.device)

    if args.eval_only is not None:
        model, saved_config = load_model(args.eval_only, device)
        contrast = float(saved_config.get("contrast", args.contrast))
        run_dir = args.eval_only.resolve().parent
        history = read_history_csv(run_dir / "history.csv")
    else:
        cfg = LGFCNNTrainConfig(
            n=args.n,
            J=args.J,
            K=args.K,
            hidden=args.hidden,
            eta_max=args.eta_max,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            device=str(device),
            contrast=args.contrast,
            lambda_u=args.lambda_u,
            lambda_r=args.lambda_r,
        )
        run_dir = args.checkpoint_root / experiment_key(cfg)
        if (run_dir / "best.pt").exists() and not args.retrain:
            print(f"found {run_dir / 'best.pt'}; use --retrain to overwrite")
            return
        model, history = train(
            cfg,
            run_dir,
            eval_every=args.eval_every,
            val_samples=args.val_samples,
        )
        contrast = cfg.contrast

    plot_training(history, run_dir / "training.png")

    prediction = predict_u1_u4(model, contrast=contrast, seed=args.test_seed)
    save_prediction_results(prediction, run_dir, args.test_seed)
    plot_u1_u4_predictions(
        prediction,
        run_dir / "u1_u4_predictions.png",
        L=model.grid.L,
    )
    print(f"[done] results: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
