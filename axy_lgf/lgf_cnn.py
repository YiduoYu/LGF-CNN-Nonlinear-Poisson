"""
LGF-CNN for -div(a(x,y) grad u) = f on a rectangle (homogeneous Dirichlet).

Fixed screened LGF filters + pointwise 1x1 MLP that learns mixing weights
and a damping factor. Shared weights across unrolled blocks by default.
No BAE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .grid import DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN, SquareGrid
from .torch_ops import ScreenedLGFBankTorch, TorchSquareOps

_MIN_COEFFICIENT = 1e-8
_TINY = 1e-300


class PointwiseMixer(nn.Module):
    """1x1 MLP that outputs mixing weights (J channels) and damping (1 channel).

    At initialisation the network is biased toward the actual Poisson filter
    (lambda=0), when present, with high damping. Without a Poisson filter,
    the initial filter weights are uniform.
    """

    def __init__(
        self,
        in_ch: int,
        J: int,
        hidden: int = 32,
        poisson_filter_index: int | None = 0,
    ):
        super().__init__()
        if in_ch < 1 or J < 1 or hidden < 1:
            raise ValueError("in_ch, J, and hidden must be positive")
        if (
            poisson_filter_index is not None
            and not 0 <= poisson_filter_index < J
        ):
            raise ValueError("poisson_filter_index must identify a bank filter")
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
            if poisson_filter_index is not None:
                last.bias[poisson_filter_index] = 5.0
            last.bias[J] = 3.0   # sigmoid(3) ~ 0.95 damping

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        alpha = torch.softmax(logits[:, : self.J], dim=1)
        eta = torch.sigmoid(logits[:, self.J : self.J + 1])
        return alpha, eta


class LGFCNN(nn.Module):
    """Unrolled LGF-CNN solver.

    Each block computes a residual and applies the fixed LGF bank. The
    pointwise mixer receives dimensionless coefficient channels plus
    magnitudes of the scale-normalized residual and bank responses. The
    signed bank responses remain in the correction, making the solver odd
    in the forcing while the learned mixture is corrected locally by 1/a.
    """

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
        if J < 1 or K < 1 or hidden < 1:
            raise ValueError("J, K, and hidden must be positive")
        if not math.isfinite(eta_max) or eta_max <= 0.0:
            raise ValueError("eta_max must be finite and positive")
        self.grid = grid
        self.K = K
        self.eta_max = float(eta_max)
        self.use_inv_a_scale = use_inv_a_scale
        self.share_weights = share_weights
        self.bank = ScreenedLGFBankTorch(
            grid,
            J=J,
            a0=a0,
            include_poisson=include_poisson,
            dtype=dtype,
            device=device,
        )

        in_ch = 3 + self.bank.J
        n_mixers = 1 if share_weights else K
        poisson_indices = torch.nonzero(
            self.bank.lambdas == 0.0, as_tuple=False,
        ).flatten()
        poisson_filter_index = (
            int(poisson_indices[0]) if poisson_indices.numel() else None
        )
        self.mixers = nn.ModuleList(
            [
                PointwiseMixer(
                    in_ch,
                    self.bank.J,
                    hidden,
                    poisson_filter_index=poisson_filter_index,
                )
                for _ in range(n_mixers)
            ]
        )

    @property
    def ops(self) -> TorchSquareOps:
        return self.bank.ops

    def _mixer(self, k: int) -> PointwiseMixer:
        if self.share_weights:
            return self.mixers[0]
        if not 0 <= k < len(self.mixers):
            raise ValueError(f"block index {k} is outside [0, {len(self.mixers)})")
        return self.mixers[k]

    def _step_count(self, K: int | None) -> int:
        steps = self.K if K is None else K
        if steps < 0:
            raise ValueError("K must be nonnegative")
        if not self.share_weights and steps > len(self.mixers):
            raise ValueError("K cannot exceed the number of unshared mixers")
        return steps

    def _step(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        f: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One correction step with diagnostics."""
        r = self.residual(u, a, f)
        z = self.bank.apply(r)  # (B, J, n, n)

        reference_coefficient = a.mean(dim=(1, 2))
        reference_map = reference_coefficient[:, None, None]
        inverse_coefficient = 1.0 / a.clamp_min(_MIN_COEFFICIENT)
        relative_inverse_coefficient = reference_map * inverse_coefficient

        residual_scale = self.ops.l2_norm(r).clamp_min(_TINY)[:, None, None]
        bank_scale = self.ops.l2_norm(z).amax(dim=1).clamp_min(_TINY)
        bank_scale = bank_scale[:, None, None, None]

        coefficient_inputs = torch.stack(
            [
                a / reference_map,
                relative_inverse_coefficient,
            ],
            dim=1,
        )
        residual_magnitude = (r / residual_scale).abs()[:, None]
        bank_magnitudes = (z / bank_scale).abs()
        mixer_inputs = torch.cat(
            [coefficient_inputs, residual_magnitude, bank_magnitudes],
            dim=1,
        )

        mixer = self._mixer(k)
        mixer_dtype = next(mixer.parameters()).dtype
        alpha, eta = mixer(mixer_inputs.to(mixer_dtype))
        alpha = alpha.to(u.dtype)
        eta = eta.to(u.dtype)

        mixed = torch.sum(alpha * z, dim=1)   # (B, n, n)
        if self.use_inv_a_scale:
            mixed = inverse_coefficient * mixed
        u_next = u + self.eta_max * eta.squeeze(1) * mixed

        return u_next, {
            "residual": r,
            "alpha": alpha,
            "eta": eta,
            "u": u_next,
        }

    def block(
        self, u: torch.Tensor, a: torch.Tensor, f: torch.Tensor, k: int
    ) -> torch.Tensor:
        """One LGF-CNN correction block. u, a, f: (B, n, n)."""
        return self._step(u, a, f, k)[0]

    def forward(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        u0: torch.Tensor | None = None,
        K: int | None = None,
    ) -> torch.Tensor:
        """Run K correction blocks, return the final batched solution."""
        u = torch.zeros_like(f) if u0 is None else u0
        for k in range(self._step_count(K)):
            u, _ = self._step(u, a, f, k)
        return u

    def forward_with_diagnostics(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        u0: torch.Tensor | None = None,
        K: int | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Forward pass returning per-block residuals and mixer fields."""
        u = torch.zeros_like(f) if u0 is None else u0
        diagnostics = []
        for k in range(self._step_count(K)):
            u, step_info = self._step(u, a, f, k)
            diagnostics.append(step_info)
        return u, diagnostics

    def residual(
        self, u: torch.Tensor, a: torch.Tensor, f: torch.Tensor
    ) -> torch.Tensor:
        """Discrete PDE residual f - A(a) u for a batch."""
        return f - self.ops.apply_div_a_grad(u, a)


@dataclass
class LGFCNNTrainConfig:
    """All model, optimisation, data, and runtime training settings."""

    n: int = 31
    J: int = 6
    K: int = 6
    hidden: int = 32
    eta_max: float = 1.0
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 8
    samples_per_epoch: int = 64
    eval_every: int = 10
    val_samples: int = 16
    seed: int = 0
    device: str = "cpu"
    contrast: float = 0.5
    domain_min: float = DEFAULT_DOMAIN_MIN
    domain_max: float = DEFAULT_DOMAIN_MAX
    loss_mode: str = "supervised"   # "supervised" or "physics"
    data_kind: str = "mms"          # "mms" or "solver"
