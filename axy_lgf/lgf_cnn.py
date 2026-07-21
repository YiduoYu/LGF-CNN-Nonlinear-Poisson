"""
LGF-CNN for -div(a(x,y) grad u) = f on a rectangle (homogeneous Dirichlet).

Fixed screened Dirichlet LGF filters + pointwise 1x1 MLP mixing / damping.
No BAE. Shared weights across unrolled blocks by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .grid import DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN, SquareGrid
from .torch_ops import ScreenedLGFBankTorch, TorchSquareOps

# Positivity floor for a; never binding for the sampled coefficients.
_MIN_COEFFICIENT = 1e-8
# Guards the per-sample scales when a field is identically zero.
_TINY = 1e-300


class PointwiseMixer(nn.Module):
    """1x1 MLP producing mixing weights (J) and damping (1).

    Mixing is a residual refinement of a Poisson-favoring default:
        alpha = softmax(bias_prior + MLP)
    so the untrained network behaves like damped (a0/a) G_0.
    """

    def __init__(self, in_ch: int, J: int, hidden: int = 32):
        super().__init__()
        if in_ch < 1 or J < 1 or hidden < 1:
            raise ValueError("in_ch, J, and hidden must be positive")
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
        """
        x: (B, C, n, n)
        returns alpha (B, J, n, n), eta (B, 1, n, n)
        """
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
            include_poisson=include_poisson,
            dtype=dtype,
            device=device,
        )

        # Channels: a, 1/a, r, z_1..z_J
        in_ch = 3 + self.bank.J
        # Mixer in float32 by default; call .double() on mixers if needed.
        if share_weights:
            self.mixers = nn.ModuleList([PointwiseMixer(in_ch, self.bank.J, hidden)])
        else:
            self.mixers = nn.ModuleList(
                [PointwiseMixer(in_ch, self.bank.J, hidden) for _ in range(K)]
            )

    @property
    def ops(self) -> TorchSquareOps:
        """Grid operations without registering the same module twice."""
        return self.bank.ops

    def _mixer(self, k: int) -> PointwiseMixer:
        if self.share_weights:
            return self.mixers[0]
        if not 0 <= k < len(self.mixers):
            raise ValueError(f"block index {k} is outside [0, {len(self.mixers)})")
        return self.mixers[k]

    def _step(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        f: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One correction step plus tensors needed for diagnostics.

        Every field handed to the mixer is made dimensionless by a per-sample
        scale, so the softmax/sigmoid see the same input whenever the problem
        is rescaled.  With alpha and eta thus 0-homogeneous, and z linear in r,
        the block is *exactly* 1-homogeneous in f and (-1)-homogeneous in a --
        the linearity of -div(a grad u) = f is built into the architecture
        instead of being left for the network to approximate.
        """
        r = f - self.ops.apply_div_a_grad(u, a)
        a0 = a.mean(dim=(1, 2))  # (B,) per-sample reference coefficient
        z = self.bank.apply(r, a0)  # (B, J, n, n)

        a0c = a0[:, None, None]
        # a0/a is both the diagonal rescaling used below and a dimensionless
        # coefficient channel.
        inv_a = a0c / a.clamp_min(_MIN_COEFFICIENT)
        scale_r = self.ops.l2_norm(r).clamp_min(_TINY)[:, None, None]
        scale_z = self.ops.l2_norm(z).amax(dim=1).clamp_min(_TINY)[:, None, None]

        feats = torch.stack([a / a0c, inv_a, r / scale_r], dim=1)  # (B, 3, n, n)
        feats = torch.cat([feats, z / scale_z[:, None]], dim=1)

        mixer = self._mixer(k)
        # Cast features to mixer dtype (float32 or float64).
        mdtype = next(mixer.parameters()).dtype
        alpha, eta = mixer(feats.to(dtype=mdtype))
        alpha = alpha.to(dtype=u.dtype)
        eta = eta.to(dtype=u.dtype)

        mixed = torch.sum(alpha * z, dim=1)  # (B, n, n)
        if self.use_inv_a_scale:
            mixed = inv_a * mixed
        u_next = u + self.eta_max * eta.squeeze(1) * mixed
        diagnostics = {
            "residual": r,
            "alpha": alpha,
            "eta": eta,
            "u": u_next,
        }
        return u_next, diagnostics

    def block(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        f: torch.Tensor,
        k: int,
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
        """
        a, f: (B, n, n)
        returns u_K: (B, n, n)
        """
        steps = self.K if K is None else K
        if steps < 0:
            raise ValueError("K must be nonnegative")
        if not self.share_weights and steps > len(self.mixers):
            raise ValueError("K cannot exceed the number of unshared mixers")
        u = torch.zeros_like(f) if u0 is None else u0
        for k in range(steps):
            u = self.block(u, a, f, k)
        return u

    def forward_with_diagnostics(
        self,
        a: torch.Tensor,
        f: torch.Tensor,
        u0: torch.Tensor | None = None,
        K: int | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Forward pass that also returns per-block residuals and mixer fields."""
        steps = self.K if K is None else K
        if steps < 0:
            raise ValueError("K must be nonnegative")
        if not self.share_weights and steps > len(self.mixers):
            raise ValueError("K cannot exceed the number of unshared mixers")
        u = torch.zeros_like(f) if u0 is None else u0
        diagnostics = []
        for k in range(steps):
            u, step_info = self._step(u, a, f, k)
            diagnostics.append(step_info)
        return u, diagnostics

    def residual(
        self, u: torch.Tensor, a: torch.Tensor, f: torch.Tensor
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
    eval_every: int = 10
    val_samples: int = 16
    seed: int = 0
    device: str = "cpu"
    contrast: float = 0.5
    domain_min: float = DEFAULT_DOMAIN_MIN
    domain_max: float = DEFAULT_DOMAIN_MAX
    loss_mode: str = "supervised"  # "supervised" or "physics"
    data_kind: str = "mms"  # "mms" or "solver", see data.make_batch
