"""PyTorch discrete operators and Dirichlet LGF bank for the rectangle problem."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import SquareGrid
from .lgf_bank import default_screening_values
from .lgf_inverse import _laplacian_eigenvalues


def dst1_ortho(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply an orthonormal DST-I along one dimension using an FFT.

    The odd extension is differentiable and avoids storing or multiplying by
    a dense n x n sine matrix.
    """
    if not x.is_floating_point():
        raise TypeError("DST input must be floating point")
    dim %= x.ndim
    y = x.movedim(dim, -1)
    n = y.shape[-1]
    if n < 1:
        raise ValueError("DST dimension must be nonempty")

    zero = torch.zeros_like(y[..., :1])
    odd_extension = torch.cat([zero, y, zero, -y.flip(-1)], dim=-1)
    spectrum = torch.fft.fft(odd_extension, dim=-1)
    scale = math.sqrt(1.0 / (2.0 * (n + 1)))
    result = -spectrum.imag[..., 1 : n + 1] * scale
    return result.movedim(-1, dim)


def dst2_ortho(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal DST-I along the last two dimensions."""
    if x.ndim < 2:
        raise ValueError("2D DST input must have at least two dimensions")
    return dst1_ortho(dst1_ortho(x, dim=-1), dim=-2)


class TorchSquareOps(nn.Module):
    """Grid operators for batches of fields shaped (B, n, n)."""

    def __init__(self, grid: SquareGrid, dtype=torch.float64, device=None):
        super().__init__()
        self.n = grid.n
        self.L = grid.L
        self.h = grid.h
        mu = _laplacian_eigenvalues(grid)
        self.register_buffer(
            "mu",
            torch.as_tensor(mu, dtype=dtype, device=device),
            persistent=True,
        )

    def apply_div_a_grad(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Batched -div(a grad u) with homogeneous Dirichlet BC.

        u, a: (B, n, n)
        """
        expected = (self.n, self.n)
        if u.ndim != 3 or a.shape != u.shape or u.shape[-2:] != expected:
            raise ValueError("u and a must have shape (B, n, n)")
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
        a0: float | torch.Tensor = 1.0,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        """Apply (a0 (-Delta) + sigma I)^{-1} via DST. rhs: (B, n, n).

        a0 may be a scalar or a per-batch tensor of shape (B,).
        """
        if rhs.ndim != 3 or rhs.shape[-2:] != (self.n, self.n):
            raise ValueError("rhs must have shape (B, n, n)")
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and nonnegative")
        rhs_hat = dst2_ortho(rhs)
        if isinstance(a0, torch.Tensor):
            if a0.numel() not in (1, rhs.shape[0]):
                raise ValueError("tensor a0 must be scalar or have shape (B,)")
            if not torch.all(torch.isfinite(a0)) or torch.any(a0 <= 0.0):
                raise ValueError("a0 must be finite and positive")
            a0b = a0.reshape(-1, 1, 1).to(dtype=rhs.dtype, device=rhs.device)
            denom = a0b * self.mu + sigma
        else:
            a0f = float(a0)
            if not math.isfinite(a0f) or a0f <= 0.0:
                raise ValueError("a0 must be finite and positive")
            denom = a0f * self.mu + sigma
        u_hat = rhs_hat / denom
        return dst2_ortho(u_hat)

    def l2_norm(self, v: torch.Tensor) -> torch.Tensor:
        """Per-batch discrete L2 norm. v: (B, n, n) -> (B,)."""
        return torch.sqrt(torch.sum(v * v, dim=(1, 2)) * self.h**2)


class ScreenedLGFBankTorch(nn.Module):
    """Fixed screened Dirichlet LGF bank."""

    def __init__(
        self,
        grid: SquareGrid,
        lambdas: np.ndarray | None = None,
        J: int = 6,
        a0: float = 1.0,
        include_poisson: bool = True,
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()
        self.ops = TorchSquareOps(grid, dtype=dtype, device=device)
        if lambdas is None:
            lambdas = default_screening_values(
                grid.h, J=J, include_poisson=include_poisson
            )
        lam = torch.as_tensor(lambdas, dtype=dtype, device=device)
        if lam.ndim != 1 or lam.numel() == 0:
            raise ValueError("lambdas must be a nonempty one-dimensional array")
        if not torch.all(torch.isfinite(lam)) or torch.any(lam < 0.0):
            raise ValueError("lambdas must be finite and nonnegative")
        if not math.isfinite(a0) or a0 <= 0.0:
            raise ValueError("a0 must be finite and positive")
        self.register_buffer("lambdas", lam, persistent=True)
        self.a0 = float(a0)

    @property
    def J(self) -> int:
        return int(self.lambdas.numel())

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """r: (B, n, n) -> z: (B, J, n, n)."""
        if r.ndim != 3 or r.shape[-2:] != (self.ops.n, self.ops.n):
            raise ValueError("r must have shape (B, n, n)")
        r_hat = dst2_ortho(r)
        denominators = self.a0 * self.ops.mu[None, :, :] + self.lambdas[:, None, None]
        z_hat = r_hat[:, None, :, :] / denominators[None, :, :, :]
        return dst2_ortho(z_hat)
