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
        """Store grid spacing and cache Laplacian eigenvalues as a buffer."""
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
        padded_u = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)

        # Face diffusivities (arithmetic mean), shapes (B, n, n+1) and (B, n+1, n)
        ax_inner = 0.5 * (a[:, :, :-1] + a[:, :, 1:])
        ax = torch.cat([a[:, :, :1], ax_inner, a[:, :, -1:]], dim=2)

        ay_inner = 0.5 * (a[:, :-1, :] + a[:, 1:, :])
        ay = torch.cat([a[:, :1, :], ay_inner, a[:, -1:, :]], dim=1)

        ux_left = (padded_u[:, 1:-1, 1:-1] - padded_u[:, 1:-1, :-2]) / h
        ux_right = (padded_u[:, 1:-1, 2:] - padded_u[:, 1:-1, 1:-1]) / h
        uy_down = (padded_u[:, 1:-1, 1:-1] - padded_u[:, :-2, 1:-1]) / h
        uy_up = (padded_u[:, 2:, 1:-1] - padded_u[:, 1:-1, 1:-1]) / h

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
        """Discrete L2 norm over the trailing grid axes, e.g. (B, n, n) -> (B,)."""
        return torch.sqrt(torch.sum(v * v, dim=(-2, -1)) * self.h**2)


class ScreenedLGFBankTorch(nn.Module):
    """Fixed screened Dirichlet LGF bank."""

    def __init__(
        self,
        grid: SquareGrid,
        lambdas: np.ndarray | None = None,
        J: int = 6,
        include_poisson: bool = True,
        dtype=torch.float64,
        device=None,
    ):
        """Create a fixed bank from explicit or default screening values."""
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
        self.register_buffer("lambdas", lam, persistent=True)

    @property
    def J(self) -> int:
        """Number of screened LGF filters."""
        return int(self.lambdas.numel())

    def apply(self, r: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        """Apply the bank with a per-sample reference coefficient.

        z[b, j] = (a0[b] (-Delta_h) + lambda_j I)^{-1} r[b]

        The screening term lambda_j is an absolute length scale (1/ell^2) and
        does not scale with a0.  The lambda_j = 0 Poisson filter is therefore
        exactly (-1)-homogeneous in a0 while the screened filters are only
        approximately so; see the scaling note in the README.

        r: (B, n, n), a0: (B,) positive -> z: (B, J, n, n)
        """
        if r.ndim != 3 or r.shape[-2:] != (self.ops.n, self.ops.n):
            raise ValueError("r must have shape (B, n, n)")
        if a0.ndim != 1 or a0.shape[0] != r.shape[0]:
            raise ValueError("a0 must have shape (B,)")
        r_hat = dst2_ortho(r)
        denominators = (
            a0[:, None, None, None] * self.ops.mu[None, None, :, :]
            + self.lambdas[None, :, None, None]
        )
        z_hat = r_hat[:, None, :, :] / denominators
        return dst2_ortho(z_hat)
