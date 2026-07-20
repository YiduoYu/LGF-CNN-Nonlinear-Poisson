"""PyTorch discrete operators and Dirichlet LGF bank for the rectangle problem."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import SquareGrid
from .lgf_bank import default_screening_values
from .lgf_inverse import _laplacian_eigenvalues


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
        self.register_buffer("S", dst_matrix(grid.n, dtype=dtype, device=device), persistent=True)
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
        h = self.h
        # Pad Dirichlet halo: (B, n+2, n+2)
        U = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)

        # Face diffusivities (arithmetic mean), shapes (B, n, n+1) and (B, n+1, n)
        ax_left = a
        ax_right = a
        ax_inner = 0.5 * (a[:, :, :-1] + a[:, :, 1:])
        ax = torch.cat([ax_left[:, :, :1], ax_inner, ax_right[:, :, -1:]], dim=2)

        ay_down = a
        ay_up = a
        ay_inner = 0.5 * (a[:, :-1, :] + a[:, 1:, :])
        ay = torch.cat([ay_down[:, :1, :], ay_inner, ay_up[:, -1:, :]], dim=1)

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
        rhs_hat = torch.einsum("ij,bjk,kl->bil", self.S, rhs, self.S)
        if isinstance(a0, torch.Tensor):
            a0b = a0.reshape(-1, 1, 1).to(dtype=rhs.dtype, device=rhs.device)
            denom = a0b * self.mu + sigma
        else:
            denom = float(a0) * self.mu + sigma
        u_hat = rhs_hat / denom
        return torch.einsum("ij,bjk,kl->bil", self.S, u_hat, self.S)

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
        self.register_buffer("lambdas", lam, persistent=True)
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
