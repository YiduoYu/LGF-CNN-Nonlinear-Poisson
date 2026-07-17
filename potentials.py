"""
Single-layer and double-layer potential matrices, and difference potential
computation.

Provides:
  - get_sl: Assemble single-layer potential matrix
  - get_dl: Assemble double-layer potential matrix
  - get_dp_fft: Compute difference potential via FFT
"""

import numpy as np


def get_sl(G, source, target):
    """Vectorized assembly of single-layer potential matrix.

    S(i,j) = G(|target_i - source_j|)

    Parameters
    ----------
    G : ndarray — LGF table (0-based distances), shape (N, N).
    source : ndarray (n, 2) — source grid indices.
    target : ndarray (m, 2) — target grid indices.

    Returns
    -------
    S : ndarray (m, n)
    """
    # Vectorized: dx[i,j] = |target[i,0] - source[j,0]|
    dx = np.abs(target[:, 0:1] - source[None, :, 0])
    dy = np.abs(target[:, 1:2] - source[None, :, 1])
    return G[dx, dy]


def get_dl(G, source, target, N_plus):
    """Vectorized double-layer potential matrix.

    D(i,j) = sum_{neighbors k of source_j}
        (G(|target_i - source_j|) - G(|target_i - k|)) * (1 - N_plus(k))
    """
    m = target.shape[0]
    n = source.shape[0]
    D = np.zeros((m, n))

    # Distances from all targets to all sources
    dx0 = np.abs(target[:, 0:1] - source[None, :, 0])
    dy0 = np.abs(target[:, 1:2] - source[None, :, 1])
    G0 = G[dx0, dy0]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for di, dj in directions:
        sn = source + np.array([di, dj])
        # Bounds mask
        valid = (sn[:, 0] >= 0) & (sn[:, 0] < N_plus.shape[0]) & \
                (sn[:, 1] >= 0) & (sn[:, 1] < N_plus.shape[1])
        if not valid.any():
            continue
        # Distances to neighbor positions
        dx_n = np.abs(target[:, 0:1] - sn[None, :, 0])
        dy_n = np.abs(target[:, 1:2] - sn[None, :, 1])
        Gn = G[dx_n, dy_n]
        weight = 1 - N_plus[sn[:, 0], sn[:, 1]]
        D += (G0 - Gn) * weight[None, :]

    return D


def get_dp_fft(eigenvector, Lh, gam, uh, N, xdom, h, sigma, full_domain=True):
    """
    Compute the difference potential for a given density on the discrete
    gamma set.

    Solves: Lh[DP] = {0, x in M^+; Lh[rho_gamma], x in M^-}
    using the FFT-based sine transform solver.

    Parameters
    ----------
    eigenvector : ndarray
        Eigenvalues of discrete Laplacian (column-major order).
    Lh : scipy.sparse matrix
        Discrete Laplacian operator (N^2 x N^2).
    gam : ndarray
        Indices of discrete gamma points (0-based), shape (n, 2).
    uh : ndarray
        Density values at gamma points, length n.
    N : int
        Grid dimension.
    xdom : ndarray
        Boolean mask: True where Lh[rho_gamma] should be applied (exterior).
    h : float
        Mesh size.
    sigma : float
        Modified Helmholtz coefficient.
    full_domain : bool
        If True, return full-domain solution; if False, return gamma values only.

    Returns
    -------
    output : ndarray
        Difference potential (full domain or gamma points only).
    """
    # Place density values on the grid
    var1 = np.zeros((N, N))
    n = gam.shape[0]
    for p in range(n):
        var1[gam[p, 0], gam[p, 1]] = uh[p]

    # Apply discrete Laplacian to the density
    var2 = var1.ravel('F')  # Fortran/column-major order
    var3 = Lh.dot(var2)
    var4 = var3.reshape(N, N, order='F')

    # Restrict to exterior domain (xdom = ~mp)
    var5 = var4 * xdom

    # Solve Poisson equation on full rectangular domain via FFT
    um = _fft_solve_internal(var5, sigma, h, N, eigenvector)

    if full_domain:
        return um
    else:
        # Return values at gamma points only
        output = np.zeros(n)
        for p in range(n):
            output[p] = um[gam[p, 0], gam[p, 1]]
        return output


# Internal FFT solver (avoid circular imports)
def _fft_solve_internal(fm, sigma, h, N, eigenvector):
    """Internal FFT solver, same as fft_solve but avoids import issues."""
    from .fft_solver import fst2d

    scale = (2.0 / (N + 1)) ** 2

    # Forward 2D sine transform
    F = fst2d(fst2d(fm, axis=0), axis=1)

    # Solve in Fourier space
    fvec = F.ravel('F')
    uvec = fvec / (eigenvector / (h ** 2) - sigma) * scale
    U = uvec.reshape(N, N, order='F')

    # Inverse 2D sine transform
    um = fst2d(fst2d(U, axis=0), axis=1)

    return um
