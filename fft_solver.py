"""
FFT-based Poisson solver on rectangular domains using fast sine transforms.

Provides:
  - fst2d: Fast sine transform (DST-I) in 2D using FFT
  - fft_solve: Solve Poisson/modified Helmholtz on rectangular domain via FFT
"""

import numpy as np


def fst2d(F, axis):
    """
    Compute the fast sine transform (DST-I) in 2D along a specified axis.

    Uses the FFT-based method: pad with zeros, compute FFT, extract
    imaginary part. This is the unnormalized DST-I.

    Parameters
    ----------
    F : ndarray
        2D input array.
    axis : int
        Axis along which to compute the transform (0 or 1).
        axis=0 corresponds to MATLAB dim=1 (columns).
        axis=1 corresponds to MATLAB dim=2 (rows).

    Returns
    -------
    u : ndarray
        Transformed array, same shape as F.
    """
    m = F.shape[axis]

    # Pad with one zero before the first element along each axis
    # (matching MATLAB padarray(F, [1 1], 'pre'))
    A = np.pad(F, ((1, 0), (1, 0)), mode='constant')

    # FFT along specified axis with length 2*m+2
    w = -np.fft.fft(A, 2 * m + 2, axis=axis)

    if axis == 0:
        u = np.imag(w[1:m + 1, 1:])
    else:  # axis == 1
        u = np.imag(w[1:, 1:m + 1])

    return u


def fft_solve(fm, sigma, dim, h, N, eigenvector):
    """
    Solve Poisson / modified Helmholtz equation on rectangular domain via FFT.

    Solves: k * laplacian(u) - sigma * u = fm   on [a,b]^2
    with homogeneous Dirichlet BC on the auxiliary domain boundary.

    Parameters
    ----------
    fm : ndarray
        Right-hand side on the N x N grid (interior values only).
    sigma : float
        Modified Helmholtz coefficient.
    dim : int
        Spatial dimension (2 for 2D).
    h : float
        Mesh size.
    N : int
        Grid dimension.
    eigenvector : ndarray
        Eigenvalues of discrete Laplacian in column-major order.

    Returns
    -------
    um : ndarray
        Solution on the N x N grid.
    """
    scale = (2.0 / (N + 1)) ** dim
    Ntotal = N ** dim

    # Forward 2D sine transform
    if dim == 2:
        F = fst2d(fst2d(fm, axis=0), axis=1)
    elif dim == 3:
        raise NotImplementedError('3D FFT solve not yet implemented in Python')
    else:
        raise ValueError(f'Dimension {dim} not supported.')

    # Solve in Fourier space
    fvec = F.ravel('F')  # Fortran/column-major order
    uvec = fvec / (eigenvector / (h ** 2) - sigma) * scale

    if dim == 2:
        U = uvec.reshape(N, N, order='F')
    elif dim == 3:
        U = uvec.reshape(N, N, N, order='F')

    # Inverse 2D sine transform (same as forward for unnormalized DST-I)
    if dim == 2:
        um = fst2d(fst2d(U, axis=0), axis=1)
    elif dim == 3:
        um = fst2d(fst2d(fst2d(U, axis=0), axis=1), axis=2)

    return um
