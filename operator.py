"""
Discrete Laplacian operator construction and eigenvalue computation.

Provides:
  - get_operator: Build sparse discrete Laplacian (2nd or 4th order)
  - get_eigen: Compute eigenvalues of discrete Laplacian for FFT solver
"""

import numpy as np
from scipy.sparse import diags, kron, eye as speye


def get_operator(k, sigma, h, N, iOrder):
    """
    Construct the sparse discrete Laplacian operator Lh.

    Lh = k * A / h^2 - sigma * I

    where A is the 2D discrete Laplacian built via Kronecker sums.

    Parameters
    ----------
    k : float
        Diffusion coefficient.
    sigma : float
        Modified Helmholtz coefficient (sigma=0 gives Poisson/Laplace).
    h : float
        Mesh size.
    N : int
        Grid dimension (N x N grid).
    iOrder : int
        Finite difference order: 2 (3-point stencil) or 4 (5-point stencil).

    Returns
    -------
    Lh : scipy.sparse.csr_matrix
        Sparse N^2 x N^2 operator matrix.
    """
    e = np.ones(N)

    if iOrder == 2:
        # 3-point stencil: [1, -2, 1]
        P = diags([e, -2 * e, e], [-1, 0, 1], shape=(N, N), format='csr')
    elif iOrder == 4:
        # 5-point stencil: [-1/12, 16/12, -30/12, 16/12, -1/12]
        P = diags(
            [-e / 12, 16 * e / 12, -30 * e / 12, 16 * e / 12, -e / 12],
            [-2, -1, 0, 1, 2],
            shape=(N, N),
            format='csr'
        )
    else:
        raise ValueError(f'Order {iOrder} not supported. Use 2 or 4.')

    I = speye(N, format='csr')

    # 2D Laplacian via Kronecker sum
    A = kron(P, I, format='csr') + kron(I, P, format='csr')

    # Modified Helmholtz operator
    Lh = k * A / (h ** 2) - sigma * speye(N ** 2, format='csr')

    return Lh


def get_eigen(N, dim, iOrder):
    """
    Compute eigenvalues of the discrete Laplacian.

    Used in the FFT-based solver to invert the operator in Fourier space.
    Returns a 1D array in column-major (Fortran) order matching MATLAB's
    reshape convention.

    Parameters
    ----------
    N : int
        Grid dimension.
    dim : int
        Spatial dimension (2 or 3).
    iOrder : int
        Finite difference order (2 or 4).

    Returns
    -------
    eigenvector : ndarray
        Eigenvalues of the discrete Laplacian, length N^dim,
        in column-major (Fortran) order.
    """
    l = np.arange(1, N + 1, dtype=np.float64)

    if iOrder == 2:
        v = (np.cos(l * np.pi / (N + 1)) - 1) * 2
    elif iOrder == 4:
        v = -(np.cos(l * np.pi / (N + 1)) - 1) * (np.cos(l * np.pi / (N + 1)) - 7) / 3
    else:
        raise ValueError(f'Order {iOrder} not supported.')

    if dim == 2:
        # Build 2D eigenvalues in column-major order matching MATLAB
        vx, vy = np.meshgrid(v, v, indexing='ij')  # like MATLAB ndgrid
        eigen_2d = vx + vy
        eigenvector = eigen_2d.ravel('F')  # Fortran order
    elif dim == 3:
        # 3D eigenvalues in column-major order
        vx, vy, vz = np.meshgrid(v, v, v, indexing='ij')
        eigen_3d = vx + vy + vz
        eigenvector = eigen_3d.ravel('F')
    else:
        raise ValueError(f'Dimension {dim} not supported.')

    return eigenvector
