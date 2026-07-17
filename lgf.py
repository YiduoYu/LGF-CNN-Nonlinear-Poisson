"""
Lattice Green's functions for the discrete Laplace and modified Helmholtz
equations in 2D.

Provides:
  - evaluate_LGF: LGF for Laplace equation (sigma=0) via integral representation
  - LGF_FFT: LGF for modified Helmholtz equation (sigma != 0) via FFT
"""

import numpy as np
from scipy.integrate import quad
from numpy import cos, sin, sinh, exp, abs as np_abs, pi


# ---------------------------------------------------------------------------
# Lattice Green's function for Laplace (sigma = 0)
# ---------------------------------------------------------------------------

def evaluate_LGF(m, n):
    """
    Evaluate the lattice Green's function G(m,n) for the discrete Laplace
    equation in 2D.

    For small distances (m^2 + n^2 <= 30), uses numerical quadrature of the
    integral representation. For large distances, uses an asymptotic expansion.

    G(m,n) = (1/2π) ∫_0^π [(e^{-|m|s} cos(n y) - 1) / sinh(s)] dy
    where cosh(s) = 2 - cos(y)

    Parameters
    ----------
    m : int or float
        Distance in x-direction (0-based grid distance).
    n : int or float
        Distance in y-direction (0-based grid distance).

    Returns
    -------
    result : float
        Lattice Green's function value G(m,n).

    Notes
    -----
    This function takes 0-based distances (unlike the MATLAB version which
    takes 1-based indices and internally subtracts 1).
    """
    mm = np.sqrt(m ** 2 + n ** 2)

    # Define the integrand
    def integrand(y):
        # s from cosh(s) = 2 - cos(y)
        s = np.arccosh(2.0 - cos(y))
        return (exp(-np_abs(m) * s) * cos(n * y) - 1.0) / sinh(s)

    if mm <= 30:
        # Numerical quadrature for small distances
        integral_result, _ = quad(integrand, 0.0, pi, epsrel=1e-13, epsabs=1e-13)
        result = integral_result / (2.0 * pi)
    else:
        # Asymptotic expansion for large distances
        gamma = 0.5772156649015328606
        theta = np.arctan2(n, m)

        result = (
            -1.0 / (2.0 * pi) * (np.log(mm) + gamma + np.log(8.0) / 2.0)
            + cos(4.0 * theta) / (mm ** 2) / 24.0 / pi
            + (25.0 * cos(8.0 * theta) + 18.0 * cos(4.0 * theta)) / (mm ** 4) / 480.0 / pi
            + (490.0 * cos(12.0 * theta) + 459.0 * cos(8.0 * theta)) / (mm ** 6) / 2016.0 / pi
        )

    return result


# ---------------------------------------------------------------------------
# Precomputed LGF table (cached for speed)
# ---------------------------------------------------------------------------

def compute_LGF_table(N):
    """
    Precompute the LGF table for all distances up to N.

    Parameters
    ----------
    N : int
        Maximum grid distance.

    Returns
    -------
    G : ndarray
        N x N array of LGF values. G[i, j] = evaluate_LGF(i, j).
    """
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            G[i, j] = evaluate_LGF(i, j)
            G[j, i] = G[i, j]
    return G


# ---------------------------------------------------------------------------
# Lattice Green's function for modified Helmholtz (sigma != 0)
# ---------------------------------------------------------------------------

def LGF_FFT(h, sigma, N):
    """
    Compute the lattice Green's function for the modified Helmholtz equation
    via inverse FFT in k-space.

    Parameters
    ----------
    h : float
        Mesh size.
    sigma : float
        Modified Helmholtz coefficient (nonzero).
    N : int
        Number of grid points (must be even).

    Returns
    -------
    G : ndarray
        Lattice Green's function on the non-negative quadrant.
        Size: (N//2+1, N//2+1) or similar.
    """
    if N % 2 != 0:
        raise ValueError('N must be even for LGF_FFT')

    # Create grid in k-space from -π to π
    k = (2.0 * pi / N) * np.arange(-N // 2, N // 2)
    kx, ky = np.meshgrid(k, k, indexing='ij')

    # Denominator in k-space
    denominator = (h ** 2) * sigma + 4.0 - 2.0 * (cos(kx) + cos(ky))

    # G(k) = 1 / denominator
    G_k = 1.0 / denominator

    # Inverse FFT
    G_grid = np.fft.ifftn(np.fft.ifftshift(G_k))
    G_grid = np.fft.fftshift(G_grid)

    # Extract non-negative quadrant
    G = G_grid[N // 2:, N // 2:]

    # Take real part to remove numerical artifacts
    G = np.real(G)

    return G
