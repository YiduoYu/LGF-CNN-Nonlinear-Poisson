"""
Lagrange interpolation basis functions in 1D and 2D.

Provides:
  - get_lagrange_1d: Evaluate 1D Lagrange basis and derivative
  - get_lagrange_2d: Evaluate 2D tensor-product Lagrange basis and partial derivatives
"""

import numpy as np


def get_lagrange_1d(xv, x, m):
    """
    Evaluate the m-th Lagrange basis polynomial and its derivative at x.

    Parameters
    ----------
    xv : ndarray
        Interpolation nodes.
    x : float
        Evaluation point.
    m : int
        Index of the basis function (0-based).

    Returns
    -------
    u : float
        Value of m-th Lagrange basis at x.
    du : float
        Value of derivative of m-th Lagrange basis at x.
    """
    n = len(xv)

    # Evaluate basis function value: L_m(x) = ∏_{i≠m} (x - x_i) / (x_m - x_i)
    u = 1.0
    for i in range(n):
        if i != m:
            u *= (x - xv[i]) / (xv[m] - xv[i])

    # Evaluate derivative: L'_m(x) = Σ_{i≠m} 1/(x_m - x_i) * ∏_{j≠i,m} (x - x_j)/(x_m - x_j)
    du = 0.0
    for i in range(n):
        if i != m:
            p = 1.0 / (xv[m] - xv[i])
            for j in range(n):
                if j != i and j != m:
                    p *= (x - xv[j]) / (xv[m] - xv[j])
            du += p

    return u, du


def get_lagrange_2d(xv, yv, eval_pt):
    """
    Evaluate 2D tensor-product Lagrange basis functions and their partial
    derivatives at a given point.

    The basis functions are tensor products of 1D Lagrange basis functions
    in x and y directions. Output is ordered in column-major (x varies first,
    then y), matching the MATLAB implementation:

        for j in range(ny):       # y direction
            for i in range(nx):   # x direction
                count = count + 1
                u[count] = Lx_i(x) * Ly_j(y)

    Parameters
    ----------
    xv : ndarray
        Interpolation nodes in x-direction.
    yv : ndarray
        Interpolation nodes in y-direction.
    eval_pt : tuple or list
        (x, y) evaluation point.

    Returns
    -------
    u : ndarray
        Basis function values, length nx*ny.
    u_x : ndarray
        x-derivatives of basis functions, length nx*ny.
    u_y : ndarray
        y-derivatives of basis functions, length nx*ny.
    """
    x, y = eval_pt[0], eval_pt[1]

    nx = len(xv)
    ny = len(yv)

    u = np.zeros(nx * ny)
    u_x = np.zeros(nx * ny)
    u_y = np.zeros(nx * ny)

    count = 0
    for j in range(ny):
        uy, dudy = get_lagrange_1d(yv, y, j)
        for i in range(nx):
            ux, dudx = get_lagrange_1d(xv, x, i)
            u[count] = ux * uy
            u_x[count] = dudx * uy
            u_y[count] = ux * dudy
            count += 1

    return u, u_x, u_y
