"""
Manufactured solutions for convergence testing.

Provides exact solutions, their gradients, and forcing terms for
verification of the solver via the method of manufactured solutions (MMS).

Available manufactured solutions:
  - 'one':    u = 1,                         f = 0
  - 'poly1':  u = 3x + 5y,                  f = 0
  - 'poly2':  u = x^2 + y^2,                f = 4
  - 'poly3':  u = x^3 + 3y^3,              f = 6x + 18y
  - 'trig1':  u = sin(x)*cos(y),            f = -2*sin(x)*cos(y)
  - 'trig2':  u = sin(3*pi*x)*cos(5*pi*y),  f = -34*pi^2*sin(3*pi*x)*cos(5*pi*y)
  - 'trig3':  u = cos(x)*cos(y),            f = -2*cos(x)*cos(y)
  - 'trig4':  u = sin(3*x)*cos(5*y),        f = -34*sin(3*x)*cos(5*y)
"""

import numpy as np


def manufactured_solution(x, y, ms):
    """
    Evaluate the manufactured solution u(x, y).

    Parameters
    ----------
    x : ndarray
        x-coordinates.
    y : ndarray
        y-coordinates.
    ms : str
        Name of manufactured solution.

    Returns
    -------
    u : ndarray
        Exact solution values.
    """
    if ms == 'poly2':
        return x ** 2 + y ** 2
    elif ms == 'poly1':
        return 3.0 * x + 5.0 * y
    elif ms == 'poly3':
        return x ** 3 + 3.0 * y ** 3
    elif ms == 'trig1':
        return np.sin(x) * np.cos(y)
    elif ms == 'trig4':
        return np.sin(3.0 * x) * np.cos(5.0 * y)
    elif ms == 'trig3':
        return np.cos(x) * np.cos(y)
    elif ms == 'one':
        return np.ones_like(x)
    elif ms == 'trig2':
        return np.sin(3.0 * np.pi * x) * np.cos(5.0 * np.pi * y)
    else:
        raise ValueError(f'Unknown manufactured solution: {ms}')


def manufactured_gradient(x, y, ms):
    """
    Evaluate the gradient (du/dx, du/dy) of the manufactured solution.

    Parameters
    ----------
    x : ndarray
        x-coordinates.
    y : ndarray
        y-coordinates.
    ms : str
        Name of manufactured solution.

    Returns
    -------
    ux : ndarray
        x-derivative of the exact solution.
    uy : ndarray
        y-derivative of the exact solution.
    """
    if ms == 'poly2':
        ux = 2.0 * x
        uy = 2.0 * y
    elif ms == 'poly1':
        ux = 3.0 * np.ones_like(x)
        uy = 5.0 * np.ones_like(x)
    elif ms == 'poly3':
        ux = 3.0 * x ** 2
        uy = 9.0 * y ** 2
    elif ms == 'trig1':
        ux = np.cos(x) * np.cos(y)
        uy = -np.sin(x) * np.sin(y)
    elif ms == 'trig4':
        ux = 3.0 * np.cos(3.0 * x) * np.cos(5.0 * y)
        uy = -5.0 * np.sin(3.0 * x) * np.sin(5.0 * y)
    elif ms == 'trig3':
        ux = -np.sin(x) * np.cos(y)
        uy = -np.cos(x) * np.sin(y)
    elif ms == 'one':
        ux = np.zeros_like(x)
        uy = np.zeros_like(x)
    elif ms == 'trig2':
        ux = 3.0 * np.pi * np.cos(3.0 * np.pi * x) * np.cos(5.0 * np.pi * y)
        uy = -5.0 * np.pi * np.sin(3.0 * np.pi * x) * np.sin(5.0 * np.pi * y)
    else:
        raise ValueError(f'Unknown manufactured solution: {ms}')

    return ux, uy


def manufactured_forcing(x, y, ms):
    """
    Evaluate the forcing term f(x, y) = laplacian(u) for the manufactured
    solution (i.e., the right-hand side of the Poisson equation,
    laplacian(u) - sigma*u = f).

    For pure Laplace/Poisson: f = laplacian(u).
    For modified Helmholtz: f = laplacian(u) - sigma*u.

    Parameters
    ----------
    x : ndarray
        x-coordinates.
    y : ndarray
        y-coordinates.
    ms : str
        Name of manufactured solution.

    Returns
    -------
    f : ndarray
        Forcing term values.
    """
    if ms == 'poly2':
        return 4.0 * np.ones_like(x)
    elif ms == 'poly3':
        return 3.0 * 2.0 * x + 3.0 * 3.0 * 2.0 * y  # 6x + 18y
    elif ms == 'trig1':
        return -2.0 * np.sin(x) * np.cos(y)
    elif ms == 'trig4':
        return -34.0 * np.sin(3.0 * x) * np.cos(5.0 * y)
    elif ms == 'trig3':
        return -2.0 * np.cos(x) * np.cos(y)
    elif ms == 'one':
        return np.zeros_like(x)
    elif ms == 'poly1':
        return np.zeros_like(x)
    elif ms == 'trig2':
        return -34.0 * np.pi ** 2 * np.sin(3.0 * np.pi * x) * np.cos(5.0 * np.pi * y)
    else:
        raise ValueError(f'Unknown manufactured solution: {ms}')
