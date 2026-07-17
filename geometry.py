"""
Geometry definitions: level set functions and boundary normals for various
domain shapes.

Provides:
  - get_geometry: Return level set function psi and (optional) analytic normals
  - get_unit_normal_star: Compute outward unit normal for star-shaped curve
"""

import numpy as np


def get_geometry(geom, ar=1.0):
    """
    Return the level set function psi(x,y) and optionally analytic normal
    functions for the specified geometry.

    The boundary is the zero level set: psi(x,y) = 0.
    The interior M^+ is: psi(x,y) <= 0.

    Parameters
    ----------
    geom : str
        Geometry type. Options:
        - 'ellipse': x^2/r1^2 + y^2/r2^2 - 1 = 0
        - 'leaf': sqrt((x-0.03)^2 + (y-0.25)^2) - sin(3*atan2(y-0.25, x-0.03))
        - 'star': sqrt(x^2+y^2) - (0.75 + 0.2*sin(5*atan2(y,x)))
        - 'butterfly': sqrt(x^2+y^2) - (1 - cos(2*atan2(y,x))*sin(4*atan2(y,x)))
    ar : float
        Aspect ratio (for ellipse: r1=1, r2=1/ar).

    Returns
    -------
    psi : callable
        Level set function psi(x, y).
    nx : callable or None
        x-component of outward unit normal (None if not available analytically).
    ny : callable or None
        y-component of outward unit normal (None if not available analytically).
    domain_bounds : tuple
        (a, b, l) — auxiliary domain parameters.
    """
    if geom == 'ellipse':
        r1 = 1.0
        r2 = 1.0 / ar

        def psi(x, y):
            return x ** 2 / (r1 ** 2) + y ** 2 / (r2 ** 2) - 1.0

        def nx(x, y):
            denom = np.sqrt((2 * x / r1 ** 2) ** 2 + (2 * y / r2 ** 2) ** 2)
            return 2 * x / (r1 ** 2) / denom

        def ny(x, y):
            denom = np.sqrt((2 * x / r1 ** 2) ** 2 + (2 * y / r2 ** 2) ** 2)
            return 2 * y / (r2 ** 2) / denom

        a, b, l = -1.0, 1.0, 0.15

    elif geom == 'leaf':

        def psi(x, y):
            return np.sqrt((x - 0.03) ** 2 + (y - 0.25) ** 2) - np.sin(3 * np.arctan2(y - 0.25, x - 0.03))

        nx = None
        ny = None
        a, b, l = -1.0, 1.0, 0.15

    elif geom == 'star':

        def psi(x, y):
            return np.sqrt(x ** 2 + y ** 2) - (0.75 + 0.2 * np.sin(5 * np.arctan2(y, x)))

        nx = None  # Use get_unit_normal_star instead
        ny = None
        a, b, l = -1.0, 1.0, 0.11

    elif geom == 'butterfly':

        def psi(x, y):
            theta = np.arctan2(y, x)
            r = np.sqrt(x ** 2 + y ** 2)
            return r - (1.0 - np.cos(2 * theta) * np.sin(4 * theta))

        nx = None
        ny = None
        a, b, l = -2.0, 2.0, 0.15

    else:
        raise ValueError(f'Unknown geometry: {geom}')

    return psi, nx, ny, (a, b, l)


def get_unit_normal_star(x, y):
    """
    Compute the outward unit normal vector for the star-shaped curve:
        r(theta) = 0.75 + 0.2 * sin(5 * theta)

    Parameters
    ----------
    x : ndarray
        x-coordinates of points on the curve.
    y : ndarray
        y-coordinates of points on the curve.

    Returns
    -------
    nx : ndarray
        x-component of outward unit normal.
    ny : ndarray
        y-component of outward unit normal.
    """
    theta = np.arctan2(y, x)

    # Verify points are on the curve
    r_actual = np.sqrt(x ** 2 + y ** 2)
    r_expected = 0.75 + 0.2 * np.sin(5 * theta)
    if np.max(np.abs(r_actual - r_expected)) > 1e-10:
        import warnings
        warnings.warn('Some points may not be on the specified star curve')

    # r(theta) and its derivative
    r = 0.75 + 0.2 * np.sin(5 * theta)
    dr_dtheta = np.cos(5 * theta)

    # Tangent vector: (dx/dtheta, dy/dtheta)
    dx_dtheta = dr_dtheta * np.cos(theta) - r * np.sin(theta)
    dy_dtheta = dr_dtheta * np.sin(theta) + r * np.cos(theta)

    # Normal vector (perpendicular to tangent, pointing outward)
    normal_x = -dy_dtheta
    normal_y = dx_dtheta

    # Normalize
    normal_mag = np.sqrt(normal_x ** 2 + normal_y ** 2)
    nx = normal_x / normal_mag
    ny = normal_y / normal_mag

    # Ensure outward direction (dot with radial direction should be positive)
    radial_x = x / np.sqrt(x ** 2 + y ** 2)
    radial_y = y / np.sqrt(x ** 2 + y ** 2)
    dot_product = nx * radial_x + ny * radial_y

    # Flip if pointing inward
    for i in range(len(dot_product)):
        if dot_product[i] < 0:
            nx[i] = -nx[i]
            ny[i] = -ny[i]

    return nx, ny
