"""
Boundary discretization: finding boundary-grid intersections, interpolation
boxes, and extrapolation for corner points.

Provides:
  - get_intersect_ptr: Find boundary-grid intersection points via Newton's method
  - get_box3_index_v5: Determine optimal 3x3 interpolation boxes
  - get_extrap: Build extrapolation matrix for extra corner points
"""

import numpy as np
from scipy.optimize import fsolve


def get_intersect_ptr(a, l, h, mp, N, psi):
    """
    Find intersection points between the boundary (zero level set of psi)
    and the background Cartesian grid.

    For each gamma_ex grid point (exterior point adjacent to boundary),
    find the closest intersection of the line between the grid point and
    its interior neighbor with the boundary using Newton's method.

    Parameters
    ----------
    a : float
        Lower bound of auxiliary domain.
    l : float
        Padding length.
    h : float
        Mesh size.
    mp : ndarray
        Boolean mask: True for interior (M^+) points.
    N : int
        Grid dimension.
    psi : callable
        Level set function psi(x, y).

    Returns
    -------
    xq : ndarray
        x-coordinates of intersection points.
    yq : ndarray
        y-coordinates of intersection points.
    ind2 : ndarray
        Half-integer indices indicating intersection direction.
    """
    # Find gamma_ex and gamma_in points
    gamma_ex_list = []
    gamma_in_list = []

    for i in range(1, N - 1):  # 0-based, skip boundaries
        for j in range(1, N - 1):
            ss = (int(mp[i, j]) + int(mp[i + 1, j]) + int(mp[i - 1, j])
                  + int(mp[i, j + 1]) + int(mp[i, j - 1]))
            if 0 < ss < 5:
                if mp[i, j]:
                    gamma_in_list.append([i, j])
                else:
                    gamma_ex_list.append([i, j])

    gam_ex = np.array(gamma_ex_list)
    ngam_ex = len(gam_ex)

    xq = np.zeros(ngam_ex)
    yq = np.zeros(ngam_ex)
    ind2 = np.zeros((ngam_ex, 2))

    def _find_root_1d(f, x0, x_left, x_right):
        """Robust 1D root-finding: try fsolve, fall back to bisection."""
        import warnings
        # Try fsolve first
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                sol = fsolve(f, x0, xtol=1e-12, maxfev=2000, full_output=True)
                x_root = sol[0][0]
                # Check convergence and bounds
                if (sol[2] == 1  # converged
                        and x_left <= x_root <= x_right
                        and abs(f(x_root)) < 1e-10):
                    return x_root
            except Exception:
                pass

        # Fall back to bisection
        try:
            f_left = f(x_left)
            f_right = f(x_right)
            # If signs differ, use bisection
            if f_left * f_right < 0:
                xa, xb = x_left, x_right
                for _ in range(60):
                    xm = (xa + xb) / 2.0
                    fm = f(xm)
                    if abs(fm) < 1e-14 or (xb - xa) < 1e-15:
                        return xm
                    if f_left * fm < 0:
                        xb = xm
                        f_right = fm
                    else:
                        xa = xm
                        f_left = fm
                return (xa + xb) / 2.0
            else:
                # Same sign — return the endpoint with smaller |f|
                if abs(f_left) < abs(f_right):
                    return x_left if abs(f_left) < 1e-10 else np.nan
                else:
                    return x_right if abs(f_right) < 1e-10 else np.nan
        except Exception:
            return np.nan

    for p in range(ngam_ex):
        jj = gam_ex[p, 0]   # 0-based grid index
        kk = gam_ex[p, 1]   # 0-based grid index

        # Physical coordinates of this grid point
        gx = a - l + (jj + 1) * h
        gy = a - l + (kk + 1) * h

        temp = np.full((4, 2), np.nan)

        # Right neighbor: root of psi(x, gy) on [gx, gx+h]
        if mp[jj + 1, kk]:
            x_left = gx
            x_right = gx + h
            x0 = (x_left + x_right) / 2.0
            g = lambda s: psi(s, gy)
            x_root = _find_root_1d(g, x0, x_left, x_right)
            temp[0, :] = [x_root, gy]

        # Left neighbor: root of psi(x, gy) on [gx-h, gx]
        if mp[jj - 1, kk]:
            x_left = gx - h
            x_right = gx
            x0 = (x_left + x_right) / 2.0
            g = lambda s: psi(s, gy)
            x_root = _find_root_1d(g, x0, x_left, x_right)
            temp[1, :] = [x_root, gy]

        # Top neighbor: root of psi(gx, y) on [gy, gy+h]
        if mp[jj, kk + 1]:
            y_left = gy
            y_right = gy + h
            y0 = (y_left + y_right) / 2.0
            g = lambda s: psi(gx, s)
            y_root = _find_root_1d(g, y0, y_left, y_right)
            temp[2, :] = [gx, y_root]

        # Bottom neighbor: root of psi(gx, y) on [gy-h, gy]
        if mp[jj, kk - 1]:
            y_left = gy - h
            y_right = gy
            y0 = (y_left + y_right) / 2.0
            g = lambda s: psi(gx, s)
            y_root = _find_root_1d(g, y0, y_left, y_right)
            temp[3, :] = [gx, y_root]

        # Find closest intersection to the grid point
        grid_pt = np.array([gx, gy])
        dist = np.sqrt((temp[:, 0] - grid_pt[0]) ** 2 + (temp[:, 1] - grid_pt[1]) ** 2)

        # Choose the closest valid intersection
        valid = ~np.isnan(dist)
        if not np.any(valid):
            # Fallback: use midpoint of the first found interior neighbor
            if mp[jj + 1, kk]:
                ind_min = 0
            elif mp[jj - 1, kk]:
                ind_min = 1
            elif mp[jj, kk + 1]:
                ind_min = 2
            elif mp[jj, kk - 1]:
                ind_min = 3
            else:
                raise ValueError(f'No interior neighbor found for gamma_ex point ({jj},{kk})')
        else:
            ind_min = np.nanargmin(dist)

        # Store half-integer index indicating intersection direction
        if ind_min == 0:
            ind2[p, :] = [jj + 0.5, kk]
        elif ind_min == 1:
            ind2[p, :] = [jj - 0.5, kk]
        elif ind_min == 2:
            ind2[p, :] = [jj, kk + 0.5]
        elif ind_min == 3:
            ind2[p, :] = [jj, kk - 0.5]

        xq[p] = temp[ind_min, 0]
        yq[p] = temp[ind_min, 1]

    return xq, yq, ind2


def get_box3_index_v5(nq, np_mask, m_gamma, ind2, gam_ex):
    """
    Determine the optimal 3x3 box of grid points for Lagrange interpolation
    near each boundary intersection point.

    Selects the box with maximum overlap with M^+ (the interior) from
    6 candidate placements.

    Parameters
    ----------
    nq : int
        Number of query/intersection points.
    np_mask : ndarray
        Boolean mask for N^+ (interior + gamma_ex).
    m_gamma : ndarray
        Boolean mask for gamma points.
    ind2 : ndarray
        Half-integer intersection indices from get_intersect_ptr.
    gam_ex : ndarray
        Indices of gamma_ex points.

    Returns
    -------
    m_cell : ndarray
        Updated mask including extra corner points for interpolation.
    ll_box3 : ndarray
        Lower-left corner indices of 3x3 interpolation boxes.
    """
    N_grid = np_mask.shape[0]
    ll_box3 = np.zeros((nq, 2), dtype=int)
    m_cell = m_gamma.copy().astype(int)

    def box_sum_safe(ji, ki):
        """Count N^+ points in a 3x3 box; return -1 if out of bounds."""
        if (ji < 0 or ki < 0 or ji + 2 >= N_grid or ki + 2 >= N_grid):
            return -1
        return int(
            np_mask[ji, ki] + np_mask[ji + 1, ki] + np_mask[ji + 2, ki]
            + np_mask[ji, ki + 1] + np_mask[ji + 1, ki + 1] + np_mask[ji + 2, ki + 1]
            + np_mask[ji, ki + 2] + np_mask[ji + 1, ki + 2] + np_mask[ji + 2, ki + 2]
        )

    for i in range(nq):
        j1 = gam_ex[i, 0]
        k1 = gam_ex[i, 1]

        j2 = ind2[i, 0]
        k2 = ind2[i, 1]

        # Determine direction flag and base index
        if (j2 - j1) > 1e-12 and abs(k2 - k1) < 1e-12:
            j = j1
            k = k1
            fl = 1
        elif (j2 - j1) < -1e-12 and abs(k2 - k1) < 1e-12:
            j = j1 - 1
            k = k1
            fl = 1
        elif abs(j2 - j1) < 1e-12 and (k2 - k1) > 1e-12:
            j = j1
            k = k1
            fl = 2
        elif abs(j2 - j1) < 1e-12 and (k2 - k1) < -1e-12:
            j = j1
            k = k1 - 1
            fl = 2
        else:
            raise ValueError(f'No valid index direction for point {i}')

        if fl == 1:
            # Horizontal intersection — 6 candidate boxes
            candidates = [
                (j, k),
                (j - 1, k),
                (j, k - 1),
                (j - 1, k - 1),
                (j, k - 2),
                (j - 1, k - 2),
            ]
        elif fl == 2:
            # Vertical intersection — 6 candidate boxes
            candidates = [
                (j, k),
                (j - 1, k),
                (j, k - 1),
                (j - 1, k - 1),
                (j - 2, k),
                (j - 2, k - 1),
            ]
        else:
            raise ValueError(f'Missing cases for point {i}')

        # Evaluate candidates, pick the valid one with most interior points
        best_score = -1
        best_ll = None
        for ji, ki in candidates:
            score = box_sum_safe(ji, ki)
            if score > best_score:
                best_score = score
                best_ll = (ji, ki)

        if best_ll is None:
            # Fallback: try a simpler approach
            # Use the box anchored at (j, k) if valid, else try shifts
            for dji in [0, -1, 1, -2, 2]:
                for dki in [0, -1, 1, -2, 2]:
                    ji = j + dji
                    ki = k + dki
                    if box_sum_safe(ji, ki) >= 0:
                        best_ll = (ji, ki)
                        break
                if best_ll is not None:
                    break

        if best_ll is None:
            raise ValueError(
                f'No valid 3x3 box found for gamma_ex point ({j1},{k1}) '
                f'with intersection direction ({j2},{k2}). '
                f'Grid size: {N_grid}, j={j}, k={k}'
            )

        ll_box3[i, :] = best_ll

        # Update m_cell: mark all 9 points of the 3x3 box
        indx = ll_box3[i, 0]
        indy = ll_box3[i, 1]
        for di in range(3):
            for dj in range(3):
                m_cell[indx + di, indy + dj] = 1

    return m_cell, ll_box3


def get_extrap(N, gam2, ind_extra_ex, m_gamma):
    """
    Build the extrapolation matrix R that implicitly extends the density
    from gamma points to "extra" corner points beyond the boundary.

    Uses 2nd-order extrapolation in each cardinal direction, averaged
    over all available directions.

    Parameters
    ----------
    N : int
        Grid dimension.
    gam2 : ndarray
        Augmented gamma set indices (gamma + extra points).
    ind_extra_ex : ndarray
        Indices of extra exterior points.
    m_gamma : ndarray
        Boolean mask for gamma points.

    Returns
    -------
    R : ndarray
        Extrapolation matrix of size (n_extra_ex, ngam2).
    """
    ngam2 = gam2.shape[0]

    # Build index lookup: ChiMat[i, j] = global index in gam2, or 0
    ChiMat = np.zeros((N, N), dtype=int)
    for i in range(ngam2):
        ChiMat[gam2[i, 0], gam2[i, 1]] = i

    n_extra = ind_extra_ex.shape[0]
    R = np.zeros((n_extra, ngam2))
    R_flag = np.zeros(n_extra)

    # 2nd-order extrapolation in each cardinal direction
    # Formula: u_extra = 2*u_neighbor - u_next (linear extrapolation)
    for i in range(n_extra):
        tx = ind_extra_ex[i, 0]
        ty = ind_extra_ex[i, 1]

        # Right direction (x+1, x+2)
        if (tx + 1 < N and ty < N
                and m_gamma[tx + 1, ty] > 0
                and tx + 2 < N
                and m_gamma[tx + 2, ty] > 0):
            R[i, ChiMat[tx + 1, ty]] = -2
            R[i, ChiMat[tx + 2, ty]] = 1
            R_flag[i] += 1

        # Left direction (x-1, x-2)
        if (tx - 1 >= 0 and m_gamma[tx - 1, ty] > 0
                and tx - 2 >= 0 and m_gamma[tx - 2, ty] > 0):
            R[i, ChiMat[tx - 1, ty]] = -2
            R[i, ChiMat[tx - 2, ty]] = 1
            R_flag[i] += 1

        # Up direction (y+1, y+2)
        if (ty + 1 < N and m_gamma[tx, ty + 1] > 0
                and ty + 2 < N and m_gamma[tx, ty + 2] > 0):
            R[i, ChiMat[tx, ty + 1]] = -2
            R[i, ChiMat[tx, ty + 2]] = 1
            R_flag[i] += 1

        # Down direction (y-1, y-2)
        if (ty - 1 >= 0 and m_gamma[tx, ty - 1] > 0
                and ty - 2 >= 0 and m_gamma[tx, ty - 2] > 0):
            R[i, ChiMat[tx, ty - 1]] = -2
            R[i, ChiMat[tx, ty - 2]] = 1
            R_flag[i] += 1

    # Average over available directions
    for i in range(n_extra):
        if R_flag[i] > 0:
            R[i, :] /= R_flag[i]

    # Set identity for the extra points themselves
    # In the augmented gamma vector, extra_ex points are at the end
    for i in range(n_extra):
        R[i, ngam2 - n_extra + i] = 1

    return R
