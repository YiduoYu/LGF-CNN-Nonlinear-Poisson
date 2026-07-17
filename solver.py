"""
Main solver for 2D Poisson and modified Helmholtz equations on arbitrary
domains using the Lattice Green's Function / Boundary Algebraic Equation
(LGF-BAE) method (also known as the Difference Potential Method).

Solves:
    k * laplacian(u) - sigma * u = f    in  M^+  (interior)
    alpha * du/dn + beta * u = g         on  Gamma (boundary)

with the boundary defined implicitly as the zero level set of psi(x,y).

Usage:
    result = solve_helmholtz_2d(N=7, iOrder=2, ms='trig1', geom='ellipse',
                                sigma=0.0, alpha=1.0, beta=1.0,
                                potential='single', plot_stuff=False)
"""

import time
import numpy as np

from .operator import get_operator, get_eigen
from .fft_solver import fft_solve
from .lgf import evaluate_LGF, LGF_FFT
from .boundary import get_intersect_ptr, get_box3_index_v5, get_extrap
from .lagrange import get_lagrange_2d
from .potentials import get_sl, get_dl, get_dp_fft
from .geometry import get_geometry, get_unit_normal_star
from .manufactured import manufactured_solution, manufactured_gradient, manufactured_forcing


def _safe_cond(A):
    """Compute condition number safely, returning inf on failure."""
    if A.size == 0:
        return 0.0
    # Check for NaN/Inf
    if np.any(~np.isfinite(A)):
        return np.inf
    try:
        return np.linalg.cond(A)
    except np.linalg.LinAlgError:
        try:
            # Fallback: use scipy's condition number estimator
            from scipy.linalg import norm
            s = np.linalg.svd(A, compute_uv=False, hermitian=False)
            return s[0] / s[-1] if s[-1] > 0 else np.inf
        except Exception:
            return np.inf


def solve_helmholtz_2d(N=7, iOrder=2, ms='one', geom='ellipse',
                       sigma=0.0, k=1.0, alpha=1.0, beta=1.0,
                       ar=1.0, potential='single', plot_stuff=False,
                       psi_func=None, nx_func=None, ny_func=None,
                       domain_bounds=None, g_func=None,
                       use_manufactured_rhs=True,
                       solver_mode=None, moving_obs_psi=None):
    # Resolve solver_mode: explicit arg > env var > default 'factor'
    if solver_mode is None:
        import os
        solver_mode = os.environ.get('LGF_SOLVER_MODE', 'factor')
    """
    Solve the 2D modified Helmholtz equation on an arbitrary domain.

    Parameters
    ----------
    N : int
        Refinement parameter. Actual grid size = 2^N - 1.
    iOrder : int
        Finite difference order: 2 (3-point stencil) or 4 (5-point stencil).
    ms : str
        Manufactured solution name: 'one', 'poly1', 'poly2', 'poly3',
        'trig1', 'trig2', 'trig3', 'trig4'.
    geom : str
        Geometry: 'ellipse', 'leaf', 'star', 'butterfly'.
        If 'custom', psi_func, nx_func, ny_func, domain_bounds must be provided.
    sigma : float
        Modified Helmholtz coefficient (0 for Poisson).
    k : float
        Diffusion coefficient (default 1).
    alpha : float
        Robin BC coefficient for du/dn.
    beta : float
        Robin BC coefficient for u.
    ar : float
        Aspect ratio for ellipse geometry.
    potential : str
        Potential formulation: 'single' or 'double'.
    plot_stuff : bool
        If True, display error plot.
    psi_func : callable or None
        Custom level set function psi(x,y). Used when geom='custom'.
    nx_func : callable or None
        Custom x-normal. Used when geom='custom'.
    ny_func : callable or None
        Custom y-normal. Used when geom='custom'.
    domain_bounds : tuple (a, b, l) or None
        Custom domain bounds. Used when geom='custom'.
    g_func : callable or None
        Boundary data function g(x,y) for the RHS of the BC equation.
        If None, uses the manufactured solution to compute BC data.
        For pure Dirichlet (alpha=0, beta=1): g_func returns the
        prescribed boundary value at each intersection point.
    use_manufactured_rhs : bool
        If True (default), use manufactured forcing for the RHS f.
        If False, set f=0 (for Laplace equation path planning).

    Returns
    -------
    err : float
        Maximum absolute error on the interior (if manufactured solution).
    c1, c2, c3 : float
        Condition numbers (for diagnostics).
    N_actual : int
        Actual grid size.
    info : dict
        Additional information (solution, grid, timing, etc.).
    """
    t_start = time.time()

    print(f'Start test for order {iOrder} with N = {N}.')

    # ------------------------------------------------------------------
    # 1. Setup: auxiliary domain and geometry
    # ------------------------------------------------------------------
    N_actual = 2 ** N - 1

    if geom == 'custom':
        psi = psi_func
        a, b, l = domain_bounds
        # nx_func, ny_func already set from parameters
    else:
        psi, nx_func, ny_func, (a, b, l) = get_geometry(geom, ar=ar)

    h = (b + l - (a - l)) / (N_actual + 1)
    xvec = a - l + h * np.arange(1, N_actual + 1)

    # Grid (like MATLAB ndgrid)
    xm, ym = np.meshgrid(xvec, xvec, indexing='ij')

    # ------------------------------------------------------------------
    # 2. Identify point sets (interior, gamma)
    # ------------------------------------------------------------------
    mp = psi(xm, ym) <= 0  # M^+ (interior)

    # Find gamma points (discrete boundary)
    gamma_in_list = []
    gamma_ex_list = []
    for i in range(1, N_actual - 1):
        for j in range(1, N_actual - 1):
            ss = (int(mp[i, j]) + int(mp[i + 1, j]) + int(mp[i - 1, j])
                  + int(mp[i, j + 1]) + int(mp[i, j - 1]))
            if 0 < ss < 5:
                if mp[i, j]:
                    gamma_in_list.append([i, j])
                else:
                    gamma_ex_list.append([i, j])

    gam_in = np.array(gamma_in_list, dtype=int)
    gam_ex = np.array(gamma_ex_list, dtype=int)
    gam = np.vstack([gam_in, gam_ex]) if len(gam_in) > 0 else gam_ex

    ngam = gam.shape[0]
    ngam_in = gam_in.shape[0]
    ngam_ex = gam_ex.shape[0]

    # N^+ = interior + gamma_ex
    N_plus = mp.copy()
    if ngam_ex > 0:
        N_plus[gam_ex[:, 0], gam_ex[:, 1]] = True

    # Gamma mask
    m_gamma = np.zeros((N_actual, N_actual), dtype=int)
    for i in range(ngam):
        m_gamma[gam[i, 0], gam[i, 1]] = 1

    # ------------------------------------------------------------------
    # 3. Find boundary-grid intersection points
    # ------------------------------------------------------------------
    xq, yq, ind2 = get_intersect_ptr(a, l, h, mp, N_actual, psi)
    nq = len(xq)

    # Determine 3x3 interpolation boxes
    m_cell, ll_box3 = get_box3_index_v5(nq, N_plus.astype(int), m_gamma, ind2, gam_ex)

    # Identify extra corner points
    extra_mat = m_cell - m_gamma
    extra_mat_in = mp.astype(int) * extra_mat
    extra_mat_ex = extra_mat - extra_mat_in
    n_extra = np.sum(extra_mat)

    # List extra_in and extra_ex points
    ind_extra_in_list = []
    ind_extra_ex_list = []
    for i in range(N_actual):
        for j in range(N_actual):
            if extra_mat_in[i, j] > 0:
                ind_extra_in_list.append([i, j])
            if extra_mat_ex[i, j] > 0:
                ind_extra_ex_list.append([i, j])

    ind_extra_in = np.array(ind_extra_in_list, dtype=int) if ind_extra_in_list else np.empty((0, 2), dtype=int)
    ind_extra_ex = np.array(ind_extra_ex_list, dtype=int) if ind_extra_ex_list else np.empty((0, 2), dtype=int)

    n_extra_in = ind_extra_in.shape[0]
    n_extra_ex = ind_extra_ex.shape[0]

    # Augmented gamma set
    gam2 = np.vstack([gam_in, ind_extra_in, gam_ex, ind_extra_ex]) if n_extra > 0 else gam
    ngam2 = gam2.shape[0]

    # gam2_in = [gam_in; ind_extra_in]
    gam2_in = np.vstack([gam_in, ind_extra_in]) if n_extra_in > 0 else gam_in

    # Grid coordinates of gamma points (0-based indices → physical coords)
    xgam_d = a - l + (gam2[:, 0] + 1) * h
    ygam_d = a - l + (gam2[:, 1] + 1) * h

    # ------------------------------------------------------------------
    # 4. Lattice Green's function
    # ------------------------------------------------------------------
    if abs(sigma) > 1e-12:
        G = LGF_FFT(h, sigma, 2 * (N_actual + 1))
    else:
        G = np.zeros((N_actual, N_actual))
        for i in range(N_actual):
            for j in range(i + 1):
                G[i, j] = evaluate_LGF(i, j)
                G[j, i] = G[i, j]

    # ------------------------------------------------------------------
    # 5. Discrete operator and particular solution (Ghf)
    # ------------------------------------------------------------------
    Lh = get_operator(k, sigma, h, N_actual, iOrder)
    eigenvector = get_eigen(N_actual, 2, iOrder)

    # Right-hand side: fm = mp .* (f(xm,ym,ms) - sigma * ue(xm,ym,ms))
    if use_manufactured_rhs:
        fm = mp.astype(float) * (manufactured_forcing(xm, ym, ms)
                                 - sigma * manufactured_solution(xm, ym, ms))
    else:
        # Laplace equation: f = 0 in the free space
        fm = np.zeros((N_actual, N_actual))

    t_fft = time.time()
    Ghf = fft_solve(fm, sigma, 2, h, N_actual, eigenvector)
    fft_t = time.time() - t_fft
    print(f'FFT solve wall-time = {fft_t:.6f} (s).')

    # Extract Ghf at augmented gamma points
    Ghf_gam2 = np.zeros(ngam2)
    for i in range(ngam2):
        Ghf_gam2[i] = Ghf[gam2[i, 0], gam2[i, 1]]

    # ------------------------------------------------------------------
    # 6. Extrapolation matrix for extra gamma_ex points
    # ------------------------------------------------------------------
    if n_extra_ex > 0:
        R = get_extrap(N_actual, gam2, ind_extra_ex, m_gamma)
        # R1: columns for gam2_in (first part)
        # R2: columns for gam_ex (middle part)
        R1 = R[:, :ngam_in + n_extra_in]
        R2 = R[:, ngam_in + n_extra_in:ngam_in + n_extra_in + ngam_ex]
    else:
        R = np.zeros((0, ngam2))
        R1 = np.zeros((0, ngam_in))
        R2 = np.zeros((0, ngam_ex))

    # ------------------------------------------------------------------
    # 7. Index mapping for augmented gamma
    # ------------------------------------------------------------------
    index_gam2 = np.zeros((N_actual, N_actual), dtype=int)
    for i in range(ngam2):
        index_gam2[gam2[i, 0], gam2[i, 1]] = i

    # ------------------------------------------------------------------
    # 8. Boundary equation projection (BEP) matrices
    # ------------------------------------------------------------------
    bm0 = np.zeros((nq, ngam2))
    bm1x = np.zeros((nq, ngam2))
    bm1y = np.zeros((nq, ngam2))

    for i in range(nq):
        # 3x3 box lower-left corner (0-based indices) → physical coords
        bx = a - l + (ll_box3[i, 0] + 1) * h
        by = a - l + (ll_box3[i, 1] + 1) * h
        xvec_box = np.array([bx, bx + h, bx + 2 * h])
        yvec_box = np.array([by, by + h, by + 2 * h])

        u_basis, dudx, dudy = get_lagrange_2d(xvec_box, yvec_box, [xq[i], yq[i]])

        # Map 3x3 basis (9 values) to global augmented gamma indices.
        # get_lagrange_2d returns y-major order (x varies fastest within fixed y),
        # matching the MATLAB convention.
        idx_local = 0
        for dj in range(3):  # y-direction outer
            for di in range(3):  # x-direction inner
                gi = ll_box3[i, 0] + di
                gj = ll_box3[i, 1] + dj
                global_idx = index_gam2[gi, gj]
                bm0[i, global_idx] = u_basis[idx_local]
                bm1x[i, global_idx] = dudx[idx_local]
                bm1y[i, global_idx] = dudy[idx_local]
                idx_local += 1

    # Get normal vectors at intersection points
    if geom == 'custom':
        # Use provided normal functions or compute from level set gradient
        if nx_func is not None and ny_func is not None:
            nxv = nx_func(xq, yq)
            nyv = ny_func(xq, yq)
        else:
            eps = 1e-6
            psi_x = (psi(xq + eps, yq) - psi(xq - eps, yq)) / (2 * eps)
            psi_y = (psi(xq, yq + eps) - psi(xq, yq - eps)) / (2 * eps)
            mag = np.sqrt(psi_x ** 2 + psi_y ** 2)
            nxv = psi_x / mag
            nyv = psi_y / mag
    elif geom == 'ellipse' and nx_func is not None:
        nxv = nx_func(xq, yq)
        nyv = ny_func(xq, yq)
    elif geom == 'star':
        nxv, nyv = get_unit_normal_star(xq, yq)
    else:
        # Compute normals from level set gradient
        eps = 1e-6
        psi_x = (psi(xq + eps, yq) - psi(xq - eps, yq)) / (2 * eps)
        psi_y = (psi(xq, yq + eps) - psi(xq, yq - eps)) / (2 * eps)
        mag = np.sqrt(psi_x ** 2 + psi_y ** 2)
        nxv = psi_x / mag
        nyv = psi_y / mag

    # Broadcast normals
    nxm = np.tile(nxv[:, np.newaxis], (1, ngam2))
    nym = np.tile(nyv[:, np.newaxis], (1, ngam2))

    # Robin BC operator: B = alpha*(bm1x.*nxm + bm1y.*nym) + beta*bm0
    B = alpha * (bm1x * nxm + bm1y * nym) + beta * bm0

    # Partition B into [bm1 | bm2 | bm3] corresponding to [gam2_in | gam_ex | extra_ex]
    bm1 = B[:, :ngam_in + n_extra_in]                            # gam2_in
    bm2 = B[:, ngam_in + n_extra_in:ngam_in + n_extra_in + ngam_ex]  # gam_ex
    bm3 = B[:, ngam_in + n_extra_in + ngam_ex:]                   # ind_extra_ex

    # RHS vector
    if g_func is not None:
        # Use custom boundary data function (e.g., for path planning)
        g_vals = g_func(xq, yq)
        bvec3 = g_vals - B @ Ghf_gam2
    else:
        # Original manufactured solution path
        ux_exact, uy_exact = manufactured_gradient(xq, yq, ms)
        ue_exact = manufactured_solution(xq, yq, ms)
        bvec3 = (alpha * (ux_exact * nxv + uy_exact * nyv)
                 + beta * ue_exact
                 - B @ Ghf_gam2)

    # ------------------------------------------------------------------
    # 9. Solve boundary equation (single or double layer)
    # ------------------------------------------------------------------
    if potential == 'single':
        print('Using single layer formulation.')
        from scipy.sparse.linalg import gmres, LinearOperator
        from scipy.linalg import lu_factor, lu_solve
        Sm = get_sl(G, gam_ex, gam_ex)           # ngex × ngex
        Sp = get_sl(G, gam_ex, gam2_in); import os; _dbg=os.environ.get("LGF_DBG_DIM","0"); (_dbg=="1") and print(f"DIMS: nq={len(xq)}, ng_in={ngam_in}, ng_ex={ngam_ex}, bm1={bm1.shape}, bm2={bm2.shape}, Sp={Sp.shape}, Sm={Sm.shape}")          # ngex × ngam2_in

        # ------------------------------------------------------------------
        # Schur complement (local update) path
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Standard full-factorization path (always used for factor mode)
        # For schur mode: classify and report, then use standard solve.
        # Full Schur acceleration is estimated in factor_compare.py;
        # implementing it in the solver requires deeper restructuring.
        # ------------------------------------------------------------------
        if solver_mode == 'schur':
            # Classify γ⁻ into static / dynamic
            if moving_obs_psi is not None:
                s_idx = []; d_idx = []
                for idx in range(gam_ex.shape[0]):
                    i, j = gam_ex[idx]
                    if abs(moving_obs_psi(xm[i,j], ym[i,j])) < 4*h:
                        d_idx.append(idx)
                    else:
                        s_idx.append(idx)
                s_idx = np.array(s_idx, dtype=int); d_idx = np.array(d_idx, dtype=int)
            else:
                if not hasattr(solve_helmholtz_2d, '_schur_coord_cache'):
                    solve_helmholtz_2d._schur_coord_cache = {}
                if N not in solve_helmholtz_2d._schur_coord_cache:
                    solve_helmholtz_2d._schur_coord_cache[N] = \
                        set((gam_ex[i,0], gam_ex[i,1]) for i in range(gam_ex.shape[0]))
                    s_idx = np.arange(gam_ex.shape[0], dtype=int)
                    d_idx = np.array([], dtype=int)
                else:
                    ref = solve_helmholtz_2d._schur_coord_cache[N]
                    s_idx = []; d_idx = []
                    for idx in range(gam_ex.shape[0]):
                        i, j = gam_ex[idx]
                        if (i, j) in ref: s_idx.append(idx)
                        else: d_idx.append(idx)
                    s_idx = np.array(s_idx, dtype=int); d_idx = np.array(d_idx, dtype=int)

            ns, nd = len(s_idx), len(d_idx)
            ng = gam_ex.shape[0]
            print(f'  Schur: |γ⁻|={ng}, static={ns} ({100*ns//ng}%), dynamic={nd} ({100*nd//ng}%)', end='')
            if nd > 0:
                # Estimate speedup: assembly O(|γ⁻|²) + factor O(|γ⁻|³)
                # Local: O(|γ⁻_s|²) precomputed + O(|γ⁻_d|²) per frame
                spd = (ng/nd)**2 if nd > 0 else 1
                print(f' → est. {spd:.0f}× speedup from local update')
            else:
                print(' → baseline call')

        # Choose solver path
        if os.environ.get('LGF_NO_PC') == '1':
            class Cnt:
                def __init__(self): self.n=0
                def __call__(self,xk): self.n+=1
            if os.environ.get('LGF_USE_RAW') == '1':
                # Raw: (bm1*Sp + bm2*Sm) @ q = bvec3, recover y = Sm @ q
                S_raw = bm1 @ Sp + bm2 @ Sm
                cnt=Cnt(); t0=time.time()
                q,info_gmres=gmres(S_raw,bvec3,rtol=1e-10,atol=1e-14,callback=cnt,maxiter=3000)
                t_gmres=time.time()-t0
                y = Sm @ q
                print(f'  RAW GMRES: {cnt.n} iters, {t_gmres*1000:.0f}ms (cond≈{np.linalg.cond(S_raw):.0f})')
                c2 = _safe_cond(S_raw)
            else:
                # S_full = bm1*Sp*Sm^{-T} + bm2
                A_s_full = np.linalg.solve(Sm.T, Sp.T).T
                S_full = bm1 @ A_s_full + bm2
                if n_extra_ex > 0:
                    S_full += bm3 @ (-R1 @ A_s_full - R2)
                cnt=Cnt(); t0=time.time()
                y,info_gmres=gmres(S_full,bvec3,rtol=1e-10,atol=1e-14,callback=cnt,maxiter=3000)
                t_gmres=time.time()-t0
                print(f'  no-PC GMRES: {cnt.n} iters, {t_gmres*1000:.0f}ms (cond≈{np.linalg.cond(S_full):.0f})')
                c2 = _safe_cond(S_full)
            SmT_lu = lu_factor(Sm.T)
            w_y = lu_solve(SmT_lu, y)
            x = Sp @ w_y
            t_lu = time.time()
            c1 = _safe_cond(Sm)
            c3 = c1
        else:
            # Standard LU-preconditioned GMRES
            SmT_lu = lu_factor(Sm.T)
            t_lu = time.time()
            def apply_S(v):
                w = lu_solve(SmT_lu, v)
                As_v = Sp @ w
                result = bm1 @ As_v + bm2 @ v
                if n_extra_ex > 0:
                    result += bm3 @ (-R1 @ As_v - R2 @ v)
                return result
            nq = bm1.shape[0]
            S_op = LinearOperator((nq, nq), matvec=apply_S, dtype=float)
            y, info_gmres = gmres(S_op, bvec3, rtol=1e-10, atol=1e-14)
            if info_gmres > 0:
                A_s_full = np.linalg.solve(Sm.T, Sp.T).T
                S_full = bm1 @ A_s_full + bm2
                if n_extra_ex > 0:
                    S_full += bm3 @ (-R1 @ A_s_full - R2)
                y = np.linalg.solve(S_full, bvec3)
            w_y = lu_solve(SmT_lu, y)
            x = Sp @ w_y
            c1 = _safe_cond(Sm)
            c2 = 1.0 if info_gmres == 0 else -1.0
            c3 = c1
            print(f'  Factor: {time.time()-t_lu:.2f}s, GMRES: {info_gmres} iters')
            A_s_full = np.linalg.solve(Sm.T, Sp.T).T
            S_full = bm1 @ A_s_full + bm2
            if n_extra_ex > 0:
                S_full += bm3 @ (-R1 @ A_s_full - R2)
            # Compare: S_full (cond~4) vs Sm (single-layer, cond~2.6e3)
            # Sm maps γ⁻ → γ⁻; use a compatible RHS (all-ones on γ⁻)
            b_sm = np.ones(Sm.shape[0])
            class Cnt:
                def __init__(self): self.n=0
                def __call__(self,xk): self.n+=1
            print(f'  S_full: {S_full.shape[0]}x{S_full.shape[1]}, cond≈{np.linalg.cond(S_full):.0f}')
            print(f'  Sm:     {Sm.shape[0]}x{Sm.shape[1]}, cond≈{np.linalg.cond(Sm):.0f}')
            for label, S, b in [('S_full',S_full,bvec3),('Sm',Sm,b_sm)]:
                cnt=Cnt()
                t0=time.time()
                q,info=gmres(S,b,rtol=1e-10,atol=1e-14,callback=cnt,maxiter=3000)
                t1=time.time()-t0
                r=np.linalg.norm(S@q-b)/np.linalg.norm(b)
                print(f'  {label}: {cnt.n} iters, {t1*1000:.0f}ms, relerr={r:.2e}, info={info}')
            print(f'  LU-PC (tol=1e-10): conv flag={info_gmres}, LU {t_factor*1000:.0f}ms')

        if n_extra_ex > 0:
            z = -R1 @ x - R2 @ y
            ugam = np.concatenate([x, y, z])
        else:
            ugam = np.concatenate([x, y])

    elif potential == 'double':
        print('Using double layer formulation.')
        Dm = get_dl(G, gam_ex, gam_ex, N_plus)   # ngex × ngex
        Dp = get_dl(G, gam_ex, gam2_in, N_plus)   # ngex × ngam2_in

        A_d = np.linalg.solve(Dm.T, Dp.T).T

        D = bm1 @ A_d + bm2
        if n_extra_ex > 0:
            D += bm3 @ (-R1 @ A_d - R2)

        Md = D @ Dm
        c1 = _safe_cond(Dm)
        c2 = _safe_cond(D)
        c3 = _safe_cond(Md)

        if beta == 0.0 and alpha != 0.0:
            y, _res, _rank, _s = np.linalg.lstsq(D, bvec3, rcond=1e-10)
        else:
            y = np.linalg.solve(D, bvec3)
        x = A_d @ y
        if n_extra_ex > 0:
            z = -R1 @ x - R2 @ y
            ugam = np.concatenate([x, y, z])
        else:
            ugam = np.concatenate([x, y])
    else:
        raise ValueError(f"Unknown potential type: '{potential}'")

    print(f'The condition numbers are ({c1:.4e}, {c2:.4e}, {c3:.4e}).')

    # ------------------------------------------------------------------
    # 10. Difference potential
    # ------------------------------------------------------------------
    # Extract density on original gamma set (gam = [gam_in; gam_ex])
    # The augmented gamma vector ugam = [x; y; z] where:
    #   x = density on gam2_in = [gam_in; ind_extra_in]
    #   y = density on gam_ex
    #   z = density on ind_extra_ex
    # The original gamma set is [gam_in; gam_ex], so skip extra_in points.
    ugam_gamma = np.concatenate([
        ugam[:ngam_in],                                          # gam_in
        ugam[ngam_in + n_extra_in:ngam_in + n_extra_in + ngam_ex]  # gam_ex
    ])

    gam_original = np.vstack([gam_in, gam_ex]) if ngam_in > 0 else gam_ex

    t_dp = time.time()
    dp = get_dp_fft(eigenvector, Lh, gam_original, ugam_gamma,
                    N_actual, ~mp, h, sigma, full_domain=True)
    dp_t = time.time() - t_dp
    print(f'DP solve (end) wall-time = {dp_t:.6f} (s).')

    # ------------------------------------------------------------------
    # 11. Final solution
    # ------------------------------------------------------------------
    um = dp + Ghf

    # Compute error (only if manufactured solution available)
    if use_manufactured_rhs:
        ue_grid = manufactured_solution(xm, ym, ms)
        error_field = np.abs(mp.astype(float) * (um - ue_grid))
        err = np.max(error_field)
    else:
        ue_grid = None
        error_field = None
        err = 0.0  # No error computation for non-manufactured problems

    t_total = time.time() - t_start
    print(f'Elapsed      wall-time = {t_total:.6f} (s)')
    print(f'Max error = {err:.6e}')
    print()

    # Optional plotting
    if plot_stuff:
        _plot_error(xm, ym, mp, um, ue_grid)

    # Return results
    info = {
        'um': um,
        'xm': xm,
        'ym': ym,
        'mp': mp,
        'dp': dp,
        'Ghf': Ghf,
        'gam': gam,
        'gam_in': gam_in,
        'gam_ex': gam_ex,
        'h': h,
        'N_actual': N_actual,
        'xq': xq,
        'yq': yq,
        'error_field': error_field,
        'timing': {'fft': fft_t, 'dp': dp_t, 'total': t_total},
    }

    return err, c1, c2, c3, N_actual, info


def _plot_error(xm, ym, mp, um, ue_grid):
    """Internal helper to display error plot."""
    try:
        import matplotlib.pyplot as plt
        error = um - ue_grid
        error[~mp] = np.nan
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xm, ym, error, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Error')
        plt.show()
    except ImportError:
        print('Matplotlib not available for plotting.')
