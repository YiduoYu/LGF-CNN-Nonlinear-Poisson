"""Optional hypre BoomerAMG V-cycle, used with the unchanged common PCG.

The local C bridge transfers the existing CSR matrix and residual vectors;
hypre performs every coarsening, interpolation, smoothing and coarse solve.
Build once with python -m axy_lgf.build_boomer. No native work is compiled
or downloaded at import time or within a measured solver invocation.
"""

from __future__ import annotations

import atexit
import ctypes as ct
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import weakref

import numpy as np

from .operators import apply_reaction_diffusion, assemble_reaction_diffusion
from .solvers import _pcg, _validate_problem

PROJECT = Path(__file__).resolve().parent
BRIDGE = PROJECT / ".deps" / ("libaxy_boomer.dylib" if sys.platform == "darwin" else "libaxy_boomer.so")
SETTINGS = {
    "coarsening": "HMIS (10)", "interpolation": "extended+i (6)",
    "P_max_elements": 4, "strong_threshold": 0.25, "aggressive_levels": 0,
    "restriction": "P.T (0)", "smoother": "symmetric hybrid Gauss-Seidel (6)",
    "relax_order": "lexicographic (0)", "pre_sweeps": 1, "post_sweeps": 1,
    "coarse_solver": "Gaussian elimination (9)", "max_coarse_size": 9,
    "max_levels": 25, "cycle": "V (1)", "inner_tolerance": 0.0,
    "inner_miniter": 1, "inner_maxiter": 1, "initial_correction": "zero on every application",
}


class BoomerUnavailableError(ImportError):
    """The optional native hypre bridge has not been built."""


@lru_cache(maxsize=1)
def _library():
    if not BRIDGE.exists():
        raise BoomerUnavailableError(
            "hypre BoomerAMG is unavailable. Build the optional serial bridge: "
            "python -m axy_lgf.build_boomer (requires a C/C++ compiler and make)."
        )
    try:
        library = ct.CDLL(str(BRIDGE))
    except OSError as exc:
        raise BoomerUnavailableError(f"Cannot load {BRIDGE}: {exc}") from exc
    floats = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    library.axy_boomer_version.argtypes = []
    library.axy_boomer_version.restype = ct.c_char_p
    library.axy_boomer_create.argtypes = [ct.c_int, integers, integers, floats, ct.POINTER(ct.c_void_p)]
    library.axy_boomer_create.restype = ct.c_int
    for name in ("axy_boomer_apply", "axy_boomer_matvec"):
        function = getattr(library, name)
        function.argtypes = [ct.c_void_p, floats, floats]
        function.restype = ct.c_int
    library.axy_boomer_destroy.argtypes = [ct.c_void_p]
    library.axy_boomer_destroy.restype = None
    library.axy_boomer_finalize.argtypes = []
    library.axy_boomer_finalize.restype = None
    atexit.register(library.axy_boomer_finalize)
    return library


def boomer_metadata() -> dict:
    library = _library()
    manifest = PROJECT / ".deps/boomer_build.json"
    return dict(hypre_version=library.axy_boomer_version().decode(), settings=SETTINGS,
                bridge_sha256=hashlib.sha256(BRIDGE.read_bytes()).hexdigest(),
                build=json.loads(manifest.read_text()))


def _check(status: int) -> None:
    if status:
        raise RuntimeError(f"hypre BoomerAMG call failed (error code {status})")


class BoomerVcycle:
    """Fixed linear hypre preconditioner, owning its matrix and hierarchy.

    Serial CPU float64, R=P.T, symmetric GS pre/post relaxation and a direct
    coarse solve. Each apply starts from zero and runs exactly one cycle.
    Vector transfer and native-call overhead belong to each apply/solve.
    """

    def __init__(self, matrix):
        self.library = _library()
        self.size = matrix.shape[0]
        if matrix.shape != (self.size, self.size) or self.size == 0:
            raise ValueError("matrix must be nonempty and square")
        csr = matrix.tocsr().astype(np.float64)
        if csr.nnz > np.iinfo(np.int32).max or self.size > np.iinfo(np.int32).max:
            raise ValueError("serial BoomerAMG bridge requires int32-sized CSR indices")
        handle = ct.c_void_p()
        _check(self.library.axy_boomer_create(
            self.size, np.ascontiguousarray(csr.indptr, dtype=np.int32),
            np.ascontiguousarray(csr.indices, dtype=np.int32),
            np.ascontiguousarray(csr.data, dtype=np.float64), ct.byref(handle)))
        self.handle = handle
        self._cleanup = weakref.finalize(self, self.library.axy_boomer_destroy, handle)
        try:
            self(np.zeros(self.size, dtype=np.float64))  # Include lazy native setup in setup time.
        except Exception:
            self.close()
            raise

    def _apply(self, rhs, function):
        if not self._cleanup.alive:
            raise RuntimeError("BoomerAMG preconditioner is closed")
        values = np.asarray(rhs, dtype=np.float64)
        if values.size != self.size:
            raise ValueError("BoomerAMG rhs size mismatch")
        flat = np.ascontiguousarray(values.ravel())
        result = np.empty(self.size, dtype=np.float64)
        _check(function(self.handle, flat, result))
        return result.reshape(values.shape)

    def __call__(self, rhs):
        return self._apply(rhs, self.library.axy_boomer_apply)

    def apply_matrix(self, rhs):
        """Action of the transferred hypre matrix, for numerical verification."""
        return self._apply(rhs, self.library.axy_boomer_matvec)

    def close(self):
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def solve_boomer_pcg(a, f, grid, *, q=0.0, a_nodes=None, tol=1e-10, maxiter=500, u0=None):
    """Same outer PCG/operator as LGF/GMG/PyAMG; only Pinv is BoomerAMG."""
    started = perf_counter()
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    matrix = assemble_reaction_diffusion(a, grid, q, a_nodes=a_nodes)
    with BoomerVcycle(matrix) as inverse:
        setup_time = perf_counter() - started
        action = lambda u: apply_reaction_diffusion(u, a, grid, q, a_nodes=a_nodes)
        result = _pcg(action, inverse, f, grid, tol=tol, maxiter=maxiter,
                      u0=u0, a0=float(np.mean(a)))
        result.setup_time = setup_time
    return result
