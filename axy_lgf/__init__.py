# axy_lgf — variable-coefficient elliptic solver on a rectangle
#
#     -div(a(x,y) grad u) = f    in (x_min, x_max)^2
#     u = 0                      on boundary
#
# Uses DST-based Dirichlet LGF (no BAE), screened LGF banks, and LGF-CNN.

from .solvers import solve_axy_pcg, solve_axy_richardson, solve_axy_lgf_bank
from .manufactured import fixed_case_solutions, make_problem
from .lgf_bank import ScreenedLGFBank


def __getattr__(name):
    # Classical NumPy/SciPy workflows do not require a PyTorch installation.
    if name in {"LGFCNN", "LGFCNNTrainConfig"}:
        from .lgf_cnn import LGFCNN, LGFCNNTrainConfig
        return {"LGFCNN": LGFCNN, "LGFCNNTrainConfig": LGFCNNTrainConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "solve_axy_pcg",
    "solve_axy_richardson",
    "solve_axy_lgf_bank",
    "make_problem",
    "fixed_case_solutions",
    "ScreenedLGFBank",
    "LGFCNN",
    "LGFCNNTrainConfig",
]
