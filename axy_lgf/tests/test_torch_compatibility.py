"""Optional legacy parity checks; no training or network changes."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

torch = pytest.importorskip("torch")

from axy_lgf.grid import SquareGrid
from axy_lgf.lgf_bank import ScreenedLGFBank
from axy_lgf.operators import apply_div_a_grad
from axy_lgf.torch_ops import ScreenedLGFBankTorch, TorchSquareOps


def test_legacy_numpy_torch_operator_and_bank_parity():
    grid = SquareGrid(15)
    rng = np.random.default_rng(27)
    a = 0.8+0.4*rng.random((15, 15))
    u = rng.normal(size=(15, 15))
    at, ut = torch.tensor(a[None]), torch.tensor(u[None])
    ops = TorchSquareOps(grid)
    assert_allclose(ops.apply_div_a_grad(ut, at).numpy()[0],
                    apply_div_a_grad(u, a, grid), atol=2e-13)
    numpy_bank = ScreenedLGFBank.from_defaults(grid, J=3)
    torch_bank = ScreenedLGFBankTorch(grid, J=3)
    assert_allclose(torch_bank.apply(ut).numpy()[0], numpy_bank.apply(u), atol=1e-15)
