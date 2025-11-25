import types
import numpy as np


def _mk_lstate(d=4):
    ls = types.SimpleNamespace()
    ls.d = d
    ls.w = np.zeros(d)
    return ls


def test_ridge_vm_fits_and_updates_w():
    from ipo.core.value_model import RidgeVM

    X = np.vstack([np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)), np.tile([-1.0, 0.0, 0.0, 0.0], (4, 1))])
    y = np.array([1] * 4 + [-1] * 4, dtype=float)
    lstate = _mk_lstate(4)
    vm = RidgeVM()
    vm.fit(lstate, X, y, 1e-3, types.SimpleNamespace())
    assert isinstance(lstate.w, np.ndarray)
    assert lstate.w.shape == (4,)


def test_get_vm_factory_returns_classes():
    from ipo.core.value_model import get_vm, RidgeVM, XGBVM, LogisticVM

    assert isinstance(get_vm("Ridge"), RidgeVM)
    assert isinstance(get_vm("XGBoost"), XGBVM)
    assert isinstance(get_vm("Logistic"), LogisticVM)
