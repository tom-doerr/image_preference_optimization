import types
import numpy as np


class _SB:
    def write(self, *a, **k):
        return None
    def metric(self, *a, **k):
        return None


class _ST:
    def __init__(self):
        self.session_state = types.SimpleNamespace()
        self.sidebar = _SB()
        # add dict-like get for session_state
        def _get(k, d=None):
            return getattr(self.session_state, k, d)
        setattr(self.session_state, 'get', _get)


class _LState:
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y


def test_get_dataset_prefers_memory():
    st = _ST()
    X = np.ones((3, 4))
    y = np.array([+1, -1, +1])
    ls = _LState(X=X, y=y)
    from ipo.ui.ui_sidebar import _get_dataset_for_display
    Xd, yd = _get_dataset_for_display(st, ls, "prompt")
    assert Xd is X and yd is y


def test_autofit_calls_ensure_when_selected_and_data_present(monkeypatch):
    st = _ST()
    st.session_state.reg_lambda = 1.0
    X = np.ones((3, 4))
    y = np.array([+1, -1, +1])
    ls = _LState(X=X, y=y)
    called = {}
    def _ens(*a, **k):
        called['ok'] = True
    import sys
    sys.modules['value_model'] = types.SimpleNamespace(ensure_fitted=_ens)
    from ipo.ui.ui_sidebar import _autofit_xgb_if_selected
    _autofit_xgb_if_selected(st, ls, "XGBoost", X, y)
    assert called.get('ok') is True
