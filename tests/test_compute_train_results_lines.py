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
        # dict-like get
        def _get(k, d=None):
            return getattr(self.session_state, k, d)
        setattr(self.session_state, 'get', _get)


class _LState:
    def __init__(self, d, X=None, y=None):
        self.d = d
        self.X = X
        self.y = y


def test_compute_train_results_lines_xgb_ok_status_and_order(monkeypatch):
    # Arrange st + dataset
    st = _ST()
    st.session_state.last_train_at = '2025-11-24T10:55:00+00:00'
    st.session_state.cv_last_at = 'n/a'
    X = np.array([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [-0.5, -0.5]])
    y = np.array([+1, -1, +1, -1])
    ls = _LState(2, X=X, y=y)

    # Stub value_scorer to report ok and return large probabilities
    import sys
    def _gvs(vm_choice, lstate, prompt, ss):
        return (lambda f: 0.9 if vm_choice == 'XGBoost' else 0.0, 'ok')
    sys.modules['value_scorer'] = types.SimpleNamespace(get_value_scorer=_gvs)

    from ipo.ui.ui_sidebar import compute_train_results_lines
    lines = compute_train_results_lines(st, ls, 'prompt', 'XGBoost')

    # Assert canonical ordering and active yes
    assert lines[0].startswith('Train score:')
    assert lines[1].startswith('CV score:')
    assert lines[2].startswith('Last CV:')
    assert lines[3].startswith('Last train:')
    assert lines[4].startswith('Value scorer status:')
    assert lines[5].startswith('Value scorer:')
    assert lines[6] == 'XGBoost active: yes'
    assert lines[7] == 'Optimization: Ridge only'

