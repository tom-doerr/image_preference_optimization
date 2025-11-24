import types
import numpy as np


class _SB:
    def write(self, *a, **k):
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
        self.width = 640
        self.height = 640


def test_batch_ui_refit_uses_memory_first(monkeypatch):
    # Prepare st, lstate with in-memory dataset
    st = _ST()
    st.session_state.prompt = 'unit test prompt'
    st.session_state.reg_lambda = 1.0
    X = np.ones((4, 3))
    y = np.array([+1, -1, +1, -1])
    ls = _LState(3, X=X, y=y)
    st.session_state.lstate = ls

    # Monkeypatch streamlit import inside batch_ui helpers
    import sys
    sys.modules['streamlit'] = st

    # Monkeypatch value_model.fit_value_model to capture inputs
    captured = {}
    def _fit(vm_choice, lstate, Xv, yv, lam, ss):
        captured['vm'] = vm_choice
        captured['rows'] = int(getattr(Xv, 'shape', (0,))[0])
    sys.modules['value_model'] = types.SimpleNamespace(fit_value_model=_fit)

    # Ensure no folder dataset fetch is attempted (would raise if called)
    sys.modules['ipo.core.persistence'] = types.SimpleNamespace(get_dataset_for_prompt_or_session=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('should not be called')))

    from ipo.ui.batch_ui import _refit_from_dataset_keep_batch
    _refit_from_dataset_keep_batch()

    # Assert Ridge trained with in-memory rows (4)
    assert captured.get('vm') == 'Ridge'
    assert captured.get('rows') == 4

