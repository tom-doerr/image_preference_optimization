import types
import unittest


class _SB:
    def __init__(self):
        pass
    def button(self, label, **k):
        # Simulate clicking the train button when asked
        return label == "Train Logistic now (sync)"
    def write(self, *a, **k):
        pass


class _ST:
    def __init__(self):
        self.session_state = types.SimpleNamespace()
        self.sidebar = _SB()
    def toast(self, msg):
        # Capture the last toast to session_state
        self.session_state._last_toast = msg


class TestLogitTrainToastText(unittest.TestCase):
    def test_toast_after_sync_fit(self):
        st = _ST()
        st.session_state.prompt = "p"
        st.session_state.reg_lambda = 1.0
        import sys
        sys.modules['streamlit'] = st
        # Minimal dataset via _get_dataset_for_display: return rows
        import numpy as np
        X = np.ones((6, 3))
        y = np.array([+1, -1, +1, -1, +1, -1])
        sys.modules['ipo.core.persistence'] = types.SimpleNamespace(
            get_dataset_for_prompt_or_session=lambda prompt, ss: (X, y)
        )
        # Minimal lstate and fit_value_model stub
        lstate = types.SimpleNamespace(d=3)
        st.session_state.lstate = lstate
        def _fit(*a, **k):
            return None
        sys.modules['value_model'] = types.SimpleNamespace(fit_value_model=_fit)
        from ipo.ui.ui_sidebar import _handle_train_section
        _handle_train_section(st, lstate, "p", "Logistic")
        self.assertEqual(getattr(st.session_state, '_last_toast', ''), "Logit: trained (sync)")


if __name__ == "__main__":
    unittest.main()

