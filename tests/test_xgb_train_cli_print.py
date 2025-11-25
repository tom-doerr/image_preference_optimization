import io
import sys
import types
import unittest


class _SB:
    def button(self, label, **k):
        return label == "Train XGBoost now (sync)"
    def write(self, *a, **k):
        pass


class _ST:
    def __init__(self):
        self.session_state = types.SimpleNamespace()
        self.sidebar = _SB()
    def toast(self, msg):
        self.session_state._last_toast = msg


class TestXgbTrainCliPrint(unittest.TestCase):
    def test_cli_print_after_sync_fit(self):
        st = _ST()
        st.session_state.prompt = "p"
        st.session_state.reg_lambda = 1.0
        import numpy as np
        import types as _t
        # Stub modules
        sys.modules['streamlit'] = st
        X = np.ones((4, 3))
        y = np.array([+1, -1, +1, -1])
        sys.modules['ipo.core.persistence'] = _t.SimpleNamespace(
            get_dataset_for_prompt_or_session=lambda prompt, ss: (X, y)
        )
        lstate = _t.SimpleNamespace(d=3)
        st.session_state.lstate = lstate
        sys.modules['value_model'] = _t.SimpleNamespace(fit_value_model=lambda *a, **k: None)
        from ipo.ui.ui_sidebar import _handle_train_section
        # Capture stdout
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _handle_train_section(st, lstate, "p", "XGBoost")
        finally:
            sys.stdout = old
        out = buf.getvalue()
        self.assertIn("[xgb] trained (sync)", out)


if __name__ == "__main__":
    unittest.main()

