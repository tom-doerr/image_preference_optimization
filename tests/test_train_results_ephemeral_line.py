import sys
import types
import unittest
import time


class _SB:
    def __init__(self, sink):
        self._sink = sink
    def write(self, msg):
        self._sink.append(str(msg))


class _ST:
    def __init__(self, sink):
        self.session_state = types.SimpleNamespace()
        self.sidebar = _SB(sink)


class TestTrainResultsEphemeralLine(unittest.TestCase):
    def test_ephemeral_last_action_inside_train_results(self):
        writes = []
        st = _ST(writes)
        sys.modules['streamlit'] = st
        # Seed last action within the 6-second window
        st.session_state.last_action_text = "XGBoost: trained (sync)"
        st.session_state.last_action_ts = float(time.time())
        # Minimal lstate and prompt
        lstate = types.SimpleNamespace(d=4)
        from ipo.ui.ui_sidebar import _emit_train_results
        lines = [
            "Train score: n/a",
            "CV score: n/a",
            "Last CV: n/a",
            "Last train: n/a",
            "Value scorer status: xgb_unavailable",
            "Value scorer: XGBoost (xgb_unavailable, rows=0)",
            "XGBoost active: no",
            "Optimization: Ridge only",
        ]
        _emit_train_results(st, lines)
        assert any("Last action: XGBoost: trained (sync)" in w for w in writes)


if __name__ == "__main__":
    unittest.main()

