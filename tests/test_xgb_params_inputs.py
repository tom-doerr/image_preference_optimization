import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_basic


class _Exp:
    def __init__(self, *a, **k):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class TestXgbParamsInputs(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit", "persistence"):
            sys.modules.pop(m, None)

    def test_availability_line_and_param_inputs(self):
        st = stub_basic()
        # Capture writes
        writes = st.sidebar_writes
        # Provide number_input that returns custom values for our labels
        def _num(label, **k):
            if label == "XGB n_estimators":
                return 20
            if label == "XGB max_depth":
                return 4
            return k.get("value")

        st.number_input = _num
        st.sidebar.number_input = staticmethod(_num)
        # Ensure expander is callable and context-managed
        st.sidebar.expander = staticmethod(lambda *a, **k: _Exp())

        # Minimal state
        st.session_state.prompt = "p"
        lstate = types.SimpleNamespace(d=4, w=0)
        st.session_state.lstate = lstate

        # Minimal persistence helpers used upstream
        P = types.ModuleType("persistence")
        P.dataset_rows_for_prompt = lambda *a, **k: 0
        P.dataset_stats_for_prompt = lambda *a, **k: {"rows": 0, "pos": 0, "neg": 0, "d": 4, "recent_labels": []}
        sys.modules["persistence"] = P

        # Import and call the internal block for simplicity
        import ui_sidebar

        ui_sidebar._sidebar_value_model_block(st, lstate, "p", "XGBoost", reg_lambda=1.0)

        # Expect availability line appeared
        self.assertTrue(any("XGBoost available:" in s for s in writes))
        # Expect params stored
        self.assertEqual(int(st.session_state.get("xgb_n_estimators", 0)), 20)
        self.assertEqual(int(st.session_state.get("xgb_max_depth", 0)), 4)


if __name__ == "__main__":
    unittest.main()

