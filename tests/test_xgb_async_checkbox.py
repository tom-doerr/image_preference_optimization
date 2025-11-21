import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_basic


class TestXgbAsyncCheckbox(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit"):
            sys.modules.pop(m, None)

    def test_checkbox_sets_xgb_async_flag(self):
        st = stub_basic()

        # Make the specific checkbox return False to disable async
        def _checkbox(label, *a, **k):
            if label == "Train XGBoost asynchronously":
                return False
            return k.get("value", False)

        st.sidebar.checkbox = staticmethod(_checkbox)  # type: ignore[attr-defined]
        st.session_state.prompt = "p"
        st.session_state.lstate = types.SimpleNamespace(d=4)

        # Minimal deps
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        import ui_sidebar
        vm_choice, selected_gen_mode, batch_size, _ = ui_sidebar.render_modes_and_value_model(st)

        self.assertIn("xgb_train_async", st.session_state)
        self.assertFalse(bool(st.session_state["xgb_train_async"]))


if __name__ == "__main__":
    unittest.main()

