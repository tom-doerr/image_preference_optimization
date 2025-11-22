import sys
import types
import unittest

from tests.helpers.st_streamlit import stub_basic


class TestXgbAsyncCheckbox(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit"):
            sys.modules.pop(m, None)

    def test_render_modes_without_async_toggle(self):
        st = stub_basic()
        # No special checkbox; simplified UI has no async toggle
        st.session_state.prompt = "p"
        st.session_state.lstate = types.SimpleNamespace(d=4)

        # Minimal deps
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        import ipo.ui.ui_sidebar as ui_sidebar
        vm_choice, selected_gen_mode, batch_size, _ = ui_sidebar.render_modes_and_value_model(st)
        # Ensure it renders and does not create the async key
        self.assertIn(vm_choice, ("XGBoost", "Ridge"))
        self.assertNotIn("xgb_train_async", st.session_state)


if __name__ == "__main__":
    unittest.main()
