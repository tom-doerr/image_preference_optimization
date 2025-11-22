import sys
import types
import unittest
import numpy as np


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class SmokeBootstrapApi(unittest.TestCase):
    def tearDown(self):
        for m in (
            "app_bootstrap",
            "app_api",
            "streamlit",
            "value_scorer",
            "ui_sidebar",
            "flux_local",
            "latent_opt",
        ):
            sys.modules.pop(m, None)

    def test_bootstrap_emits_early_lines(self):
        # Stub Streamlit
        writes = []
        st = types.ModuleType("streamlit")
        st.session_state = Session()
        st.set_page_config = lambda **_: None
        st.text_input = lambda *_, value="": value
        class SB:
            write = staticmethod(lambda x: writes.append(str(x)))
            text_input = staticmethod(lambda *_, value="": value)
        st.sidebar = SB()
        sys.modules["streamlit"] = st
        # Minimal stubs used by emit_early_sidebar
        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer = lambda *a, **k: (None, "ridge_untrained")
        sys.modules["value_scorer"] = vs
        uis = types.ModuleType("ui_sidebar")
        uis._emit_train_results = lambda st_mod, lines, sidebar_only=False: [st_mod.sidebar.write(ln) for ln in lines]
        sys.modules["ui_sidebar"] = uis
        fl = types.ModuleType("flux_local")
        fl.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = fl

        import app_bootstrap as ab

        ab.init_page_and_logging()
        ab.emit_early_sidebar()
        out = "\n".join(writes)
        self.assertIn("Value model:", out)
        self.assertIn("Step scores: n/a", out)
        self.assertIn("Latent dim:", out)

    def test_apply_state_sets_pair(self):
        # Stub Streamlit
        st = types.ModuleType("streamlit")
        st.session_state = Session()
        sys.modules["streamlit"] = st
        # Stub latent_opt proposer
        lo = types.ModuleType("latent_opt")
        lo.propose_next_pair = lambda state, prompt: (np.ones(2), -np.ones(2))
        sys.modules["latent_opt"] = lo
        import app_api as api

        state = types.SimpleNamespace(d=2, mu=np.zeros(2), rng=np.random.default_rng(0), sigma=1.0)
        api._apply_state(state)
        self.assertIn("lz_pair", st.session_state)
        a, b = st.session_state.lz_pair
        self.assertTrue(np.allclose(a, np.ones(2)))
        self.assertTrue(np.allclose(b, -np.ones(2)))


if __name__ == "__main__":
    unittest.main()

