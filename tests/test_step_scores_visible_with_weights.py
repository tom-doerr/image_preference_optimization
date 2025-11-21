import sys
import types
import unittest
import numpy as np


class StepScoresVisibleTest(unittest.TestCase):
    def tearDown(self):
        for m in (
            "ui",
            "latent_logic",
            "value_scorer",
            "streamlit",
        ):
            sys.modules.pop(m, None)

    def test_step_scores_render_when_weights_nonzero(self):
        from tests.helpers import st_streamlit

        st, writes = st_streamlit.stub_with_writes()
        st.session_state.prompt = "scores"

        # Stub latent_logic and value_scorer used by compute_step_scores
        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        sys.modules["latent_logic"] = ll

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (
            lambda f: float(np.sum(f)),
            "ok",
        )
        sys.modules["value_scorer"] = vs

        sys.modules["streamlit"] = st
        import ui

        lstate = types.SimpleNamespace(
            d=4, w=np.ones(4), sigma=1.0, mu=np.zeros(4)
        )

        ui.render_iter_step_scores(
            st, lstate, st.session_state.prompt, "Ridge", iter_steps=3, iter_eta=None, trust_r=None
        )

        out = "\n".join(writes)
        self.assertIn("Step scores:", out)
        self.assertRegex(out, r"Step scores:\s*[-0-9\., ]+")


if __name__ == "__main__":
    unittest.main()
