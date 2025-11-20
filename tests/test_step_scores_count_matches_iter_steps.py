import sys
import types
import unittest
import numpy as np


class StepScoresCountTest(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_metrics", "latent_logic", "value_scorer"):
            sys.modules.pop(m, None)

    def test_compute_step_scores_respects_iter_steps(self):
        # Stub latent prompt and scorer
        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        sys.modules["latent_logic"] = ll

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (
            lambda f: float(np.sum(f)),
            "ok",
        )
        sys.modules["value_scorer"] = vs

        import ui_metrics

        lstate = types.SimpleNamespace(d=4, w=np.ones(4), sigma=1.0)
        scores = ui_metrics.compute_step_scores(
            lstate,
            prompt="p",
            vm_choice="Ridge",
            iter_steps=5,
            iter_eta=None,
            trust_r=None,
            session_state={},
        )
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores), 5)
        np.testing.assert_allclose(scores, [0.4, 0.8, 1.2, 1.6, 2.0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
