import unittest
import numpy as np
from latent_opt import init_latent_state, state_summary


class TestStateSummary(unittest.TestCase):
    def test_basic_summary_fields(self):
        st = init_latent_state(width=320, height=256, seed=0)
        st.mu[:] = 1.0
        st.w[:] = 0.5
        st.step = 3
        s = state_summary(st)
        self.assertEqual(s["width"], 320)
        self.assertEqual(s["height"], 256)
        self.assertEqual(s["d"], 4 * (256 // 8) * (320 // 8))
        self.assertEqual(s["step"], 3)
        self.assertAlmostEqual(s["mu_norm"], float(np.linalg.norm(np.ones(st.d))))
        self.assertAlmostEqual(s["w_norm"], float(np.linalg.norm(np.full(st.d, 0.5))))


if __name__ == "__main__":
    unittest.main()
