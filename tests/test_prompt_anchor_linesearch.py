import unittest
import numpy as np
from latent_opt import init_latent_state
from latent_logic import propose_pair_prompt_anchor_linesearch


class TestPromptAnchorLineSearch(unittest.TestCase):
    def test_picks_largest_within_radius(self):
        st = init_latent_state(width=320, height=256, seed=0)
        w = np.zeros(st.d); w[0] = 1.0
        st.w = w
        z1, z2 = propose_pair_prompt_anchor_linesearch(st, "p", trust_r=0.8, gamma=0.0, mags=[0.2, 0.5, 1.5])
        # Expect |Δ| ≈ trust_r (largest allowable)
        # Check via half-distance from midpoint
        mid = 0.5 * (z1 + z2)
        d = float(np.linalg.norm(z1 - mid))
        self.assertAlmostEqual(d, 0.8, places=6)


if __name__ == '__main__':
    unittest.main()

