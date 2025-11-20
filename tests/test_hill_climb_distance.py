import unittest
import numpy as np
from latent_state import init_latent_state
from latent_logic import hill_climb_mu_distance, z_from_prompt


class TestHillClimbDistance(unittest.TestCase):
    def test_moves_toward_positive_away_from_negative(self):
        st = init_latent_state(width=512, height=512, seed=0)
        prompt = 'hill_climb_test'
        z_p = z_from_prompt(st, prompt)
        d = st.d
        # Construct one positive near +e1 and one negative near -e1
        e1 = np.zeros(d)
        e1[0] = 3.0
        z_pos = z_p + e1
        z_neg = z_p - e1
        X = np.vstack([z_pos - z_p, z_neg - z_p])
        y = np.array([+1.0, -1.0], dtype=float)
        # Start mu at prompt anchor
        st.mu = z_p.copy()
        # Baseline distances
        dpos0 = float(np.linalg.norm(st.mu - z_pos))
        dneg0 = float(np.linalg.norm(st.mu - z_neg))
        hill_climb_mu_distance(st, prompt, X, y, eta=0.3, gamma=0.5, trust_r=None)
        dpos1 = float(np.linalg.norm(st.mu - z_pos))
        dneg1 = float(np.linalg.norm(st.mu - z_neg))
        self.assertLess(dpos1, dpos0)
        self.assertGreater(dneg1, dneg0)


if __name__ == '__main__':
    unittest.main()
