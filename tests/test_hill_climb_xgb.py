import unittest
import numpy as np

from latent_opt import init_latent_state, z_from_prompt
from latent_logic import hill_climb_mu_xgb


class TestHillClimbXGB(unittest.TestCase):
    def test_mu_moves_along_w_direction(self):
        st = init_latent_state(width=256, height=256, seed=0)
        d = st.d
        # Simple ridge direction: all ones
        st.w = np.ones(d, dtype=float)
        prompt = "xgb hill test"
        z_p = z_from_prompt(st, prompt)
        st.mu = z_p.copy()

        # Scorer: prefers larger first coordinate in f = z − z_p
        def scorer(fvec: np.ndarray) -> float:
            return float(fvec[0])

        hill_climb_mu_xgb(st, prompt, scorer, steps=3, step_scale=0.5, trust_r=None)
        delta = st.mu - z_p
        self.assertGreater(delta[0], 0.0)


if __name__ == "__main__":
    unittest.main()
