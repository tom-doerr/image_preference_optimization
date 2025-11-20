import unittest
import numpy as np

from latent_state import init_latent_state
from latent_logic import sample_z_xgb_hill, z_from_prompt


class TestSampleZXgbHill(unittest.TestCase):
    def test_sample_moves_along_w_direction_when_scored(self):
        st = init_latent_state(width=256, height=256, seed=0)
        d = st.d
        st.w = np.ones(d, dtype=float)
        prompt = "sample z xgb hill"
        z_p = z_from_prompt(st, prompt)

        # Scorer prefers larger first coordinate in f = z − z_p
        def scorer(fvec: np.ndarray) -> float:
            return float(fvec[0])

        z = sample_z_xgb_hill(st, prompt, scorer, steps=3, step_scale=0.5, trust_r=None)
        delta = z - z_p
        self.assertGreater(delta[0], 0.0)


if __name__ == "__main__":
    unittest.main()

