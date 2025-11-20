import unittest
import numpy as np
from latent_opt import init_latent_state, z_from_prompt


class TestPromptAnchorProposal(unittest.TestCase):
    def test_centered_around_prompt(self):
        from latent_opt import propose_pair_prompt_anchor

        st = init_latent_state(width=320, height=256, seed=0)
        st.w = np.ones(st.d)  # simple direction
        z1, z2 = propose_pair_prompt_anchor(st, "p", alpha=0.5, beta=0.5)
        zp = z_from_prompt(st, "p")
        mid = 0.5 * (z1 + z2)
        self.assertTrue(np.allclose(mid, zp, atol=1e-6))

    def test_fit_direction_aligns_with_choice(self):
        # Using update_latent_ridge with feats = z - zp makes w favor the chosen delta
        from latent_opt import update_latent_ridge, propose_pair_prompt_anchor

        st = init_latent_state(width=320, height=256, seed=1)
        zp = z_from_prompt(st, "p2")
        # Create a synthetic pair around zp
        d = np.zeros(st.d)
        d[0] = 1.0
        za = zp + d
        zb = zp - d
        update_latent_ridge(st, za, zb, "a", feats_a=(za - zp), feats_b=(zb - zp))
        z1, z2 = propose_pair_prompt_anchor(st, "p2", alpha=0.5, beta=0.5)
        # w should align with +d direction
        self.assertGreater(float(st.w[0]), 0.0)
        # z1 should be closer to +d than z2
        self.assertGreater(float(np.dot(z1 - zp, d)), float(np.dot(z2 - zp, d)))


if __name__ == "__main__":
    unittest.main()
