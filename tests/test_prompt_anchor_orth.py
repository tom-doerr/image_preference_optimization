import unittest
import numpy as np
from latent_opt import init_latent_state, z_from_prompt
from latent_logic import propose_pair_prompt_anchor_iterative


class TestPromptAnchorOrth(unittest.TestCase):
    def test_midpoint_still_prompt_with_orth(self):
        st = init_latent_state(width=320, height=256, seed=0)
        st.w = np.ones(st.d)
        z1, z2 = propose_pair_prompt_anchor_iterative(
            st, "p", steps=2, trust_r=1.0, gamma=0.3
        )
        zp = z_from_prompt(st, "p")
        self.assertTrue(np.allclose(0.5 * (z1 + z2), zp, atol=1e-6))

    def test_has_orth_component(self):
        st = init_latent_state(width=320, height=256, seed=1)
        w = np.zeros(st.d)
        w[0] = 1.0
        st.w = w
        z1, z2 = propose_pair_prompt_anchor_iterative(
            st, "q", steps=2, trust_r=1.0, gamma=0.3
        )
        zp = z_from_prompt(st, "q")
        d1 = w / (np.linalg.norm(w) + 1e-12)
        v = z1 - zp
        # projection magnitude should be less than total norm if orth component present
        proj = float(np.dot(v, d1))
        self.assertLess(abs(proj), float(np.linalg.norm(v)))


if __name__ == "__main__":
    unittest.main()
