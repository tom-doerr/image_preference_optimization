import unittest
import numpy as np
from latent_opt import init_latent_state, propose_latent_pair_ridge


class TestSmokeRidgeProposal(unittest.TestCase):
    def test_propose_returns_two_vectors_and_clamps(self):
        st = init_latent_state(width=320, height=256, seed=0)
        st.w[:] = 0.0
        st.w[0] = 1.0
        z1, z2 = propose_latent_pair_ridge(st, alpha=2.0, beta=2.0, trust_r=1.0)
        self.assertEqual(len(z1), st.d)
        self.assertEqual(len(z2), st.d)
        self.assertLessEqual(float(np.linalg.norm(z1)), 1.000001)
        self.assertLessEqual(float(np.linalg.norm(z2)), 1.000001)


if __name__ == "__main__":
    unittest.main()
