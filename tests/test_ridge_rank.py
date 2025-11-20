import unittest
import numpy as np
from latent_opt import init_latent_state, update_latent_ridge, propose_latent_pair_ridge


class TestRidgeRank(unittest.TestCase):
    def test_ridge_updates_w_direction(self):
        st = init_latent_state(seed=0)
        # True direction prefers +e0 (use only first dim)
        z_a = np.zeros(st.d)
        z_a[0] = 1.0
        z_b = -z_a.copy()
        update_latent_ridge(st, z_a, z_b, "a", lr_mu=0.0, lam=1e-2)
        self.assertGreater(st.w[0], 0)

    def test_propose_ridge_hillclimb(self):
        st = init_latent_state(seed=0)
        st.w[:] = 0.0
        st.w[0] = 2.0
        z1, z2 = propose_latent_pair_ridge(st, alpha=1.0, beta=1.0)
        d1 = z1 - st.mu
        d2 = z2 - st.mu
        # Movement along +w has positive projection
        self.assertGreater(float(np.dot(d1, st.w)), 0.0)
        # Orthogonality between the two proposal directions
        self.assertAlmostEqual(float(np.dot(d1, d2)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
