import unittest
import numpy as np
from latent_opt import init_latent_state, propose_latent_pair_ridge


class TestTrustRadius(unittest.TestCase):
    def test_trust_radius_clamps(self):
        st = init_latent_state(seed=0)
        st.w[:] = 0.0
        st.w[0] = 1.0
        z1, z2 = propose_latent_pair_ridge(st, alpha=10.0, beta=10.0, trust_r=1.0)
        self.assertLessEqual(np.linalg.norm(z1), 1.000001)
        self.assertLessEqual(np.linalg.norm(z2), 1.000001)

    # Logistic path removed; ridge-only tests remain.


if __name__ == '__main__':
    unittest.main()
