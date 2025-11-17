import unittest
import numpy as np
from latent_opt import init_latent_state, propose_latent_pair_ridge


class TestRidgeDirections(unittest.TestCase):
    def test_directions_approximately_orthogonal(self):
        st = init_latent_state(seed=0)
        st.w[:] = 0.0
        st.w[0] = 3.0
        st.w[1] = 2.0
        st.w[2] = 1.0
        z1, z2 = propose_latent_pair_ridge(st, alpha=1.0, beta=1.0)
        d1 = z1 - st.mu
        d2 = z2 - st.mu
        dot = float(np.dot(d1, d2))
        self.assertAlmostEqual(dot, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
