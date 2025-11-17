import unittest
import numpy as np
from latent_opt import init_latent_state, update_latent_ridge, propose_latent_pair_ridge


class TestRidgeWithExtendedFeatures(unittest.TestCase):
    def test_proposals_slice_to_d_dims(self):
        st = init_latent_state(seed=0)
        # Create extended features (latent z plus 3 extra dims)
        z_a = np.zeros(st.d)
        z_b = np.zeros(st.d)
        feats_a = np.concatenate([z_a, np.array([1.0, 2.0, 3.0])])
        feats_b = np.concatenate([z_b, np.array([0.0, 0.0, 0.0])])
        update_latent_ridge(st, z_a, z_b, 'a', feats_a=feats_a, feats_b=feats_b)
        # state.w is now length 7; proposals should still return z of length 4
        z1, z2 = propose_latent_pair_ridge(st, alpha=1.0, beta=1.0)
        self.assertEqual(len(z1), st.d)
        self.assertEqual(len(z2), st.d)


if __name__ == '__main__':
    unittest.main()
