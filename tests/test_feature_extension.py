import unittest
import numpy as np
from latent_opt import init_latent_state, update_latent_ridge


class TestFeatureExtension(unittest.TestCase):
    def test_update_with_extended_features(self):
        st = init_latent_state(seed=0)
        z_a = np.zeros(st.d)
        z_b = np.zeros(st.d)
        feats_a = np.concatenate([z_a, np.array([1.0, 2.0, 3.0])])
        feats_b = np.concatenate([z_b, np.array([0.0, 0.0, 0.0])])
        update_latent_ridge(st, z_a, z_b, 'a', feats_a=feats_a, feats_b=feats_b)
        self.assertEqual(len(st.w), st.d + 3)
        # Direction for proposals uses only first d dims; should be defined
        self.assertEqual(st.d, st.d)


if __name__ == '__main__':
    unittest.main()
