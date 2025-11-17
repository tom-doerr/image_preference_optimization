import unittest
import numpy as np
from latent_opt import init_latent_state, update_latent_ridge, save_state, load_state


class TestFullPersistenceHistory(unittest.TestCase):
    def test_save_load_includes_pairs_choices_mu_hist(self):
        st = init_latent_state(seed=0)
        z_a = np.zeros(st.d)
        z_a[0] = 1.0
        z_b = -z_a.copy()
        update_latent_ridge(st, z_a, z_b, 'a')
        update_latent_ridge(st, z_b, z_a, 'b')
        # sanity shapes before save
        self.assertEqual(st.z_pairs.shape, (2, 2, st.d))
        self.assertEqual(st.choices.shape, (2,))
        self.assertTrue(st.mu_hist is not None and st.mu_hist.shape[0] >= 2)

        import tempfile
        import os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'state_full.npz')
            save_state(st, path)
            st2 = load_state(path)
            np.testing.assert_allclose(st2.z_pairs, st.z_pairs)
            np.testing.assert_allclose(st2.choices, st.choices)
            np.testing.assert_allclose(st2.mu_hist, st.mu_hist)


if __name__ == '__main__':
    unittest.main()
