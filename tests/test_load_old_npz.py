import os
import tempfile
import unittest
import numpy as np
from latent_opt import init_latent_state, load_state, update_latent_ridge


class TestLoadOldNPZ(unittest.TestCase):
    def test_load_npz_without_new_keys(self):
        st = init_latent_state(width=320, height=256, seed=0)
        # Create an "old" npz missing z_pairs/choices/mu_hist/X/y
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'old_state.npz')
            np.savez_compressed(
                path,
                width=st.width,
                height=st.height,
                d=st.d,
                mu=st.mu,
                sigma=st.sigma,
                w=st.w,
                step=st.step,
            )
            st2 = load_state(path)
            # New fields should be None until first update
            self.assertIsNone(st2.X)
            self.assertIsNone(st2.y)
            self.assertIsNone(st2.z_pairs)
            self.assertIsNone(st2.choices)
            self.assertIsNone(st2.mu_hist)

            # After one update they should be initialized
            za = np.zeros(st2.d)
            za[0] = 1.0
            zb = -za.copy()
            update_latent_ridge(st2, za, zb, 'a')
            self.assertEqual(st2.z_pairs.shape, (1, 2, st2.d))
            self.assertEqual(st2.choices.shape, (1,))
            self.assertEqual(st2.mu_hist.shape[1], st2.d)


if __name__ == '__main__':
    unittest.main()
