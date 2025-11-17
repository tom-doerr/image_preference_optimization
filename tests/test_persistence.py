import os
import tempfile
import unittest
import numpy as np
from latent_opt import init_latent_state, save_state, load_state


class TestPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        st = init_latent_state(width=320, height=256, seed=0)
        st.mu[:] = np.arange(8)
        st.w[:] = -1
        st.sigma = 0.42
        st.step = 7
        # Add some history (X,y)
        st.X = (np.arange(16, dtype=float).reshape(2, 8))
        st.y = np.array([1.0, -1.0])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'state.npz')
            save_state(st, path)
            st2 = load_state(path)
            self.assertEqual(st2.width, st.width)
            self.assertEqual(st2.height, st.height)
            self.assertEqual(st2.d, st.d)
            np.testing.assert_allclose(st2.mu, st.mu)
            np.testing.assert_allclose(st2.w, st.w)
            self.assertAlmostEqual(st2.sigma, st.sigma)
            self.assertEqual(st2.step, st.step)
            np.testing.assert_allclose(st2.X, st.X)
            np.testing.assert_allclose(st2.y, st.y)


if __name__ == '__main__':
    unittest.main()
