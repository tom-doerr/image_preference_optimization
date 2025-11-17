import unittest
import numpy as np
from latent_opt import init_latent_state, dumps_state, loads_state


class TestPersistenceBytes(unittest.TestCase):
    def test_roundtrip_bytes(self):
        st = init_latent_state(width=320, height=256, seed=0)
        st.mu[:] = np.arange(8)
        st.w[:] = np.linspace(0, 1, 8)
        st.sigma = 0.33
        st.step = 5
        st.X = (np.arange(24, dtype=float).reshape(3, 8))
        st.y = np.array([1.0, -1.0, 1.0])
        data = dumps_state(st)
        st2 = loads_state(data)
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
