import unittest
import numpy as np
from latent_opt import init_latent_state, z_to_latents


class TestNoiseBlendInLatents(unittest.TestCase):
    def test_zero_z_produces_nonzero_latents(self):
        st = init_latent_state(width=320, height=256, seed=42)
        z = np.zeros(st.d, dtype=float)
        lat = z_to_latents(st, z)
        self.assertGreater(float(np.abs(lat).mean()), 0.0)

    def test_deterministic_given_seed(self):
        st1 = init_latent_state(width=320, height=256, seed=123)
        st2 = init_latent_state(width=320, height=256, seed=123)
        z = np.zeros(st1.d, dtype=float)
        l1 = z_to_latents(st1, z)
        l2 = z_to_latents(st2, z)
        self.assertTrue(np.allclose(l1, l2))


if __name__ == "__main__":
    unittest.main()
