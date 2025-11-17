import unittest
import numpy as np
from latent_opt import init_latent_state, z_to_latents


class TestZeroMeanLatents(unittest.TestCase):
    def test_channelwise_zero_mean(self):
        st = init_latent_state(width=320, height=256, seed=42)
        z = st.rng.standard_normal(st.d)
        lat = z_to_latents(st, z)
        # Channel-wise mean close to zero
        ch_means = lat.mean(axis=(0, 2, 3))
        self.assertLess(float(np.abs(ch_means).max()), 1e-5)


if __name__ == '__main__':
    unittest.main()
