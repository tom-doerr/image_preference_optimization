import unittest
import numpy as np
from latent_opt import init_latent_state, z_from_prompt


class TestZFromPrompt(unittest.TestCase):
    def test_deterministic_same_prompt(self):
        st = init_latent_state(width=320, height=256, seed=0)
        z1 = z_from_prompt(st, "a prompt")
        z2 = z_from_prompt(st, "a prompt")
        self.assertTrue(np.allclose(z1, z2))

    def test_different_prompts_differ(self):
        st = init_latent_state(width=320, height=256, seed=0)
        z1 = z_from_prompt(st, "a prompt")
        z2 = z_from_prompt(st, "another prompt")
        # Very high probability different
        self.assertGreater(float(np.linalg.norm(z1 - z2)), 1e-6)


if __name__ == '__main__':
    unittest.main()

