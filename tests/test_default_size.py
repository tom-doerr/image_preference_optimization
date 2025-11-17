import unittest
from latent_opt import init_latent_state


class TestDefaultSize(unittest.TestCase):
    def test_init_latent_state_defaults(self):
        st = init_latent_state()
        self.assertEqual(st.width, 512)
        self.assertEqual(st.height, 512)
        self.assertEqual(st.d, 4 * (st.height // 8) * (st.width // 8))
        self.assertEqual(st.d, 4 * 64 * 64)


if __name__ == '__main__':
    unittest.main()

