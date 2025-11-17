import unittest
import numpy as np
from latent_opt import init_latent_state, update_latent_ridge, dumps_state, loads_state


class TestBackwardCompatMissingAttrs(unittest.TestCase):
    def test_update_and_dump_without_new_fields(self):
        st = init_latent_state(seed=0)
        # Simulate an older in-memory object missing new attributes
        for attr in ('z_pairs', 'choices', 'mu_hist'):
            if hasattr(st, attr):
                delattr(st, attr)

        z_a = np.zeros(st.d)
        z_a[0] = 1.0
        z_b = -z_a.copy()
        update_latent_ridge(st, z_a, z_b, 'a')

        # Attributes should be (re)created
        self.assertTrue(hasattr(st, 'z_pairs'))
        self.assertEqual(st.z_pairs.shape, (1, 2, st.d))
        self.assertEqual(st.choices.shape, (1,))
        self.assertTrue(hasattr(st, 'mu_hist'))
        self.assertEqual(st.mu_hist.shape[1], st.d)

        # Dump and load roundtrip without errors
        data = dumps_state(st)
        st2 = loads_state(data)
        np.testing.assert_allclose(st2.z_pairs, st.z_pairs)
        np.testing.assert_allclose(st2.choices, st.choices)
        np.testing.assert_allclose(st2.mu_hist, st.mu_hist)


if __name__ == '__main__':
    unittest.main()
