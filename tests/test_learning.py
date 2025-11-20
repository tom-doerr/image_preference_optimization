import unittest
import numpy as np
from learning import (
    init_state,
    propose_pair,
    update,
    feature_to_image,
    estimate_preferred_feature,
)


class TestLearning(unittest.TestCase):
    def test_propose_pair_bounds(self):
        s = init_state(seed=1)
        a, b = propose_pair(s)
        self.assertTrue(np.all(a >= 0) and np.all(a <= 1))
        self.assertTrue(np.all(b >= 0) and np.all(b <= 1))

    def test_update_direction(self):
        s = init_state(seed=2)
        a = np.array([0.9, 0.1, 0.1])
        b = np.array([0.1, 0.9, 0.9])
        update(s, a, b, "a", lr=1.0)
        np.testing.assert_allclose(s.w, a - b)

    def test_training_converges_with_random_pairs(self):
        rng = np.random.default_rng(0)
        s = init_state(seed=0)
        u_true = np.array([0.8, 0.15, 0.05])
        base_dot = np.dot(s.w, u_true)
        for _ in range(400):
            a = rng.random(3)
            b = rng.random(3)
            choice = "a" if np.dot(a, u_true) >= np.dot(b, u_true) else "b"
            update(s, a, b, choice, lr=0.1)
        self.assertGreater(np.dot(s.w, u_true), base_dot + 0.2)

    def test_feature_to_image_shape(self):
        img = feature_to_image(np.array([0.5, 0.6, 0.7]), size=32)
        self.assertEqual(img.shape, (32, 32, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_estimate_preferred_feature_bounds(self):
        s = init_state(seed=3)
        pref = estimate_preferred_feature(s)
        self.assertTrue(np.all(pref >= 0) and np.all(pref <= 1))


if __name__ == "__main__":
    unittest.main()
