import unittest
import numpy as np
from metrics import pair_metrics


class TestMetrics(unittest.TestCase):
    def test_pair_metrics_basic(self):
        w = np.array([1.0, 0.0])
        za = np.array([0.0, 0.0])
        zb = np.array([1.0, 0.0])
        m = pair_metrics(w, za, zb)
        self.assertAlmostEqual(m['za_norm'], 0.0)
        self.assertAlmostEqual(m['zb_norm'], 1.0)
        self.assertAlmostEqual(m['diff_norm'], 1.0)
        self.assertAlmostEqual(m['cos_w_diff'], 1.0)

    def test_pair_metrics_nan_cos(self):
        w = np.array([0.0, 0.0])
        za = np.array([0.0, 0.0])
        zb = np.array([0.0, 0.0])
        m = pair_metrics(w, za, zb)
        self.assertTrue(np.isnan(m['cos_w_diff']))


if __name__ == '__main__':
    unittest.main()

