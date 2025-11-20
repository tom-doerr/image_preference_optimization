import sys
import types
import unittest
import numpy as np


class LS:
    def __init__(self, d=4):
        self.d = d
        self.sigma = 1.0
        self.rng = np.random.default_rng(0)


class TestSampleOneForBatch(unittest.TestCase):
    def test_uses_xgb_hill_when_available(self):
        import batch_ui as bu

        # Stub latent_logic.sample_z_xgb_hill to return a sentinel vector
        ll_orig = sys.modules.get("latent_logic")
        try:
            ll = types.ModuleType("latent_logic")
            sentinel = np.ones(4, dtype=np.float32) * 7
            ll.sample_z_xgb_hill = lambda *a, **k: sentinel
            sys.modules["latent_logic"] = ll
            z = bu._sample_one_for_batch(
                LS(4), "p", True, object(), steps=3, lr_mu=0.3, trust_r=None
            )
            self.assertTrue(np.allclose(z, sentinel))
        finally:
            if ll_orig is not None:
                sys.modules["latent_logic"] = ll_orig
            else:
                sys.modules.pop("latent_logic", None)

    def test_falls_back_to_around_prompt_when_no_xgb(self):
        import batch_ui as bu

        # Monkeypatch the around_prompt helper
        called = {}

        def fake_sample():
            called["x"] = True
            return np.full(4, 2.0, dtype=np.float32)

        bu._sample_around_prompt = lambda scale=0.8: fake_sample()
        z = bu._sample_one_for_batch(
            LS(4), "p", False, None, steps=3, lr_mu=0.3, trust_r=None
        )
        self.assertTrue(called.get("x"))
        self.assertTrue(np.allclose(z, np.full(4, 2.0)))


if __name__ == "__main__":
    unittest.main()
