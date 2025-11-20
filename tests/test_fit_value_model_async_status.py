import sys
import types
import unittest
import numpy as np


class FitValueModelAsyncStatusTest(unittest.TestCase):
    def tearDown(self):
        for m in ("value_model", "xgb_value", "background"):
            sys.modules.pop(m, None)

    def test_async_flag_sets_status_and_cache(self):
        from tests.helpers.st_streamlit import Session

        ss = Session()
        ss.xgb_train_async = True
        ss.xgb_cache = {}
        ss.reg_lambda = 0.001
        ss.xgb_n_estimators = 20
        ss.xgb_max_depth = 3
        ss.last_train_at = None
        ss.vm_choice = "XGBoost"

        # Dummy latent state with ridge weights
        lstate = types.SimpleNamespace(w=np.zeros(2))
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([1.0, -1.0, 1.0])

        # Stub XGB trainer
        xv = types.ModuleType("xgb_value")
        xv.fit_xgb_classifier = lambda Xd, yd, n_estimators=50, max_depth=3: "mdl"
        sys.modules["xgb_value"] = xv

        # Background executor that runs inline
        class _Exec:
            def submit(self, fn):
                res = fn()
                return types.SimpleNamespace(done=lambda: True, result=lambda: res)

        bg = types.ModuleType("background")
        bg.get_executor = lambda: _Exec()
        sys.modules["background"] = bg

        vm = __import__("value_model")
        vm.fit_value_model("XGBoost", lstate, X, y, 0.001, ss)

        self.assertIn("model", getattr(ss, "xgb_cache", {}))
        self.assertEqual(ss.xgb_cache["model"], "mdl")
        self.assertEqual(ss.xgb_train_status.get("state"), "ok")
        self.assertEqual(ss.xgb_last_updated_rows, X.shape[0])


if __name__ == "__main__":
    unittest.main()
