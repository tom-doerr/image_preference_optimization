import sys
import types
import numpy as np


class DummyFuture:
    def __init__(self, done_flag=False):
        self._done = done_flag

    def done(self):
        return bool(self._done)


def test_fit_value_model_sync_always():
    from value_model import fit_value_model
    from constants import Keys

    # Stub background executors
    bg = types.ModuleType("background")
    bg.get_train_executor = lambda: types.SimpleNamespace(submit=lambda fn: DummyFuture(False))
    bg.get_executor = bg.get_train_executor
    sys.modules["background"] = bg

    # Stub xgb trainer
    xv = types.ModuleType("xgb_value")
    xv.fit_xgb_classifier = lambda X, y, n_estimators=50, max_depth=3: object()
    sys.modules["xgb_value"] = xv

    lstate = types.SimpleNamespace(d=4, w=np.zeros(4))
    X = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=float)
    y = np.array([1, -1], dtype=int)

    # Regardless of toggle, XGB trains synchronously now
    for flag in (True, False):
        ss = {Keys.XGB_TRAIN_ASYNC: flag}
        fit_value_model("XGBoost", lstate, X, y, 1.0, ss)
        assert ss.get(Keys.XGB_FIT_FUTURE) is None
        assert ss.get(Keys.XGB_TRAIN_STATUS, {}).get("state") == "ok"
        assert isinstance(ss.get(Keys.XGB_CACHE, {}).get("model"), object)
