import io
import sys
import types
import numpy as np


class DummyFuture:
    def __init__(self, done_flag=False):
        self._done = done_flag

    def done(self):
        return bool(self._done)


def _capture(func, *a, **k):
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        func(*a, **k)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_xgb_fit_runs_sync_even_when_future_present():
    from value_model import fit_value_model
    from constants import Keys

    # Minimal stubs
    lstate = types.SimpleNamespace(d=4, w=np.zeros(4), w_lock=None)
    X = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=float)
    y = np.array([1, -1], dtype=int)
    ss = {Keys.XGB_FIT_FUTURE: DummyFuture(done_flag=False)}
    out = _capture(fit_value_model, "XGBoost", lstate, X, y, 1.0, ss)
    assert "[xgb] train start rows=" in out
    assert ss.get(Keys.XGB_FIT_FUTURE) is None or not hasattr(ss.get(Keys.XGB_FIT_FUTURE), 'done')
    assert ss.get(Keys.XGB_TRAIN_STATUS, {}).get("state") == "ok"
