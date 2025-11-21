import sys
import types
import numpy as np


def test_xgb_training_status_returns_zero_scorer():
    # session with running future and no cache
    class SS(dict):
        __getattr__ = dict.get
    ss = SS()
    class F:
        def done(self):
            return False
    ss["xgb_fit_future"] = F()

    lstate = types.SimpleNamespace(d=4, w=np.zeros(4))
from value_scorer import get_value_scorer

    scorer, status = get_value_scorer("XGBoost", lstate, "p", ss)
    assert status in ("xgb_training", "xgb_unavailable")
    assert scorer is None


def test_ridge_status_ok_when_w_nonzero():
from value_scorer import get_value_scorer

    lstate = types.SimpleNamespace(d=4, w=np.ones(4))
    scorer, status = get_value_scorer("Ridge", lstate, "p", {})
    assert status == "Ridge"
    assert isinstance(float(scorer(np.ones(4))), float)
