import sys
import types
import numpy as np


def test_value_scorer_uses_live_xgb_model():
    # Provide a dummy XGB model via session_state
    ss = types.SimpleNamespace()
    ss.XGB_MODEL = object()
    # Stub the scorer primitive to return a deterministic value
    xgb_mod = types.ModuleType("ipo.core.xgb_value")
    def _score_xgb_proba(mdl, x):
        # Return sigmoid(sum(x)) for variety
        s = float(np.sum(np.asarray(x, dtype=float)))
        return 1.0 / (1.0 + np.exp(-s))
    xgb_mod.score_xgb_proba = _score_xgb_proba
    sys.modules['ipo.core.xgb_value'] = xgb_mod

    from ipo.core import value_scorer as vs
    lstate = types.SimpleNamespace(d=3, w=np.zeros(3))
    scorer, tag = vs.get_value_scorer("XGBoost", lstate, "p", ss)
    assert tag == "XGB"
    assert callable(scorer)
    v = scorer(np.ones(3))
    assert 0.0 < v < 1.0

