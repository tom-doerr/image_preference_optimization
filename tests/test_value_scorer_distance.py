import sys
import types
import numpy as np


def test_distance_scorer_respects_exponent():
    from value_scorer import get_value_scorer_with_status
    # lstate with d=3
    lstate = types.SimpleNamespace(d=3)
    ss = {"dist_exp": 2.0}
    scorer, st = get_value_scorer_with_status("Distance", lstate, "p", ss)
    assert st == "ok"
    v2 = scorer(np.array([1.0, 2.0, 0.0]))  # - (1^2+2^2) = -5
    ss["dist_exp"] = 1.0
    scorer1, _ = get_value_scorer_with_status("Distance", lstate, "p", ss)
    v1 = scorer1(np.array([1.0, 2.0, 0.0]))  # - (|1|+|2|) = -3
    assert v2 < v1  # more negative for p=2 vs p=1 on same vector

