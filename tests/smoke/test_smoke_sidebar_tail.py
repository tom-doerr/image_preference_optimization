import sys
import types

import numpy as np

from constants import Keys
from ui_sidebar import render_sidebar_tail
from tests.helpers.st_streamlit import stub_with_writes


class _LS:
    width = 512
    height = 512
    d = 4
    w = np.zeros(4, dtype=float)
    X = None
    y = None
    mu_hist = np.zeros((0, 4), dtype=float)
    mu = np.zeros(4, dtype=float)
    sigma = 1.0
    step = 0


def test_sidebar_tail_renders_minimal():
    st, writes = stub_with_writes()
    # Stub flux_local.set_model to avoid real loads
    sys.modules["flux_local"] = types.SimpleNamespace(set_model=lambda *a, **k: None)
    st.session_state[Keys.REG_LAMBDA] = 1e-3
    st.session_state[Keys.VM_CHOICE] = "Ridge"
    render_sidebar_tail(
        st,
        _LS(),
        "demo prompt",
        "data/demo/latent_state.npz",
        "Ridge",
        3,
        None,
        "stabilityai/sd-turbo",
        lambda *_: None,
        None,
    )
    text = "\n".join(map(str, writes))
    assert ("Value model" in text) or ("Training data" in text)
