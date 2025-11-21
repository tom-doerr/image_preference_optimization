import io
import sys
import types
import numpy as np


def _capture(func, *a, **k):
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        func(*a, **k)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_ensure_fitted_ridge_debug_and_updates_weights():
    from value_model import ensure_fitted

    # Stub ridge_fit to avoid heavy math
    ll = types.ModuleType("latent_logic")
    ll.ridge_fit = lambda X, y, lam: np.ones(X.shape[1])
    sys.modules["latent_logic"] = ll

    lstate = types.SimpleNamespace(d=4, w=np.zeros(4), w_lock=None)
    X = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=float)
    y = np.array([1, -1], dtype=int)
    ss = {}

    out = _capture(ensure_fitted, "Ridge", lstate, X, y, 1.0, ss)
    assert "[ensure] ridge sync fit" in out
    assert float(np.linalg.norm(lstate.w)) > 0.0


def test_ridge_scorer_status_prints():
    from value_scorer import get_value_scorer_with_status

    lstate = types.SimpleNamespace(d=4, w=np.zeros(4))
    out1 = _capture(get_value_scorer_with_status, "Ridge", lstate, "p", {})
    assert "[ridge-scorer]" in out1 and "ridge_untrained" in out1
    lstate.w = np.ones(4)
    out2 = _capture(get_value_scorer_with_status, "Ridge", lstate, "p", {})
    assert "[ridge-scorer]" in out2 and "status=ok" in out2


def test_batch_replace_at_debug_print():
    # Stub streamlit
    from tests.helpers.st_streamlit import stub_basic

    st = stub_basic()
    st.session_state.prompt = "dbg-replace"
    st.session_state.lstate = types.SimpleNamespace(d=4, sigma=1.0)
    st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
    st.session_state.cur_labels = [None, None]
    st.session_state.cur_batch_nonce = 7
    sys.modules["streamlit"] = st

    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
    sys.modules["latent_logic"] = ll

    import batch_ui

    out = _capture(batch_ui._curation_replace_at, 1)
    assert "[batch] replace_at idx=1" in out


def test_data_save_debug_print():
    from tests.helpers.st_streamlit import stub_basic

    st = stub_basic()
    st.session_state.prompt = "dbg-save"
    st.session_state.lstate = types.SimpleNamespace(d=4, sigma=1.0, rng=np.random.default_rng(0))
    sys.modules["streamlit"] = st

    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
    sys.modules["latent_logic"] = ll

    import batch_ui

    z = np.ones(4)
    out = _capture(batch_ui._curation_add, +1, z, None)
    assert "[data] append label=1" in out
    assert "[data] saved row=" in out
