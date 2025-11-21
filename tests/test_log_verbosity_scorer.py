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


def _setup_stub(vm_choice: str):
    from tests.helpers.st_streamlit import stub_basic

    st = stub_basic()
    st.session_state.prompt = "lv-test"
    st.session_state.steps = 1
    st.session_state.guidance_eff = 0.0
    st.session_state.lstate = types.SimpleNamespace(
        width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0), w=np.zeros(4)
    )
    st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
    st.session_state.cur_labels = [None, None]
    st.session_state.vm_choice = vm_choice
    sys.modules["streamlit"] = st

    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
    ll.z_to_latents = lambda *a, **k: np.zeros((1, 1, 2, 2))
    sys.modules["latent_logic"] = ll

    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "img"
    sys.modules["flux_local"] = fl
    return st


def test_scorer_logs_suppressed_when_log_verbosity_zero():
    st = _setup_stub("Ridge")
    st.session_state["log_verbosity"] = 0
    import batch_ui

    out = _capture(batch_ui._render_batch_ui)
    assert "[scorer]" not in out


def test_scorer_logs_present_when_log_verbosity_default():
    st = _setup_stub("Ridge")
    # default is 1
    import batch_ui

    out = _capture(batch_ui._render_batch_ui)
    assert "[scorer]" in out

