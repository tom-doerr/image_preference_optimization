import sys
import types
import numpy as np


def test_captions_ridge_zero_w_explicit_zero():
    from tests.helpers.st_streamlit import stub_capture_images

    st, images = stub_capture_images()
    st.session_state.prompt = "ridge-zero-caption"
    st.session_state.steps = 1
    st.session_state.guidance_eff = 0.0
    st.session_state.lstate = types.SimpleNamespace(
        width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0), w=np.zeros(4)
    )
    st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
    st.session_state.cur_labels = [None, None]
    st.session_state.vm_choice = "Ridge"
    sys.modules["streamlit"] = st

    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
    # Accept both call signatures used in production code
    ll.z_to_latents = lambda *a, **k: np.zeros((1, 1, 2, 2))
    sys.modules["latent_logic"] = ll

    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "img"
    sys.modules["flux_local"] = fl

    # Ridge scorer stub: returns 0.0 when w==0
    vs = types.ModuleType("value_scorer")
    vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
    sys.modules["value_scorer"] = vs

    import batch_ui

    batch_ui._render_batch_ui()
    # With ||w|| == 0, captions should be n/a under 199f
    assert any("Value: n/a" in c for c in images)
