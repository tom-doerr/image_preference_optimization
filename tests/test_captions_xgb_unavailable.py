import sys
import types
import numpy as np


def test_captions_n_a_when_xgb_unavailable():
    from tests.helpers.st_streamlit import stub_capture_images

    st, images = stub_capture_images()
    st.session_state.prompt = "xgb-na"
    st.session_state.steps = 1
    st.session_state.guidance_eff = 0.0
    st.session_state.lstate = types.SimpleNamespace(
        width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0)
    )
    st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
    st.session_state.cur_labels = [None, None]
    st.session_state.vm_choice = "XGBoost"
    sys.modules["streamlit"] = st

    # Minimal latent helpers
    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
    ll.z_to_latents = lambda lstate, z: z.reshape(1, 1, 2, 2)
    sys.modules["latent_logic"] = ll

    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "img"
    sys.modules["flux_local"] = fl

    # No xgb_cache present → scorer unavailable
import batch_ui

    batch_ui._render_batch_ui()
    assert any("Value: n/a" in c for c in images)

