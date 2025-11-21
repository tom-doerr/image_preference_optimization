import sys
import types
import numpy as np


def test_captions_include_ridge_and_numeric_when_w_nonzero():
    from tests.helpers.st_streamlit import stub_capture_images

    st, images = stub_capture_images()
    st.session_state.prompt = "ridge-caption"
    st.session_state.steps = 1
    st.session_state.guidance_eff = 0.0
    st.session_state.lstate = types.SimpleNamespace(
        width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0), w=np.ones(4)
    )
    st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
    st.session_state.cur_labels = [None, None]
    st.session_state.vm_choice = "Ridge"
    st.session_state.use_ridge_captions = True
    sys.modules["streamlit"] = st

    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
    ll.z_to_latents = lambda lstate, z: z.reshape(1, 1, 2, 2)
    sys.modules["latent_logic"] = ll

    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "img"
    sys.modules["flux_local"] = fl

    import batch_ui

    batch_ui._render_batch_ui()
    assert any("[Ridge]" in c and "Value:" in c for c in images)
    assert any(any(ch.isdigit() for ch in c) for c in images)
