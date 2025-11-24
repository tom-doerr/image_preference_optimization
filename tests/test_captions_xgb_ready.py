import sys
import types
import numpy as np


def test_captions_include_xgb_and_numeric_after_cache():
    # Capture image captions
    from tests.helpers.st_streamlit import stub_capture_images

    st, images = stub_capture_images()
    st.session_state.prompt = "xgb-caption"
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

    # Flux decode stub
    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "img"
    sys.modules["flux_local"] = fl

    # Provide an XGB scorer via cache + score function
    xv = types.ModuleType("xgb_value")
    xv.score_xgb_proba = lambda mdl, f: 0.6
    sys.modules["xgb_value"] = xv
    st.session_state.xgb_cache = {"model": object(), "n": 2}

    # Provide value_scorer that uses xgb_value proba
    vs = types.ModuleType("value_scorer")
    vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: xv.score_xgb_proba("stub", f), "ok")
    sys.modules["value_scorer"] = vs
import batch_ui

    batch_ui._render_batch_ui()
    # Expect two image captions with a numeric Value
    assert any("Value:" in c for c in images)
    assert any(any(ch.isdigit() for ch in c) for c in images)
