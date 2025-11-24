import sys
import types
import numpy as np


def test_batch_mode_smoke_decodes_and_captures_captions():
    # Stub flux_local.generate_flux_image_latents to avoid GPU/deps
    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
    sys.modules["flux_local"] = fl

    # Stub latent logic
    ll = types.ModuleType("latent_logic")
    ll.z_to_latents = lambda *a, **k: (a[1] if len(a) > 1 else a[0])
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(int(getattr(lstate, "d", 8)))
    sys.modules["latent_logic"] = ll

    # Capture image captions
    from tests.helpers.st_streamlit import stub_capture_images
    st, images = stub_capture_images()
    sys.modules["streamlit"] = st

    # Prepare session state
    st.session_state["batch_size"] = 3
    st.session_state["steps"] = 1
    st.session_state["guidance_eff"] = 0.0

    # Minimal lstate
    class LS:
        width = 64
        height = 64
        d = 25600  # match full-latent default to avoid shape mismatch
        sigma = 1.0

    st.session_state.lstate = LS()
    st.session_state.prompt = "p"

    from ipo.ui import batch_ui

    # Run flow (no clicks in stub, so just renders)
    batch_ui.run_batch_mode()

    # We should have captured some captions for the batch
    assert any("Item" in c for c in images)
