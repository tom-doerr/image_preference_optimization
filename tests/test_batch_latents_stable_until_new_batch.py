import sys
import types
import numpy as np


def test_latents_do_not_change_until_new_batch():
    from tests.helpers.st_streamlit import stub_basic

    st = stub_basic()
    st.session_state.prompt = "stable-batch"
    st.session_state.steps = 1
    st.session_state.guidance_eff = 0.0
    lstate = types.SimpleNamespace(width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0))
    st.session_state.lstate = lstate
    # Seed a deterministic batch
    st.session_state.cur_batch = [np.arange(4, dtype=float), np.arange(4, dtype=float) + 10]
    st.session_state.cur_labels = [None, None]
    sys.modules["streamlit"] = st

    # Minimal latent helpers and decode stub
    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda l, p: np.zeros(l.d)
    ll.z_to_latents = lambda *a, **k: np.zeros((1, 1, 2, 2))
    sys.modules["latent_logic"] = ll

    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "img"
    sys.modules["flux_local"] = fl

    import batch_ui

    before = [b.copy() for b in st.session_state.cur_batch]
    # Render twice: latents must remain identical
    batch_ui._render_batch_ui()
    after1 = [b.copy() for b in st.session_state.cur_batch]
    batch_ui._render_batch_ui()
    after2 = [b.copy() for b in st.session_state.cur_batch]
    assert all(np.array_equal(a, b) for a, b in zip(before, after1))
    assert all(np.array_equal(a, b) for a, b in zip(after1, after2))

    # New batch: latents should change
    batch_ui._curation_new_batch()
    new = st.session_state.cur_batch
    assert not all(np.array_equal(a, b) for a, b in zip(before, new))

