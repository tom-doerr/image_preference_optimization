import sys
import types


def test_sidebar_tail_smoke_renders_core_lines():
    from tests.helpers.st_streamlit import stub_with_writes
    from ipo.infra.constants import Keys

    # Stub Streamlit
    st, writes = stub_with_writes()

    # Minimal lstate
    class LS:  # simple namespace
        width = 64
        height = 64
        d = 25600

    lstate = LS()
    prompt = "latex, neon punk city, women with short hair, standing in the rain"
    state_path = "latent_state_dummy.npz"
    vm_choice = "XGBoost"
    iter_steps = 3
    iter_eta = 0.0
    selected_model = "stabilityai/sd-turbo"

    # Stub flux_local.set_model and get_last_call to avoid heavy imports
    fl = types.ModuleType("flux_local")
    fl.set_model = lambda *a, **k: None
    fl.get_last_call = lambda: {}
    sys.modules["flux_local"] = fl

    from ipo.ui.ui_sidebar import render_sidebar_tail

    # Call with no-op apply/rerun cbs
    render_sidebar_tail(
        st,
        lstate,
        prompt,
        state_path,
        vm_choice,
        iter_steps,
        iter_eta,
        selected_model,
        lambda *a, **k: None,
        lambda *a, **k: None,
    )

    # Assert key baseline lines are present
    text = "\n".join(writes)
    assert "Train score:" in text
    assert "CV score:" in text
    assert "XGBoost" in text or "Ridge" in text
    assert "Optimization: Ridge only" in text

