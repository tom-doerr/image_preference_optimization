import sys
import types
import numpy as np


def test_train_xgb_now_button_sets_cache():
    from tests.helpers.st_streamlit import stub_basic
from constants import Keys

    st = stub_basic()
    st.session_state.prompt = "xgb-train-now"
    st.session_state.lstate = types.SimpleNamespace(d=4, w=np.zeros(4))

    # Sidebar button: only return True for the Train XGBoost now (sync) action
    class SB(st.sidebar.__class__):
        @staticmethod
        def button(label, *a, **k):
            return label == "Train XGBoost now (sync)"

    st.sidebar = SB()
    sys.modules["streamlit"] = st

    # Dataset provider for sidebar
    p = types.ModuleType("persistence")
    p.get_dataset_for_prompt_or_session = lambda prompt, ss: (
        np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=float),
        np.array([1, -1], dtype=int),
    )
    p.dataset_stats_for_prompt = lambda prompt: {"pos": 1, "neg": 1, "d": 4}
    p.dataset_rows_for_prompt_dim = lambda prompt, d: 2
    p.dataset_rows_for_prompt = lambda prompt: 2
    sys.modules["persistence"] = p

    # Stub flux model set
    fl = types.ModuleType("flux_local")
    fl.set_model = lambda *a, **k: None
    sys.modules["flux_local"] = fl

    # XGB trainer stub
    xv = types.ModuleType("xgb_value")
    xv.fit_xgb_classifier = lambda X, y, n_estimators=50, max_depth=3: object()
    sys.modules["xgb_value"] = xv

    # Provide minimal latent_opt to satisfy ui import chain
    lo = types.ModuleType("latent_opt")
    lo.z_from_prompt = lambda st, p: np.zeros(4)
    sys.modules["latent_opt"] = lo
    # Provide a minimal latent_logic for ui helper imports
    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda st, p: np.zeros(4)
    ll.z_to_latents = lambda st, z: z.reshape(1, 1, 1, 1)
    sys.modules["latent_logic"] = ll

    # Stub persistence_ui import used by ui_sidebar
    pui = types.ModuleType("persistence_ui")
    pui.render_metadata_panel = lambda *a, **k: None
    pui.render_persistence_controls = lambda *a, **k: None
    sys.modules["persistence_ui"] = pui

    import ipo.ui.ui_sidebar as ui_sidebar

    st.session_state[Keys.VM_CHOICE] = "XGBoost"
    ui_sidebar.render_sidebar_tail(
        st,
        st.session_state.lstate,
        st.session_state.prompt,
        "data/hash/latent_state.npz",
        "XGBoost",
        10,
        0.1,
        "stabilityai/sd-turbo",
        lambda *a, **k: None,
        lambda *a, **k: None,
    )
    assert isinstance(st.session_state.get("xgb_cache", {}).get("model"), object)
