import sys
import types
import numpy as np


class _DummyXGB:
    def __init__(self, **k):
        self.k = k

    def fit(self, X, y):
        self.n = int(getattr(X, "shape", (0,))[0])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.tile(np.array([[0.05, 0.95]], dtype=float), (X.shape[0], 1))


def test_integration_xgb_training_switches_captions_to_xgb():
    # Stub xgboost
    xgb = types.ModuleType("xgboost")
    xgb.XGBClassifier = _DummyXGB
    sys.modules["xgboost"] = xgb

    # Stub streamlit (capture image captions) and ensure modules import it
    from tests.helpers.st_streamlit import stub_capture_images
    st, images = stub_capture_images()
    sys.modules["streamlit"] = st

    # Stub flux + latent logic for fast decode
    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
    sys.modules["flux_local"] = fl
    ll = types.ModuleType("latent_logic")
    ll.z_to_latents = lambda *a, **k: (a[1] if len(a) > 1 else a[0])
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(int(getattr(lstate, "d", 8)))
    sys.modules["latent_logic"] = ll

    # Minimal lstate with an in‑memory dataset (both classes)
    class LS:
        width = 64
        height = 64
        d = 256
        sigma = 1.0
        X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
        y = np.array([+1, -1, +1, -1], dtype=float)

    lstate = LS()
    st.session_state.lstate = lstate
    st.session_state.prompt = "p"

    # Trigger XGB sync training via the sidebar button path
    def _btn(label, *a, **k):
        return label == "Train XGBoost now (sync)"

    st.sidebar.button = _btn
    from ipo.ui.ui_sidebar import _handle_train_section
    _handle_train_section(st, lstate, "p", "XGBoost")

    # Disable further button clicks
    st.sidebar.button = lambda *a, **k: False

    # Set VM choice to XGBoost so the scorer/tag is XGB
    from ipo.infra.constants import Keys
    st.session_state[Keys.VM_CHOICE] = "XGBoost"

    # Run batch mode and assert at least one caption uses [XGB]
    from ipo.ui import batch_ui
    # Keep batch small for speed
    st.session_state["batch_size"] = 2
    st.session_state["steps"] = 1
    st.session_state["guidance_eff"] = 0.0
    batch_ui.run_batch_mode()
    assert any("[XGB]" in cap for cap in images)
