import sys
import types
import numpy as np


def test_ui_flow_sync_xgb_changes_captions():
    # Use the lightweight Streamlit image-capture stub
    from tests.helpers.st_streamlit import stub_capture_images

    st, images = stub_capture_images()
    st.session_state.prompt = "ui-flow-xgb-sync"
    st.session_state.steps = 1
    st.session_state.guidance_eff = 0.0
    # Minimal latent state and batch
    lstate = types.SimpleNamespace(
        width=64, height=64, d=4, sigma=1.0, rng=np.random.default_rng(0), w=np.zeros(4)
    )
    st.session_state.lstate = lstate
    st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
    st.session_state.cur_labels = [None, None]
    st.session_state.vm_choice = "XGBoost"
    # Provide both classes so a sync fit is meaningful
    dsX = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    dsY = np.array([+1.0, -1.0, +1.0, -1.0], dtype=float)
    st.session_state.dataset_X = dsX
    st.session_state.dataset_y = dsY
    # Also expose the dataset on lstate so the sync-fit button path picks it up
    lstate.X = dsX
    lstate.y = dsY
    sys.modules["streamlit"] = st

    # Latent helpers and flux stub
    ll = types.ModuleType("latent_logic")
    ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
    ll.z_to_latents = lambda *a, **k: np.zeros((1, 1, 2, 2))
    sys.modules["latent_logic"] = ll

    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "img"
    fl.set_model = lambda *a, **k: None
    sys.modules["flux_local"] = fl

    # Before training, captions should be n/a for XGB
    import batch_ui

    images.clear()
    batch_ui._render_batch_ui()
    assert any("Value: n/a" in c for c in images)

    # Stub value_model.fit_value_model to perform a synchronous fit that writes cache
    from constants import Keys

    vm = types.ModuleType("value_model")

    def fit_value_model(vm_choice, _lstate, X, y, lam, session_state):
        session_state[Keys.XGB_CACHE] = {"model": "stub", "n": int(X.shape[0])}
        session_state[Keys.XGB_TRAIN_STATUS] = {
            "state": "ok",
            "rows": int(X.shape[0]),
            "lam": float(lam),
        }

    def ensure_fitted(*a, **k):
        return None

    vm.fit_value_model = fit_value_model
    vm.ensure_fitted = ensure_fitted
    sys.modules["value_model"] = vm

    # Stub xgb_value scorer to return a probability so captions become numeric
    xv = types.ModuleType("xgb_value")
    xv.score_xgb_proba = lambda model, fvec: 0.73
    sys.modules["xgb_value"] = xv

    # Click the sync-train button via sidebar stub
    # Stub lightweight UI helpers used by ui_sidebar to avoid latent_opt imports
    ui_mod = types.ModuleType("ui")
    ui_mod.sidebar_metric_rows = lambda rows, per_row=2: None
    sys.modules["ui"] = ui_mod

    uim = types.ModuleType("ui_metrics")
    uim.render_iter_step_scores = lambda *a, **k: None
    uim.render_mu_value_history = lambda *a, **k: None
    sys.modules["ui_metrics"] = uim

    # Provide a value_scorer that reads the XGB cache and delegates to xgb_value
    vs = types.ModuleType("value_scorer")
    def _xgb_scorer_with_status(*a, **k):
        return (lambda f: xv.score_xgb_proba("stub", f), "ok")
    vs.get_value_scorer_with_status = _xgb_scorer_with_status
    sys.modules["value_scorer"] = vs

    import ui_sidebar

    def _btn(label, *a, **k):
        return label == "Train XGBoost now (sync)"

    st.sidebar.button = _btn
    images.clear()
    ui_sidebar.render_sidebar_tail(
        st,
        lstate,
        st.session_state.prompt,
        "data/dda60c595a/latent_state.npz",
        "XGBoost",
        10,
        0.1,
        "stabilityai/sd-turbo",
        lambda *a, **k: None,
        lambda *a, **k: None,
    )
    # After sync fit, re-render batch to see updated captions
    images.clear()
    batch_ui._render_batch_ui()
    assert any("Value:" in c and "n/a" not in c for c in images)
