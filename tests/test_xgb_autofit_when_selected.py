import types
import numpy as np


class _Exp:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _SB:
    def subheader(self, *a, **k):
        return None
    def write(self, *a, **k):
        return None

    def metric(self, *a, **k):
        return None

    def expander(self, *a, **k):
        return _Exp()

    def download_button(self, *a, **k):
        return None

    def number_input(self, *a, **k):
        # Return the default value when provided
        return k.get("value")

    def slider(self, *a, **k):
        return k.get("value")

    def button(self, *a, **k):
        return False


class _ST:
    def __init__(self):
        self.session_state = types.SimpleNamespace()
        self.sidebar = _SB()

    def toast(self, *a, **k):
        return None


class _LState:
    def __init__(self, d):
        self.d = d
        self.width = 640
        self.height = 640
        self.w = np.zeros(d, dtype=float)


def test_autofit_xgb_when_selected_trains_and_sets_cache():
    # Prepare small in-memory dataset with both classes
    d = 8
    X = np.zeros((4, d), dtype=float)
    X[0, 0] = 1.0
    X[1, 1] = -1.0
    X[2, 2] = 2.0
    X[3, 3] = -2.0
    y = np.array([+1, -1, +1, -1], dtype=float)

    st = _ST()
    # Minimal keys used by the sidebar code
    from constants import Keys

    setattr(st.session_state, Keys.PROMPT, "unit test prompt")
    setattr(st.session_state, Keys.REG_LAMBDA, 1.0)
    # Attach latent state with in-memory dataset
    ls = _LState(d)
    ls.X = X
    ls.y = y
    st.session_state.lstate = ls
    st.session_state.state_path = "data/test/latent_state.npz"

    # Call the sidebar tail; this should auto-fit XGB when selected
    # Stub flux_local.set_model to avoid importing heavy deps and xgboost lib
    import sys
    sys.modules['flux_local'] = types.SimpleNamespace(set_model=lambda *a, **k: None)
    # Provide a tiny xgboost stub so core xgb_value can import and fit
    class _DummyXGB:
        def __init__(self, **k):
            self.bias = 0.0

        def fit(self, X, y):
            yy = np.asarray(y).astype(float)
            self.bias = float((yy > 0).mean())

        def predict_proba(self, X):
            n = int(np.asarray(X).shape[0])
            p = np.clip(self.bias, 0.0, 1.0)
            # return two-class proba with fixed p
            return np.vstack([1.0 - p * np.ones(n), p * np.ones(n)]).T

    sys.modules['xgboost'] = types.SimpleNamespace(XGBClassifier=_DummyXGB)
    from ipo.ui.ui_sidebar import render_sidebar_tail

    render_sidebar_tail(
        st,
        ls,
        getattr(st.session_state, Keys.PROMPT),
        st.session_state.state_path,
        "XGBoost",
        1,
        0.0,
        "stabilityai/sd-turbo",
        lambda *a, **k: None,
        lambda *a, **k: None,
    )

    # Assert cache exists and matches row count
    cache = getattr(st.session_state, Keys.XGB_CACHE, None)
    assert isinstance(cache, dict), "xgb_cache should be a dict after auto-fit"
    assert cache.get("model") is not None, "xgb_cache.model must be set"
    assert int(cache.get("n")) == 4, "xgb_cache.n should equal dataset rows"
