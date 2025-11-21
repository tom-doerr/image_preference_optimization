import sys
import types


class Expander:
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.fn()
        return self

    def __exit__(self, *a):
        return False


def test_ui_details_shows_xgb_available_line():
    from tests.helpers.st_streamlit import stub_with_writes
    st, writes = stub_with_writes()
    st.sidebar.expander = lambda *a, **k: Expander(lambda: None)
    st.session_state.prompt = "xgb-available"
    st.session_state.lstate = types.SimpleNamespace(d=4)
    sys.modules["streamlit"] = st

    # Stub persistence rows/stats
    p = types.ModuleType("persistence")
    p.dataset_rows_for_prompt = lambda prompt: 0
    p.dataset_rows_for_prompt_dim = lambda prompt, d: 0
    p.dataset_stats_for_prompt = lambda prompt: {"pos": 0, "neg": 0, "d": 4}
    sys.modules["persistence"] = p

    # Stub xgboost import to force available=yes
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "xgboost":
            return types.SimpleNamespace(__name__="xgboost")
        return real_import(name, *a, **k)

    builtins.__import__ = _fake_import
    try:
        import ui_sidebar

        ui_sidebar._sidebar_value_model_block(
            st, st.session_state.lstate, st.session_state.prompt, "XGBoost", 1.0
        )
    finally:
        builtins.__import__ = real_import

    assert any("XGBoost available: yes" in w for w in writes)

