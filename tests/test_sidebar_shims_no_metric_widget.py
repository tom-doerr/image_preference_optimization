import types


def test_ensure_sidebar_shims_provides_metric_without_widget():
    from ipo.ui.ui_sidebar_misc import ensure_sidebar_shims

    # Minimal stub for streamlit-like object
    st = types.SimpleNamespace()
    st.sidebar_writes = []
    st.sidebar = types.SimpleNamespace()

    # Only 'subheader' exists; no 'metric' on purpose
    st.sidebar.subheader = lambda *_a, **_k: None

    # Should not raise and should attach a callable 'metric'
    ensure_sidebar_shims(st)
    assert hasattr(st.sidebar, "metric") and callable(st.sidebar.metric)

    # The shim should write into sidebar_writes (minimal contract)
    st.sidebar.metric("CV (Ridge)", "n/a")
    assert any("CV (Ridge): n/a" in s for s in st.sidebar_writes)

