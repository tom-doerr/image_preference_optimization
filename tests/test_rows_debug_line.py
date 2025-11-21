import io
import sys
import types


def _capture(func, *a, **k):
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        func(*a, **k)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_rows_debug_line_prints_disp():
    from tests.helpers.st_streamlit import stub_basic
    import ui_sidebar

    st = stub_basic()
    st.session_state.prompt = "rows-dbg"
    st.session_state.dataset_y = [1, 1, -1]
    st.session_state.lstate = types.SimpleNamespace(d=4)
    sys.modules["streamlit"] = st

    # Stub persistence rows counter
    p = types.ModuleType("persistence")
    p.dataset_rows_for_prompt_dim = lambda prompt, d: 5
    p.dataset_rows_for_prompt = lambda prompt: 5
    p.dataset_stats_for_prompt = lambda prompt: {"pos": 0, "neg": 0, "d": 0}
    sys.modules["persistence"] = p

    out = _capture(ui_sidebar.render_rows_and_last_action, st, st.session_state.prompt, st.session_state.lstate)
    assert "[rows] live=3 disk=5 disp=5" in out

