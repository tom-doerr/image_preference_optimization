def test_emit_train_results_no_lstate_does_not_crash_and_writes_line():
    from tests.helpers.st_streamlit import stub_with_writes
    st, writes = stub_with_writes()

    # Do not set st.session_state.lstate on purpose (regression guard for NameError)
    from ipo.ui.ui_sidebar import _emit_train_results

    lines = ["Train score: n/a"]
    _emit_train_results(st, lines, sidebar_only=True)

    # At least the provided line should be written to the sidebar
    assert any("Train score:" in w for w in writes)

