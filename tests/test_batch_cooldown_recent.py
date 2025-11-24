def test_cooldown_recent_true_then_false():
    from tests.helpers.st_streamlit import stub_basic
    from ipo.infra.constants import Keys
    from ipo.ui.batch_ui import _cooldown_recent
    from datetime import datetime, timedelta, timezone

    st = stub_basic()

    # Recent timestamp within the window -> True
    st.session_state["min_train_interval_s"] = 60
    st.session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    assert _cooldown_recent(st) is True

    # Older timestamp -> False
    st.session_state[Keys.LAST_TRAIN_AT] = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat(timespec="seconds")
    assert _cooldown_recent(st) is False

