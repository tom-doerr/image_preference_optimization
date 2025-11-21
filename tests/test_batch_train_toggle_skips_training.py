import sys
import types
import numpy as np


def test_curation_train_skipped_when_toggle_off():
    from tests.helpers.st_streamlit import stub_basic
    st = stub_basic()
    st.session_state.prompt = "skip-train"
    st.session_state.lstate = types.SimpleNamespace(
        d=4, w=np.zeros(4), sigma=1.0, rng=np.random.default_rng(0), width=64, height=64
    )
    st.session_state.train_on_new_data = False
    st.session_state.cur_batch = [np.zeros(4), np.ones(4)]
    st.session_state.cur_labels = [None, None]
    sys.modules["streamlit"] = st

    # Provide dataset
    p = types.ModuleType("persistence")
    p.get_dataset_for_prompt_or_session = lambda prompt, ss: (
        np.array([[1.0, 0.0, 0.0, 0.0]]),
        np.array([1.0]),
    )
    sys.modules["persistence"] = p

    # Import and run
    import batch_ui

    # Spy: value_model should not be imported/called
    sys.modules.pop("value_model", None)
    batch_ui._curation_train_and_next()
    # No status should be written, dataset unchanged aside from new batch
    assert st.session_state.get("xgb_train_status") is None
