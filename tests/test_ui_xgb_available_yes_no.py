import sys
import types

from tests.helpers.st_streamlit import stub_with_writes


def _call_details_block():
    # Prepare a minimal streamlit stub with captured writes
    st, writes = stub_with_writes()
    st.session_state.prompt = "xgb-avail"
    st.session_state.lstate = types.SimpleNamespace(d=4)
    sys.modules["streamlit"] = st
    # Call the details helper directly
    from ipo.ui.ui_sidebar import _vm_details_xgb
    _vm_details_xgb(st, {})
    return writes


def test_xgb_available_yes_when_module_importable():
    # Insert a dummy xgboost module to simulate availability
    sys.modules["xgboost"] = types.SimpleNamespace(__name__="xgboost")
    try:
        writes = _call_details_block()
        assert any("XGBoost available: yes" in w for w in writes)
    finally:
        sys.modules.pop("xgboost", None)


def test_xgb_available_no_when_module_missing():
    # Ensure no xgboost module is present
    sys.modules.pop("xgboost", None)
    writes = _call_details_block()
    assert any("XGBoost available: no" in w for w in writes)

