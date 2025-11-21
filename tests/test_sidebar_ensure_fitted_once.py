import sys
import types
import unittest
import numpy as np


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def _stub_st():
    st = types.ModuleType("streamlit")
    st.session_state = Session()
    class _Exp:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    st.sidebar = types.SimpleNamespace(
        write=lambda *a, **k: None,
        metric=lambda *a, **k: None,
        subheader=lambda *a, **k: None,
        selectbox=lambda *a, **k: "Batch curation",
        expander=lambda *a, **k: _Exp(),
        download_button=lambda *a, **k: None,
        file_uploader=lambda *a, **k: None,
        text_input=lambda *a, **k: "",
        checkbox=lambda *a, **k: False,
        button=lambda *a, **k: False,
    )
    st.text_input = lambda *a, **k: ""
    st.number_input = lambda *a, **k: 0
    st.slider = lambda *a, **k: 0
    st.columns = lambda n: (types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None),) * max(1, int(n))
    st.write = lambda *a, **k: None
    return st


class TestSidebarEnsureFittedOnce(unittest.TestCase):
    def tearDown(self):
        for m in ("ui_sidebar", "streamlit", "value_model", "persistence"):
            sys.modules.pop(m, None)

    def test_sidebar_uses_ensure_fitted_not_train(self):
        st = _stub_st()
        st.session_state.prompt = "p"
        # Minimal lstate
        lstate = types.SimpleNamespace(d=4, w=np.zeros(4))
        st.session_state.lstate = lstate

        # Stub persistence to return a small dataset
        P = types.ModuleType("persistence")
        P.get_dataset_for_prompt_or_session = lambda *a, **k: (np.ones((2, 4)), np.array([+1.0, -1.0]))
        P.dataset_rows_for_prompt = lambda *a, **k: 2
        P.dataset_stats_for_prompt = lambda *a, **k: {"rows": 2, "pos": 1, "neg": 1, "d": 4, "recent_labels": [1, -1]}
        sys.modules["persistence"] = P

        # Stub value_model.ensure_fitted and train_and_record
        called = {"ensure": 0}

        def _ensure(vm_choice, lstate, X, Y, lam, ss):
            called["ensure"] += 1
            ss["auto_fit_done"] = True

        VM = types.ModuleType("value_model")
        VM.ensure_fitted = _ensure
        VM.train_and_record = lambda *a, **k: (_ for _ in ()).throw(AssertionError("train_and_record should not be called"))
        sys.modules["value_model"] = VM

        # Stub flux_local + persistence_ui deps
        FL = types.ModuleType("flux_local")
        FL.set_model = lambda *a, **k: None
        sys.modules["flux_local"] = FL

        PUI = types.ModuleType("persistence_ui")
        PUI.render_metadata_panel = lambda *a, **k: None
        sys.modules["persistence_ui"] = PUI

        sys.modules["streamlit"] = st

        import ui_sidebar

        ui_sidebar.render_sidebar_tail(
            st,
            lstate,
            st.session_state.prompt,
            state_path="/tmp/x",
            vm_choice="XGBoost",
            iter_steps=1,
            iter_eta=None,
            selected_model="stabilityai/sd-turbo",
            apply_state_cb=lambda *a, **k: None,
            rerun_cb=lambda *a, **k: None,
        )

        self.assertEqual(called["ensure"], 1)


if __name__ == "__main__":
    unittest.main()
