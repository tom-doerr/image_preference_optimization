import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_with_writes


class DummyFuture:
    def result(self):
        return "ok-image"


class TestQueueToastReject(unittest.TestCase):
    def tearDown(self):
        for m in ("queue_ui", "streamlit", "latent_state", "batch_ui", "flux_local"):
            sys.modules.pop(m, None)

    def test_reject_toast(self):
        st, writes = stub_with_writes()
        st.toast = lambda msg: st.sidebar.write(str(msg))
        sys.modules["streamlit"] = st

        import latent_state

        st.session_state.lstate = latent_state.init_latent_state()
        st.session_state.prompt = "toast-reject"
        st.session_state.queue = [
            {"z": np.zeros(st.session_state.lstate.d), "future": DummyFuture(), "label": None}
        ]

        # Minimal batch_ui stub so _curation_add works
        bu = types.ModuleType("batch_ui")
        bu._curation_add = lambda lbl, z, img=None: None
        bu._curation_train_and_next = lambda: None
        sys.modules["batch_ui"] = bu

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        sys.modules["flux_local"] = fl

        import queue_ui

        queue_ui._queue_label(0, -1, img="ok-image")
        out = "\n".join(writes)
        self.assertIn("Rejected (-1)", out)


if __name__ == "__main__":
    unittest.main()
