import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_basic


class TestPairProposerToggle(unittest.TestCase):
    def test_toggle_cosinehill_logs_pair(self):
        st = stub_basic()
        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, index=0):
                if 'Value model' in label:
                    return 'CosineHill'
                if 'Generation mode' in label:
                    return 'Batch curation'
                return options[index]
        st.sidebar = SB()
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        # Value model drives proposer; set CosineHill
        st.session_state['vm_choice'] = 'CosineHill'
        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        # Inject a tiny dataset so cosinehill proposer is used
        X = np.zeros((2, app.st.session_state.lstate.d)); X[0,0]=1; X[1,0]=-1
        y = np.array([1.0, -1.0], dtype=float)
        app.st.session_state.dataset_X = X
        app.st.session_state.dataset_y = y
        # Re-import to trigger _apply_state and log
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401
        log = app.st.session_state.get('pair_log') or []
        self.assertTrue(log and log[-1].get('proposer') == 'CosineHill')


if __name__ == '__main__':
    unittest.main()
