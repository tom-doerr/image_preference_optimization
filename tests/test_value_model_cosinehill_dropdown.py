import sys
import types
import unittest
import numpy as np
from tests.helpers.st_streamlit import stub_with_writes


class TestValueModelCosineHill(unittest.TestCase):
    def test_dropdown_sets_cosinehill_and_scores_present(self):
        st, writes = stub_with_writes()
        class SB(st.sidebar.__class__):
            @staticmethod
            def selectbox(label, options, index=0):
                if 'Value model' in label:
                    return 'CosineHill'
                if 'Generation mode' in label:
                    return 'Batch curation'
                return options[index]
        st.sidebar = SB()
        st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")
        st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image = lambda *a, **kw: 'ok-text'
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        if 'app' in sys.modules:
            del sys.modules['app']
        import app
        # Inject small dataset
        X = np.zeros((2, app.st.session_state.lstate.d)); X[0,0]=1; X[1,0]=-1
        y = np.array([1.0, -1.0], dtype=float)
        app.st.session_state.dataset_X = X
        app.st.session_state.dataset_y = y
        if 'app' in sys.modules:
            del sys.modules['app']
        import app as app2  # noqa: F401

        out = "\n".join(writes)
        self.assertIn('Value model: CosineHill', out)
        self.assertIn('V(left):', out)
        self.assertIn('V(right):', out)


if __name__ == '__main__':
    unittest.main()

