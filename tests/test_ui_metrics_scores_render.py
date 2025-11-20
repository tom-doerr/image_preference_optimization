import sys
import types
import unittest
import numpy as np


class TestUIMetricsScoresRender(unittest.TestCase):
    def test_render_iter_step_scores_writes(self):
        # Stub streamlit with sidebar.write capture
        writes = []
        st = types.ModuleType('streamlit')
        class SS(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
        st.session_state = SS()
        class Sidebar:
            @staticmethod
            def write(s):
                writes.append(str(s))
        st.sidebar = Sidebar()
        sys.modules['streamlit'] = st
        # lstate with non-zero w and sigma
        d = 8
        lstate = types.SimpleNamespace(d=d, w=np.ones(d, dtype=float), sigma=1.0)
        # Session dataset
        X = np.zeros((2, d), dtype=float)
        X[0, 0] = 1.0
        X[1, 1] = -1.0
        y = np.array([+1.0, -1.0], dtype=float)
        st.session_state['dataset_X'] = X
        st.session_state['dataset_y'] = y
        from ui_metrics import render_iter_step_scores
        render_iter_step_scores(st, lstate, prompt='p', vm_choice='DistanceHill', iter_steps=2, iter_eta=None, trust_r=None)
        self.assertTrue(any('Step scores:' in w for w in writes))


if __name__ == '__main__':
    unittest.main()
