import types
import unittest


class TestUIControlsFallbacks(unittest.TestCase):
    def test_size_controls_fallback_when_stubs_return_none(self):
        import ui_controls as ui
        # Minimal streamlit stub
        st = types.ModuleType('streamlit')
        class SS(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
        st.session_state = SS()
        # Top-level slider/number_input return provided default value
        st.slider = lambda *a, **k: k.get('value')
        st.number_input = lambda *a, **k: k.get('value')
        # Sidebar without slider to trigger fallback; number_input returns None
        class Sidebar:
            @staticmethod
            def number_input(*a, **k):
                return None
        st.sidebar = Sidebar()
        # Minimal latent state
        from latent_state import init_latent_state
        lstate = init_latent_state()
        w, h, steps, guidance, apply_clicked = ui.build_size_controls(st, lstate)
        self.assertEqual(w, lstate.width)
        self.assertEqual(h, lstate.height)
        self.assertEqual(steps, 6)
        self.assertAlmostEqual(guidance, 3.5)
        self.assertFalse(apply_clicked)

    def test_batch_queue_controls_work_without_sidebar_slider(self):
        import ui_controls as ui
        st = types.ModuleModule('streamlit') if hasattr(types, 'ModuleModule') else types.ModuleType('streamlit')
        class Sidebar:
            pass  # no slider on sidebar
        st.sidebar = Sidebar()
        # Top-level slider provides value
        st.slider = lambda *a, **k: k.get('value')
        # Batch default now 25 (see constants.DEFAULT_BATCH_SIZE)
        self.assertEqual(ui.build_batch_controls(st, expanded=False), 25)
        self.assertEqual(ui.build_queue_controls(st, expanded=False), 6)


if __name__ == '__main__':
    unittest.main()
