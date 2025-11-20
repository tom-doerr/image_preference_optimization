import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestFragmentsToggle(unittest.TestCase):
    def _setup(self):
        st = stub_basic()
        st.session_state.prompt = 'frag-toggle'
        st.session_state.batch_size = 2
        # Minimal flux stub (avoid real decode)
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        sys.modules['flux_local'] = fl
        sys.modules['streamlit'] = st
        from latent_state import init_latent_state
        import batch_ui as bu
        st.session_state.lstate = init_latent_state()
        bu._curation_init_batch()
        # Minimal decode settings used by batch renderer
        st.session_state.steps = 6
        st.session_state.guidance_eff = 0.0
        return st, bu

    def test_fragments_used_when_enabled(self):
        st, bu = self._setup()
        calls = {'n': 0}

        def frag(fn):
            def wrapped(*a, **k):
                calls['n'] += 1
                return fn(*a, **k)
            return wrapped

        st.fragment = frag
        st.session_state.use_fragments = True
        bu._render_batch_ui()
        self.assertGreater(calls['n'], 0)

    def test_fragments_not_used_when_disabled(self):
        st, bu = self._setup()
        calls = {'n': 0}

        def frag(fn):
            def wrapped(*a, **k):
                calls['n'] += 1
                return fn(*a, **k)
            return wrapped

        st.fragment = frag
        st.session_state.use_fragments = False
        bu._render_batch_ui()
        self.assertEqual(calls['n'], 0)


if __name__ == '__main__':
    unittest.main()
