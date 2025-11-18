import sys
import types
import unittest


class TestPairUIGenerateDefaults(unittest.TestCase):
    def test_generate_initializes_state_and_images(self):
        # Stub streamlit
        st = types.ModuleType('streamlit')
        class SS(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
        st.session_state = SS()
        st.image = lambda *a, **k: None
        sys.modules['streamlit'] = st
        # Stub flux_local
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        sys.modules['flux_local'] = fl

        import pair_ui
        pair_ui.generate_pair()
        self.assertIsNotNone(st.session_state.get('lstate'))
        imgs = st.session_state.get('images')
        # pair_ui sets images via latents decode path
        self.assertEqual(imgs, ('ok-image', 'ok-image'))


if __name__ == '__main__':
    unittest.main()

