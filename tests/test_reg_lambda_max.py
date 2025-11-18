import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestRegLambdaMax(unittest.TestCase):
    def test_slider_and_input_allow_large_lambda(self):
        st = stub_basic()
        captured = {"slider_max": None, "num_max": None}

        class SB(st.sidebar.__class__):
            @staticmethod
            def slider(label, *a, **k):
                if 'Ridge λ' in label and len(a) >= 2:
                    captured['slider_max'] = float(a[1])
                return k.get('value', 0.0)

        st.sidebar = SB()

        def number_input(label, **k):
            if 'Ridge λ' in label:
                captured['num_max'] = float(k.get('max_value', 0.0))
            return k.get('value', 0.0)

        st.number_input = number_input
        sys.modules['streamlit'] = st

        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **kw: 'ok-image'
        fl.set_model = lambda *a, **kw: None
        sys.modules['flux_local'] = fl

        # Import app to trigger sidebar construction
        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401

        self.assertIsNotNone(captured['slider_max'])
        self.assertIsNotNone(captured['num_max'])
        self.assertGreaterEqual(captured['slider_max'], 1e5)
        self.assertGreaterEqual(captured['num_max'], 1e5)
        # Clean up to avoid polluting subsequent tests
        if 'app' in sys.modules:
            del sys.modules['app']


if __name__ == '__main__':
    unittest.main()
