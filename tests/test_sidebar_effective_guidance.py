import types
import unittest

from constants import Keys


class TestSidebarEffectiveGuidance(unittest.TestCase):
    def test_effective_guidance_clamped_for_turbo(self):
        from tests.helpers.st_streamlit import stub_with_writes
        import ui_sidebar as ui

        st, writes = stub_with_writes()
        lstate = types.SimpleNamespace(width=512, height=512)
        selected_model, _, _, _, _, _ = ui.render_model_decode_settings(st, lstate)
        self.assertEqual(selected_model, "stabilityai/sd-turbo")
        # Guidance should be clamped to 0 and written to the sidebar
        self.assertIn("Effective guidance: 0.00", writes)
        self.assertEqual(st.session_state.get(Keys.GUIDANCE_EFF), 0.0)


if __name__ == "__main__":
    unittest.main()
