import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes


class TestLatentCreationPanel(unittest.TestCase):
    def test_explains_latent_construction(self):
        st, writes = stub_with_writes()
        st.text_input = lambda *_, value="": "explain latents"
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image = lambda *a, **k: "ok-text"
        fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        import app  # noqa: F401

        out = "\n".join(writes)
        self.assertIn("Latents shape:", out)
        self.assertIn("z_prompt = RNG(prompt_sha1)", out)
        self.assertIn("Batch sample: z = z_prompt + σ · 0.8 · r", out)


if __name__ == "__main__":
    unittest.main()
