import io
import numpy as np
import hashlib
import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_with_writes
from persistence import state_path_for_prompt
from latent_opt import init_latent_state, save_state


class TestSidebarPromptHash(unittest.TestCase):
    def test_prompt_hash_line_present(self):
        prompt = "meta prompt hash"
        path = state_path_for_prompt(prompt)
        st0 = init_latent_state(d=4, seed=0)
        save_state(st0, path)
        # add minimal meta so the block renders
        with np.load(path) as data:
            items = {k: data[k] for k in data.files}
        items["created_at"] = np.array("2025-11-13T00:00:00+00:00")
        items["app_version"] = np.array("0.1.0")
        buf = io.BytesIO()
        np.savez_compressed(buf, **items)
        with open(path, "wb") as f:
            f.write(buf.getvalue())

        if "app" in sys.modules:
            del sys.modules["app"]
        st, writes = stub_with_writes()
        st.text_input = lambda *_, value="": prompt
        sys.modules["streamlit"] = st
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.generate_flux_image = lambda *a, **kw: "ok-image-text"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl
        import app  # noqa: F401

        joined = "\n".join(writes)
        self.assertIn("prompt_hash:", joined)
        h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        self.assertIn(h, joined)


if __name__ == "__main__":
    unittest.main()
