import sys
import types
import unittest
import numpy as np


class TestImageServerToggle(unittest.TestCase):
    def test_flux_local_uses_server_when_enabled(self):
        # Stub image_server module
        srv = types.ModuleType("image_server")
        srv.generate_image = lambda *a, **k: "ok-image"
        srv.generate_image_latents = lambda *a, **k: "ok-image"
        sys.modules["image_server"] = srv

        import flux_local as fl

        fl.use_image_server(True, url="http://localhost:9999")
        img = fl.generate_flux_image("p", width=64, height=64, steps=1, guidance=0.0)
        self.assertEqual(img, "ok-image")
        lat = np.zeros((1, 4, 8, 8), dtype=np.float32)
        img2 = fl.generate_flux_image_latents(
            "p", lat, width=64, height=64, steps=1, guidance=0.0
        )
        self.assertEqual(img2, "ok-image")


if __name__ == "__main__":
    unittest.main()
