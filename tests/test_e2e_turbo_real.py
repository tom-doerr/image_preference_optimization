import os
import unittest


class TestE2ETurboReal(unittest.TestCase):
    def setUp(self):
        if os.getenv("E2E_TURBO") != "1":
            self.skipTest("Set E2E_TURBO=1 to run real SD‑Turbo generation test")
        try:
            import torch  # type: ignore
        except Exception as e:
            self.skipTest(f"torch not installed: {e}")
        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            self.skipTest("CUDA not available")

    def test_real_generation_sd_turbo(self):
        from latent_opt import (
            init_latent_state,
            propose_latent_pair_ridge,
            z_to_latents,
        )
        from flux_local import set_model, generate_flux_image_latents

        model_id = os.getenv("E2E_TURBO_MODEL", "stabilityai/sd-turbo")
        set_model(model_id)

        st = init_latent_state(width=512, height=512, seed=0)
        za, _ = propose_latent_pair_ridge(st, alpha=1.0, beta=1.0)
        lat = z_to_latents(st, za)

        img = generate_flux_image_latents(
            "latex, neon punk city, women with short hair, standing in the rain",
            latents=lat,
            width=512,
            height=512,
            steps=6,
            guidance=2.5,
        )
        ok = False
        try:
            from PIL import Image  # type: ignore

            ok = ok or isinstance(img, Image.Image)
        except Exception:
            pass
        ok = ok or hasattr(img, "size") or hasattr(img, "shape")
        self.assertTrue(ok, "SD‑Turbo generator did not return an image-like object")


if __name__ == "__main__":
    unittest.main()
