import os
import unittest


class TestE2EGpuGenerate(unittest.TestCase):
    def setUp(self):
        if os.getenv("E2E_GENERATE") != "1":
            self.skipTest("Set E2E_GENERATE=1 to run real generation tests")
        try:
            import torch  # type: ignore
        except Exception as e:
            self.skipTest(f"torch not installed: {e}")
        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            self.skipTest("CUDA not available")

    def test_single_image_generation(self):
        from latent_opt import init_latent_state, z_to_latents
        from flux_local import set_model, generate_flux_image_latents

        model_id = os.getenv("E2E_FLUX_MODEL") or os.getenv("FLUX_LOCAL_MODEL")
        if not model_id:
            self.skipTest(
                "Set E2E_FLUX_MODEL or FLUX_LOCAL_MODEL to a local/HF model id"
            )

        set_model(model_id)

        st = init_latent_state(width=512, height=384, seed=0)
        from latent_opt import propose_latent_pair_ridge

        za, zb = propose_latent_pair_ridge(st, alpha=1.0, beta=1.0)
        lat = z_to_latents(st, za)

        img = generate_flux_image_latents(
            "neon punk city, women with short hair, standing in the rain",
            latents=lat,
            width=512,
            height=384,
            steps=8,
            guidance=3.5,
        )
        # Accept PIL.Image or array-like with size/shape
        ok = False
        try:
            from PIL import Image  # type: ignore

            ok = ok or isinstance(img, Image.Image)
        except Exception:
            pass
        ok = ok or hasattr(img, "size") or hasattr(img, "shape")
        self.assertTrue(ok, "Generator did not return an image-like object")


if __name__ == "__main__":
    unittest.main()
