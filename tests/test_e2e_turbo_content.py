import os
import unittest


class TestE2ETurboContent(unittest.TestCase):
    def setUp(self):
        if os.getenv("E2E_TURBO") != "1":
            self.skipTest("Set E2E_TURBO=1 to run real SD‑Turbo content test")
        try:
            import torch  # type: ignore
        except Exception as e:
            self.skipTest(f"torch not installed: {e}")
        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            self.skipTest("CUDA not available")

    def _to_np(self, img):
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore

            if isinstance(img, Image.Image):
                return np.asarray(img)
        except Exception:
            pass
        # Fall back to attributes exposed by diffusers outputs
        if hasattr(img, "numpy"):
            return img.numpy()
        if hasattr(img, "to"):  # torch tensor
            return img.detach().cpu().numpy()
        # If nothing matches, wrap in list for len checks
        return img

    def test_images_are_nontrivial_and_different(self):
        import numpy as np  # type: ignore
        from latent_opt import init_latent_state, z_to_latents
        from flux_local import set_model, generate_flux_image_latents

        model_id = os.getenv("E2E_TURBO_MODEL", "stabilityai/sd-turbo")
        set_model(model_id)

        st = init_latent_state(width=512, height=512, seed=0)
        # Two opposite z’s for maximal difference
        z = st.rng.standard_normal(st.d)
        za, zb = z, -z
        la = z_to_latents(st, za)
        lb = z_to_latents(st, zb)

        img_a = generate_flux_image_latents(
            "neon punk city, women with short hair, standing in the rain",
            latents=la,
            width=512,
            height=512,
            steps=6,
            guidance=2.5,
        )
        img_b = generate_flux_image_latents(
            "neon punk city, women with short hair, standing in the rain",
            latents=lb,
            width=512,
            height=512,
            steps=6,
            guidance=2.5,
        )

        a = self._to_np(img_a)
        b = self._to_np(img_b)
        # Ensure both are image-like arrays
        self.assertTrue(hasattr(a, "shape") and a.size > 1000)
        self.assertTrue(hasattr(b, "shape") and b.size > 1000)
        # Non-trivial content: per-image std above a tiny threshold
        self.assertGreater(float(np.asarray(a).std()), 5.0)
        self.assertGreater(float(np.asarray(b).std()), 5.0)
        # Distinct outputs: mean absolute difference above a small value
        mad = float(
            np.mean(
                np.abs(
                    np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
                )
            )
        )
        self.assertGreater(mad, 2.0, f"images too similar (MAD={mad:.3f})")


if __name__ == "__main__":
    unittest.main()
