import os
import unittest
import numpy as np


@unittest.skipUnless(
    os.environ.get("SMOKE_GPU") == "1" or os.environ.get("E2E_TURBO") == "1",
    "GPU smoke disabled (set SMOKE_GPU=1 or E2E_TURBO=1)",
)
class TestSmokeNonConstantImage(unittest.TestCase):
    def test_latent_decode_is_not_near_constant(self):
from constants import DEFAULT_PROMPT
        from latent_opt import init_latent_state, z_from_prompt, z_to_latents
        from flux_local import set_model, generate_flux_image_latents

        model = os.environ.get("FLUX_LOCAL_MODEL", "stabilityai/sd-turbo")
        set_model(model)

        st = init_latent_state(width=512, height=512, seed=123)
        z = z_from_prompt(st, DEFAULT_PROMPT)
        lat = z_to_latents(st, z)

        img = generate_flux_image_latents(
            DEFAULT_PROMPT,
            latents=lat,
            width=st.width,
            height=st.height,
            steps=6,
            guidance=2.5,
        )

        arr = np.asarray(img).astype(np.float32)
        # Convert to grayscale for stats
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        else:
            gray = arr.squeeze()

        p10 = np.percentile(gray, 10)
        p90 = np.percentile(gray, 90)
        self.assertGreaterEqual(
            p90 - p10, 5.0, "Dynamic range too small; image nearly constant"
        )

        vals = gray.round().astype(np.int16).ravel()
        if vals.size == 0:
            self.fail("Decoded image empty")
        counts = np.bincount(vals - vals.min())
        max_frac = counts.max() / vals.size
        self.assertLess(max_frac, 0.98, ">98% pixels share (nearly) the same value")


if __name__ == "__main__":
    unittest.main()
