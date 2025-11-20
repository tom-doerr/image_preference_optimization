import os
import unittest


class TestSmokePairNotConstant(unittest.TestCase):
    def setUp(self):
        if os.getenv('SMOKE_GPU') != '1':
            self.skipTest('Set SMOKE_GPU=1 to run GPU pair smoke test')
        try:
            import torch  # type: ignore
        except Exception as e:
            self.skipTest(f'torch not installed: {e}')
        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            self.skipTest('CUDA not available')

    def _to_np(self, img):
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
            if isinstance(img, Image.Image):
                return np.asarray(img)
        except Exception:
            pass
        if hasattr(img, 'numpy'):
            return img.numpy()
        if hasattr(img, 'to'):
            return img.detach().cpu().numpy()
        return img

    def test_pair_images_have_content(self):
        import numpy as np  # type: ignore
        from latent_opt import init_latent_state, z_to_latents, ProposerOpts, propose_next_pair
        from flux_local import set_model, generate_flux_image_latents

        prompt = 'neon punk city, women with short hair, standing in the rain'
        model_id = os.getenv('SMOKE_MODEL', 'stabilityai/sd-turbo')
        set_model(model_id)

        st = init_latent_state(width=512, height=512, seed=0)
        # Get a realistic pair around the prompt anchor
        z_a, z_b = propose_next_pair(st, prompt, opts=ProposerOpts(mode='line', trust_r=st.sigma))
        lat_a = z_to_latents(st, z_a)
        lat_b = z_to_latents(st, z_b)

        A = self._to_np(generate_flux_image_latents(prompt, latents=lat_a, width=512, height=512, steps=6, guidance=2.5))
        B = self._to_np(generate_flux_image_latents(prompt, latents=lat_b, width=512, height=512, steps=6, guidance=2.5))
        self.assertGreater(float(np.asarray(A).std()), 5.0)
        self.assertGreater(float(np.asarray(B).std()), 5.0)
        # Pair should not be nearly identical
        self.assertGreater(float(np.abs(np.asarray(A, dtype=float) - np.asarray(B, dtype=float)).std()), 1.0)


if __name__ == '__main__':
    unittest.main()
