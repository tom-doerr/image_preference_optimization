import os
import unittest
import numpy as np


class TestE2EPairContentGPU(unittest.TestCase):
    def setUp(self):
        if os.getenv('E2E_GPU') != '1' and os.getenv('SMOKE_GPU') != '1':
            self.skipTest('Set E2E_GPU=1 (or SMOKE_GPU=1) to run GPU e2e content test')
        try:
            import torch  # type: ignore
        except Exception as e:
            self.skipTest(f'torch not installed: {e}')
        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            self.skipTest('CUDA not available')

    def _to_np(self, img):
        try:
            from PIL import Image  # type: ignore
            if isinstance(img, Image.Image):
                return np.asarray(img)
        except Exception:
            pass
        if hasattr(img, 'numpy'):
            return img.numpy()
        if hasattr(img, 'to'):
            return img.detach().cpu().numpy()
        return np.asarray(img)

    def test_pair_images_have_content(self):
        from latent_opt import init_latent_state, z_to_latents, ProposerOpts, propose_next_pair
        from flux_local import set_model, generate_flux_image_latents

        model_id = os.getenv('SMOKE_MODEL', os.getenv('FLUX_LOCAL_MODEL', 'stabilityai/sd-turbo'))
        set_model(model_id)

        st = init_latent_state(width=512, height=512, seed=0)
        prompt = 'neon punk city, women with short hair, standing in the rain'
        za, zb = propose_next_pair(st, prompt, opts=ProposerOpts(mode='line', trust_r=st.sigma))
        la = z_to_latents(st, za)
        lb = z_to_latents(st, zb)

        A = self._to_np(generate_flux_image_latents(prompt, latents=la, width=512, height=512, steps=6, guidance=2.5))
        B = self._to_np(generate_flux_image_latents(prompt, latents=lb, width=512, height=512, steps=6, guidance=2.5))
        self.assertGreater(float(np.asarray(A).std()), 5.0)
        self.assertGreater(float(np.asarray(B).std()), 5.0)
        self.assertGreater(float(np.abs(A.astype(float) - B.astype(float)).std()), 1.0)


if __name__ == '__main__':
    unittest.main()

