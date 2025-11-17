import os
import unittest


class TestSmokeFirstDecodeContent(unittest.TestCase):
    def setUp(self):
        if os.getenv('SMOKE_GPU') != '1':
            self.skipTest('Set SMOKE_GPU=1 to run GPU content smoke test')
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

    def test_first_prompt_latent_decode_has_content(self):
        import numpy as np  # type: ignore
        from latent_opt import init_latent_state, z_from_prompt, z_to_latents
        from flux_local import set_model, generate_flux_image_latents

        model_id = os.getenv('SMOKE_MODEL', 'stabilityai/sd-turbo')
        set_model(model_id)

        st = init_latent_state(width=512, height=512, seed=0)
        z = z_from_prompt(st, 'neon punk city, women with short hair, standing in the rain')
        lat = z_to_latents(st, z)
        img = generate_flux_image_latents(
            'neon punk city, women with short hair, standing in the rain',
            latents=lat, width=512, height=512, steps=6, guidance=2.5,
        )
        A = self._to_np(img)
        self.assertTrue(hasattr(A, 'shape') and getattr(A, 'size', 0) > 1000)
        self.assertGreater(float(np.asarray(A).std()), 5.0)


if __name__ == '__main__':
    unittest.main()

