import os
import sys
import types
import unittest
import importlib
import numpy as np


class TestPairNotConstantStub(unittest.TestCase):
    def setUp(self):
        for m in ("torch", "diffusers", "flux_local"):
            sys.modules.pop(m, None)
        os.environ["FLUX_LOCAL_MODEL"] = "dummy/model"

    def _stub_stack(self):
        # torch stub with cuda and Tensor
        torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                return True

        torch.cuda = _Cuda()
        torch.float16 = object()

        class Tensor:
            def __init__(self, arr):
                self.arr = np.asarray(arr, dtype=np.float32)

            def to(self, device=None, dtype=None):
                return self

            def std(self):
                v = float(self.arr.std())
                return types.SimpleNamespace(item=lambda: v)

            def numel(self):
                return int(self.arr.size)

        torch.tensor = lambda x, dtype=None, device=None: Tensor(x)
        sys.modules["torch"] = torch

        # diffusers stub: convert latents to a pseudo-image with visible variation
        recorded = {}
        diffusers = types.ModuleType("diffusers")

        class _Out:
            def __init__(self, img):
                self.images = [img]

        class _Sched:
            init_noise_sigma = 1.0

        class _Pipe:
            def __init__(self):
                self.scheduler = _Sched()

            def to(self, device):
                return self

            def __call__(self, **kwargs):
                recorded.update(kwargs)
                L = kwargs["latents"].arr  # (1,4,h,w)
                r = L[0, 0]
                g = L[0, 1]
                b = (L[0, 2] + L[0, 3]) * 0.5
                img = np.clip(np.stack([r, g, b], axis=-1) * 32 + 128, 0, 255).astype(
                    np.uint8
                )
                return _Out(img)

        class _DP:
            @staticmethod
            def from_pretrained(mid, torch_dtype=None, **kwargs):
                return _Pipe()

        diffusers.DiffusionPipeline = _DP
        sys.modules["diffusers"] = diffusers
        return recorded

    def test_pair_images_have_variance_stub(self):
        self._stub_stack()
        import flux_local

        importlib.reload(flux_local)
        from latent_opt import (
            init_latent_state,
            z_to_latents,
            ProposerOpts,
            propose_next_pair,
        )

        st = init_latent_state(width=320, height=256, seed=0)
        prompt = "stub pair variance"
        za, zb = propose_next_pair(
            st, prompt, opts=ProposerOpts(mode="line", trust_r=st.sigma)
        )
        la = z_to_latents(st, za)
        lb = z_to_latents(st, zb)

        A = flux_local.generate_flux_image_latents(
            prompt, la, width=320, height=256, steps=4, guidance=2.0
        )
        B = flux_local.generate_flux_image_latents(
            prompt, lb, width=320, height=256, steps=4, guidance=2.0
        )
        An = np.asarray(A)
        Bn = np.asarray(B)
        self.assertGreater(float(An.std()), 1.0)
        self.assertGreater(float(Bn.std()), 1.0)
        self.assertGreater(
            float(np.abs(An.astype(float) - Bn.astype(float)).std()), 0.5
        )


if __name__ == "__main__":
    unittest.main()
