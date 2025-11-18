import types
import unittest


class _DummyPipe:
    def __init__(self, ctr):
        self.ctr = ctr
    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None):
        self.ctr['n'] = self.ctr.get('n', 0) + 1
        # Return two dummy arrays; callers don't inspect shapes here
        import numpy as np
        return np.zeros((1, 77, 768), dtype=float), np.zeros((1, 77, 768), dtype=float)


class TestPromptEncodeCache(unittest.TestCase):
    def test_encode_called_once_per_prompt(self):
        import flux_local as fl
        # Monkeypatch the pipeline and internals
        ctr = {}
        fl.PIPE = _DummyPipe(ctr)
        fl.CURRENT_MODEL_ID = 'dummy-model'
        fl.PROMPT_CACHE.clear()
        fl._ensure_pipe = lambda mid=None: None
        # Avoid real _run_pipe
        fl._run_pipe = lambda **kw: 'ok'

        img1 = fl.generate_flux_image('hello world', steps=1, guidance=2.5)
        img2 = fl.generate_flux_image('hello world', steps=1, guidance=2.5)
        self.assertEqual(ctr.get('n', 0), 1)
        self.assertEqual(img1, 'ok')
        self.assertEqual(img2, 'ok')


if __name__ == '__main__':
    unittest.main()
