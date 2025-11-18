"""
Test-only shim: runs the Streamlit app with a stubbed `flux_local` so Playwright
can exercise the UI without GPU/diffusers. Keep it minimal; not used in prod.
Run: streamlit run scripts/app_stubbed.py --server.headless true --server.port 8597
"""
import sys
import types
import numpy as np


def _fake_img(w, h, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    R = (X * 255).astype(np.uint8)
    G = (Y * 255).astype(np.uint8)
    B = (rng.random((h, w)) * 255).astype(np.uint8)
    return np.stack([R, G, B], axis=-1)


fl = types.ModuleType('flux_local')

_LAST = {
    'event': 'stub_init',
    'model_id': 'stub/model',
    'width': 512,
    'height': 512,
    'latents_std': 1.0,
    'latents_mean': 0.0,
    'latents_shape': (1, 4, 64, 64),
}


def set_model(model_id):  # noqa: D401
    _LAST.update({'event': 'set_model', 'model_id': model_id})


def generate_flux_image(prompt, seed=None, width=512, height=512, steps=6, guidance=3.5):
    _LAST.update({'event': 'text_call', 'width': width, 'height': height})
    return _fake_img(width, height, 1)


def generate_flux_image_latents(prompt, latents, width=512, height=512, steps=6, guidance=3.5):
    _LAST.update({'event': 'latents_call', 'width': width, 'height': height})
    return _fake_img(width, height, 2)


def get_last_call():
    return dict(_LAST)


fl.set_model = set_model
fl.generate_flux_image = generate_flux_image
fl.generate_flux_image_latents = generate_flux_image_latents
fl.get_last_call = get_last_call
sys.modules['flux_local'] = fl

# Import the real app
import app  # noqa: E402,F401

