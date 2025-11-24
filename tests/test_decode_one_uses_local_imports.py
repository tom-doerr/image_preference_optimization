def test_decode_one_does_not_raise_when_time_and_imports_local():
    import sys, types, numpy as np
    from types import SimpleNamespace

    # Stub flux_local.generate_flux_image_latents to avoid heavy deps
    fl = types.ModuleType("flux_local")
    fl.generate_flux_image_latents = lambda *a, **k: "ok-image"
    sys.modules["flux_local"] = fl

    # Stub latent_logic.z_to_latents (identity mapping)
    ll = types.ModuleType("latent_logic")
    ll.z_to_latents = lambda *a, **k: (a[1] if len(a) > 1 else a[0])
    sys.modules["latent_logic"] = ll

    from ipo.ui.batch_ui import _decode_one

    lstate = SimpleNamespace(width=64, height=64)
    z = np.zeros(8)
    img = _decode_one(0, lstate, "p", z, steps=1, guidance_eff=0.0)
    assert img == "ok-image"

