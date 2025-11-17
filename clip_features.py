import numpy as np


_clip_loaded = False
_clip_model = None
_clip_preprocess = None
_clip_device = None


def _load_clip():
    global _clip_loaded, _clip_model, _clip_preprocess, _clip_device
    if _clip_loaded:
        return
    import torch  # type: ignore
    import open_clip  # type: ignore
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    _clip_model = model
    _clip_preprocess = preprocess
    _clip_device = device
    _clip_loaded = True


def image_embedding(img) -> np.ndarray:
    """Return L2-normalized CLIP image embedding as np.float32 array.

    Requires: torch, open_clip; image as PIL Image or ndarray HxWxC (uint8).
    """
    from PIL import Image
    import torch  # type: ignore

    _load_clip()
    if not isinstance(img, Image.Image):
        if hasattr(img, 'dtype'):
            img = Image.fromarray(img)
        else:
            raise ValueError('image_embedding expects PIL.Image or ndarray')
    with torch.no_grad():
        x = _clip_preprocess(img).unsqueeze(0).to(_clip_device)
        feats = _clip_model.encode_image(x)  # type: ignore[attr-defined]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).detach().cpu().numpy().astype(np.float32)
