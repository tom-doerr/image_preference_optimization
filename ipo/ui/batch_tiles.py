from __future__ import annotations

from typing import Any


def render_batch_tile_body(
    i: int,
    render_nonce: int,
    lstate: Any,
    prompt: str,
    steps: int,
    guidance_eff: float,
    best_of: bool,
    scorer,
    fut_running: bool,
    cur_batch,
    z_p,
) -> None:
    """Render one batch tile (image + caption + buttons).

    Extracted from batch_ui to reduce file complexity; behavior/strings unchanged.
    """
    import streamlit as st

    # Late import to avoid circular import at module load time
    from .batch_ui import _decode_one, _render_good_bad_buttons, _tile_value_text  # type: ignore

    z_i = cur_batch[i]

    def _tile_visual():
        img = _decode_one(i, lstate, prompt, z_i, steps, guidance_eff)
        v_text = _tile_value_text(st, z_p, z_i, scorer)
        cap = f"Item {i} • {v_text}"
        return img, cap

    img_i, cap_txt = _tile_visual()
    st.image(img_i, caption=cap_txt, width="stretch")

    # Best-of path removed in app; always render Good/Bad buttons
    btn_cols = getattr(st, "columns", lambda x: [None] * x)(2)
    gcol = btn_cols[0] if btn_cols and len(btn_cols) > 0 else None
    bcol = btn_cols[1] if btn_cols and len(btn_cols) > 1 else None
    nonce = int(st.session_state.get("cur_batch_nonce", 0))
    _render_good_bad_buttons(st, i, z_i, img_i, nonce, gcol, bcol)


def render_row_inline(st, idxs, lstate, prompt: str, steps: int, guidance_eff: float, best_of: bool, scorer, cur_batch, z_p) -> None:
    """Minimal inline row renderer used as a fallback in batch UI.

    It renders each tile via render_batch_tile_body inside column containers.
    """
    try:
        rn = int(st.session_state.get("render_nonce", 0))
    except Exception:
        rn = 0
    cols = getattr(st, "columns", lambda x: [None] * x)(len(idxs))
    for col, i in zip(cols, idxs):
        if col is not None:
            with col:
                render_batch_tile_body(i, rn, lstate, prompt, steps, guidance_eff, best_of, scorer, False, cur_batch, z_p)
        else:
            render_batch_tile_body(i, rn, lstate, prompt, steps, guidance_eff, best_of, scorer, False, cur_batch, z_p)
