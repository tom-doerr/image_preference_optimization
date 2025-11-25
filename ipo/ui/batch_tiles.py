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
    from .batch_ui import _decode_one, _tile_value_text, _render_good_bad_buttons  # type: ignore

    z_i = cur_batch[i]

    def _tile_visual():
        img = _decode_one(i, lstate, prompt, z_i, steps, guidance_eff)
        v_text = _tile_value_text(st, z_p, z_i, scorer)
        cap = f"Item {i} • {v_text}"
        return img, cap

    img_i, cap_txt = _tile_visual()
    st.image(img_i, caption=cap_txt, width="stretch")

    if best_of:
        if st.button(f"Choose {i}", key=f"choose_{i}", width="stretch"):
            # Delegate to batch_ui's handler via _render_good_bad_buttons semantics
            for j, z_j in enumerate(cur_batch):
                lbl = 1 if j == i else -1
                img_j = img_i if j == i else None
                # Reuse the good/bad path to record/save and replace
                # (batch_ui._render_good_bad_buttons calls internal helpers)
                _render_good_bad_buttons(st, j, z_j, img_j, int(st.session_state.get("cur_batch_nonce", 0)), None, None)
    else:
        btn_cols = getattr(st, "columns", lambda x: [None] * x)(2)
        gcol = btn_cols[0] if btn_cols and len(btn_cols) > 0 else None
        bcol = btn_cols[1] if btn_cols and len(btn_cols) > 1 else None
        nonce = int(st.session_state.get("cur_batch_nonce", 0))
        _render_good_bad_buttons(st, i, z_i, img_i, nonce, gcol, bcol)

