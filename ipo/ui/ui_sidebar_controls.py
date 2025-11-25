from __future__ import annotations

from typing import Any


def render_rows_and_last_action(st: Any, base_prompt: str, lstate: Any | None = None) -> None:
    from ipo.ui.ui_sidebar import (
        _emit_dim_mismatch,
        _emit_last_action_recent,
        _rows_refresh_tick,
        _render_rows_counters,
        _debug_saves_section,
    )
    st.sidebar.subheader("Training data & scores")
    _emit_dim_mismatch(st)
    _emit_last_action_recent(st)
    _rows_refresh_tick(st)
    _render_rows_counters(st, lstate, base_prompt)
    _debug_saves_section(st, base_prompt, lstate)


def render_model_decode_settings(st: Any, lstate: Any):
    from ipo.ui.ui_sidebar import (
        _build_size_controls,
        safe_write,
        Keys as K,
    )
    from ipo.infra.util import safe_set

    st.sidebar.header("Model & decode settings")
    try:
        st.session_state.pop(K.USE_FRAGMENTS, None)
    except Exception:
        pass
    try:
        width, height, steps, guidance, apply_clicked = _build_size_controls(st, lstate)
    except Exception:
        width = getattr(lstate, "width", 512)
        height = getattr(lstate, "height", 512)
        steps = 6
        guidance = 0.0
        apply_clicked = False
    selected_model = "stabilityai/sd-turbo"
    try:
        eff_guidance = 0.0 if isinstance(selected_model, str) and "turbo" in selected_model else float(guidance)
        safe_set(st.session_state, K.GUIDANCE_EFF, eff_guidance)
        safe_write(st, f"Effective guidance: {eff_guidance:.2f}")
    except Exception:
        pass
    return selected_model, int(width), int(height), int(steps), float(guidance), bool(apply_clicked)

