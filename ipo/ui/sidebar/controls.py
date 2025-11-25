from __future__ import annotations

from typing import Any


def _render_rows_counters(st: Any, lstate: Any | None, base_prompt: str) -> None:
    try:
        from latent_opt import state_summary  # type: ignore
        from ipo.ui.ui import sidebar_metric, sidebar_metric_rows
        disp_plain = st.session_state.get('ROWS_DISPLAY', '0')
        sidebar_metric("Dataset rows", disp_plain)
        sidebar_metric("Rows (disk)", int(disp_plain or 0))
        if lstate is not None:
            info = state_summary(lstate)
            sidebar_metric_rows([("Pairs:", info.get("pairs_logged", 0)), ("Choices:", info.get("choices_logged", 0))], per_row=2)
    except Exception:
        pass


def _debug_saves_section(st: Any, base_prompt: str, lstate: Any | None) -> None:
    """Debug (saves) removed: no-op to keep call sites stable."""
    return


def render_rows_and_last_action(st: Any, base_prompt: str, lstate: Any | None = None) -> None:
    from ipo.ui.sidebar.misc import emit_dim_mismatch as _emit_dim_mismatch, emit_last_action_recent as _emit_last_action_recent, rows_refresh_tick as _rows_refresh_tick
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
