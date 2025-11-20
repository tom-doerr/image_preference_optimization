from __future__ import annotations

from typing import Any, Tuple


def build_controls(st: Any, lstate: Any, base_prompt: str) -> Tuple[
    str, str | None, str, int, int, int, float, float, int, float, bool
]:
    """Assemble sidebar controls and return key values.

    Returns (vm_choice, selected_gen_mode, selected_model, width, height,
    steps, guidance, reg_lambda, iter_steps, iter_eta, async_queue_mode).
    Minimal re-export to keep app.py small.
    """
    from constants import Keys
    from ui_sidebar_modes import render_modes_and_value_model  # lazy import
    from ui_sidebar_extra import render_rows_and_last_action, render_model_decode_settings
    from ui_controls_extra import render_advanced_controls

    # Mode & value model
    vm_choice, selected_gen_mode, _batch_size, _queue_size = render_modes_and_value_model(st)

    # Optional random anchor
    try:
        _rand_anchor_default = bool(getattr(st.session_state, Keys.USE_RANDOM_ANCHOR, True))
    except Exception:
        _rand_anchor_default = True
    try:
        _rand_anchor_cb = getattr(st.sidebar, "checkbox", lambda *a, **k: _rand_anchor_default)(
            "Use random anchor (ignore prompt)", value=_rand_anchor_default
        )
    except Exception:
        _rand_anchor_cb = _rand_anchor_default
    try:
        use_random_anchor = bool(_rand_anchor_cb)
        st.session_state[Keys.USE_RANDOM_ANCHOR] = use_random_anchor
        setattr(lstate, "use_random_anchor", use_random_anchor)
        if use_random_anchor:
            try:
                delattr(lstate, "random_anchor_z")
            except Exception:
                setattr(lstate, "random_anchor_z", None)
    except Exception:
        pass

    # Data strip and auto-refreshing rows metric
    render_rows_and_last_action(st, base_prompt, lstate)

    # Model & decode settings
    selected_model, width, height, steps, guidance, apply_clicked = render_model_decode_settings(st, lstate)
    try:
        st.session_state[Keys.STEPS] = int(steps)
        st.session_state[Keys.GUIDANCE] = float(guidance)
    except Exception:
        pass
    if apply_clicked:
        # Defer actual apply/reset to app.py; we only return values
        pass

    # Advanced proposer/value controls (kept minimal; uses session_state)
    adv_exp = None
    try:
        if bool(st.session_state.get(Keys.SIDEBAR_COMPACT, False)) and callable(getattr(st.sidebar, "expander", None)):
            adv_exp = st.sidebar.expander("Advanced", expanded=False)
    except Exception:
        adv_exp = None
    if adv_exp is not None:
        try:
            with adv_exp:
                render_advanced_controls(st, lstate, base_prompt, vm_choice, selected_gen_mode)
        except TypeError:
            render_advanced_controls(st, lstate, base_prompt, vm_choice, selected_gen_mode)
    else:
        render_advanced_controls(st, lstate, base_prompt, vm_choice, selected_gen_mode)

    # Best-of batch toggle (only in Batch mode)
    try:
        if selected_gen_mode is None or selected_gen_mode == "Batch curation":
            best_of = bool(
                getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                    "Best-of batch (one winner)", value=bool(getattr(st.session_state, "batch_best_of", False))
                )
            )
            st.session_state["batch_best_of"] = best_of
    except Exception:
        pass

    # Resolve modes
    def _resolve_modes() -> tuple[bool, bool]:
        if selected_gen_mode is not None:
            return (selected_gen_mode == "Batch curation", selected_gen_mode == "Async queue")
        return (False, False)

    _curation_mode, async_queue_mode = _resolve_modes()

    # Ridge λ and iterative controls
    _sb_num = getattr(st.sidebar, "number_input", st.number_input)
    try:
        reg_lambda = float(_sb_num("Ridge λ", value=1e-3, step=1e-3, format="%.6f"))
    except Exception:
        reg_lambda = 1e-3
    try:
        st.session_state[Keys.REG_LAMBDA] = float(reg_lambda)
    except Exception:
        pass
    try:
        eta_default = float(getattr(st.session_state, Keys.ITER_ETA, 0.1))
    except Exception:
        eta_default = 0.1
    iter_eta_num = _sb_num("Iterative step (eta)", value=eta_default, step=0.01, format="%.2f")
    try:
        st.session_state[Keys.ITER_ETA] = float(iter_eta_num)
    except Exception:
        pass
    try:
        iter_eta = float(getattr(st.session_state, Keys.ITER_ETA, iter_eta_num))
    except Exception:
        iter_eta = float(iter_eta_num)
    try:
        from constants import DEFAULT_ITER_STEPS as _DEF_STEPS
    except Exception:
        _DEF_STEPS = 10
    try:
        steps_default = int(getattr(st.session_state, Keys.ITER_STEPS, _DEF_STEPS))
    except Exception:
        steps_default = int(_DEF_STEPS)
    iter_steps_num = _sb_num("Optimization steps (latent)", value=steps_default, step=1)
    try:
        st.session_state[Keys.ITER_STEPS] = int(iter_steps_num)
    except Exception:
        pass
    try:
        iter_steps = int(getattr(st.session_state, Keys.ITER_STEPS, iter_steps_num))
    except Exception:
        iter_steps = int(iter_steps_num)

    # Effective guidance used by decode paths (Turbo forces 0.0 in callers)
    try:
        st.session_state[Keys.GUIDANCE_EFF] = 0.0
    except Exception:
        pass

    return (
        vm_choice,
        selected_gen_mode,
        selected_model,
        int(width),
        int(height),
        int(steps),
        float(guidance),
        float(reg_lambda),
        int(iter_steps),
        float(iter_eta),
        bool(async_queue_mode),
    )

