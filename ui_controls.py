from typing import Tuple, Optional
from constants import DEFAULT_ITER_STEPS


def _sb_num(st):
    return getattr(st.sidebar, "number_input", st.number_input)


def _sb_sld(st):
    return getattr(st.sidebar, "slider", st.slider)


def build_size_controls(st, lstate) -> Tuple[int, int, int, float, bool]:
    num = _sb_num(st)
    sld = _sb_sld(st)
    width = num("Width", min_value=256, max_value=1024, step=64, value=lstate.width)
    height = num("Height", min_value=256, max_value=1024, step=64, value=lstate.height)
    steps = sld("Steps", min_value=0, max_value=50, value=6)
    guidance = sld("Guidance", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    # Fallbacks for stubs that return None
    width = lstate.width if width is None else width
    height = lstate.height if height is None else height
    steps = 6 if steps is None else steps
    guidance = 3.5 if guidance is None else guidance
    apply_clicked = False
    # Auto-apply when size sliders differ from current latent state
    if int(width) != int(lstate.width) or int(height) != int(lstate.height):
        apply_clicked = True
    try:
        if st.sidebar.button("Apply size now"):
            apply_clicked = True
    except Exception:
        pass
    return int(width), int(height), int(steps), float(guidance), bool(apply_clicked)


def build_pair_controls(st, expanded: bool = False):
    sld = _sb_sld(st)
    expander = getattr(st.sidebar, "expander", None)
    ctx = expander("Pair controls", expanded=expanded) if callable(expander) else None
    if ctx is not None:
        ctx.__enter__()
    # Brief, plain explanation to make sliders self-explanatory
    try:
        st.sidebar.write(
            "Proposes the next A/B around the prompt: Alpha scales d1 (∥ w), Beta scales d2 (⟂ d1); "
            "Trust radius clamps ‖y‖; lr_μ is the μ update step; γ adds orthogonal exploration."
        )
    except Exception:
        pass
    alpha = sld(
        "Alpha (ridge d1)", min_value=0.0, max_value=3.0, value=0.5, step=0.05,
        help="Scale step along d1 (∥ w; value-gradient)."
    )
    beta = sld(
        "Beta (ridge d2)", min_value=0.0, max_value=3.0, value=0.5, step=0.05,
        help="Scale step along d2 (⟂ d1; orthogonal explore)."
    )
    trust_r = sld(
        "Trust radius (||y||)", min_value=0.0, max_value=5.0, value=2.5, step=0.1,
        help="Clamp proposal offset norm ‖y‖ ≤ r around anchor."
    )
    lr_mu_ui = sld(
        "Step size (lr_μ)", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
        help="How far μ moves toward the winner per click."
    )
    gamma_orth = sld(
        "Orth explore (γ)", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Add small orthogonal component (⊥ d1) to avoid stagnation."
    )
    # Optimization steps (latent) are now driven by a numeric input in app.py;
    # here we simply read the shared session_state value and pass it through.
    try:
        sess = getattr(st, "session_state", None)
    except Exception:
        sess = None
    steps_default = DEFAULT_ITER_STEPS
    if sess is not None:
        try:
            steps_default = int(sess.get("iter_steps", steps_default))
        except Exception:
            pass
    iter_steps = steps_default
    # eta (Iterative step) is also driven by a numeric input; reuse shared state.
    eta_default = 0.1
    if sess is not None:
        try:
            eta_default = float(sess.get("iter_eta", eta_default))
        except Exception:
            pass
    iter_eta = eta_default
    if ctx is not None:
        ctx.__exit__(None, None, None)
    return float(alpha), float(beta), float(trust_r), float(lr_mu_ui), float(gamma_orth), int(iter_steps), float(iter_eta)


def build_batch_controls(st, expanded: bool = False) -> int:
    sld = _sb_sld(st)
    batch_size = sld("Batch size", min_value=0, max_value=64, value=25, step=1)
    return int(batch_size)


def build_queue_controls(st, expanded: bool = False) -> int:
    expander = getattr(st.sidebar, "expander", None)
    ctx = expander("Queue controls", expanded=expanded) if callable(expander) else None
    if ctx is not None:
        ctx.__enter__()
    sld = _sb_sld(st)
    queue_size = sld("Queue size", min_value=0, max_value=16, value=6, step=1)
    if ctx is not None:
        ctx.__exit__(None, None, None)
    return int(queue_size)


def build_mode_select(st) -> Tuple[Optional[str], bool, bool]:
    """Return (selected_mode, curation_mode, async_queue_mode). Dropdown is authoritative.

    selected_mode is one of ["Pair (A/B)", "Batch curation", "Async queue"] or None if selectbox not available.
    """
    gen_opts = ["Pair (A/B)", "Batch curation", "Async queue"]
    sb_sel = getattr(st.sidebar, "selectbox", None)
    selected_mode = None
    if callable(sb_sel):
        try:
            selected_mode = sb_sel("Generation mode", gen_opts, index=1)
            if selected_mode not in gen_opts:
                selected_mode = None
        except Exception:
            selected_mode = None
    # Legacy toggles (keep for tests)
    def _legacy():
        cm = st.sidebar.checkbox("Batch curation mode", value=False)
        aq = st.sidebar.checkbox("Async queue mode", value=False)
        return bool(cm), bool(aq)

    if selected_mode is not None and callable(getattr(st.sidebar, "expander", None)):
        with st.sidebar.expander("Advanced (legacy mode toggles)"):
            cm, aq = _legacy()
    else:
        cm, aq = _legacy()

    if selected_mode is not None:
        return selected_mode, (selected_mode == gen_opts[1]), (selected_mode == gen_opts[2])
    return selected_mode, cm, aq
