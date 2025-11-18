from typing import Tuple, Optional


def _sb_num(st):
    return getattr(st.sidebar, "number_input", st.number_input)


def _sb_sld(st):
    return getattr(st.sidebar, "slider", st.slider)


def build_size_controls(st, lstate) -> Tuple[int, int, int, float, bool]:
    num = _sb_num(st)
    sld = _sb_sld(st)
    width = num("Width", min_value=256, max_value=1024, step=64, value=lstate.width)
    height = num("Height", min_value=256, max_value=1024, step=64, value=lstate.height)
    steps = sld("Steps", 1, 50, 6)
    guidance = sld("Guidance", 0.0, 10.0, 3.5, 0.1)
    # Fallbacks for stubs that return None
    width = lstate.width if width is None else width
    height = lstate.height if height is None else height
    steps = 6 if steps is None else steps
    guidance = 3.5 if guidance is None else guidance
    apply_clicked = False
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
    alpha = sld("Alpha (ridge d1)", 0.05, 3.0, 0.5, 0.05)
    beta = sld("Beta (ridge d2)", 0.05, 3.0, 0.5, 0.05)
    trust_r = sld("Trust radius (||y||)", 0.5, 5.0, 2.5, 0.1)
    lr_mu_ui = sld("Step size (lr_μ)", 0.05, 1.0, 0.3, 0.05)
    gamma_orth = sld("Orth explore (γ)", 0.0, 1.0, 0.2, 0.05)
    iter_steps = sld("Optimization steps (latent)", 1, 10, 1, 1)
    iter_eta = sld("Iterative step (eta)", 0.0, 1.0, 0.0, 0.05)
    if ctx is not None:
        ctx.__exit__(None, None, None)
    return float(alpha), float(beta), float(trust_r), float(lr_mu_ui), float(gamma_orth), int(iter_steps), float(iter_eta)


def build_batch_controls(st, expanded: bool = False) -> int:
    expander = getattr(st.sidebar, "expander", None)
    ctx = expander("Batch controls", expanded=expanded) if callable(expander) else None
    if ctx is not None:
        ctx.__enter__()
    sld = _sb_sld(st)
    batch_size = sld("Batch size", 2, 12, value=6, step=1)
    if ctx is not None:
        ctx.__exit__(None, None, None)
    return int(batch_size)


def build_queue_controls(st, expanded: bool = False) -> int:
    expander = getattr(st.sidebar, "expander", None)
    ctx = expander("Queue controls", expanded=expanded) if callable(expander) else None
    if ctx is not None:
        ctx.__enter__()
    sld = _sb_sld(st)
    queue_size = sld("Queue size", 2, 16, value=6, step=1)
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
