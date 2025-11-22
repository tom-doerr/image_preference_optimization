from typing import Tuple, Optional
from constants import DEFAULT_ITER_STEPS, DEFAULT_BATCH_SIZE

def build_batch_controls(st, expanded: bool = False) -> int:
    from ui_sidebar import build_batch_controls as _bbc
    return int(_bbc(st, expanded=expanded))


def _sb_num(st):
    return getattr(st.sidebar, "number_input", st.number_input)


def _sb_sld(st):
    return getattr(st.sidebar, "slider", st.slider)


def build_size_controls(st, lstate) -> Tuple[int, int, int, float, bool]:
    num = _sb_num(st)
    sld = _sb_sld(st)
    width = num("Width", step=64, value=lstate.width)
    height = num("Height", step=64, value=lstate.height)
    steps = sld("Steps", value=6)
    guidance = sld("Guidance", value=3.5, step=0.1)
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
    from ui_sidebar import build_pair_controls as _bpc
    return _bpc(st, expanded=expanded)


def build_size_controls(st, lstate) -> Tuple[int, int, int, float, bool]:
    from ui_sidebar import _build_size_controls as _bsc
    return _bsc(st, lstate)


def build_queue_controls(st, expanded: bool = False) -> int:
    expander = getattr(st.sidebar, "expander", None)
    ctx = expander("Queue controls", expanded=expanded) if callable(expander) else None
    if ctx is not None:
        ctx.__enter__()
    sld = _sb_sld(st)
    queue_size = sld("Queue size", value=6, step=1)
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
        return (
            selected_mode,
            (selected_mode == gen_opts[1]),
            (selected_mode == gen_opts[2]),
        )
    return selected_mode, cm, aq
