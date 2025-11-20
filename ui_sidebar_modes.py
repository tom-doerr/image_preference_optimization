from __future__ import annotations

from typing import Any, Tuple

from constants import Keys
from ui_controls import build_batch_controls, build_queue_controls


def render_modes_and_value_model(st: Any) -> Tuple[str, str | None, int | None, int | None]:
    """Render 'Mode & value model' section and batch/queue controls.

    Returns (vm_choice, selected_gen_mode, batch_size, queue_size).
    Labels/strings match existing tests.
    """
    st.sidebar.subheader("Mode & value model")
    _sb_sel = getattr(st.sidebar, "selectbox", None)
    _gen_opts = ["Batch curation", "Async queue", "Upload latents"]
    selected_gen_mode = None
    if callable(_sb_sel):
        try:
            selected_gen_mode = _sb_sel("Generation mode", _gen_opts, index=0)
            if selected_gen_mode not in _gen_opts:
                selected_gen_mode = None
        except Exception:
            selected_gen_mode = None
    _vm_opts = ["XGBoost", "DistanceHill", "Ridge", "CosineHill"]
    vm_choice = str(st.session_state.get(Keys.VM_CHOICE, "XGBoost"))
    if callable(_sb_sel):
        try:
            # Prefer existing session choice when available by selecting its index
            try:
                idx = _vm_opts.index(vm_choice)
            except Exception:
                idx = 0
            _sel = _sb_sel("Value model", _vm_opts, index=idx)
            if _sel in _vm_opts:
                vm_choice = _sel
        except Exception:
            vm_choice = vm_choice or "XGBoost"
    st.session_state[Keys.VM_CHOICE] = vm_choice
    st.session_state[Keys.VM_TRAIN_CHOICE] = vm_choice
    # Batch/queue controls
    batch_size = None
    queue_size = None
    if selected_gen_mode == _gen_opts[0]:
        batch_size = build_batch_controls(st, expanded=True)
    elif selected_gen_mode == _gen_opts[1]:
        queue_size = build_queue_controls(st, expanded=True)
    else:
        batch_size = build_batch_controls(st, expanded=False)
        queue_size = build_queue_controls(st, expanded=False)
    try:
        if queue_size is not None:
            st.session_state[Keys.QUEUE_SIZE] = int(queue_size)
    except Exception:
        pass
    try:
        if batch_size is not None:
            st.session_state[Keys.BATCH_SIZE] = int(batch_size)
    except Exception:
        pass
    return vm_choice, selected_gen_mode, batch_size, queue_size
