from __future__ import annotations

from typing import Any


def _button_key(st, prefix: str, nonce: int, idx: int) -> str:
    try:
        rnd = int(st.session_state.get("render_nonce", 0))
    except Exception:
        rnd = 0
    try:
        rcount = int(st.session_state.get("render_count", 0))
    except Exception:
        rcount = 0
    try:
        seq = int(st.session_state.get("btn_seq", 0)) + 1
    except Exception:
        seq = 1
    st.session_state["btn_seq"] = seq
    return f"{prefix}_{rcount}_{rnd}_{nonce}_{idx}_{seq}"


def _toast_and_record(st, msg: str) -> None:
    try:
        getattr(st, "toast", lambda *a, **k: None)(msg)
        from ipo.infra.constants import Keys as _K
        import time as __t
        st.session_state[_K.LAST_ACTION_TEXT] = msg
        st.session_state[_K.LAST_ACTION_TS] = float(__t.time())
    except Exception:
        pass


def _label_and_replace(i: int, lbl: int, z_i, img_i, st) -> None:
    # Import inside function to avoid cycles
    from .batch_ui import _curation_add, _refit_from_dataset_keep_batch, _curation_replace_at, _log  # type: ignore
    import time as _time
    t0b2 = _time.perf_counter()
    _curation_add(lbl, z_i, img_i)
    st.session_state.cur_labels[i] = lbl
    _refit_from_dataset_keep_batch()
    _curation_replace_at(i)
    _log(f"[perf] {'good' if lbl>0 else 'bad'}_label item={i} took {(_time.perf_counter() - t0b2) * 1000:.1f} ms")
    try:
        rr = getattr(st, "rerun", None)
        if callable(rr):
            rr()
    except Exception:
        pass


def render_good_bad_buttons(st, i: int, z_i, img_i, nonce: int, gcol, bcol) -> None:
    gkey = _button_key(st, "good", nonce, i)
    bkey = _button_key(st, "bad", nonce, i)
    # Good (+1)
    if gcol is not None:
        with gcol:
            if st.button(f"Good (+1) {i}", key=gkey, width="stretch"):
                _label_and_replace(i, 1, z_i, img_i, st)
                _toast_and_record(st, "Labeled Good (+1)")
    else:
        if st.button(f"Good (+1) {i}", key=gkey, width="stretch"):
            _label_and_replace(i, 1, z_i, img_i, st)
            _toast_and_record(st, "Labeled Good (+1)")
    # Bad (-1)
    if bcol is not None:
        with bcol:
            if st.button(f"Bad (-1) {i}", key=bkey, width="stretch"):
                _label_and_replace(i, -1, z_i, img_i, st)
                _toast_and_record(st, "Labeled Bad (-1)")
    else:
        if st.button(f"Bad (-1) {i}", key=bkey, width="stretch"):
            _label_and_replace(i, -1, z_i, img_i, st)
            _toast_and_record(st, "Labeled Bad (-1)")


def handle_best_of(st, i: int, img_i, cur_batch) -> None:
    # Import inside to avoid cycles
    from .batch_ui import _curation_add, _curation_train_and_next, _log  # type: ignore
    import time as _time
    t0b = _time.perf_counter()
    for j, z_j in enumerate(cur_batch):
        lbl = 1 if j == i else -1
        img_j = img_i if j == i else None
        _curation_add(lbl, z_j, img_j)
        st.session_state.cur_labels[j] = lbl
    _curation_train_and_next()
    try:
        getattr(st, "toast", lambda *a, **k: None)(f"Best-of: chose {i}")
    except Exception:
        pass
    _log(f"[perf] best_of choose item={i} took {(_time.perf_counter() - t0b) * 1000:.1f} ms")

