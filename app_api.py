from __future__ import annotations

from typing import Any, Optional, Tuple


def generate_pair(st: Any, base_prompt: str) -> None:
    from pair_ui import generate_pair as _gen
    from constants import Config, Keys
    try:
        _gen()
        imgs = st.session_state.get(Keys.IMAGES)
        if not imgs or imgs[0] is None or imgs[1] is None:
            try:
                from flux_local import generate_flux_image  # type: ignore

                if callable(generate_flux_image):
                    img = generate_flux_image(
                        base_prompt,
                        width=st.session_state.lstate.width,
                        height=st.session_state.lstate.height,
                        steps=Config.DEFAULT_STEPS,
                        guidance=Config.DEFAULT_GUIDANCE,
                    )
                    st.session_state[Keys.IMAGES] = (img, img)
            except Exception:
                pass
    except Exception:
        pass


def _prefetch_next_for_generate() -> None:
    from pair_ui import _prefetch_next_for_generate as _pf

    _pf()


def _pair_scores() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    from pair_ui import _pair_scores as _impl

    return _impl()


def _curation_init_batch() -> None:
    import batch_ui as _b

    _b._curation_init_batch()


def _curation_new_batch() -> None:
    import batch_ui as _b

    _b._curation_new_batch()


def _sample_around_prompt(scale: float = 0.8):
    import batch_ui as _b

    return _b._sample_around_prompt(scale)


def _curation_replace_at(idx: int) -> None:
    import batch_ui as _b

    _b._curation_replace_at(idx)


def _curation_add(label: int, z, img=None) -> None:
    import batch_ui as _b

    _b._curation_add(label, z, img)


def _curation_train_and_next() -> None:
    import batch_ui as _b

    _b._curation_train_and_next()


def _refit_from_dataset_keep_batch() -> None:
    import batch_ui as _b
    try:
        _b._refit_from_dataset_keep_batch()
    except Exception:
        pass


def _label_and_persist(z, label: int, retrain: bool = True) -> None:
    _curation_add(int(label), z)
    if retrain:
        try:
            _curation_train_and_next()
        except Exception:
            pass


def _queue_fill_up_to(st: Any) -> None:
    import queue_ui as _q
    from constants import Keys as _K

    _q._queue_fill_up_to()
    try:
        qsrc = st.session_state.get(_K.QUEUE) or st.session_state.get("queue")
        q = list(qsrc or [])
        st.session_state[_K.QUEUE] = q
        st.session_state["queue"] = q
    except Exception:
        pass


def _queue_label(idx: int, label: int) -> None:
    import queue_ui as _q

    _q._queue_label(idx, label)


def run_batch_mode() -> None:
    import batch_ui as _b

    _b.run_batch_mode()


def run_upload_mode(st: Any) -> None:
    import batch_ui as _b
    from upload_ui import run_upload_mode as _run_upload

    lstate, prompt = _b._lstate_and_prompt()
    _run_upload(lstate, prompt)

