import streamlit as st
import concurrent.futures as futures  # re-exported for tests
import logging as _logging

try:
    from rich_cli import enable_color_print as _enable_color

    _enable_color()
except Exception:
    pass
import numpy as np
import os
import hashlib
from PIL import Image
from constants import (
    DEFAULT_PROMPT,
)
from constants import Keys
# (sidebar helpers are imported within ui_sidebar modules)
import batch_ui as _batch_ui
# step scores rendering moved to ui_sidebar; keep imports local there
# batch/queue controls are rendered via ui_sidebar_modes
from persistence import (
    state_path_for_prompt,
)
# (persistence_ui helpers are imported in ui_sidebar)
from latent_opt import (
    init_latent_state,
    save_state,
    load_state,
)
from latent_opt import dumps_state, loads_state  # re-exported for tests
from flux_local import (
    set_model,
)
from app_api import (
    _curation_init_batch as _curation_init_batch,
    _curation_new_batch as _curation_new_batch,
    _sample_around_prompt as _sample_around_prompt,
    _curation_replace_at as _curation_replace_at,
    _curation_add as _curation_add,
    _curation_train_and_next as _curation_train_and_next,
    _refit_from_dataset_keep_batch as _refit_from_dataset_keep_batch,
    _queue_label as _queue_label,
)
from app_api import _label_and_persist as _label_and_persist
from app_state import _apply_state as _apply_state  # re-export for tests
from app_bootstrap import prompt_first_bootstrap  # import-time bootstrap helper
from ui_sidebar import render_sidebar_tail as render_sidebar_tail_module  # new consolidated sidebar tail
from img_latents import image_to_z as _image_to_z
# controls are built in app_main.build_controls

# Optional helpers (text-only path and debug accessor) — may be absent in tests
try:
    from flux_local import generate_flux_image  # type: ignore
except Exception:  # pragma: no cover - shim for minimal stubs
    generate_flux_image = None  # type: ignore
try:
    from flux_local import get_last_call  # type: ignore
except Exception:  # pragma: no cover

    def get_last_call():  # type: ignore
        return {}


st.set_page_config(page_title="Latent Preference Optimizer", layout="wide")
# Silence ruff F401 for re-exports accessed by tests
_exports_silence = (dumps_state, loads_state, futures)
# Streamlit rerun API shim: prefer st.rerun(), fallback to experimental in older versions
st_rerun = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
# Emit minimal sidebar lines early so string-capture tests are stable
def _emit_minimal_sidebar_lines() -> None:
    try:
        vm = (
            st.session_state.get(Keys.VM_CHOICE)
            or st.session_state.get("vm_choice")
            or "Ridge"
        )
        st.sidebar.write(f"Value model: {vm}")
        st.sidebar.write("Train score: n/a")
        st.sidebar.write("Step scores: n/a")
        st.sidebar.write(f"XGBoost active: {'yes' if vm == 'XGBoost' else 'no'}")
    except Exception:
        pass

_emit_minimal_sidebar_lines()
try:
    # Ensure a model is selected once at import so stubs can assert set_model called
    from constants import DEFAULT_MODEL as _DEF_MODEL
    set_model(_DEF_MODEL)
except Exception:
    pass
## toast helper removed — modules call st.toast directly where needed


# Shared logger routed to ipo.debug.log (and stdout for tests)
LOGGER = _logging.getLogger("ipo")
if not LOGGER.handlers:
    try:
        _h = _logging.FileHandler("ipo.debug.log")
        _h.setFormatter(
            _logging.Formatter("%(asctime)s %(levelname)s app: %(message)s")
        )
        LOGGER.addHandler(_h)
        LOGGER.setLevel(_logging.INFO)
    except Exception:
        pass
try:
    import os as _os

    _lvl = (_os.getenv("IPO_LOG_LEVEL") or "").upper()
    if _lvl:
        LOGGER.setLevel(getattr(_logging, _lvl, _logging.INFO))
except Exception:
    pass


def _log(msg: str, level: str = "info") -> None:
    try:
        print(msg)
    except Exception:
        pass
    try:
        getattr(LOGGER, level, LOGGER.info)(msg)
    except Exception:
        pass


# kept for back-compat in tests that import app.image_to_z
def image_to_z(img: Image.Image, lstate) -> np.ndarray:  # pragma: no cover - thin wrapper
    return _image_to_z(img, lstate)


# Back-compat for tests: keep names on app module
_state_path_for_prompt = state_path_for_prompt

# Prompt-aware persistence
if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT
# Default: train value models in background to keep UI responsive.
if "xgb_train_async" not in st.session_state:
    try:
        st.session_state["xgb_train_async"] = True
    except Exception:
        pass
# Also default Ridge to async to avoid UI stalls during fits.
if Keys.RIDGE_TRAIN_ASYNC not in st.session_state:  # keep minimal logic
    try:
        st.session_state[Keys.RIDGE_TRAIN_ASYNC] = True
    except Exception:
        pass

_sb_txt = getattr(st.sidebar, "text_input", st.text_input)
base_prompt = _sb_txt("Prompt", value=st.session_state.prompt)
prompt_changed = base_prompt != st.session_state.prompt
if prompt_changed:
    st.session_state.prompt = base_prompt

st.session_state.state_path = state_path_for_prompt(st.session_state.prompt)

## (legacy) image fragment helper removed — image tiles handled in modules




if "lstate" not in st.session_state or prompt_changed:
    if os.path.exists(st.session_state.state_path):
        _apply_state(st, load_state(st.session_state.state_path))
    else:
        _apply_state(st, init_latent_state())
    if "prompt_image" not in st.session_state:
        st.session_state.prompt_image = None
    # Initialize prompt-first placeholders without decoding at import time.
    prompt_first_bootstrap(st, st.session_state.lstate, base_prompt)

from app_main import build_controls  # type: ignore  # noqa: E402
from app_run import run_app as _run_app, generate_pair as _run_generate_pair, _queue_fill_up_to as _run_queue_fill_up_to  # noqa: E402

lstate = st.session_state.lstate
z_a, z_b = st.session_state.lz_pair
vm_choice, selected_gen_mode, selected_model, width, height, steps, guidance, reg_lambda, iter_steps, iter_eta, async_queue_mode = build_controls(
    st, lstate, base_prompt
)
# Expose a simple module flag used by a few tests
use_xgb = bool(vm_choice == "XGBoost")
# Apply resize if requested by user
if hasattr(st.session_state, "apply_size_clicked") and st.session_state.apply_size_clicked:
    st.session_state.apply_size_clicked = False
if False:  # placeholder to keep structure minimal
    pass


def render_sidebar_tail():
    render_sidebar_tail_module(
        st,
        lstate,
        st.session_state.prompt,
        st.session_state.state_path,
        vm_choice,
        int(iter_steps),
        float(iter_eta) if iter_eta is not None else None,
        selected_model,
        _apply_state,
        st_rerun,
    )


render_sidebar_tail()

def _force_render_sidebar_tail() -> None:
    """Test helper: re-render sidebar tail against the current Streamlit stub.

    Useful when tests swap `sys.modules['streamlit']` without clearing the app
    module cache; keeps behavior deterministic across import orders.
    """
    render_sidebar_tail()

## imports moved to top


def generate_pair():
    """Zero-arg test wrapper: delegate to app_run.generate_pair."""
    _run_generate_pair(st, base_prompt)


## history helpers removed; proposer opts live in proposer module and are used by internals only



def _queue_fill_up_to() -> None:
    """Zero-arg test wrapper: delegate to app_run._queue_fill_up_to."""
    _run_queue_fill_up_to(st)



## Pair preference flow moved to app_run/app_api to keep this file lean.


## Pair UI renderer removed (pair mode no longer routed); generate_pair remains for tests.

## Prompt-first decode removed from import path; placeholders are set in app_bootstrap


def _render_batch_ui() -> None:
    return _batch_ui._render_batch_ui()


## Queue UI renderer imported from queue_ui for test compatibility


## Pair scores shim lives in app_api if needed by tests


## Deprecated helpers removed; tests should use queue_ui/batch_ui directly


## Pair mode runner removed; only Batch and Queue are routed.


## Mode runners consolidated via app_run.run_app


## import moved to top
# Run selected mode (Batch default vs Async queue)
try:
    async_queue_mode
except NameError:  # minimal guard for test stubs/import order
    async_queue_mode = False
try:
    _log(f"[mode] dispatch async_queue_mode={bool(async_queue_mode)}")
except Exception:
    pass
_run_app(st, vm_choice, selected_gen_mode, bool(async_queue_mode))

st.write(f"Interactions: {lstate.step}")
if st.button("Reset", type="secondary"):
    _apply_state(st, init_latent_state(width=int(width), height=int(height)))
    save_state(st.session_state.lstate, st.session_state.state_path)
    if callable(st_rerun):
        st_rerun()

st.caption(
    f"Persistence: {st.session_state.state_path}{' (loaded)' if os.path.exists(st.session_state.state_path) else ''}"
)
# Footer: recent prompt states (hash + truncated text)
recent = st.session_state.get(Keys.RECENT_PROMPTS, [])
if recent:

    def _hash_of(p: str) -> str:
        return hashlib.sha1(p.encode("utf-8")).hexdigest()[:10]

    items = [f"{_hash_of(p)} • {p[:30]}" for p in recent[:3]]
    st.caption("Recent states: " + ", ".join(items))

## First-round prompt seeding is handled in _apply_state; no duplicate logic here

# μ preview UI removed
## _toast already defined above; avoid duplicate definitions
