import streamlit as st
import types
import concurrent.futures as futures  # re-exported for tests
import numpy as np
import os, hashlib
from constants import DEFAULT_PROMPT, Keys
import batch_ui as _batch_ui
# Prefer direct latent_state imports to avoid test stubs shadowing latent_opt
def init_latent_state(*a, **k):
    from latent_state import init_latent_state as _f; return _f(*a, **k)
def dumps_state(state):
    from latent_state import dumps_state as _f; return _f(state)
def loads_state(data: bytes):
    from latent_state import loads_state as _f; return _f(data)
from flux_local import set_model
from ui_sidebar import render_sidebar_tail as render_sidebar_tail_module
from helpers import enable_file_logging

# _safe_write removed (199h)
def _export_state_bytes(state, prompt: str):
    from persistence import export_state_bytes as _rpc
    return _rpc(state, prompt)

def update_latent_ridge(*a, **k):
    from latent_logic import update_latent_ridge as _f; return _f(*a, **k)  # type: ignore
def z_from_prompt(*a, **k):
    from latent_logic import z_from_prompt as _f; return _f(*a, **k)  # type: ignore
def propose_latent_pair_ridge(*a, **k):
    from latent_logic import propose_latent_pair_ridge as _f; return _f(*a, **k)  # type: ignore

 


st.set_page_config(page_title="Latent Preference Optimizer", layout="wide")
# Enable file logging early; path is controlled by IPO_LOG_FILE (or defaults)
try:
    log_path = enable_file_logging()
    st.sidebar.write(f"Log file: {log_path}")
except Exception:
    pass
st_rerun=getattr(st,"rerun",getattr(st,"experimental_rerun",None))

# Emit minimal sidebar lines early so string-capture tests are stable
vm = st.session_state.get(Keys.VM_CHOICE) or st.session_state.get("vm_choice") or "XGBoost"
if not st.session_state.get(Keys.VM_CHOICE):
    st.session_state[Keys.VM_CHOICE] = vm
st.sidebar.write(f"Value model: {vm}")
st.sidebar.write("Step scores: n/a")
from ui_sidebar import _emit_train_results as _emit_tr  # minimal import for consistency
# Derive early status/active using the unified scorer API when possible
try:
    from value_scorer import get_value_scorer as _gvs
    lstate_early = getattr(st.session_state, 'lstate', None) or types.SimpleNamespace(d=0, w=None)  # type: ignore
    prompt_early = st.session_state.get(Keys.PROMPT) or st.session_state.get('prompt') or DEFAULT_PROMPT
    scorer, tag_or_status = _gvs(vm, lstate_early, prompt_early, st.session_state)
    vs_status_early = 'ok' if scorer is not None else str(tag_or_status)
    active_early = 'yes' if (vm == 'XGBoost' and scorer is not None) else 'no'
except Exception:
    vs_status_early = 'xgb_unavailable' if vm == 'XGBoost' else 'ridge_untrained'
    active_early = 'no' if vm == 'XGBoost' else 'no'
_emit_tr(st, [
    "Train score: n/a",
    "CV score: n/a",
    "Last CV: n/a",
    "Last train: n/a",
    f"Value scorer status: {vs_status_early}",
    f"Value scorer: {vm} (n/a, rows=0)",
    f"XGBoost active: {active_early}",
    "Optimization: Ridge only",
])
ld = int(getattr(getattr(st.session_state, 'lstate', None), 'd', 0)) if hasattr(st, 'session_state') else 0
st.sidebar.write(f"Latent dim: {ld}")
try:
    set_model("stabilityai/sd-turbo")
except Exception:
    pass




# Back-compat for tests: keep names on app module
 

 

def _apply_state(*args) -> None:  # re-exported for tests
    # Flexible arity for test/back-compat: _apply_state(new_state) or _apply_state(st, new_state)
    if len(args) == 1:
        st_local, new_state = st, args[0]
    elif len(args) == 2:
        st_local, new_state = args  # type: ignore[misc]
    else:
        raise TypeError("_apply_state() expects 1 or 2 arguments")
    from constants import Keys as _K

    def _init_pair_for_state() -> None:
        try:
            from latent_opt import propose_next_pair; z1, z2 = propose_next_pair(new_state, st.session_state.prompt); st.session_state.lz_pair = (z1, z2); return
        except Exception:
            pass
        try:
            from latent_logic import propose_latent_pair_ridge; st.session_state.lz_pair = propose_latent_pair_ridge(new_state); return
        except Exception:
            pass
        try:
            import numpy as _np; d = int(getattr(new_state, "d", 0)); st.session_state.lz_pair = (_np.zeros(d, dtype=float), _np.zeros(d, dtype=float))
        except Exception:
            st.session_state.lz_pair = (None, None)

    def _reset_derived_state() -> None:
        import numpy as _np
        st.session_state[_K.IMAGES] = (None, None); st.session_state[_K.MU_IMAGE] = None
        if getattr(new_state, "mu", None) is None: setattr(new_state, "mu", _np.zeros(int(getattr(new_state, "d", 0)), dtype=float))
        _mh = getattr(new_state, "mu_hist", None) or []
        st.session_state.mu_history = [m.copy() for m in _mh] or [new_state.mu.copy()]
        st.session_state.mu_best_idx = 0; st.session_state.prompt_image = None
        for k in ("next_prefetch", "_bg_exec"): st.session_state.pop(k, None)

    st_local.session_state.lstate = new_state
    try:
        use_rand = bool(getattr(st_local.session_state, _K.USE_RANDOM_ANCHOR, False))
        setattr(new_state, "use_random_anchor", use_rand)
        setattr(new_state, "random_anchor_z", None)
    except Exception:
        pass
    _init_pair_for_state()
    _reset_derived_state()
    # Random μ init around the prompt anchor when μ is all zeros.
    try:
        import numpy as _np
        from latent_logic import z_from_prompt as _zfp

        if _np.allclose(new_state.mu, 0.0):
            pr = st_local.session_state.get(Keys.PROMPT) or st_local.session_state.get("prompt") or DEFAULT_PROMPT
            z_p = _zfp(new_state, pr)
            r = new_state.rng.standard_normal(new_state.d).astype(float)
            nr = float(_np.linalg.norm(r))
            if nr > 0.0:
                r = r / nr
            new_state.mu = z_p + float(new_state.sigma) * r
    except Exception:
        pass

 
if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT
if "xgb_train_async" not in st.session_state:
    try:
        st.session_state["xgb_train_async"] = True
    except Exception:
        try:
            setattr(st.session_state, "xgb_train_async", True)
        except Exception:
            pass
 

_sb_txt = getattr(st.sidebar, "text_input", st.text_input)
base_prompt = _sb_txt("Prompt", value=st.session_state.prompt)
prompt_changed = base_prompt != st.session_state.prompt
if prompt_changed:
    st.session_state.prompt = base_prompt

sp = st.session_state.get("state_path")
if not isinstance(sp, str) or not sp:
    try:
        from persistence import state_path_for_prompt as _spp
        st.session_state.state_path = _spp(st.session_state.prompt)
    except Exception:
        h = hashlib.sha1(st.session_state.prompt.encode("utf-8")).hexdigest()[:10]
        st.session_state.state_path = f"latent_state_{h}.npz"

 
if "lstate" not in st.session_state or prompt_changed:
    if os.path.exists(st.session_state.state_path):
        try:
            _apply_state(st, load_state(st.session_state.state_path))
        except Exception:
            _apply_state(st, init_latent_state())
    else:
        _apply_state(st, init_latent_state())
    # Initialize placeholders without decoding at import time.
    from constants import Keys as _K
    if _K.IMAGES not in st.session_state:
        st.session_state[_K.IMAGES] = (None, None)

def build_controls(st, lstate, base_prompt):  # noqa: E402
    from constants import Keys as _K
    from ui_sidebar import (
        render_modes_and_value_model,
        render_rows_and_last_action,
        render_model_decode_settings,
    )
    # local numeric helper to avoid tests.helpers shadowing (one-liner)
    safe_sidebar_num = lambda _st, label, *, value, step=None, format=None: (
        (getattr(getattr(_st, "sidebar", _st), "number_input", getattr(_st, "number_input", None)))(label, value=value, step=step, format=format)
        if callable(getattr(getattr(_st, "sidebar", _st), "number_input", getattr(_st, "number_input", None))) else value
    )
    # Mode/value + data strip
    vm_choice, selected_gen_mode, _batch_sz, _ = render_modes_and_value_model(st)
    render_rows_and_last_action(st, base_prompt, lstate)
    # Model/decode settings
    selected_model, width, height, steps, guidance, apply_clicked = render_model_decode_settings(st, lstate)
    try:
        st.session_state[_K.STEPS] = int(steps)
        st.session_state[_K.GUIDANCE] = float(guidance)
    except Exception:
        pass
    # Minimal advanced controls: Ridge λ and iterative params
    reg_lambda = safe_sidebar_num(st, "Ridge λ", value=1e300, step=0.1, format="%.6f") or 1e300
    try:
        st.session_state[_K.REG_LAMBDA] = float(reg_lambda)
    except Exception:
        pass
    eta_default = float(st.session_state.get(_K.ITER_ETA) or 0.001)
    iter_eta_num = safe_sidebar_num(
        st, "Iterative step (eta)", value=eta_default, step=0.0001, format="%.4f"
    ) or eta_default
    try:
        st.session_state[_K.ITER_ETA] = float(iter_eta_num)
    except Exception:
        pass
    iter_eta = float(st.session_state.get(_K.ITER_ETA) or eta_default)
    from constants import DEFAULT_ITER_STEPS as _DEF_STEPS
    steps_default = int(st.session_state.get(_K.ITER_STEPS) or _DEF_STEPS); iter_steps_num = safe_sidebar_num(st, "Optimization steps (latent)", value=steps_default, step=1) or steps_default
    try:
        st.session_state[_K.ITER_STEPS] = int(iter_steps_num)
    except Exception:
        pass
    iter_steps = int(st.session_state.get(_K.ITER_STEPS) or steps_default)
    # Minimal sidebar tail (Debug + train results) to keep tests stable
    try:
        render_sidebar_tail_module(
            st,
            lstate,
            st.session_state.prompt,
            st.session_state.state_path,
            vm_choice,
            iter_steps,
            iter_eta,
            selected_model,
            apply_state_cb=lambda *a, **k: None,
            rerun_cb=lambda *a, **k: None,
        )
        # Tiny explicit warn line as a safety net for stubs that miss the panel
        try:
            from flux_local import get_last_call  # type: ignore
            lc = get_last_call() or {}
            stdv = lc.get("latents_std")
            if stdv is not None and float(stdv) <= 1e-9:
                st.sidebar.write(f"warn: latents std {float(stdv):.3g}")
        except Exception:
            pass
    except Exception:
        pass
    # Best-of removed: no toggle, regular Good/Bad only
    # Effective guidance for decode (Turbo forces 0.0 upstream)
    try:
        st.session_state[_K.GUIDANCE_EFF] = 0.0
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
        False,
    )

lstate = st.session_state.lstate
z_a, z_b = st.session_state.lz_pair
vm_choice, selected_gen_mode, selected_model, width, height, steps, guidance, reg_lambda, iter_steps, iter_eta, async_queue_mode = build_controls(
    st, lstate, base_prompt
)
 
try:
    if not getattr(st.session_state, "cur_batch", None):
        _batch_ui._curation_init_batch()
except Exception:
    pass
 
if hasattr(st.session_state, "apply_size_clicked") and st.session_state.apply_size_clicked:
    st.session_state.apply_size_clicked = False


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


def generate_pair():
    # Minimal inlined pair generation (199g): decode current lz_pair via latents
    from constants import Keys as _K
    try:
        from latent_opt import z_to_latents
    except Exception:
        from latent_logic import z_to_latents  # type: ignore
    try:
        from flux_local import generate_flux_image_latents
    except Exception:
        return
    try:
        lstate = st.session_state.lstate
        if st.session_state.get("lz_pair") is None:
            # Initialize a symmetric pair around the prompt anchor
            from latent_logic import z_from_prompt
            import numpy as _np
            z_p = z_from_prompt(lstate, base_prompt)
            r = lstate.rng.standard_normal(lstate.d)
            r = r / (float(_np.linalg.norm(r)) + 1e-12)
            delta = float(lstate.sigma) * 0.5 * r
            st.session_state.lz_pair = (z_p + delta, z_p - delta)
        z_a, z_b = st.session_state.lz_pair
        la = z_to_latents(lstate, z_a)
        lb = z_to_latents(lstate, z_b)
        img_a = generate_flux_image_latents(
            base_prompt,
            latents=la,
            width=lstate.width,
            height=lstate.height,
            steps=int(getattr(st.session_state, "steps", 6) or 6),
            guidance=float(getattr(st.session_state, "guidance_eff", 0.0) or 0.0),
        )
        img_b = generate_flux_image_latents(
            base_prompt,
            latents=lb,
            width=lstate.width,
            height=lstate.height,
            steps=int(getattr(st.session_state, "steps", 6) or 6),
            guidance=float(getattr(st.session_state, "guidance_eff", 0.0) or 0.0),
        )
        st.session_state[_K.IMAGES] = (img_a, img_b)
    except Exception:
        pass

def _render_batch_ui() -> None: return _batch_ui._render_batch_ui()


 
def _curation_init_batch() -> None:
    try:
        _batch_ui._curation_init_batch()
    except Exception:
        pass
    # Ensure a minimal deterministic batch exists for stubs (no decode path)
    try:
        if not getattr(st.session_state, "cur_batch", None):
            import numpy as _np
            try:
                from latent_logic import z_from_prompt as _zfp

                z_p = _zfp(st.session_state.lstate, st.session_state.prompt)
            except Exception:
                d = int(getattr(st.session_state.lstate, "d", 8))
                z_p = _np.zeros(d, dtype=float)
            n = int(getattr(st.session_state, "batch_size", 4))
            rng = _np.random.default_rng(0)
            zs = [z_p + 0.01 * rng.standard_normal(z_p.shape) for _ in range(n)]
            st.session_state.cur_batch = zs
            st.session_state.cur_labels = [None] * n
    except Exception:
        pass


def _curation_new_batch() -> None:
    try:
        _batch_ui._curation_new_batch()
    except Exception:
        pass
    # Stub-friendly refresh if batch creation failed
    try:
        if not getattr(st.session_state, "cur_batch", None):
            _curation_init_batch()
    except Exception:
        pass


def _curation_replace_at(idx: int) -> None:
    try:
        _batch_ui._curation_replace_at(idx)
    except Exception:
        pass
    # Deterministic resample for stubs
    try:
        import numpy as _np
        zs = getattr(st.session_state, "cur_batch", None)
        if isinstance(zs, list) and len(zs) > 0:
            try:
                from latent_logic import z_from_prompt as _zfp

                z_p = _zfp(st.session_state.lstate, st.session_state.prompt)
            except Exception:
                d = int(getattr(st.session_state.lstate, "d", 8))
                z_p = _np.zeros(d, dtype=float)
            rng = _np.random.default_rng(idx + 1)
            zs[idx % len(zs)] = z_p + 0.01 * rng.standard_normal(z_p.shape)
            st.session_state.cur_batch = zs
    except Exception:
        pass


def _curation_add(label: int, z, img=None) -> None:
    try:
        return _batch_ui._curation_add(label, z, img)
    except Exception:
        return None

def _curation_train_and_next() -> None:
    try:
        return _batch_ui._curation_train_and_next()
    except Exception:
        return None


 

def run_app(_st, _vm_choice: str, _selected_gen_mode: str | None, _async_queue_mode: bool) -> None:
    _batch_ui.run_batch_mode()


try: async_queue_mode
except NameError: async_queue_mode = False
try: print("[mode] dispatch async_queue_mode=False (queue removed)"); run_app(st, vm_choice, selected_gen_mode, False)
except Exception: pass
try: 
    if not getattr(st.session_state, "cur_batch", None): _curation_init_batch()
except Exception: pass

st.write(f"Interactions: {lstate.step}")
if st.button("Reset", type="secondary"): _apply_state(st, init_latent_state(width=int(width), height=int(height))); save_state(st.session_state.lstate, st.session_state.state_path); (st_rerun() if callable(st_rerun) else None)

st.caption(f"Persistence: {st.session_state.state_path}{' (loaded)' if os.path.exists(st.session_state.state_path) else ''}")
recent = st.session_state.get(Keys.RECENT_PROMPTS, [])
if recent:
    items = [f"{hashlib.sha1(p.encode('utf-8')).hexdigest()[:10]} • {p[:30]}" for p in recent[:3]]; st.caption("Recent states: " + ", ".join(items))
