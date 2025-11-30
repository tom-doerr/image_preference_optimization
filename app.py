import os

import streamlit as st

from ipo.infra.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ITER_ETA,
    DEFAULT_ITER_STEPS,
    DEFAULT_PROMPT,
    DEFAULT_SPACE_MODE,
    DEFAULT_XGB_OPTIM_MODE,
    Keys,
)
from ipo.ui.app_api import _apply_state
from ipo.ui.app_api import build_controls as _build_controls
from ipo.ui.app_api import run_app as _run_app_impl


def init_latent_state(*a, **k):
    from ipo.core.latent_state import init_latent_state as _f
    return _f(*a, **k)

st.set_page_config(page_title="Latent Preference Optimizer", layout="wide")
if Keys.VM_CHOICE not in st.session_state:
    st.session_state[Keys.VM_CHOICE] = "XGBoost"
if Keys.ITER_ETA not in st.session_state:
    st.session_state[Keys.ITER_ETA] = DEFAULT_ITER_ETA

if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_PROMPT
base_prompt = st.sidebar.text_input(
    "Prompt", value=st.session_state.get("prompt") or DEFAULT_PROMPT)
st.session_state.prompt = base_prompt
# Space mode selection
space_modes = ["PooledEmbed", "PromptEmbed", "Latent"]
space_m = st.session_state.get(Keys.SPACE_MODE) or DEFAULT_SPACE_MODE
st.session_state[Keys.SPACE_MODE] = st.sidebar.selectbox(
    "Space", space_modes, index=space_modes.index(space_m))
# Noise seed for generation
noise_seed = int(st.session_state.get(Keys.NOISE_SEED) or 42)
st.session_state[Keys.NOISE_SEED] = st.sidebar.number_input("Noise Seed", value=noise_seed)
# Delta scale for PooledEmbed mode
delta_scale = float(st.session_state.get(Keys.DELTA_SCALE) or 0.1)
st.session_state[Keys.DELTA_SCALE] = st.sidebar.number_input(
    "Delta Scale", min_value=0.0, value=delta_scale, step=0.01, format="%.2f")
# Value function algo selection
vm_opts = ["Ridge", "XGBoost", "Gaussian"]
vm_idx = vm_opts.index(st.session_state.get(Keys.VM_CHOICE) or "XGBoost")
st.session_state[Keys.VM_CHOICE] = st.sidebar.selectbox("Value Model", vm_opts, index=vm_idx)
st.sidebar.markdown("---")
st.sidebar.subheader("XGBoost")
xgb_n = int(st.session_state.get(Keys.XGB_N_ESTIMATORS) or 50)
st.session_state[Keys.XGB_N_ESTIMATORS] = st.sidebar.number_input("Trees", min_value=1, value=xgb_n)
xgb_d = int(st.session_state.get(Keys.XGB_MAX_DEPTH) or 8)
st.session_state[Keys.XGB_MAX_DEPTH] = st.sidebar.number_input("Depth", min_value=1, value=xgb_d)
trust_r = float(st.session_state.get(Keys.TRUST_R) or 200.0)
st.session_state[Keys.TRUST_R] = st.sidebar.number_input("Max Dist", 0.0, value=trust_r, step=0.1)
xgb_modes = ["Grad", "Line", "Hill"]
xgb_m = st.session_state.get(Keys.XGB_OPTIM_MODE) or DEFAULT_XGB_OPTIM_MODE
st.session_state[Keys.XGB_OPTIM_MODE] = st.sidebar.selectbox(
    "Optim", xgb_modes, index=xgb_modes.index(xgb_m))
st.session_state[Keys.XGB_MOMENTUM] = st.sidebar.checkbox(
    "Momentum", value=st.session_state.get(Keys.XGB_MOMENTUM, True))
samp_modes = ["AvgGood", "GoodDist", "Prompt+AvgGood", "Prompt", "Random"]
samp_m = st.session_state.get(Keys.SAMPLE_MODE) or "Random"
st.session_state[Keys.SAMPLE_MODE] = st.sidebar.selectbox(
    "Start", samp_modes, index=samp_modes.index(samp_m))
st.session_state[Keys.REGEN_ALL] = st.sidebar.checkbox(
    "Regen All", value=st.session_state.get(Keys.REGEN_ALL, False))
st.session_state[Keys.BATCH_LABEL] = st.sidebar.checkbox(
    "Batch Label", value=st.session_state.get(Keys.BATCH_LABEL, True))
# Ridge alpha (regularization)
alpha_val = float(st.session_state.get(Keys.REG_LAMBDA) or 1000)
st.session_state[Keys.REG_LAMBDA] = st.sidebar.number_input(
    "Ridge Alpha", min_value=0.0, value=alpha_val, format="%.4f")
# Latent optimization steps
iter_val = int(st.session_state.get(Keys.ITER_STEPS) or DEFAULT_ITER_STEPS)
st.session_state[Keys.ITER_STEPS] = st.sidebar.number_input(
    "Optim Steps", min_value=0, value=iter_val)
eta_val = max(0.0001, float(st.session_state.get(Keys.ITER_ETA) or DEFAULT_ITER_ETA))
st.session_state[Keys.ITER_ETA] = st.sidebar.number_input(
    "Step Size", min_value=0.0001, value=eta_val, format="%.4f")
# Diffusion steps
diff_steps = int(st.session_state.get(Keys.STEPS) or 10)
st.session_state[Keys.STEPS] = st.sidebar.number_input("Diff Steps", min_value=1, value=diff_steps)
# Batch size
batch_val = int(st.session_state.get(Keys.BATCH_SIZE) or DEFAULT_BATCH_SIZE)
st.session_state[Keys.BATCH_SIZE] = st.sidebar.number_input(
    "Batch Size", min_value=1, value=batch_val)
# Images per row (-1 = auto)
ipr_val = int(st.session_state.get(Keys.IMAGES_PER_ROW) or 8)
st.session_state[Keys.IMAGES_PER_ROW] = st.sidebar.number_input(
    "Imgs/Row", min_value=-1, value=ipr_val)
# Training stats
st.sidebar.markdown("---")
st.sidebar.subheader("Training Stats")
from ipo.core.persistence import state_path_for_prompt  # noqa: E402

st.session_state.state_path = state_path_for_prompt(base_prompt)
if "lstate" not in st.session_state:
    from ipo.core.latent_state import load_state
    p = st.session_state.state_path
    if os.path.exists(p):
        try:
            _apply_state(st, load_state(p))
        except Exception:
            sm = st.session_state.get(Keys.SPACE_MODE, "Latent")
            _apply_state(st, init_latent_state(space_mode=sm))
    else:
        sm = st.session_state.get(Keys.SPACE_MODE, "Latent")
        _apply_state(st, init_latent_state(space_mode=sm))
lstate = st.session_state.lstate
import numpy as np  # noqa: E402

from ipo.core.persistence import get_dataset_for_prompt_or_session  # noqa: E402

X, y = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
st.session_state[Keys.DATASET_X] = X
st.session_state[Keys.DATASET_Y] = y
print(f"[app] prompt='{base_prompt[:30]}...' n={X.shape[0] if X is not None else 0}")
# Train on page load if data exists
if X is not None and X.shape[0] > 0:
    print(f"[app] training {X.shape[0]} samples...")
    from ipo.core.value_model import fit_value_model
    vm = st.session_state.get(Keys.VM_CHOICE) or "Ridge"
    alpha = float(st.session_state.get(Keys.REG_LAMBDA) or 1000)
    fit_value_model(vm, lstate, X, y, alpha, st.session_state)
    print("[app] training done")
# Display training stats

w = getattr(lstate, "w", None)
w_norm = float(np.linalg.norm(w)) if w is not None else 0.0
st.sidebar.text(f"Ridge |w|: {w_norm:.4f}")
xgb_cache = st.session_state.get(Keys.XGB_CACHE) or {}
xgb_n = xgb_cache.get("n", 0)
st.sidebar.text(f"XGB samples: {xgb_n}")
last_train = st.session_state.get(Keys.LAST_TRAIN_AT) or "never"
st.sidebar.text(f"Last train: {last_train}")
if st.sidebar.button("HP Search"):
    from ipo.core.hparam_search import xgb_hparam_search
    if X is not None and X.shape[0] >= 6:
        best_p, best_sc = xgb_hparam_search(X, y)
        st.sidebar.success(f"Best: n={best_p['n']} d={best_p['d']} lr={best_p['lr']}")
# Dataset stats
n_total = 0 if X is None else X.shape[0]
n_pos = int((y > 0).sum()) if y is not None else 0
n_neg = int((y < 0).sum()) if y is not None else 0
st.sidebar.text(f"Samples: {n_total} (+{n_pos} / -{n_neg})")
(
    vm_choice,
    selected_gen_mode,
    selected_model,
    width,
    height,
    steps,
    guidance,
    reg_lambda,
    iter_steps,
    iter_eta,
    _,
) = _build_controls(st, lstate, base_prompt)
_run_app_impl(st, vm_choice, selected_gen_mode)

st.write(f"Interactions: {getattr(lstate, 'step', 0)}")
from ipo.core.latent_state import save_state  # noqa: E402

if st.button("Reset", type="secondary"):
    sm = st.session_state.get(Keys.SPACE_MODE, "Latent")
    _apply_state(st, init_latent_state(width=int(width), height=int(height), space_mode=sm))
    save_state(st.session_state.lstate, st.session_state.state_path)
    st.rerun()
