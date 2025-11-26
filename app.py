import os

import streamlit as st

from ipo.infra.constants import DEFAULT_PROMPT, Keys
from ipo.ui.app_api import _apply_state
from ipo.ui.app_api import build_controls as _build_controls
from ipo.ui.app_api import run_app as _run_app_impl


def init_latent_state(*a, **k):
    from ipo.core.latent_state import init_latent_state as _f; return _f(*a, **k)

st.set_page_config(page_title="Latent Preference Optimizer", layout="wide")
if Keys.VM_CHOICE not in st.session_state: st.session_state[Keys.VM_CHOICE] = "XGBoost"
if Keys.ITER_ETA not in st.session_state: st.session_state[Keys.ITER_ETA] = 0.00001

if "prompt" not in st.session_state: st.session_state.prompt = DEFAULT_PROMPT
base_prompt = st.sidebar.text_input("Prompt", value=st.session_state.get("prompt") or DEFAULT_PROMPT)
st.session_state.prompt = base_prompt
# Value function algo selection
vm_opts = ["Ridge", "XGBoost"]
vm_idx = vm_opts.index(st.session_state.get(Keys.VM_CHOICE) or "XGBoost")
st.session_state[Keys.VM_CHOICE] = st.sidebar.selectbox("Value Model", vm_opts, index=vm_idx)
# Latent optimization steps
iter_val = int(st.session_state.get(Keys.ITER_STEPS) or 10)
st.session_state[Keys.ITER_STEPS] = st.sidebar.number_input("Optim Steps", min_value=0, value=iter_val)
# Batch size
batch_val = int(st.session_state.get(Keys.BATCH_SIZE) or 3)
st.session_state[Keys.BATCH_SIZE] = st.sidebar.number_input("Batch Size", min_value=1, max_value=20, value=batch_val)
# Training stats
st.sidebar.markdown("---")
st.sidebar.subheader("Training Stats")
from ipo.core.persistence import state_path_for_prompt

st.session_state.state_path = state_path_for_prompt(base_prompt)
if "lstate" not in st.session_state:
    from ipo.core.latent_state import load_state
    p = st.session_state.state_path
    if os.path.exists(p):
        try: _apply_state(st, load_state(p))
        except: _apply_state(st, init_latent_state())
    else: _apply_state(st, init_latent_state())
lstate = st.session_state.lstate
import numpy as np
from ipo.core.persistence import get_dataset_for_prompt_or_session
X, y = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
print(f"[app] prompt='{base_prompt[:30]}...' X={X.shape if X is not None else None}")
# Train on page load if data exists
if X is not None and X.shape[0] > 0:
    from ipo.core.value_model import fit_value_model
    vm = st.session_state.get(Keys.VM_CHOICE) or "Ridge"
    fit_value_model(vm, lstate, X, y, 1e300, st.session_state)
# Display training stats

w = getattr(lstate, "w", None)
w_norm = float(np.linalg.norm(w)) if w is not None else 0.0
st.sidebar.text(f"Ridge |w|: {w_norm:.4f}")
xgb_cache = st.session_state.get(Keys.XGB_CACHE) or {}
xgb_n = xgb_cache.get("n", 0)
st.sidebar.text(f"XGB samples: {xgb_n}")
last_train = st.session_state.get(Keys.LAST_TRAIN_AT) or "never"
st.sidebar.text(f"Last train: {last_train}")
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
from ipo.core.latent_state import save_state

if st.button("Reset", type="secondary"):
    _apply_state(st, init_latent_state(width=int(width), height=int(height)))
    save_state(st.session_state.lstate, st.session_state.state_path)
    st.rerun()
