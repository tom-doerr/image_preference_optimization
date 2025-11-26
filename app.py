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
iter_val = int(st.session_state.get(Keys.ITER_STEPS) or 0)
st.session_state[Keys.ITER_STEPS] = st.sidebar.number_input("Optim Steps", 0, 1000, iter_val)
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
