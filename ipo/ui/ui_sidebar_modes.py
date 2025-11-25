from __future__ import annotations

from typing import Any


def _select_generation_mode(st: Any) -> str | None:
    _sb_sel = getattr(st.sidebar, "selectbox", None)
    opts = ["Batch curation"]
    if not callable(_sb_sel):
        return None
    try:
        sel = _sb_sel("Generation mode", opts, index=0)
        return sel if sel in opts else None
    except Exception:
        return None


def _select_value_model(st: Any, vm_choice: str) -> str:
    _sb_sel = getattr(st.sidebar, "selectbox", None)
    opts = ["XGBoost", "Logistic", "Ridge"]
    if callable(_sb_sel):
        try:
            idx = opts.index(vm_choice) if vm_choice in opts else 0
            sel = _sb_sel("Value model", opts, index=idx)
            if sel in opts:
                return sel
        except Exception:
            return vm_choice or "XGBoost"
    return vm_choice


def _toggle_random_anchor(st: Any) -> bool:
    try:
        from ipo.infra.constants import Keys
        use_rand = bool(
            getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                "Use random anchor (ignore prompt)",
                value=bool(st.session_state.get(Keys.USE_RANDOM_ANCHOR, False)),
            )
        )
        st.session_state[Keys.USE_RANDOM_ANCHOR] = use_rand
        try:
            ls = getattr(st.session_state, "lstate", None)
            if ls is not None:
                setattr(ls, "use_random_anchor", use_rand)
                setattr(ls, "random_anchor_z", None)
        except Exception:
            pass
        return use_rand
    except Exception:
        return False


def build_batch_controls(st, expanded: bool = False) -> int:
    sld = getattr(st.sidebar, "slider", st.slider)
    try:
        from ipo.infra.constants import DEFAULT_BATCH_SIZE
    except Exception:
        DEFAULT_BATCH_SIZE = 4
    batch_size = sld("Batch size", value=DEFAULT_BATCH_SIZE, step=1)
    return int(batch_size)


def build_pair_controls(st, expanded: bool = False):
    sld = getattr(st.sidebar, "slider", st.slider)
    expander = getattr(st.sidebar, "expander", None)
    ctx = expander("Pair controls", expanded=expanded) if callable(expander) else None
    if ctx is not None:
        ctx.__enter__()
    try:
        st.sidebar.write(
            "Proposes the next A/B around the prompt: Alpha scales d1 (∥ w), Beta scales d2 (⟂ d1); Trust radius clamps ‖y‖; lr_μ is the μ update step; γ adds orthogonal exploration."
        )
    except Exception:
        pass
    alpha = sld("Alpha (ridge d1)", value=0.5, step=0.05)
    beta = sld("Beta (ridge d2)", value=0.5, step=0.05)
    trust_r = sld("Trust radius (||y||)", value=2.5, step=0.1)
    lr_mu_ui = sld("Step size (lr_μ)", value=0.001, step=0.001)
    gamma_orth = sld("Orth explore (γ)", value=0.2, step=0.05)
    sess = getattr(st, "session_state", None)
    if sess is not None and hasattr(sess, "get"):
        steps_default = int((sess.get("iter_steps") or 1000))
        eta_default = float((sess.get("iter_eta") or 0.00001))
    else:
        steps_default = 1000
        eta_default = 0.00001
    if ctx is not None:
        ctx.__exit__(None, None, None)
    return float(alpha), float(beta), float(trust_r), float(lr_mu_ui), float(gamma_orth), int(steps_default), float(eta_default)


def render_modes_and_value_model(st: Any) -> tuple[str, str | None, int | None, int | None]:
    from ipo.infra.constants import Keys
    st.sidebar.subheader("Mode & value model")
    selected_gen_mode = _select_generation_mode(st)
    vm_choice = str(st.session_state.get(Keys.VM_CHOICE, "XGBoost"))
    vm_choice = _select_value_model(st, vm_choice)
    st.session_state[Keys.VM_CHOICE] = vm_choice
    st.session_state[Keys.VM_TRAIN_CHOICE] = vm_choice
    batch_size = build_batch_controls(st, expanded=True)
    _toggle_random_anchor(st)
    try:
        st.session_state[Keys.BATCH_SIZE] = int(batch_size)
    except Exception:
        pass
    return vm_choice, selected_gen_mode, batch_size, None

