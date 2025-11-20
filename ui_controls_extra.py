from __future__ import annotations

from typing import Any


def render_advanced_controls(st: Any, lstate: Any, base_prompt: str, vm_choice: str, selected_gen_mode: str | None) -> None:
    from constants import Keys
    from ui_controls import build_pair_controls  # reuse existing controls
    try:
        from latent_logic import hill_climb_mu_distance as _hc_dist
    except Exception:  # tests may stub latent_logic without this func
        def _hc_dist(*_a, **_k):  # type: ignore[misc]
            return None
    import latent_logic as ll  # for xgb hill climb
    from latent_opt import save_state
    from persistence import get_dataset_for_prompt_or_session

    # Pair controls (alpha/beta/trust/lr_mu/orth/iter params)
    st.sidebar.subheader("Latent optimization")
    _alpha, _beta, _trust_r, _lr_mu_ui, _gamma_orth, _iter_steps, _iter_eta = build_pair_controls(st, expanded=False)
    for k, v in (("alpha", _alpha), ("trust_r", _trust_r), ("lr_mu_ui", _lr_mu_ui)):
        try:
            st.session_state[k] = float(v)
        except Exception:
            pass

    # Hill-climb μ
    st.sidebar.subheader("Hill-climb μ")
    _eta_mu = getattr(st.sidebar, "number_input", st.number_input)("η (step)", value=0.2, step=0.01, format="%.2f")
    _gamma_mu = getattr(st.sidebar, "number_input", st.number_input)("γ (sigmoid)", value=0.5, step=0.1, format="%.1f")
    _trust_mu = getattr(st.sidebar, "number_input", st.number_input)("Trust radius r (0=off)", value=0.0, step=1.0, format="%.1f")
    _hill_label = "Hill-climb μ (XGBoost)" if vm_choice == "XGBoost" else "Hill-climb μ (distance)"
    if getattr(st.sidebar, "button", st.button)(_hill_label):
        Xd, yd = get_dataset_for_prompt_or_session(base_prompt, st.session_state)
        if Xd is not None and yd is not None and getattr(Xd, "shape", (0,))[0] > 0:
            r_val = None if float(_trust_mu) <= 0.0 else float(_trust_mu)
            if vm_choice == "XGBoost":
                try:
                    from value_scorer import get_value_scorer
                    cache = st.session_state.get(Keys.XGB_CACHE) or {}
                    mdl = cache.get("model")
                    if mdl is not None:
                        scorer = get_value_scorer("XGBoost", lstate, base_prompt, st.session_state)
                        step_scale = float(_lr_mu_ui) * float(getattr(lstate, "sigma", 1.0))
                        steps_now = int(st.session_state.get(Keys.ITER_STEPS, _iter_steps))
                        ll.hill_climb_mu_xgb(lstate, base_prompt, scorer, steps=steps_now, step_scale=step_scale, trust_r=r_val)
                        save_state(lstate, st.session_state.state_path)
                except Exception:
                    pass
            else:
                _hc_dist(lstate, base_prompt, Xd, yd, eta=float(_eta_mu), gamma=float(_gamma_mu), trust_r=r_val)
                save_state(lstate, st.session_state.state_path)
        if callable(getattr(st, 'rerun', None)):
            st.rerun()

    # Best-of toggle (Batch only)
    try:
        if selected_gen_mode is None or selected_gen_mode == "Batch curation":
            best_of = bool(getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                "Best-of batch (one winner)", value=bool(getattr(st.session_state, "batch_best_of", False))
            ))
            st.session_state["batch_best_of"] = best_of
    except Exception:
        pass
