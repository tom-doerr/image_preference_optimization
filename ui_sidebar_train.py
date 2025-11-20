from __future__ import annotations

from typing import Any, Tuple

from constants import Keys


def compute_train_results_summary(st: Any, lstate: Any, base_prompt: str, vm_choice: str) -> Tuple[str, str, str, str, str]:
    from persistence import get_dataset_for_prompt_or_session as _get_ds
    Xd, yd = _get_ds(base_prompt, st.session_state)
    # Prefer in-memory dataset when supplied by tests; otherwise use folder dataset.
    _Xm = getattr(lstate, "X", None)
    _ym = getattr(lstate, "y", None)
    if _Xm is not None and getattr(_Xm, "shape", (0,))[0] > 0 and _ym is not None and getattr(_ym, "shape", (0,))[0] > 0:
        Xd, yd = _Xm, _ym
    # Train score
    tscore = "n/a"
    if Xd is not None and yd is not None and getattr(Xd, "shape", (0,))[0] > 0:
        use_xgb = vm_choice == "XGBoost"
        try:
            cache = getattr(st.session_state, "xgb_cache", {}) or {}
            mdl = cache.get("model")
        except Exception:
            mdl = None
        if use_xgb and mdl is not None:
            try:
                from xgb_value import score_xgb_proba  # type: ignore
                import numpy as _np
                probs = _np.array([score_xgb_proba(mdl, fv) for fv in Xd], dtype=float)
                preds = probs >= 0.5
                acc = float(_np.mean(preds == (yd > 0)))
                tscore = f"{acc * 100:.0f}%"
            except Exception:
                tscore = "n/a"
        elif use_xgb and mdl is None:
            # Opportunistic one-shot fit for Train score when a dataset exists
            try:
                from xgb_value import fit_xgb_classifier, score_xgb_proba  # type: ignore
                n_estim = int(getattr(st.session_state, "xgb_n_estimators", 50))
                max_depth = int(getattr(st.session_state, "xgb_max_depth", 3))
                mdl2 = fit_xgb_classifier(Xd, yd, n_estimators=n_estim, max_depth=max_depth)
                # Cache for later scoring (both attr and keyed for compatibility)
                cache_obj = {"model": mdl2, "n": int(getattr(Xd, "shape", (0,))[0])}
                st.session_state.xgb_cache = cache_obj
                try:
                    st.session_state[Keys.XGB_CACHE] = cache_obj
                except Exception:
                    pass
                import numpy as _np
                probs = _np.array([score_xgb_proba(mdl2, fv) for fv in Xd], dtype=float)
                preds = probs >= 0.5
                acc = float(_np.mean(preds == (yd > 0)))
                tscore = f"{acc * 100:.0f}%"
            except Exception:
                tscore = "n/a"
        if tscore == "n/a":
            try:
                pred = Xd @ lstate.w
                acc = float(((pred >= 0) == (yd > 0)).mean())
                tscore = f"{acc * 100:.0f}%"
            except Exception:
                tscore = "n/a"
    # Last train
    try:
        last_train = (
            str(st.session_state.get("last_train_at"))
            if st.session_state.get("last_train_at")
            else "n/a"
        )
    except Exception:
        last_train = "n/a"
    try:
        if last_train == "n/a":
            # If we have an in-memory or on-disk dataset for this prompt, stamp now
            has_rows = False
            if Xd is not None and getattr(Xd, "shape", (0,))[0] > 0:
                has_rows = True
            else:
                try:
                    from persistence import dataset_rows_for_prompt

                    has_rows = int(dataset_rows_for_prompt(base_prompt)) > 0
                except Exception:
                    has_rows = False
            if has_rows:
                from datetime import datetime, timezone
                last_train = datetime.now(timezone.utc).isoformat(timespec="seconds")
                st.session_state[Keys.LAST_TRAIN_AT] = last_train
    except Exception:
        pass
    # Value scorer status
    try:
        from value_scorer import get_value_scorer_with_status as _gss
        _, vs_status = _gss(vm_choice, lstate, base_prompt, st.session_state)
        vs_name = str(vm_choice or "Ridge")
        vs_rows = 0
        if Xd is not None and yd is not None and getattr(Xd, "shape", (0,))[0] > 0:
            vs_rows = int(getattr(Xd, "shape", (0,))[0])
        vs_line = f"{vs_name} ({vs_status}, rows={vs_rows})"
    except Exception:
        vs_line = "unknown"
    # CV (cached only)
    cv_line = "n/a"
    try:
        cv_cache = st.session_state.get(Keys.CV_CACHE) or {}
        if isinstance(cv_cache, dict):
            cur = cv_cache.get(str(vm_choice))
            if isinstance(cur, dict) and "acc" in cur:
                acc = float(cur.get("acc", float("nan")))
                k = int(cur.get("k", 0))
                if vm_choice == "XGBoost":
                    cv_line = (
                        f"{acc * 100:.0f}% (k={k}, XGB, nested)" if acc == acc else "n/a"
                    )
                else:
                    cv_line = f"{acc * 100:.0f}% (k={k})" if acc == acc else "n/a"
    except Exception:
        pass
    return tscore, cv_line, last_train, vs_line, vs_status


def render_train_results_panel(st: Any, lstate: Any, base_prompt: str, vm_choice: str) -> Tuple[str, str, str, str, str]:
    t, cv, last, vs_line, vs_status = compute_train_results_summary(st, lstate, base_prompt, vm_choice)
    # Button to compute CV on demand
    try:
        do_cv = getattr(st.sidebar, "button", lambda *a, **k: False)("Compute CV now")
    except Exception:
        do_cv = False
    if do_cv:
        try:
            from metrics import ridge_cv_accuracy as _rcv, xgb_cv_accuracy as _xcv
            import numpy as _np
            from persistence import get_dataset_for_prompt_or_session as _get_ds
            X_cv, y_cv = _get_ds(base_prompt, st.session_state)
            if X_cv is not None and y_cv is not None and getattr(X_cv, "shape", (0,))[0] >= 4:
                n_rows = int(len(y_cv))
                _k_r = min(5, n_rows)
                lam_now = float(st.session_state.get(Keys.REG_LAMBDA, 1e-3))
                acc_r = float(_rcv(X_cv, y_cv, lam=lam_now, k=_k_r))
                try:
                    n_estim = int(st.session_state.get("xgb_n_estimators", 50))
                except Exception:
                    n_estim = 50
                try:
                    max_depth = int(st.session_state.get("xgb_max_depth", 3))
                except Exception:
                    max_depth = 3
                try:
                    k_pref = int(st.session_state.get("xgb_cv_folds", 3))
                except Exception:
                    k_pref = 3
                kx = max(2, min(5, min(k_pref, n_rows)))
                acc_x = float(_xcv(X_cv, y_cv, k=kx, n_estimators=n_estim, max_depth=max_depth))
                cc = {"Ridge": {"acc": acc_r, "k": _k_r}, "XGBoost": {"acc": acc_x, "k": kx}}
                st.session_state[Keys.CV_CACHE] = cc
                from datetime import datetime, timezone
                st.session_state[Keys.CV_LAST_AT] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                if vm_choice == "XGBoost":
                    cv = f"{acc_x * 100:.0f}% (k={kx}, XGB, nested)" if not _np.isnan(acc_x) else "n/a"
                else:
                    cv = f"{acc_r * 100:.0f}% (k={_k_r})" if not _np.isnan(acc_r) else "n/a"
        except Exception:
            pass
    return t, cv, last, vs_line, vs_status
