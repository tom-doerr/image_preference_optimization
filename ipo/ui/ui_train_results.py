from __future__ import annotations

from typing import Any


def compute_train_results_lines(
    st: Any, lstate: Any, prompt: str, vm_choice: str
) -> list[str]:
    """Compute canonical Train-results lines in fixed order.

    Delegates dataset resolution to ui_sidebar._get_dataset_for_display to avoid
    duplicating logic and keep behavior stable.
    """
    try:
        from value_scorer import get_value_scorer as _gvs
        from ipo.ui.ui_sidebar import _get_dataset_for_display  # reuse helper
        import numpy as _np
    except Exception:
        return [
            "Train score: n/a",
            "CV score: n/a",
            "Last CV: n/a",
            "Last train: n/a",
            "Value scorer status: ridge_untrained",
            "Value scorer: Ridge (ridge_untrained, rows=0)",
            "XGBoost active: no",
            "Optimization: Ridge only",
        ]

    def _last_times():
        lt = str(getattr(st.session_state, "last_train_at", "n/a") or st.session_state.get("last_train_at", "n/a"))
        lc = str(st.session_state.get("cv_last_at", "n/a"))
        return lt, lc

    Xd, yd = _get_dataset_for_display(st, lstate, prompt)
    tscore = "n/a"
    vs_status = (
        "xgb_unavailable" if str(vm_choice) == "XGBoost" else "ridge_untrained"
    )
    rows = int(getattr(Xd, "shape", (0,))[0]) if Xd is not None else 0
    scorer, tag = _gvs(vm_choice, lstate, prompt, st.session_state)
    if scorer is not None and Xd is not None and yd is not None and rows > 0:
        vs_status = "ok"
        scores = _np.asarray([scorer(x) for x in Xd], dtype=float)
        yhat = scores >= (0.5 if str(vm_choice) == "XGBoost" else 0.0)
        acc = float((yhat == (yd > 0)).mean())
        tscore = f"{acc * 100:.0f}%"
    else:
        try:
            w = getattr(lstate, "w", _np.zeros(getattr(Xd, "shape", (0, 0))[1]))
            if Xd is not None and rows > 0 and getattr(w, "size", 0) > 0:
                yhat = (Xd @ w) >= 0.0
                acc = float((yhat == (yd > 0)).mean())
                tscore = f"{acc * 100:.0f}%"
                vs_status = vs_status
        except Exception:
            pass

    vs_line = f"{vm_choice or 'Ridge'} ({vs_status}, rows={rows})"
    last_train, last_cv = _last_times()
    active = "yes" if (str(vm_choice) == "XGBoost" and vs_status == "ok") else "no"
    return [
        f"Train score: {tscore}",
        "CV score: n/a",
        f"Last CV: {last_cv}",
        f"Last train: {last_train}",
        f"Value scorer status: {vs_status}",
        f"Value scorer: {vs_line}",
        f"XGBoost active: {active}",
        "Optimization: Ridge only",
    ]


def emit_train_result_lines(st: Any, lines: list[str], sidebar_only: bool = False) -> None:
    """Write canonical Train results lines to sidebar and/or capture sink."""
    if sidebar_only:
        for ln in lines:
            try:
                st.sidebar.write(ln)
            except Exception:
                pass
    else:
        for ln in lines:
            try:
                from ipo.infra.util import safe_write
                safe_write(st, ln)
            except Exception:
                try:
                    st.sidebar.write(ln)
                except Exception:
                    pass
