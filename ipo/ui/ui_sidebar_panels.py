from __future__ import annotations

from typing import Any


def sidebar_value_model_block(st: Any, lstate: Any, prompt: str, vm_choice: str, reg_lambda: float) -> None:
    """Render the Value model block in the sidebar.

    Delegates utilities (cached CV, details writers, status lines) to ui_sidebar helpers
    to preserve strings and behavior.
    """
    from .ui_sidebar import (
        _cached_cv_lines,
        safe_write,
        _emit_cv_metrics,
        _xgb_status_line,
        _vm_details_ridge,
        _vm_details_distance,
        _vm_details_xgb,
        _cv_on_demand,
        Keys,
    )

    def _emit_cv_all() -> None:
        xl, rl = _cached_cv_lines(st)
        safe_write(st, xl)
        safe_write(st, rl)
        _emit_cv_metrics(st, xl, rl)

    def _with_details_expander(render_fn) -> None:
        exp = getattr(st.sidebar, "expander", None)
        if not callable(exp):
            _emit_cv_all()
            return
        with exp("Details", expanded=False):
            render_fn()

    vm = "Ridge" if vm_choice not in ("XGBoost", "Ridge", "Distance") else vm_choice
    cache: dict = {}
    safe_write(st, f"Value model: {vm}")
    _emit_cv_all()

    def _render_details() -> None:
        try:
            rows_n = int((cache or {}).get("n") or 0)
        except Exception:
            rows_n = 0
        status = "ok" if rows_n > 0 else "unavailable"
        try:
            xst = st.session_state.get(Keys.XGB_TRAIN_STATUS)
            if isinstance(xst, dict) and isinstance(xst.get("state"), str):
                status = str(xst.get("state"))
        except Exception:
            pass
        _xgb_status_line(st, rows_n, status)
        if vm == "Ridge":
            _vm_details_ridge(st, lstate, prompt, reg_lambda)
        elif vm == "Distance":
            _vm_details_distance(st)
        else:
            _vm_details_xgb(st, cache)

    _with_details_expander(_render_details)
    _cv_on_demand(st, lstate, prompt, vm)


def handle_train_section(st: Any, lstate: Any, prompt: str, vm_choice: str) -> None:
    """Render the small train-controls section (sync-only fits).

    Delegates dataset selection and the actual train handlers back to ui_sidebar
    to avoid duplicating logic and to keep strings/behavior identical.
    """
    try:
        # Resolve dataset (prefers in-memory, falls back to folder dataset)
        from ipo.ui.ui_sidebar import _get_dataset_for_display, _autofit_xgb_if_selected
        Xd, yd = _get_dataset_for_display(st, lstate, prompt)
        _autofit_xgb_if_selected(st, lstate, vm_choice, Xd, yd)  # no-op by design
        button = getattr(st.sidebar, "button", lambda *a, **k: False)
        if str(vm_choice) == "XGBoost" and button("Train XGBoost now (sync)"):
            from ipo.ui.ui_sidebar import _xgb_train_controls
            _xgb_train_controls(st, lstate, Xd, yd)
        elif str(vm_choice) == "Logistic" and button("Train Logistic now (sync)"):
            from ipo.ui.ui_sidebar import _logit_train_controls
            _logit_train_controls(st, lstate, Xd, yd)
    except Exception:
        pass
