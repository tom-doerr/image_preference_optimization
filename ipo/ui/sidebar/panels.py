from __future__ import annotations

from typing import Any

from ipo.infra.util import SAFE_EXC


def sidebar_value_model_block(st: Any, lstate: Any, prompt: str, vm_choice: str, reg_lambda: float) -> None:
    """Render the Value model block in the sidebar.

    Delegates utilities (cached CV, details writers, status lines) to ui_sidebar helpers
    to preserve strings and behavior.
    """
    from .ui_sidebar import (
        Keys,
        _cached_cv_lines,
        _cv_on_demand,
        _emit_cv_metrics,
        _vm_details_distance,
        _vm_details_ridge,
        _vm_details_xgb,
        _xgb_status_line,
        safe_write,
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
        except SAFE_EXC:
            rows_n = 0
        status = "ok" if rows_n > 0 else "unavailable"
        try:
            xst = st.session_state.get(Keys.XGB_TRAIN_STATUS)
            if isinstance(xst, dict) and isinstance(xst.get("state"), str):
                status = str(xst.get("state"))
        except SAFE_EXC:
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
        from ipo.ui.ui_sidebar import _autofit_xgb_if_selected, _get_dataset_for_display
        Xd, yd = _get_dataset_for_display(st, lstate, prompt)
        _autofit_xgb_if_selected(st, lstate, vm_choice, Xd, yd)  # no-op by design
        button = getattr(st.sidebar, "button", lambda *a, **k: False)
        if str(vm_choice) == "XGBoost" and button("Train XGBoost now (sync)"):
            from ipo.ui.ui_sidebar import _xgb_train_controls
            _xgb_train_controls(st, lstate, Xd, yd)
        elif str(vm_choice) == "Logistic" and button("Train Logistic now (sync)"):
            from ipo.ui.ui_sidebar import _logit_train_controls
            _logit_train_controls(st, lstate, Xd, yd)
    except SAFE_EXC:
        pass


# Moved from ui_sidebar.py to reduce that module's complexity.
def _xgb_train_controls(st: Any, lstate: Any, Xd, yd) -> None:
    from value_model import fit_value_model as _fit_vm
    def _select_dataset():
        Xm = getattr(lstate, 'X', None)
        ym = getattr(lstate, 'y', None)
        has_mem = (
            Xm is not None and getattr(Xm, 'shape', (0,))[0] > 0 and ym is not None and getattr(ym, 'shape', (0,))[0] > 0
        )
        return (Xm, ym) if has_mem else (Xd, yd)

    def _count_pos_neg(Ys) -> tuple[int, int]:
        try:
            yy = [int(v) for v in list(Ys)] if Ys is not None else []
            p = sum(1 for v in yy if v > 0)
            n = sum(1 for v in yy if v < 0)
            return int(p), int(n)
        except Exception:
            return (0, 0)

    def _train_now(Xs, Ys) -> None:
        from ipo.infra.constants import Keys
        lam_now = float(st.session_state.get(Keys.REG_LAMBDA, 1.0))
        _fit_vm("XGBoost", lstate, Xs, Ys, lam_now, st.session_state)
        try:
            getattr(st, "toast", lambda *a, **k: None)("XGBoost: trained (sync)")
        except Exception:
            pass
        try:
            print("[xgb] trained (sync)")
        except Exception:
            pass
        # Record ephemeral last action for sidebar Train results
        try:
            import time as _time
            st.session_state[Keys.LAST_ACTION_TEXT] = "XGBoost: trained (sync)"
            st.session_state[Keys.LAST_ACTION_TS] = float(_time.time())
        except Exception:
            pass

    Xs, Ys = _select_dataset()
    pos, neg = _count_pos_neg(Ys)
    if Xs is not None and Ys is not None and getattr(Xs, 'shape', (0,))[0] > 1 and pos > 0 and neg > 0:
        _train_now(Xs, Ys)


def _logit_train_controls(st: Any, lstate: Any, Xd, yd) -> None:
    from value_model import fit_value_model as _fit_vm
    Xm = getattr(lstate, 'X', None)
    ym = getattr(lstate, 'y', None)
    Xs, Ys = (Xm, ym) if (
        Xm is not None and getattr(Xm, 'shape', (0,))[0] > 0 and ym is not None and getattr(ym, 'shape', (0,))[0] > 0
    ) else (Xd, yd)
    if Xs is not None and Ys is not None and getattr(Xs, 'shape', (0,))[0] > 1:
        from ipo.infra.constants import Keys
        lam_now = float(st.session_state.get(Keys.REG_LAMBDA, 1.0))
        _fit_vm("Logistic", lstate, Xs, Ys, lam_now, st.session_state)
        try:
            getattr(st, "toast", lambda *a, **k: None)("Logit: trained (sync)")
        except Exception:
            pass
        try:
            import time as _time
            st.session_state[Keys.LAST_ACTION_TEXT] = "Logit: trained (sync)"
            st.session_state[Keys.LAST_ACTION_TS] = float(_time.time())
        except Exception:
            pass


def _vm_details_xgb(st: Any, cache: dict) -> None:
    try:
        import importlib.util as _ilu
        avail = "yes" if _ilu.find_spec("xgboost") is not None else "no"
        st.sidebar.write(f"XGBoost available: {avail}")
    except SAFE_EXC:
        pass
    try:
        from ipo.infra.util import safe_sidebar_num as _num
    except SAFE_EXC:
        _num = None
    n_fit = cache.get("n") or 0
    try:
        n_estim = int(st.session_state.get("xgb_n_estimators", 50))
        max_depth = int(st.session_state.get("xgb_max_depth", 3))
    except Exception:
        n_estim, max_depth = 50, 3
    try:
        if callable(_num):
            n_estim = int(_num(st, "XGB n_estimators", value=n_estim, step=1))
            max_depth = int(_num(st, "XGB max_depth", value=max_depth, step=1))
            st.session_state["xgb_n_estimators"] = n_estim
            st.session_state["xgb_max_depth"] = max_depth
    except SAFE_EXC:
        pass
    st.sidebar.write(f"fit_rows={int(n_fit)}, n_estimators={n_estim}, depth={max_depth}")


def _sidebar_training_data_block(st: Any, prompt: str, lstate: Any) -> None:
    try:
        exp = getattr(st.sidebar, "expander", None)
        if callable(exp):
            with exp("Training data", expanded=False):
                st.sidebar.write("Pos")
                st.sidebar.write("Neg")
                st.sidebar.write("Feat dim")
                st.sidebar.write("Pairs:")
                st.sidebar.write("Choices:")
        # Always also emit a compact strip on the sidebar root
        try:
            from ipo.ui.ui_sidebar import _mem_dataset_stats
            stats = _mem_dataset_stats(st, lstate)
        except Exception:
            stats = {"rows": 0, "pos": 0, "neg": 0, "d": int(getattr(lstate, 'd', 0))}
        st.sidebar.write("Training data & scores")
        st.sidebar.write(f"Dataset rows: {stats.get('rows', 0)}")
        st.sidebar.write("Rows (disk): 0")  # disk scan removed in simplified path
        st.sidebar.write("Pairs:: 0")
        st.sidebar.write("Choices:: 0")
    except Exception:
        pass
