from __future__ import annotations

from typing import Any


def sidebar_cv_on_demand(st: Any, lstate: Any, prompt: str, vm: str) -> None:
    """On-demand CV computation (no auto-CV on import/rerun)."""
    def _get_k() -> int:
        k_local = 3
        try:
            num = getattr(st.sidebar, "number_input", None)
            if callable(num):
                k_val = int(num("CV folds", value=3, step=1))
                if k_val < 2:
                    k_val = 2
                if k_val > 5:
                    k_val = 5
                k_local = k_val
        except Exception:
            k_local = 3
        return int(k_local)

    def _clicked() -> bool:
        try:
            btn = getattr(st.sidebar, "button", None)
            return bool(btn and btn("Compute CV now"))
        except Exception:
            return False

    def _dataset_for_cv():
        try:
            Xm = getattr(lstate, "X", None)
            ym = getattr(lstate, "y", None)
            if Xm is not None and ym is not None and getattr(Xm, "shape", (0,))[0] > 0:
                return Xm, ym
        except Exception:
            pass
        try:
            from ipo.ui.ui_sidebar import _get_dataset_for_display as _gdf
            return _gdf(st, lstate, prompt)
        except Exception:
            return None, None

    def _compute_cv_acc(Xd, yd, k: int) -> float | None:
        try:
            if Xd is None or yd is None or getattr(Xd, "shape", (0,))[0] <= 1:
                return None
            if vm == "XGBoost":
                from metrics import xgb_cv_accuracy as _cv
            else:
                from metrics import ridge_cv_accuracy as _cv
            return float(_cv(Xd, yd, k=k))
        except Exception:
            return None

    def _record_result(acc: float | None, k: int) -> None:
        try:
            from ipo.infra.constants import Keys
            cache2 = st.session_state.get(Keys.CV_CACHE) or {}
            if not isinstance(cache2, dict):
                cache2 = {}
            if acc is not None:
                cache2[vm] = {"acc": acc, "k": k}
            st.session_state[Keys.CV_CACHE] = cache2
        except Exception:
            pass
        try:
            import datetime as _dt
            from ipo.infra.constants import Keys
            st.session_state[Keys.CV_LAST_AT] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        except Exception:
            pass

    try:
        if not _clicked():
            return
        k = _get_k()
        Xd, yd = _dataset_for_cv()
        acc = _compute_cv_acc(Xd, yd, k)
        _record_result(acc, k)
    except Exception:
        return

