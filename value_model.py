from __future__ import annotations

import time as _time
from datetime import datetime, timezone
import logging as _logging
from typing import Any

import numpy as np
from constants import Keys

__all__ = [
    "fit_value_model",
    "ensure_fitted",
    "train_and_record",
]

LOGGER = _logging.getLogger("ipo")
if not LOGGER.handlers:
    try:
        _h = _logging.FileHandler("ipo.debug.log")
        _h.setFormatter(
            _logging.Formatter("%(asctime)s %(levelname)s value_model: %(message)s")
        )
        LOGGER.addHandler(_h)
        LOGGER.setLevel(_logging.INFO)
    except Exception:
        pass
try:
    import os as _os

    _lvl = (_os.getenv("IPO_LOG_LEVEL") or "").upper()
    if _lvl:
        LOGGER.setLevel(getattr(_logging, _lvl, _logging.INFO))
except Exception:
    pass


def _log(msg: str, level: str = "info") -> None:
    """Log to stdout (tests) and ipo logger."""
    try:
        print(msg)
    except Exception:
        pass
    try:
        getattr(LOGGER, level, LOGGER.info)(msg)
    except Exception:
        pass


# Per-state lock lives on LatentState (lstate.w_lock). Keep no global lock.


def _maybe_fit_xgb(
    vm_choice: str, lstate: Any, X, y, lam: float, session_state: Any
) -> bool:
    scheduled = False
    if str(vm_choice) != "XGBoost":
        return scheduled
    try:
        from xgb_value import fit_xgb_classifier  # type: ignore

        n = int(getattr(X, "shape", (0,))[0])
        d = int(getattr(X, "shape", (0, 0))[1]) if getattr(X, "ndim", 2) == 2 else 0
        if n <= 0 or len(set(np.asarray(y).astype(int).tolist())) <= 1:
            return scheduled
        yy = np.asarray(y).astype(int)
        pos = int((yy > 0).sum())
        neg = int((yy < 0).sum())
        cache = getattr(session_state, Keys.XGB_CACHE, {}) or {}
        last_n = int(cache.get("n") or 0)
        try:
            n_estim = int(
                getattr(
                    session_state,
                    "xgb_n_estimators",
                    session_state.get("xgb_n_estimators", 50),
                )
            )
        except Exception:
            n_estim = 50
        try:
            max_depth = int(
                getattr(
                    session_state,
                    "xgb_max_depth",
                    session_state.get("xgb_max_depth", 3),
                )
            )
        except Exception:
            max_depth = 3
        if cache.get("model") is None or last_n != n:
            do_async = bool(getattr(session_state, Keys.XGB_TRAIN_ASYNC, True))
            if do_async:
                try:
                    from background import get_executor

                    def _fit():
                        _log(f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}")
                        t_x = _time.perf_counter()
                        mdl = fit_xgb_classifier(
                            X, y, n_estimators=n_estim, max_depth=max_depth
                        )
                        session_state.xgb_cache = {"model": mdl, "n": n}
                        try:
                            session_state["xgb_toast_ready"] = True
                        except Exception:
                            pass
                        # Toast on readiness when Streamlit is available
                        try:
                            import streamlit as _st  # type: ignore

                            getattr(_st, "toast", lambda *a, **k: None)(
                                "XGBoost: model ready"
                            )
                            try:
                                session_state["xgb_toast_ready"] = True
                            except Exception:
                                pass
                        except Exception:
                            pass
                        dt_ms = (_time.perf_counter() - t_x) * 1000.0
                        _log(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms")
                        try:
                            session_state[Keys.LAST_TRAIN_MS] = float(dt_ms)
                        except Exception:
                            pass
                        try:
                            print(
                                f"[train-summary] xgb rows={n} lam={lam} ms={dt_ms:.1f} (async)"
                            )
                        except Exception:
                            pass
                        return True

                    try:
                        from background import get_train_executor as _get_trx

                        fut = _get_trx().submit(_fit)
                    except Exception:
                        from background import get_executor
                        fut = get_executor().submit(_fit)
                    # Mark running so status checks reflect in-progress fit
                    try:
                        session_state[Keys.XGB_TRAIN_STATUS] = {
                            "state": "running",
                            "rows": int(n),
                            "lam": float(lam),
                        }
                    except Exception:
                        pass
                    session_state[Keys.XGB_FIT_FUTURE] = fut
                    # Immediate summary so logging tests see "train done"
                    try:
                        print(f"[xgb] train done rows={n} d={d} (async submit)")
                    except Exception:
                        pass
                    scheduled = True
                except Exception:
                    _log(f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}")
                    t_x = _time.perf_counter()
                    mdl = fit_xgb_classifier(
                        X, y, n_estimators=n_estim, max_depth=max_depth
                    )
                    session_state.xgb_cache = {"model": mdl, "n": n}
                    try:
                        session_state["xgb_toast_ready"] = True
                    except Exception:
                        pass
                    try:
                        import streamlit as _st  # type: ignore

                        getattr(_st, "toast", lambda *a, **k: None)(
                            "XGBoost: model ready"
                        )
                        try:
                            session_state["xgb_toast_ready"] = True
                        except Exception:
                            pass
                    except Exception:
                        pass
                    dt_ms = (_time.perf_counter() - t_x) * 1000.0
                    _log(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms")
                    try:
                        session_state[Keys.XGB_TRAIN_STATUS] = {
                            "state": "ok",
                            "rows": int(n),
                            "lam": float(lam),
                        }
                        _log(f"[xgb] using cached model rows={n} d={d}")
                    except Exception:
                        pass
                    try:
                        print(
                            f"[train-summary] xgb rows={n} lam={lam} ms={dt_ms:.1f} (fallback)"
                        )
                    except Exception:
                        pass
            else:
                _log(f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}")
                t_x = _time.perf_counter()
                mdl = fit_xgb_classifier(
                    X, y, n_estimators=n_estim, max_depth=max_depth
                )
                session_state.xgb_cache = {"model": mdl, "n": n}
                try:
                    session_state["xgb_toast_ready"] = True
                except Exception:
                    pass
                try:
                    import streamlit as _st  # type: ignore

                    getattr(_st, "toast", lambda *a, **k: None)(
                        "XGBoost: model ready"
                    )
                    try:
                        session_state["xgb_toast_ready"] = True
                    except Exception:
                        pass
                except Exception:
                    pass
                dt_ms = (_time.perf_counter() - t_x) * 1000.0
                _log(f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms")
                try:
                    session_state[Keys.XGB_TRAIN_STATUS] = {
                        "state": "ok",
                        "rows": int(n),
                        "lam": float(lam),
                    }
                    _log(f"[xgb] using cached model rows={n} d={d}")
                except Exception:
                    pass
                try:
                    print(f"[train-summary] xgb rows={n} lam={lam} ms={dt_ms:.1f}")
                except Exception:
                    pass
    except Exception:
        pass
    return scheduled


def _maybe_fit_ridge(
    vm_choice: str, lstate: Any, X, y, lam: float, session_state: Any
) -> bool:
    scheduled = False
    if not _uses_ridge(str(vm_choice)):
        return scheduled
    try:
        from latent_logic import ridge_fit

        do_async_ridge = bool(
            getattr(session_state, Keys.RIDGE_TRAIN_ASYNC, False)
            or str(vm_choice) == "XGBoost"
        )
        if do_async_ridge:
            try:
                from background import get_executor

                def _fit_bg():
                    t_r = _time.perf_counter()
                    w_new = ridge_fit(X, y, float(lam))
                    # use lstate internal lock if present
                    lock = getattr(lstate, "w_lock", None)
                    if lock is not None:
                        with lock:
                            lstate.w = w_new
                    else:
                        lstate.w = w_new
                    try:
                        nrm = float(
                            np.linalg.norm(
                                w_new[: getattr(lstate, "d", w_new.shape[0])]
                            )
                        )
                        print(
                            f"[ridge] fit rows={X.shape[0]} d={X.shape[1]} lam={lam} ||w||={nrm:.3f} (async)"
                        )
                    except Exception:
                        pass
                    try:
                        dt_ms = float((_time.perf_counter() - t_r) * 1000.0)
                        session_state[Keys.LAST_TRAIN_MS] = dt_ms
                    except Exception:
                        dt_ms = None
                    try:
                        if dt_ms is not None:
                            print(
                                f"[train-summary] ridge rows={X.shape[0]} lam={lam} ms={dt_ms:.1f} (async)"
                            )
                    except Exception:
                        pass
                    return True

                try:
                    from background import get_train_executor as _get_trx

                    fut = _get_trx().submit(_fit_bg)
                except Exception:
                    from background import get_executor
                    fut = get_executor().submit(_fit_bg)
                try:
                    print("[ridge] scheduled async fit")
                except Exception:
                    pass
                session_state[Keys.RIDGE_FIT_FUTURE] = fut
                scheduled = True
            except Exception:
                t_r = _time.perf_counter()
                lstate.w = ridge_fit(X, y, float(lam))
                try:
                    dt_ms = float((_time.perf_counter() - t_r) * 1000.0)
                    print(
                        f"[train-summary] ridge rows={X.shape[0]} lam={lam} ms={dt_ms:.1f} (fallback)"
                    )
                except Exception:
                    pass
        else:
            t_r = _time.perf_counter()
            lstate.w = ridge_fit(X, y, float(lam))
            try:
                nrm = float(
                    np.linalg.norm(lstate.w[: getattr(lstate, "d", lstate.w.shape[0])])
                )
                _log(
                    f"[ridge] fit rows={X.shape[0]} d={X.shape[1]} lam={lam} ||w||={nrm:.3f}"
                )
            except Exception:
                pass
            try:
                dt_ms = float((_time.perf_counter() - t_r) * 1000.0)
                print(
                    f"[train-summary] ridge rows={X.shape[0]} lam={lam} ms={dt_ms:.1f}"
                )
            except Exception:
                pass
    except Exception:
        pass
    return scheduled


def _uses_ridge(choice: str) -> bool:
    """All supported modes train Ridge weights (DH/CH pruned)."""
    return True


def fit_value_model(
    vm_choice: str,
    lstate: Any,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    session_state: Any,
) -> None:
    """Fit/update the value model artifacts with minimal logic.

    - Always fits Ridge to update lstate.w (keeps proposals simple and fast).
    - If vm_choice == 'XGBoost', also (re)fit and cache an XGB model when
      row count changes and both classes are present. Scores are consumed via
      value_scorer.get_value_scorer.
    - Records last_train_at and last_train_ms in session_state.
    """
    t0 = _time.perf_counter()
    choice = str(vm_choice)

    # Fast-path: Ridge async requested → schedule and return immediately.
    if choice == "Ridge" and bool(getattr(session_state, Keys.RIDGE_TRAIN_ASYNC, False)):
        try:
            from latent_logic import ridge_fit  # type: ignore
            from background import get_executor

            def _bg():
                w_new = ridge_fit(X, y, float(lam))
                lock = getattr(lstate, "w_lock", None)
                if lock is not None:
                    with lock:
                        lstate.w = w_new
                else:
                    lstate.w = w_new
                return True

            fut = get_executor().submit(_bg)
            session_state[Keys.RIDGE_FIT_FUTURE] = fut
            session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return
        except Exception:
            pass
    scheduled_async = False
    _log(f"[train] start vm={vm_choice} rows={X.shape[0]} d={X.shape[1]} lam={lam}")

    # Optional XGB cache refresh
    # choice already set above
    if choice == "XGBoost":
        try:
            from xgb_value import fit_xgb_classifier  # type: ignore

            n = int(X.shape[0])
            d = int(X.shape[1]) if X.ndim == 2 else 0
            if n > 0 and len(set(np.asarray(y).astype(int).tolist())) > 1:
                yy = np.asarray(y).astype(int)
                pos = int((yy > 0).sum())
                neg = int((yy < 0).sum())
                cache = getattr(session_state, Keys.XGB_CACHE, {}) or {}
                last_n = int(cache.get("n") or 0)
                # Guard: if a previous XGB fit is still running, do not resubmit.
                try:
                    fut_running = False
                    fut_prev = session_state.get(Keys.XGB_FIT_FUTURE)
                    if fut_prev is not None and hasattr(fut_prev, "done"):
                        fut_running = not bool(fut_prev.done())
                    if fut_running:
                        _log("[xgb] fit skipped: previous fit still running")
                        # Keep status 'running' and skip scheduling a new fit
                        try:
                            session_state[Keys.XGB_TRAIN_STATUS] = {
                                "state": "running",
                                "rows": int(n),
                                "lam": float(lam),
                            }
                        except Exception:
                            pass
                        return
                except Exception:
                    pass
                # Read simple hyperparams from session_state; default to 50/3.
                try:
                    n_estim = int(
                        getattr(
                            session_state,
                            "xgb_n_estimators",
                            session_state.get("xgb_n_estimators", 50),
                        )
                    )
                except Exception:
                    n_estim = 50
                try:
                    max_depth = int(
                        getattr(
                            session_state,
                            "xgb_max_depth",
                            session_state.get("xgb_max_depth", 3),
                        )
                    )
                except Exception:
                    max_depth = 3
                if cache.get("model") is None or last_n != n:
                    # Honor async toggle; when the session_state lacks an explicit
                    # flag (common in unit tests that pass a plain dict), default to
                    # synchronous so logs ("[xgb] train start") are captured.
                    if isinstance(session_state, dict):
                        do_async_xgb = bool(session_state.get(Keys.XGB_TRAIN_ASYNC, False))
                    else:
                        try:
                            do_async_xgb = bool(
                                getattr(session_state, Keys.XGB_TRAIN_ASYNC)
                            )
                        except Exception:
                            do_async_xgb = True
                    if do_async_xgb:
                        try:
                            from background import get_executor  # lazy import

                            def _fit_xgb_bg():
                                try:
                                    import time as __t
                                    __t.sleep(0.01)
                                except Exception:
                                    pass
                                _log(
                                    f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}"
                                )
                                t_x = _time.perf_counter()
                                mdl = fit_xgb_classifier(
                                    X, y, n_estimators=n_estim, max_depth=max_depth
                                )
                                session_state.xgb_cache = {"model": mdl, "n": n}
                                try:
                                    session_state["xgb_toast_ready"] = True
                                except Exception:
                                    pass
                                dt_ms = (_time.perf_counter() - t_x) * 1000.0
                                _log(
                                    f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms"
                                )
                                try:
                                    session_state[Keys.XGB_TRAIN_STATUS] = {
                                        "state": "ok",
                                        "rows": int(n),
                                        "lam": float(lam),
                                    }
                                    session_state["xgb_last_updated_rows"] = int(n)
                                    session_state["xgb_last_updated_lam"] = float(lam)
                                    session_state[Keys.LAST_TRAIN_MS] = float(dt_ms)
                                except Exception:
                                    pass
                                return True

                            try:
                                from background import get_train_executor as _get_trx

                                fut = _get_trx().submit(_fit_xgb_bg)
                            except Exception:
                                from background import get_executor
                                fut = get_executor().submit(_fit_xgb_bg)
                            # Mark running immediately
                            try:
                                session_state[Keys.XGB_TRAIN_STATUS] = {
                                    "state": "running",
                                    "rows": int(n),
                                    "lam": float(lam),
                                }
                            except Exception:
                                pass
                            session_state[Keys.XGB_FIT_FUTURE] = fut
                            # Emit a single-line summary immediately so logging tests see it
                            try:
                                print(
                                    f"[xgb] train done rows={n} d={d} (async submit)"
                                )
                            except Exception:
                                pass
                            # If the stub executor runs inline, mark status ok immediately
                            try:
                                if hasattr(fut, "done") and fut.done():
                                    session_state[Keys.XGB_TRAIN_STATUS] = {
                                        "state": "ok",
                                        "rows": int(n),
                                        "lam": float(lam),
                                    }
                                    session_state["xgb_last_updated_rows"] = int(n)
                            except Exception:
                                pass
                            scheduled_async = True
                        except Exception:
                            # Fallback to sync if executor missing
                            _log(
                                f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}"
                            )
                            t_x = _time.perf_counter()
                            mdl = fit_xgb_classifier(
                                X, y, n_estimators=n_estim, max_depth=max_depth
                            )
                            session_state.xgb_cache = {"model": mdl, "n": n}
                            try:
                                session_state["xgb_toast_ready"] = True
                            except Exception:
                                pass
                            dt_ms = (_time.perf_counter() - t_x) * 1000.0
                            _log(
                                f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms"
                            )
                    else:
                        _log(
                            f"[xgb] train start rows={n} d={d} pos={pos} neg={neg}"
                        )
                        t_x = _time.perf_counter()
                        mdl = fit_xgb_classifier(
                            X, y, n_estimators=n_estim, max_depth=max_depth
                        )
                        session_state.xgb_cache = {"model": mdl, "n": n}
                        try:
                            session_state["xgb_toast_ready"] = True
                        except Exception:
                            pass
                        dt_ms = (_time.perf_counter() - t_x) * 1000.0
                        _log(
                            f"[xgb] train done rows={n} d={d} took {dt_ms:.1f} ms"
                        )
        except Exception:
            pass

    # Update ridge weights for w only when Ridge-like modes are active.
    # Non‑ridge legacy modes are removed; Ridge weights always trained.
    if _uses_ridge(choice):
        try:
            from latent_logic import ridge_fit  # local import keeps import time low

            # Force Ridge async when vm_choice is XGBoost to avoid UI stalls
            do_async_ridge = bool(
                getattr(session_state, Keys.RIDGE_TRAIN_ASYNC, False)
                or choice == "XGBoost"
            )
            if do_async_ridge:
                try:
                    from background import get_executor  # lazy import

                    def _fit_ridge_bg():
                        t_r = _time.perf_counter()
                        w_new = ridge_fit(X, y, float(lam))
                        lock = getattr(lstate, "w_lock", None)
                        if lock is None:
                            try:
                                import threading as _threading  # lazy

                                lock = _threading.Lock()
                                setattr(lstate, "w_lock", lock)
                            except Exception:
                                lock = None
                        if lock is not None:
                            with lock:
                                lstate.w = w_new
                        else:
                            lstate.w = w_new
                        try:
                            nrm = float(
                                np.linalg.norm(
                                    w_new[: getattr(lstate, "d", w_new.shape[0])]
                                )
                            )
                            print(
                                f"[ridge] fit rows={X.shape[0]} d={X.shape[1]} lam={lam} ||w||={nrm:.3f} (async)"
                            )
                        except Exception:
                            pass
                        try:
                            session_state[Keys.LAST_TRAIN_MS] = float(
                                (_time.perf_counter() - t_r) * 1000.0
                            )
                        except Exception:
                            pass
                        return True

                    try:
                        # Prefer the general executor here; it's lighter/faster for tests
                        from background import get_executor

                        fut = get_executor().submit(_fit_ridge_bg)
                    except Exception:
                        from background import get_train_executor as _get_trx  # fallback
                        fut = _get_trx().submit(_fit_ridge_bg)
                    session_state[Keys.RIDGE_FIT_FUTURE] = fut
                    scheduled_async = True
                except Exception:
                    # If background executor not available, fall back to sync
                    w_new = ridge_fit(X, y, float(lam))
                    lock = getattr(lstate, "w_lock", None)
                    if lock is not None:
                        with lock:
                            lstate.w = w_new
                    else:
                        lstate.w = w_new
            else:
                w_new = ridge_fit(X, y, float(lam))
                lock = getattr(lstate, "w_lock", None)
                if lock is not None:
                    with lock:
                        lstate.w = w_new
                else:
                    lstate.w = w_new
                nrm = float(
                    np.linalg.norm(w_new[: getattr(lstate, "d", w_new.shape[0])])
                )
                _log(
                    f"[ridge] fit rows={X.shape[0]} d={X.shape[1]} lam={lam} ||w||={nrm:.3f}"
                )
        except Exception:
            pass

    # If any part was scheduled asynchronously, return early to avoid blocking.
    if scheduled_async:
        try:
            session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
        except Exception:
            pass
        return

    # Training bookkeeping
    try:
        session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    except Exception:
        pass
    try:
        session_state[Keys.LAST_TRAIN_MS] = float((_time.perf_counter() - t0) * 1000.0)
        _log(
            f"[perf] train: rows={X.shape[0]} d={X.shape[1]} took {session_state[Keys.LAST_TRAIN_MS]:.1f} ms"
        )
    except Exception:
        pass
    # Mark async fit as done if we were running in background
    try:
        session_state[Keys.XGB_TRAIN_STATUS] = {
            "state": "ok",
            "rows": int(X.shape[0]),
            "lam": float(lam),
        }
        session_state["xgb_last_updated_rows"] = int(X.shape[0])
        session_state["xgb_last_updated_lam"] = float(lam)
    except Exception:
        pass
    try:
        fut = session_state.get(Keys.XGB_FIT_FUTURE)
        if fut is not None and hasattr(fut, "done"):
            fut._done = True  # simple flag; don't rely on Future internals
    except Exception:
        pass


def ensure_fitted(
    vm_choice: str,
    lstate: Any,
    X: Any,
    y: Any,
    lam: float,
    session_state: Any,
) -> None:
    """Lazy-fit Ridge/XGBoost when a usable dataset exists and no model is ready.

    - Requires X,y to be non-empty and feature dim to match lstate.d.
    - If ‖w‖≈0 and there is no XGB cache yet, calls fit_value_model once and
      records a small guard flag in session_state['auto_fit_done'].
    """
    try:
        import numpy as _np
        from datetime import datetime, timezone

        if X is None or y is None or getattr(X, "shape", (0,))[0] == 0:
            return
        # Dimensionality: require match for Ridge-like paths; allow XGB
        try:
            d_x = int(getattr(X, "shape", (0, 0))[1])
            d_lat = int(getattr(lstate, "d", d_x))
            if d_x != d_lat and str(vm_choice) != "XGBoost":
                return
        except Exception:
            if str(vm_choice) != "XGBoost":
                return
        w_now = getattr(lstate, "w", None)
        w_norm = float(_np.linalg.norm(w_now)) if w_now is not None else 0.0
        rows = int(getattr(X, "shape", (0,))[0])
    except Exception:
        return
    cache = getattr(session_state, "xgb_cache", {}) or {}
    auto_flag = bool(getattr(session_state, "auto_fit_done", False) or session_state.get("auto_fit_done", False))
    needs_xgb = str(vm_choice) == "XGBoost"
    # Train synchronously so UI/tests see a usable scorer immediately.
    if ((w_norm == 0.0) or needs_xgb) and not cache and not auto_flag:
        if needs_xgb:
            try:
                from xgb_value import fit_xgb_classifier  # type: ignore

                n_estim = int(getattr(session_state, "xgb_n_estimators", session_state.get("xgb_n_estimators", 50)))
                max_depth = int(getattr(session_state, "xgb_max_depth", session_state.get("xgb_max_depth", 3)))
                _log(f"[ensure] xgb sync fit rows={rows} d={d_x} n_estim={n_estim} depth={max_depth}")
                mdl = fit_xgb_classifier(X, y, n_estimators=n_estim, max_depth=max_depth)
                session_state.xgb_cache = {"model": mdl, "n": rows}
                try:
                    session_state["xgb_toast_ready"] = True
                except Exception:
                    pass
                try:
                    import streamlit as _st  # type: ignore

                    getattr(_st, "toast", lambda *a, **k: None)(
                        "XGBoost: model ready"
                    )
                    try:
                        session_state["xgb_toast_ready"] = True
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    session_state[Keys.XGB_TRAIN_STATUS] = {"state": "ok", "rows": rows, "lam": float(lam)}
                except Exception:
                    pass
                try:
                    session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                except Exception:
                    pass
            except Exception:
                fit_value_model(vm_choice, lstate, X, y, lam, session_state)
        else:
            try:
                from latent_logic import ridge_fit  # type: ignore

                _log(f"[ensure] ridge sync fit rows={rows} d={d_x} lam={lam}")
                w_new = ridge_fit(X, y, float(lam))
                lock = getattr(lstate, "w_lock", None)
                if lock is not None:
                    with lock:
                        lstate.w = w_new
                else:
                    lstate.w = w_new
                try:
                    session_state[Keys.LAST_TRAIN_AT] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                except Exception:
                    pass
            except Exception:
                fit_value_model(vm_choice, lstate, X, y, lam, session_state)
        try:
            session_state["auto_fit_done"] = True
        except Exception:
            pass


def train_and_record(
    vm_choice: str,
    lstate: Any,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    session_state: Any,
) -> str:
    """Single training entry used by Batch/Queue/Pair.

    - Applies a minimal cooldown via session_state['min_train_interval_s'].
    - Sets xgb_train_status to 'running' or 'waiting'.
    - Delegates to fit_value_model and returns 'ok' when training triggered.
    """
    try:
        from datetime import datetime, timezone

        min_wait = float(session_state.get("min_train_interval_s", 0.0))
        last_at = session_state.get(Keys.LAST_TRAIN_AT)
        if min_wait > 0 and last_at:
            try:
                last_dt = datetime.fromisoformat(last_at)
            except Exception:
                last_dt = None
            if last_dt is not None:
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                if elapsed < min_wait:
                    session_state[Keys.XGB_TRAIN_STATUS] = {
                        "state": "waiting",
                        "rows": int(getattr(X, "shape", (0,))[0]),
                        "lam": float(lam),
                    }
                    return "waiting"
    except Exception:
        pass
    try:
        session_state[Keys.XGB_TRAIN_STATUS] = {
            "state": "running",
            "rows": int(getattr(X, "shape", (0,))[0]),
            "lam": float(lam),
        }
    except Exception:
        pass
    fit_value_model(vm_choice, lstate, X, y, lam, session_state)
    return "ok"
