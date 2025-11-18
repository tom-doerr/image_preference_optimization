from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from typing import Any

import numpy as np


def fit_value_model(vm_choice: str,
                    lstate: Any,
                    X: np.ndarray,
                    y: np.ndarray,
                    lam: float,
                    session_state: Any) -> None:
    """Fit/update the value model artifacts with minimal logic.

    - Always fits Ridge to update lstate.w (keeps proposals simple and fast).
    - If vm_choice == 'XGBoost', also (re)fit and cache an XGB model when
      row count changes and both classes are present. Scores are consumed via
      value_scorer.get_value_scorer.
    - Records last_train_at and last_train_ms in session_state.
    """
    t0 = _time.perf_counter()

    # Optional XGB cache refresh
    if str(vm_choice) == 'XGBoost':
        try:
            from xgb_value import fit_xgb_classifier  # type: ignore
            n = int(X.shape[0])
            if n > 0 and len(set(np.asarray(y).astype(int).tolist())) > 1:
                cache = getattr(session_state, 'xgb_cache', {}) or {}
                last_n = int(cache.get('n') or 0)
                if cache.get('model') is None or last_n != n:
                    mdl = fit_xgb_classifier(X, y)
                    session_state.xgb_cache = {'model': mdl, 'n': n}
        except Exception:
            pass

    # Always update ridge weights for w
    try:
        from latent_logic import ridge_fit  # local import keeps import time low
        lstate.w = ridge_fit(X, y, float(lam))
    except Exception:
        pass

    # Training bookkeeping
    try:
        session_state['last_train_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    except Exception:
        pass
    try:
        session_state['last_train_ms'] = float((_time.perf_counter() - t0) * 1000.0)
        print(f"[perf] train: rows={X.shape[0]} d={X.shape[1]} took {session_state['last_train_ms']:.1f} ms")
    except Exception:
        pass

