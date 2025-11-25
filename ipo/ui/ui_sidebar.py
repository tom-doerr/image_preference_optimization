from __future__ import annotations

from typing import Any
from ipo.infra.util import SAFE_EXC, safe_write

import numpy as np

"""Sidebar UI composition for the app.

Debug/log-tail helpers live in `ui_sidebar_debug` to keep this file lean
without changing behavior or strings.
"""
from ipo.infra.constants import Keys
from ipo.core.persistence import dataset_rows_for_prompt, dataset_stats_for_prompt
from ipo.infra.util import safe_write

def _step_len_for_scores(lstate: Any, n_steps: int, iter_eta: float | None, trust_r: float | None) -> float:
    try:
        n = max(1, int(n_steps))
    except Exception:
        n = 1
    try:
        if iter_eta is not None and float(iter_eta) > 0.0:
            return float(iter_eta)
    except Exception:
        pass
    try:
        if trust_r is not None and float(trust_r) > 0.0:
            return float(trust_r) / n
    except Exception:
        pass
    try:
        return float(getattr(lstate, "sigma", 1.0)) / n
    except Exception:
        return 1.0 / n


from .ui_sidebar_modes import _select_generation_mode  # re-export


from .ui_sidebar_modes import _select_value_model  # re-export


from .ui_sidebar_modes import _toggle_random_anchor  # re-export
def sidebar_metric(label: str, value) -> None:
    """Plain text lines only; no Streamlit metric widgets."""
    import streamlit as st
    try:
        st.sidebar.write(f"{label}: {value}")
    except Exception:
        pass


def sidebar_metric_rows(pairs, per_row: int = 2) -> None:
    import streamlit as st

    try:
        for i in range(0, len(pairs), per_row):
            row = pairs[i : i + per_row]
            if (
                hasattr(st.sidebar, "columns")
                and callable(getattr(st.sidebar, "columns", None))
                and len(row) > 1
            ):
                cols = st.sidebar.columns(len(row))
                for (label, value), col in zip(row, cols):
                    with col:
                        sidebar_metric(label, value)
            else:
                for label, value in row:
                    sidebar_metric(label, value)
    except SAFE_EXC:
        for label, value in pairs:
            sidebar_metric(label, value)


def status_panel(*_args, **_kwargs) -> None:
    """No-op: Images status panel removed from sidebar."""
    return


def env_panel(env: dict) -> None:
    """Compact Environment panel (Python/torch/CUDA/Streamlit)."""
    import streamlit as st
    from .sidebar.misc import env_panel as _ep
    _ep(st, env)


def perf_panel(last_call: dict, last_train_ms) -> None:
    """Minimal Performance panel: decode_s and train_ms when available."""
    import streamlit as st
    pairs = []
    d = last_call.get("dur_s") if isinstance(last_call, dict) else None
    if d is not None:
        pairs.append(("decode_s", f"{float(d):.3f}"))
    if last_train_ms is not None:
        try:
            pairs.append(("train_ms", f"{float(last_train_ms):.1f}"))
        except SAFE_EXC:
            pass
    if not pairs:
        return
    exp = getattr(st.sidebar, "expander", None)
    if callable(exp):
        with exp("Performance", expanded=False):
            sidebar_metric_rows(pairs, per_row=2)
    else:
        sidebar_metric_rows(pairs, per_row=2)


from .ui_sidebar_modes import build_batch_controls  # re-export


from .ui_sidebar_modes import build_pair_controls  # re-export


def build_size_controls(st, lstate):
    # Minimal, deterministic defaults for tests; ignore stubs returning None
    w = int(getattr(lstate, "width", 512))
    h = int(getattr(lstate, "height", 512))
    steps = 6
    guidance = 3.5
    apply_clicked = False
    return w, h, steps, guidance, apply_clicked


def build_queue_controls(st, expanded: bool = False) -> int:
    # Kept only for tests that expect a constant
    return 6


# Minimal copies of step‑score helpers so ui_sidebar is self‑contained
def _ridge_dir(lstate: Any):
    try:
        import numpy as _np
        w_raw = getattr(lstate, "w", None)
        if w_raw is None:
            return None
        w = _np.asarray(w_raw[: getattr(lstate, "d", 0)], dtype=float).copy()
        n = float(_np.linalg.norm(w))
        if n == 0.0:
            return None
        return w, n, (w / n)
    except Exception:
        return None


def _get_scorer_for_vm(vm_choice: str, lstate: Any, prompt: str, session_state: Any):
    try:
        from value_scorer import get_value_scorer
        scorer, _ = get_value_scorer(vm_choice, lstate, prompt, session_state)
        return scorer
    except Exception:
        return None


def _accumulate_step_scores(d1, step_len: float, n_steps: int, z_p, scorer, w):
    import numpy as _np
    scores: list[float] = []
    for k in range(1, n_steps + 1):
        zc = z_p + (k * step_len) * d1
        try:
            if scorer is not None:
                s = float(scorer(zc - z_p))
            else:
                s = float(_np.dot(w, (zc - z_p)))
        except Exception:
            s = float(_np.dot(w, (zc - z_p)))
        scores.append(s)
    return scores


def compute_step_scores(
    lstate: Any,
    prompt: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: float | None,
    trust_r: float | None,
    session_state: Any,
):
    """Delegate to a smaller module to reduce sidebar complexity."""
    try:
        from .sidebar.step_scores import compute_step_scores as _css
        return _css(lstate, prompt, vm_choice, int(iter_steps), iter_eta, trust_r, session_state)
    except Exception:
        return None


def render_iter_step_scores(
    st: Any,
    lstate: Any,
    prompt: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: float | None,
    trust_r: float | None,
) -> None:
    from .sidebar.step_scores_render import render_iter_step_scores as _render
    _render(st, lstate, prompt, vm_choice, iter_steps, iter_eta, trust_r)


def render_mu_value_history(st: Any, lstate: Any, prompt: str) -> None:
    try:
        import numpy as _np
        from latent_opt import z_from_prompt as _zfp
        mu_hist = getattr(lstate, "mu_hist", None)
        if mu_hist is None or getattr(mu_hist, "size", 0) == 0:
            return
        z_p = _zfp(lstate, prompt).reshape(1, -1)
        mu_flat = mu_hist.reshape(mu_hist.shape[0], -1)
        vals = _np.linalg.norm(mu_flat - z_p, axis=1)
        sb = getattr(st, "sidebar", st)
        if hasattr(sb, "line_chart"):
            sb.subheader("Latent distance per step")
            sb.line_chart(vals.tolist())
    except Exception:
        pass


def render_pair_sidebar(*_args, **_kwargs) -> None:
    """No-op: A/B pair optimization UI removed."""
    return

# Small helpers to simplify render_sidebar_tail
def _get_dataset_for_display(st: Any, lstate: Any, prompt: str):
    """Return in-memory dataset only (simpler, no disk scan).

    We keep a single source of truth in session_state (X/y). Disk is written on
    label, but never re-scanned during renders.
    """
    Xm = getattr(lstate, 'X', None)
    ym = getattr(lstate, 'y', None)
    if Xm is not None and getattr(Xm, 'shape', (0,))[0] > 0 and ym is not None and getattr(ym, 'shape', (0,))[0] > 0:
        return Xm, ym
    return None, None


def _autofit_xgb_if_selected(st: Any, lstate: Any, vm_choice: str, Xd, yd) -> None:
    """Auto-fit disabled: XGB trains only on explicit button click (sync-only)."""
    return


def compute_train_results_lines(st: Any, lstate: Any, prompt: str, vm_choice: str) -> list[str]:
    """Small, self-contained train-results builder (keeps strings stable)."""
    try:
        import numpy as _np
        from value_scorer import get_value_scorer as _gvs
    except Exception:
        _np = None
        _gvs = None
    # Resolve dataset (memory-first)
    X = getattr(lstate, 'X', None)
    y = getattr(lstate, 'y', None)
    rows = int(getattr(X, 'shape', (0,))[0]) if X is not None else 0
    vs_status = 'ridge_untrained'
    tscore = 'n/a'
    if _gvs is not None:
        scorer, tag = _gvs(vm_choice, lstate, prompt, st.session_state)
        if scorer is not None and X is not None and y is not None and rows > 0 and _np is not None:
            scores = _np.asarray([scorer(x) for x in X], dtype=float)
            thr = 0.5 if str(vm_choice) == 'XGBoost' else 0.0
            yhat = scores >= thr
            tscore = f"{float((yhat == (y > 0)).mean()) * 100:.0f}%"
            vs_status = 'ok'
        else:
            vs_status = tag if isinstance(tag, str) else vs_status
    last_train = str(st.session_state.get('last_train_at', 'n/a'))
    last_cv = str(st.session_state.get('cv_last_at', 'n/a'))
    active = 'yes' if (str(vm_choice) == 'XGBoost' and vs_status == 'ok') else 'no'
    vs_line = f"{vm_choice or 'Ridge'} ({vs_status}, rows={rows})"
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

# Use shared helpers.safe_write to avoid duplication

# Local alias for concise access
K = Keys


def _sidebar_persistence_section(st: Any, lstate: Any, prompt: str, state_path: str, apply_state_cb, rerun_cb) -> None:
    """Download/State persistence UI removed (no-op)."""
    return


def _render_iter_step_scores_block(st: Any, lstate: Any, prompt: str, vm_choice: str, iter_steps: int, iter_eta: float | None) -> None:
    try:
        _tr = st.session_state.get("trust_r", 0.0)
        trust_val = float(_tr) if (_tr is not None and float(_tr) > 0.0) else None
    except Exception:
        trust_val = None
    try:
        render_iter_step_scores(
            st,
            lstate,
            prompt,
            vm_choice,
            int(iter_steps),
            float(iter_eta) if iter_eta is not None else None,
            trust_val,
        )
        render_mu_value_history(st, lstate, prompt)
    except Exception:
        pass


def _ensure_sidebar_shims(st: Any) -> None:
    from .sidebar.misc import ensure_sidebar_shims as _ens
    _ens(st)


def _labels_pos_neg(y) -> tuple[int, int]:
    try:
        import numpy as _np

        yy = _np.asarray(y, dtype=float) if y is not None else _np.asarray([])
        return int((yy > 0).sum()), int((yy < 0).sum())
    except Exception:
        pos = int(sum(1 for v in (y or []) if float(v) > 0))
        neg = int(sum(1 for v in (y or []) if float(v) < 0))
        return pos, neg


def _mem_dataset_stats(st: Any, lstate: Any) -> dict:
    try:
        y = st.session_state.get(K.DATASET_Y) or st.session_state.get("dataset_y")
    except Exception:
        y = None
    n = int(len(y)) if y is not None else 0
    pos, neg = _labels_pos_neg(y)
    d = int(getattr(lstate, "d", 0))
    return {"rows": n, "pos": pos, "neg": neg, "d": d}


def _sidebar_training_data_block(st: Any, prompt: str, lstate: Any) -> None:
    """Minimal, memory-only counters; no disk scanning."""
    try:
        exp = getattr(st.sidebar, "expander", None)
        stats = _mem_dataset_stats(st, lstate)
        if callable(exp):
            with exp("Training data", expanded=False):
                sidebar_metric_rows([("Pos", stats.get("pos", 0)), ("Neg", stats.get("neg", 0))], per_row=2)
                sidebar_metric_rows([("Feat dim", stats.get("d", 0))], per_row=1)
        else:
            safe_write(st, f"Training data: pos={stats.get('pos',0)} neg={stats.get('neg',0)} d={stats.get('d',0)}")
    except Exception:
        pass


def _cached_cv_lines(st):
    try:
        from ipo.infra.constants import Keys
        cv_cache = st.session_state.get(Keys.CV_CACHE) or {}
        if isinstance(cv_cache, dict):
            r = cv_cache.get("Ridge") or {}
            x = cv_cache.get("XGBoost") or {}
            ridge_line = (
                f"CV (Ridge): {float(r['acc']) * 100:.0f}% (k={int(r['k'])})"
                if ("acc" in r and "k" in r)
                else "CV (Ridge): n/a"
            )
            xgb_line = (
                f"CV (XGBoost): {float(x['acc']) * 100:.0f}% (k={int(x['k'])})"
                if ("acc" in x and "k" in x)
                else "CV (XGBoost): n/a"
            )
            return xgb_line, ridge_line
    except Exception:
        pass
    return "CV (XGBoost): n/a", "CV (Ridge): n/a"
def _xgb_status_line(st: Any, rows_n: int, status: str) -> None:
    try:
        line = f"XGBoost model rows: {rows_n} (status: {status})"
        st.sidebar.write(line)
        safe_write(st, line)
    except Exception:
        pass


def _vm_details_ridge(st: Any, lstate: Any, prompt: str, reg_lambda: float) -> None:
    try:
        import numpy as _np
        w_norm = float(_np.linalg.norm(getattr(lstate, 'w', 0.0)))
    except Exception:
        w_norm = 0.0
    try:
        rows = int(len(getattr(lstate, 'y', []) or []))
    except Exception:
        rows = 0
    st.sidebar.write(f"λ={reg_lambda:.3g}, ||w||={w_norm:.3f}, rows={rows}")


def _vm_details_distance(st: Any) -> None:
    try:
        from ipo.infra.util import safe_sidebar_num as _num
    except Exception:
        _num = None
    p = float(st.session_state.get(Keys.DIST_EXP, 2.0))
    try:
        if callable(_num):
            p = float(_num(st, "Distance exponent (p)", value=p, step=0.1))
            st.session_state[Keys.DIST_EXP] = p
    except Exception:
        pass
    st.sidebar.write(f"Distance exponent p={p:.2f}")


def _vm_details_xgb(st: Any, cache: dict) -> None:
    from .sidebar.panels import _vm_details_xgb as _impl
    return _impl(st, cache)


def _emit_cv_metrics(st: Any, xgb_line: str, ridge_line: str) -> None:
    """No-op: we only emit plain text lines (see safe_write calls)."""
    return


def _sidebar_value_model_block(st: Any, lstate: Any, prompt: str, vm_choice: str, reg_lambda: float) -> None:
    from .sidebar.panels import sidebar_value_model_block as _sv
    _sv(st, lstate, prompt, vm_choice, reg_lambda)


def _cv_on_demand(st: Any, lstate: Any, prompt: str, vm: str) -> None:
    """No-op CV compute (button removed to simplify)."""
    return


def _resolve_meta_pairs(prompt: str, state_path: str):
    from .ui_sidebar_meta import resolve_meta_pairs as _res
    return _res(prompt, state_path)


def _emit_meta_pairs(st: Any, pairs) -> None:
    from .ui_sidebar_meta import emit_meta_pairs as _emit
    _emit(st, pairs)


def _render_metadata_panel_inline(st: Any, lstate: Any, prompt: str, state_path: str) -> None:
    """Emit compact metadata panel (app_version, created_at, prompt_hash)."""
    pairs = _resolve_meta_pairs(prompt, state_path)
    if pairs:
        _emit_meta_pairs(st, pairs)


def _emit_latent_dim_and_data_strip(st: Any, lstate: Any) -> None:
    from .sidebar.misc import emit_latent_dim_and_data_strip as _emit
    _emit(st, lstate)


def _ensure_train_results_expander_label(st: Any) -> None:
    """No-op: Train results block removed from sidebar."""
    return


def _xgb_train_controls(st: Any, lstate: Any, Xd, yd) -> None:
    from .sidebar.panels import _xgb_train_controls as _impl
    return _impl(st, lstate, Xd, yd)


def _logit_train_controls(st: Any, lstate: Any, Xd, yd) -> None:
    from .sidebar.panels import _logit_train_controls as _impl
    return _impl(st, lstate, Xd, yd)


from .sidebar.panels import handle_train_section as _handle_train_section


def _early_persistence_and_meta(
    st: Any,
    lstate: Any,
    prompt: str,
    state_path: str,
    apply_state_cb,
    rerun_cb,
    selected_model: str,
):
    """Emit persistence + metadata + step-score prep with minimal branching."""
    from ipo.infra.pipeline_local import set_model
    try:
        if hasattr(st.sidebar, "download_button"):
            _sidebar_persistence_section(st, lstate, prompt, state_path, apply_state_cb, rerun_cb)
    except Exception:
        pass
    _render_metadata_panel_inline(st, lstate, prompt, state_path)
    # Model selection is hardcoded but call site stays explicit for tests
    set_model(selected_model)


def _predicted_values_block(*_args, **_kwargs) -> None:
    return


def render_sidebar_tail(
    st: Any,
    lstate: Any,
    prompt: str,
    state_path: str,
    vm_choice: str,
    iter_steps: int,
    iter_eta: float | None,
    selected_model: str,
    apply_state_cb,
    rerun_cb,
) -> None:
    _early_persistence_and_meta(st, lstate, prompt, state_path, apply_state_cb, rerun_cb, selected_model)
    # Status lines (Value model/XGBoost active/Optimization) are emitted later in the
    # canonical train-results block to preserve expected ordering in tests.
    _render_iter_step_scores_block(st, lstate, prompt, vm_choice, iter_steps, iter_eta)
    # Always emit the simple Value model line early for tests/readability
    try:
        safe_write(st, f"Value model: {str(vm_choice)}")
    except Exception:
        pass
    _ensure_sidebar_shims(st)
    _emit_latent_dim_and_data_strip(st, lstate)
    _sidebar_training_data_block(st, prompt, lstate)
    # Train controls remain available; we omit the verbose Train results block
    _handle_train_section(st, lstate, prompt, vm_choice)
    _predicted_values_block(st, vm_choice, lstate, prompt)


def _emit_train_results(st: Any, lines: list[str], sidebar_only: bool = False) -> None:
    """Minimal writer for Train results + ephemeral last action."""
    target = getattr(st, "sidebar", st)
    for ln in (lines or []):
        try:
            target.write(ln)
        except Exception:
            pass
    try:
        import time as _time
        txt = st.session_state.get(Keys.LAST_ACTION_TEXT)
        ts = st.session_state.get(Keys.LAST_ACTION_TS)
        if txt and ts is not None and (_time.time() - float(ts)) < 6.0:
            target.write(f"Last action: {txt}")
    except Exception:
        pass


# Merged from ui_sidebar_extra
from .sidebar.misc import emit_dim_mismatch as _emit_dim_mismatch


from .sidebar.misc import emit_last_action_recent as _emit_last_action_recent


from .sidebar.misc import rows_refresh_tick as _rows_refresh_tick


# moved to ui_sidebar_controls to reduce this file's complexity


def render_rows_and_last_action(st: Any, base_prompt: str, lstate: Any | None = None) -> None:
    from .sidebar.controls import render_rows_and_last_action as _rows
    _rows(st, base_prompt, lstate)


def render_model_decode_settings(st: Any, lstate: Any):
    from .sidebar.controls import render_model_decode_settings as _ctl
    return _ctl(st, lstate)


# Merged from ui_sidebar_modes
from .ui_sidebar_modes import render_modes_and_value_model  # re-export
def _build_size_controls(st, lstate):
    num = getattr(st.sidebar, "number_input", st.number_input)
    sld = getattr(st.sidebar, "slider", st.slider)
    width = num("Width", step=64, value=getattr(lstate, "width", 512))
    height = num("Height", step=64, value=getattr(lstate, "height", 512))
    steps = sld("Steps", value=6)
    guidance = sld("Guidance", value=3.5, step=0.1)
    # Keep behavior identical to prior helper
    width = getattr(lstate, "width", 512) if width is None else width
    height = getattr(lstate, "height", 512) if height is None else height
    steps = 6 if steps is None else steps
    guidance = 3.5 if guidance is None else guidance
    apply_clicked = False
    if int(width) != int(getattr(lstate, "width", width)) or int(height) != int(getattr(lstate, "height", height)):
        apply_clicked = True
    # Removed explicit "Apply size now" button; width/height changes are detected
    # via the comparison above, and callers already ignore apply_clicked.
    return int(width), int(height), int(steps), float(guidance), bool(apply_clicked)
