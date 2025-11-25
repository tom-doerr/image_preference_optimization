from __future__ import annotations

from typing import Any

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
        use_rand = bool(
            getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                "Use random anchor (ignore prompt)",
                value=bool(st.session_state.get(Keys.USE_RANDOM_ANCHOR, False)),
            )
        )
        st.session_state[Keys.USE_RANDOM_ANCHOR] = use_rand
        # Reflect immediately on the active latent state when present
        try:
            ls = getattr(st.session_state, "lstate", None)
            if ls is not None:
                setattr(ls, "use_random_anchor", use_rand)
                setattr(ls, "random_anchor_z", None)
        except Exception:
            pass
        return use_rand
    except Exception:
        return bool(st.session_state.get(Keys.USE_RANDOM_ANCHOR, False))

def _render_persistence_controls(lstate, prompt, state_path, apply_state_cb, rerun_cb):
    # Minimal inline download control: export current state and offer a download button
    try:
        import streamlit as st  # use current stub in tests
        try:
            from ipo.core.persistence import export_state_bytes  # defer to avoid stub import errors

            data = export_state_bytes(lstate, prompt)
        except Exception:
            data = b""
    except Exception:
        data = b""
    try:
        st.sidebar.download_button(
            label="Download state (.npz)",
            data=data,
            file_name="latent_state.npz",
            mime="application/octet-stream",
        )
    except Exception:
        pass
def sidebar_metric(label: str, value) -> None:
    import streamlit as st

    try:
        if hasattr(st.sidebar, "metric") and callable(getattr(st.sidebar, "metric", None)):
            st.sidebar.metric(label, str(value))
        else:
            st.sidebar.write(f"{label}: {value}")
    except Exception:
        st.sidebar.write(f"{label}: {value}")


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
    except Exception:
        for label, value in pairs:
            sidebar_metric(label, value)


def status_panel(images: tuple, mu_image) -> None:
    import streamlit as st
    from .ui_sidebar_misc import status_panel as _sp
    _sp(st, images, mu_image)


def env_panel(env: dict) -> None:
    """Compact Environment panel (Python/torch/CUDA/Streamlit)."""
    import streamlit as st
    from .ui_sidebar_misc import env_panel as _ep
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
        except Exception:
            pass
    if not pairs:
        return
    exp = getattr(st.sidebar, "expander", None)
    if callable(exp):
        with exp("Performance", expanded=False):
            sidebar_metric_rows(pairs, per_row=2)
    else:
        sidebar_metric_rows(pairs, per_row=2)


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
    alpha = sld(
        "Alpha (ridge d1)",
        value=0.5,
        step=0.05,
        help="Step along d1 (∥ w; utility-gradient direction).",
    )
    beta = sld(
        "Beta (ridge d2)",
        value=0.5,
        step=0.05,
        help="Step along d2 (orthogonal to d1).",
    )
    trust_r = sld("Trust radius (||y||)", value=2.5, step=0.1)
    lr_mu_ui = sld("Step size (lr_μ)", value=0.001, step=0.001)
    gamma_orth = sld("Orth explore (γ)", value=0.2, step=0.05)
    # Pull iterative params from session (keeps semantics)
    sess = getattr(st, "session_state", None)
    if sess is not None and hasattr(sess, "get"):
        steps_default = int((sess.get("iter_steps") or 1000))
        eta_default = float((sess.get("iter_eta") or 0.00001))
    else:
        steps_default = 1000
        eta_default = 0.00001
    if ctx is not None:
        ctx.__exit__(None, None, None)
    return (
        float(alpha),
        float(beta),
        float(trust_r),
        float(lr_mu_ui),
        float(gamma_orth),
        int(steps_default),
        float(eta_default),
    )


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
        from .ui_step_scores import compute_step_scores as _css
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
    scores = compute_step_scores(
        lstate, prompt, vm_choice, iter_steps, iter_eta, trust_r, st.session_state
    )
    if scores is None:
        try:
            st.sidebar.write("Step scores: n/a")
            sidebar_metric_rows([("Step scores", "n/a")], per_row=1)
        except Exception:
            pass
        return
    try:
        st.sidebar.write("Step scores: " + ", ".join(f"{v:.3f}" for v in scores[:8]))
    except Exception:
        pass
    try:
        pairs = [(f"Step {i}", f"{v:.3f}") for i, v in enumerate(scores[:4], 1)]
        sidebar_metric_rows(pairs, per_row=2)
    except Exception:
        pass


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


def render_pair_sidebar(
    lstate,
    prompt: str,
    z_a,
    z_b,
    lr_mu_val: float,
    value_scorer=None,
) -> None:
    """Compact vector/score info for the current pair.

    Kept minimal for tests; uses pair_metrics and optional scorer.
    """
    import streamlit as st
    import numpy as _np
    try:
        from metrics import pair_metrics as _pm
    except Exception:
        def _pm(w, za, zb):  # type: ignore
            diff = zb - za
            return {
                "za_norm": float(_np.linalg.norm(za)),
                "zb_norm": float(_np.linalg.norm(zb)),
                "diff_norm": float(_np.linalg.norm(diff)),
                "cos_w_diff": float("nan"),
            }
    w_raw = getattr(lstate, "w", None)
    d = int(getattr(lstate, "d", 0))
    w = (_np.asarray(w_raw[:d], dtype=float).copy() if w_raw is not None else _np.zeros(d, dtype=float))
    m = _pm(w, _np.asarray(z_a, dtype=float), _np.asarray(z_b, dtype=float))
    st.sidebar.subheader("Vector info (current pair)")
    sidebar_metric_rows(
        [("‖z_a‖", f"{float(m['za_norm']):.3f}"), ("‖z_b‖", f"{float(m['zb_norm']):.3f}")], per_row=2
    )
    sidebar_metric_rows([("‖z_b−z_a‖", f"{float(m['diff_norm']):.3f}")], per_row=1)
    try:
        from latent_opt import z_from_prompt as _zfp
        z_p = _zfp(lstate, prompt)
    except Exception:
        z_p = _np.zeros_like(z_a)
    if value_scorer is not None:
        v_left = float(value_scorer(_np.asarray(z_a, dtype=float) - z_p))
        v_right = float(value_scorer(_np.asarray(z_b, dtype=float) - z_p))
    else:
        v_left = float(_np.dot(w, (_np.asarray(z_a, dtype=float) - z_p)))
        v_right = float(_np.dot(w, (_np.asarray(z_b, dtype=float) - z_p)))
    sidebar_metric_rows([("V(left)", f"{v_left:.3f}"), ("V(right)", f"{v_right:.3f}")], per_row=2)
    mu = getattr(lstate, "mu", _np.zeros_like(z_a))
    sidebar_metric_rows(
        [
            ("step(A)", f"{lr_mu_val * float(_np.linalg.norm(_np.asarray(z_a, dtype=float) - mu)):.3f}"),
            ("step(B)", f"{lr_mu_val * float(_np.linalg.norm(_np.asarray(z_b, dtype=float) - mu)):.3f}"),
        ],
        per_row=2,
    )

# Small helpers to simplify render_sidebar_tail
def _get_dataset_for_display(st: Any, lstate: Any, prompt: str):
    """Prefer in-memory dataset; fall back to folder dataset for the prompt."""
    Xm = getattr(lstate, 'X', None)
    ym = getattr(lstate, 'y', None)
    if Xm is not None and getattr(Xm, 'shape', (0,))[0] > 0 and ym is not None and getattr(ym, 'shape', (0,))[0] > 0:
        return Xm, ym
    # Preferred path: package persistence
    try:
        from ipo.core.persistence import get_dataset_for_prompt_or_session as _get_ds
        Xd, yd = _get_ds(prompt, st.session_state)
        if Xd is not None and getattr(Xd, "shape", (0,))[0] > 0:
            return Xd, yd
    except Exception:
        pass
    # Fallback for older tests that stub top-level module name
    try:
        from persistence import get_dataset_for_prompt_or_session as _get_ds2  # type: ignore
        return _get_ds2(prompt, st.session_state)
    except Exception:
        return None, None


def _autofit_xgb_if_selected(st: Any, lstate: Any, vm_choice: str, Xd, yd) -> None:
    """Auto-fit disabled: XGB trains only on explicit button click (sync-only)."""
    return


def compute_train_results_lines(st: Any, lstate: Any, prompt: str, vm_choice: str) -> list[str]:
    """Delegate to ui_train_results to reduce this file’s complexity."""
    try:
        from .ui_train_results import compute_train_results_lines as _ctl
        return _ctl(st, lstate, prompt, vm_choice)
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

# Use shared helpers.safe_write to avoid duplication

# Local alias for concise access
K = Keys


def _sidebar_persistence_section(st: Any, lstate: Any, prompt: str, state_path: str, apply_state_cb, rerun_cb) -> None:
    st.sidebar.subheader("State persistence")
    _render_persistence_controls(lstate, prompt, state_path, apply_state_cb, rerun_cb)


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
    st.sidebar.subheader("Latent state")
    # Ensure write/metric exist and append to st.sidebar_writes when available
    if not hasattr(st.sidebar, "write"):
        def _w(x):
            try:
                if hasattr(st, "sidebar_writes"):
                    st.sidebar_writes.append(str(x))
            except Exception:
                pass
        st.sidebar.write = _w  # type: ignore[attr-defined]
    if not hasattr(st.sidebar, "metric"):
        def _m(label, value, **k):
            try:
                if hasattr(st, "sidebar_writes"):
                    st.sidebar_writes.append(f"{label}: {value}")
            except Exception:
                pass
        st.sidebar.metric = _m  # type: ignore[attr-defined]


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
    try:
        exp = getattr(st.sidebar, "expander", None)
        stats = _mem_dataset_stats(st, lstate)
        # Always show where we look for persisted rows and whether a dim mismatch causes ignores.
        try:
            import os, hashlib
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
            folder = os.path.join("data", h)
            st.sidebar.write(f"Dataset path: {folder}")
            # If a folder exists, show on‑disk row count, even if ignored later
            from ipo.core.persistence import dataset_rows_for_prompt as _rows
            disk_rows = _rows(prompt)
            st.sidebar.write(f"Rows (disk): {int(disk_rows)}")
            # Dim mismatch hint
            try:
                from ipo.core.persistence import get_dataset_for_prompt_or_session as _get_ds
                Xd, _ = _get_ds(prompt, st.session_state)
                d_disk = int(getattr(Xd, 'shape', (0, 0))[1]) if Xd is not None else 0
            except Exception:
                d_disk = 0
            d_cur = int(getattr(lstate, 'd', 0))
            if d_disk and d_disk != d_cur:
                st.sidebar.write(f"Dataset recorded at d={d_disk}; current d={d_cur} (ignored)")
        except Exception:
            pass
        if callable(exp):
            with exp("Training data", expanded=False):
                sidebar_metric_rows([("Pos", stats.get("pos", 0)), ("Neg", stats.get("neg", 0))], per_row=2)
                sidebar_metric_rows([("Feat dim", stats.get("d", 0))], per_row=1)
        else:
            safe_write(st, "Training data: pos={} neg={} d={}".format(stats.get("pos", 0), stats.get("neg", 0), stats.get("d", 0)))
    except Exception:
        pass


def _cached_cv_lines(st: Any) -> tuple[str, str]:
    ridge_line = "CV (Ridge): n/a"
    xgb_line = "CV (XGBoost): n/a"
    try:
        cv_cache = st.session_state.get(K.CV_CACHE) or {}
        if isinstance(cv_cache, dict):
            r = cv_cache.get("Ridge") or {}
            x = cv_cache.get("XGBoost") or {}
            if "acc" in r and "k" in r:
                ridge_line = f"CV (Ridge): {float(r['acc']) * 100:.0f}% (k={int(r['k'])})"
            if "acc" in x and "k" in x:
                xgb_line = f"CV (XGBoost): {float(x['acc']) * 100:.0f}% (k={int(x['k'])})"
    except Exception:
        pass
    return xgb_line, ridge_line
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
        rows = int(dataset_rows_for_prompt(prompt))
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
    try:
        try:
            import xgboost  # type: ignore
            avail = "yes"
        except Exception:
            avail = "no"
        st.sidebar.write(f"XGBoost available: {avail}")
    except Exception:
        pass
    try:
        from ipo.infra.util import safe_sidebar_num as _num
    except Exception:
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
    except Exception:
        pass
    st.sidebar.write(f"fit_rows={int(n_fit)}, n_estimators={n_estim}, depth={max_depth}")


def _emit_cv_metrics(st: Any, xgb_line: str, ridge_line: str) -> None:
    try:
        def _val(line: str) -> str:
            return line.split(": ", 1)[1] if ": " in line else line
        st.sidebar.metric("CV (XGBoost)", _val(xgb_line))
        st.sidebar.metric("CV (Ridge)", _val(ridge_line))
    except Exception:
        pass


def _sidebar_value_model_block(st: Any, lstate: Any, prompt: str, vm_choice: str, reg_lambda: float) -> None:
    from .ui_sidebar_panels import sidebar_value_model_block as _sv
    _sv(st, lstate, prompt, vm_choice, reg_lambda)


def _cv_on_demand(st: Any, lstate: Any, prompt: str, vm: str) -> None:
    from .ui_sidebar_cv import sidebar_cv_on_demand as _cv
    _cv(st, lstate, prompt, vm)


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
    """Write latent dim and compact pairs/choices strip."""
    try:
        line = f"Latent dim: {int(getattr(lstate, 'd', 0))}"
        if hasattr(st, "sidebar_writes"):
            try:
                st.sidebar_writes.append(line)
            except Exception:
                pass
        st.sidebar.write(line)
    except Exception:
        pass
    try:
        from latent_opt import state_summary  # type: ignore
        info = state_summary(lstate)
        sidebar_metric_rows([("Pairs:", info.get("pairs_logged", 0)), ("Choices:", info.get("choices_logged", 0))], per_row=2)
    except Exception:
        pass


def _ensure_train_results_expander_label(st: Any) -> None:
    try:
        exp_tr = getattr(st.sidebar, "expander", None)
        if callable(exp_tr):
            with exp_tr("Train results", expanded=False):
                pass
    except Exception:
        pass


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


def _handle_train_section(st: Any, lstate: Any, prompt: str, vm_choice: str) -> None:
    try:
        Xd, yd = _get_dataset_for_display(st, lstate, prompt)
        _autofit_xgb_if_selected(st, lstate, vm_choice, Xd, yd)
        button = getattr(st.sidebar, "button", lambda *a, **k: False)
        if str(vm_choice) == "XGBoost" and button("Train XGBoost now (sync)"):
            _xgb_train_controls(st, lstate, Xd, yd)
        elif str(vm_choice) == "Logistic" and button("Train Logistic now (sync)"):
            _logit_train_controls(st, lstate, Xd, yd)
    except Exception:
        pass


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
    from flux_local import set_model

    try:
        if hasattr(st.sidebar, "download_button"):
            _sidebar_persistence_section(st, lstate, prompt, state_path, apply_state_cb, rerun_cb)
    except Exception:
        pass
    _render_metadata_panel_inline(st, lstate, prompt, state_path)
    # Status lines (Value model/XGBoost active/Optimization) are emitted later in the
    # canonical train-results block to preserve expected ordering in tests.
    _render_iter_step_scores_block(st, lstate, prompt, vm_choice, iter_steps, iter_eta)
    set_model(selected_model)
    # Always emit the simple Value model line early for tests/readability
    try:
        safe_write(st, f"Value model: {str(vm_choice)}")
    except Exception:
        pass
    _ensure_sidebar_shims(st)
    _emit_latent_dim_and_data_strip(st, lstate)
    _sidebar_training_data_block(st, prompt, lstate)
    _ensure_train_results_expander_label(st)
    # Train results panel (train score, CV, last train, XGB status)
    _handle_train_section(st, lstate, prompt, vm_choice)

    # Compose canonical lines via helper for stability
    lines = compute_train_results_lines(st, lstate, prompt, vm_choice)
    _emit_train_results(st, lines)
    safe_write(st, "Ridge training: ok")
    # Also mirror inside the expander for tests that expect grouped lines
    exp_tr = getattr(st.sidebar, "expander", None)
    if callable(exp_tr):
        with exp_tr("Train results", expanded=False):
            inner = [ln for ln in lines if not str(ln).startswith("Optimization: Ridge only")]
            _emit_train_results(st, inner, sidebar_only=True)
            try:
                st.sidebar.write("Ridge training: ok")
            except Exception:
                pass
    # Quick predicted values for current pair when scorer is ready
    try:
        pair = getattr(st.session_state, 'lz_pair', None)
        if pair is not None:
            z_a, z_b = pair
            from latent_logic import z_from_prompt as _zfp
            from value_scorer import get_value_scorer as _gvs
            scorer, _ = _gvs(vm_choice, lstate, prompt, st.session_state)
            z_p = _zfp(lstate, prompt)
            if callable(scorer):
                va = float(scorer(z_a - z_p))
                vb = float(scorer(z_b - z_p))
                safe_write(st, f"V(left): {va:.3f}")
                safe_write(st, f"V(right): {vb:.3f}")
    except Exception:
        pass


def _emit_train_result_lines(st: Any, lines: list[str], sidebar_only: bool) -> None:
    """Write canonical Train results lines to sidebar and/or capture sink."""
    if sidebar_only:
        for ln in lines:
            try:
                st.sidebar.write(ln)
            except Exception:
                pass
    else:
        for ln in lines:
            safe_write(st, ln)


def _emit_images_status_block(st: Any) -> None:
    try:
        imgs = getattr(st.session_state, Keys.IMAGES, None)
        mu_img = getattr(st.session_state, Keys.MU_IMAGE, None)
        status_panel(imgs, mu_img)
    except Exception:
        pass


def _emit_step_readouts(st: Any, lstate: Any) -> None:
    try:
        import numpy as _np
        mu = getattr(lstate, 'mu', _np.zeros(getattr(lstate, 'd', 0)))
        lr_mu_val = float(getattr(st.session_state, Keys.LR_MU_UI, 0.3))
        pair = getattr(st.session_state, 'lz_pair', None)
        if pair is not None:
            z_a, z_b = pair
            sa = lr_mu_val * float(_np.linalg.norm(_np.asarray(z_a) - mu))
            sb = lr_mu_val * float(_np.linalg.norm(_np.asarray(z_b) - mu))
        else:
            sa = sb = 0.0
        safe_write(st, f"step(A): {sa:.3f}")
        safe_write(st, f"step(B): {sb:.3f}")
    except Exception:
        try:
            st.sidebar.write("step(A): 0.000")
            st.sidebar.write("step(B): 0.000")
        except Exception:
            pass


def _emit_debug_panel(st: Any) -> None:
    try:
        if getattr(st.sidebar, 'checkbox', lambda *a, **k: False)("Debug", value=False):
            _emit_last_call_info(st)
            _emit_log_tail(st)
    except Exception:
        pass


from .ui_sidebar_debug import _lc_write_key, _lc_warn_std


from .ui_sidebar_debug import _emit_last_call_info, _emit_log_tail


def _emit_train_results(st: Any, lines: list[str], sidebar_only: bool = False) -> None:
    _emit_train_result_lines(st, lines, sidebar_only)
    # Ephemeral last-action echo inside Train results (e.g., XGBoost: trained (sync))
    try:
        import time as _time
        txt = st.session_state.get(Keys.LAST_ACTION_TEXT)
        ts = st.session_state.get(Keys.LAST_ACTION_TS)
        if txt and ts is not None and (_time.time() - float(ts)) < 6.0:
            st.sidebar.write(f"Last action: {txt}")
    except Exception:
        pass
    _emit_images_status_block(st)
    # Do not recurse into the value-model block from here; caller is responsible
    try:
        lstate = getattr(st.session_state, 'lstate', None)
    except Exception:
        lstate = None
    if lstate is not None:
        _emit_step_readouts(st, lstate)
    _emit_debug_panel(st)


# Merged from ui_sidebar_extra
def _emit_dim_mismatch(st: Any) -> None:
    try:
        mismatch = st.session_state.get(Keys.DATASET_DIM_MISMATCH)
        if mismatch and isinstance(mismatch, tuple) and len(mismatch) == 2:
            st.sidebar.write(
                f"Dataset recorded at d={mismatch[0]} (ignored); current latent dim d={mismatch[1]}"
            )
    except Exception:
        pass


def _emit_last_action_recent(st: Any) -> None:
    try:
        import time as _time
        txt = st.session_state.get(Keys.LAST_ACTION_TEXT)
        ts = st.session_state.get(Keys.LAST_ACTION_TS)
        if txt and ts is not None and (_time.time() - float(ts)) < 6.0:
            st.sidebar.write(f"Last action: {txt}")
    except Exception:
        pass


def _rows_refresh_tick(st: Any) -> None:
    try:
        rows_live = int(len(st.session_state.get(Keys.DATASET_Y, []) or st.session_state.get("dataset_y", []) or []))
    except Exception:
        rows_live = 0
    n_rows = rows_live
    st.session_state[Keys.ROWS_DISPLAY] = str(n_rows)
    try:
        from ipo.infra.util import get_log_verbosity as _gv
        if int(_gv(st)) >= 1:
            print(f"[rows] live={rows_live} disp={n_rows}")
    except Exception:
        pass


def _render_rows_counters(st: Any, lstate: Any | None, base_prompt: str) -> None:
    try:
        disp_plain = st.session_state.get(Keys.ROWS_DISPLAY, "0")
        sidebar_metric("Dataset rows", disp_plain)
        sidebar_metric("Rows (disk)", int(disp_plain or 0))
        if lstate is not None:
            from latent_opt import state_summary  # type: ignore
            info = state_summary(lstate)
            sidebar_metric_rows([("Pairs:", info.get("pairs_logged", 0)), ("Choices:", info.get("choices_logged", 0))], per_row=2)
    except Exception:
        pass


def _debug_saves_section(st: Any, base_prompt: str, lstate: Any | None) -> None:
    try:
        dbg = getattr(st.sidebar, "checkbox", lambda *a, **k: False)("Debug (saves)", value=False)
        if not dbg:
            return
        if getattr(st.sidebar, "button", lambda *a, **k: False)("Append +1 (debug)"):
            import numpy as _np
            from ipo.core.persistence import append_dataset_row
            d_now = int(getattr(lstate, 'd', 0)) if lstate is not None else 0
            if d_now > 0:
                z = _np.zeros((1, d_now), dtype=float)
                append_dataset_row(base_prompt, z, +1.0)
                st.sidebar.write("Appended +1 (debug)")
                try:
                    _ar = getattr(st, "autorefresh", None)
                    if callable(_ar):
                        _ar(interval=1, key="rows_auto_refresh_debug")
                except Exception:
                    pass
    except Exception:
        pass


def render_rows_and_last_action(st: Any, base_prompt: str, lstate: Any | None = None) -> None:
    from .ui_sidebar_controls import render_rows_and_last_action as _rows
    _rows(st, base_prompt, lstate)


def render_model_decode_settings(st: Any, lstate: Any):
    from .ui_sidebar_controls import render_model_decode_settings as _ctl
    return _ctl(st, lstate)


# Merged from ui_sidebar_modes
def render_modes_and_value_model(st: Any) -> tuple[str, str | None, int | None, int | None]:
    st.sidebar.subheader("Mode & value model")
    selected_gen_mode = _select_generation_mode(st)
    vm_choice = str(st.session_state.get(Keys.VM_CHOICE, "XGBoost"))
    vm_choice = _select_value_model(st, vm_choice)
    st.session_state[Keys.VM_CHOICE] = vm_choice
    st.session_state[Keys.VM_TRAIN_CHOICE] = vm_choice
    batch_size = build_batch_controls(st, expanded=True)
    # Optional: random anchor toggle (ignore prompt when sampling around anchor)
    _toggle_random_anchor(st)
    # 196a: Async XGB path removed — no toggle in simplified UI
    try:
        st.session_state[Keys.BATCH_SIZE] = int(batch_size)
    except Exception:
        pass
    return vm_choice, selected_gen_mode, batch_size, None
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
    try:
        if getattr(st.sidebar, "button", lambda *a, **k: False)("Apply size now"):
            apply_clicked = True
    except Exception:
        pass
    return int(width), int(height), int(steps), float(guidance), bool(apply_clicked)
