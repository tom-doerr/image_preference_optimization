from __future__ import annotations

import os
from typing import Any

from constants import Keys
from env_info import get_env_summary
from ui import env_panel, perf_panel, sidebar_metric_rows
from persistence_ui import render_metadata_panel


def render_sidebar_extras(st: Any, lstate: Any, base_prompt: str, is_turbo: bool, guidance_eff: float, selected_model: str) -> None:
    """Render environment/perf panels, state metrics, Debug expander, and metadata panel.

    Kept minimal to avoid behavior changes.
    """
    # Environment block
    try:
        _exp = getattr(st.sidebar, 'expander', None)
        if callable(_exp):
            with _exp("Environment", expanded=False):
                env_panel(get_env_summary())
        else:
            env_panel(get_env_summary())
    except Exception:
        env_panel(get_env_summary())

    # Performance panel
    try:
        try:
            from flux_local import get_last_call  # type: ignore
        except Exception:
            def get_last_call():
                return {}
        perf_panel(get_last_call() or {}, st.session_state.get('last_train_ms'))
    except Exception:
        pass

    # Latent state summary
    try:
        from latent_opt import state_summary  # type: ignore
        info = state_summary(lstate)
        pairs_state = [("Latent dim", f"{info['d']}")]
        pairs_state += [(k, f"{info[k]}") for k in ('width','height','step','sigma','mu_norm','w_norm','pairs_logged','choices_logged')]
        sidebar_metric_rows(pairs_state, per_row=2)
    except Exception:
        pass

    # Debug panel (collapsible): pipeline stats and OOM retry
    try:
        try:
            from flux_local import get_last_call  # type: ignore
        except Exception:
            def get_last_call():
                return {}
        expander = getattr(st.sidebar, 'expander', None)
        if callable(expander):
            with expander("Debug", expanded=False):
                last = get_last_call() or {}
                dbg_pairs = []
                try:
                    lat_depth = 4
                    lat_shape = f"1x{lat_depth}x{max(1, lstate.height//8)}x{max(1, lstate.width//8)}"
                    dbg_pairs.append(("latent_depth", str(lat_depth)))
                    dbg_pairs.append(("latent_shape", lat_shape))
                except Exception:
                    pass
                for k in ("model_id", "width", "height", "steps", "guidance", "latents_std", "init_sigma", "img0_std", "img0_min", "img0_max"):
                    if k in last and last[k] is not None:
                        dbg_pairs.append((k, str(last[k])))
                if is_turbo:
                    dbg_pairs.append(("guidance_eff", str(guidance_eff)))
                try:
                    retry_val = os.getenv("RETRY_ON_OOM", "0")
                    dbg_pairs.append(("RETRY_ON_OOM", retry_val))
                    toggle = getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                        "Enable OOM retry (env RETRY_ON_OOM)",
                        value=(retry_val not in ("0", "false", "False", "")),
                    )
                    os.environ["RETRY_ON_OOM"] = "1" if toggle else "0"
                except Exception:
                    pass
                if dbg_pairs:
                    sidebar_metric_rows(dbg_pairs, per_row=2)
                try:
                    ls = last.get('latents_std')
                    if ls is not None and float(ls) <= 1e-6:
                        st.sidebar.write('warn: latents std ~0')
                except Exception:
                    pass
        else:
            st.sidebar.subheader("Debug")
            last = get_last_call() or {}
            dbg_pairs = []
            try:
                lat_depth = 4
                lat_shape = f"1x{lat_depth}x{max(1, lstate.height//8)}x{max(1, lstate.width//8)}"
                dbg_pairs.append(("latent_depth", str(lat_depth)))
                dbg_pairs.append(("latent_shape", lat_shape))
            except Exception:
                pass
            for k in ("model_id", "width", "height", "steps", "guidance", "latents_std", "init_sigma", "img0_std", "img0_min", "img0_max"):
                if k in last and last[k] is not None:
                    dbg_pairs.append((k, str(last[k])))
            if is_turbo:
                dbg_pairs.append(("guidance_eff", str(guidance_eff)))
            try:
                retry_val = os.getenv("RETRY_ON_OOM", "0")
                dbg_pairs.append(("RETRY_ON_OOM", retry_val))
                toggle = getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                    "Enable OOM retry (env RETRY_ON_OOM)",
                    value=(retry_val not in ("0", "false", "False", "")),
                )
                os.environ["RETRY_ON_OOM"] = "1" if toggle else "0"
            except Exception:
                pass
            if dbg_pairs:
                sidebar_metric_rows(dbg_pairs, per_row=2)
            try:
                ls = last.get('latents_std')
                if ls is not None and float(ls) <= 1e-6:
                    st.sidebar.write('warn: latents std ~0')
            except Exception:
                pass
    except Exception:
        pass

    # State metadata panel and file paths
    try:
        render_metadata_panel(st.session_state.get(Keys.STATE_PATH), st.session_state.get(Keys.PROMPT))
    except Exception:
        pass

