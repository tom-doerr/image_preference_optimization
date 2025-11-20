from __future__ import annotations

from typing import Any

from constants import Keys
from constants import DEFAULT_MODEL, MODEL_CHOICES


def render_rows_and_last_action(st: Any, base_prompt: str, lstate: Any | None = None) -> None:
    """Render Last action and the auto-refreshing rows metric in the sidebar.

    Keeps Streamlit constraints: compute values inside an optional fragment,
    but write to the sidebar only outside the fragment.
    """
    try:
        st.sidebar.subheader("Training data & scores")
        try:
            from constants import Keys as _K

            mismatch = st.session_state.get(_K.DATASET_DIM_MISMATCH)
            if mismatch and isinstance(mismatch, tuple) and len(mismatch) == 2:
                st.sidebar.write(
                    f"Dataset recorded at d={mismatch[0]} (ignored); current latent dim d={mismatch[1]}"
                )
        except Exception:
            pass
        try:
            import time as _time

            txt = st.session_state.get(Keys.LAST_ACTION_TEXT)
            ts = st.session_state.get(Keys.LAST_ACTION_TS)
            if txt and ts is not None and (_time.time() - float(ts)) < 6.0:
                st.sidebar.write(f"Last action: {txt}")
        except Exception:
            pass

        def _rows_refresh_tick() -> None:
            try:
                rows_live = int(len(st.session_state.get(Keys.DATASET_Y, []) or []))
            except Exception:
                rows_live = 0
            try:
                if lstate is not None:
                    try:
                        from persistence import dataset_rows_for_prompt_dim  # local import

                        rows_disk = int(dataset_rows_for_prompt_dim(base_prompt, getattr(lstate, "d", 0)))
                    except Exception:
                        from persistence import dataset_rows_for_prompt  # type: ignore  # local import

                        rows_disk = int(dataset_rows_for_prompt(base_prompt))
                else:
                    from persistence import dataset_rows_for_prompt  # local import

                    rows_disk = int(dataset_rows_for_prompt(base_prompt))
            except Exception:
                rows_disk = 0
            n_rows = max(rows_live, rows_disk)
            try:
                import time as _time

                _spin = "|/-\\"
                _art = _spin[int(_time.time()) % len(_spin)]
                disp = f"{n_rows} {_art}"
            except Exception:
                disp = str(n_rows)
            st.session_state[Keys.ROWS_DISPLAY] = disp
            try:
                print(f"[rows] live={rows_live} disk={rows_disk} disp={disp}")
            except Exception:
                pass
            try:
                _ar = getattr(st, "autorefresh", None)
                if callable(_ar):
                    _ar(interval=1000, key="rows_auto_refresh")
            except Exception:
                pass

        _frag = getattr(st, "fragment", None)
        if callable(_frag):
            try:
                _frag(_rows_refresh_tick)()
            except TypeError:
                _rows_refresh_tick()
        else:
            _rows_refresh_tick()

        # Render metrics outside any fragment
        try:
            from ui import sidebar_metric

            disp = st.session_state.get(Keys.ROWS_DISPLAY, "0")
            sidebar_metric("Dataset rows", disp)
            try:
                if lstate is not None:
                    try:
                        from persistence import dataset_rows_for_prompt_dim as _rows_dim

                        rows_disk_now = int(_rows_dim(base_prompt, getattr(lstate, "d", 0)))
                    except Exception:
                        from persistence import dataset_rows_for_prompt as _rows_d  # type: ignore

                        rows_disk_now = int(_rows_d(base_prompt))
                else:
                    from persistence import dataset_rows_for_prompt as _rows_d

                    rows_disk_now = int(_rows_d(base_prompt))
                sidebar_metric("Rows (disk)", rows_disk_now)
            except Exception:
                pass
            # Also show pairs/choices when lstate is provided
            if lstate is not None:
                try:
                    from latent_opt import state_summary  # type: ignore
                    info = state_summary(lstate)
                    from ui import sidebar_metric_rows
                    sidebar_metric_rows([("Pairs:", info.get("pairs_logged", 0)), ("Choices:", info.get("choices_logged", 0))], per_row=2)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass


def render_model_decode_settings(st: Any, lstate: Any):
    """Render 'Model & decode settings' and return (selected_model, width, height, steps, guidance, apply_clicked).

    Keeps behavior and labels stable for existing tests.
    """
    st.sidebar.header("Model & decode settings")
    # Fragment toggle for image tiles
    try:
        use_frags = bool(
            getattr(st.sidebar, "checkbox", lambda *a, **k: True)(
                "Use fragments (isolate image tiles)",
                value=bool(st.session_state.get(Keys.USE_FRAGMENTS, True)),
            )
        )
        st.session_state[Keys.USE_FRAGMENTS] = use_frags
    except Exception:
        pass
    # Optional image server
    try:
        use_srv = bool(
            getattr(st.sidebar, "checkbox", lambda *a, **k: False)(
                "Use image server",
                value=bool(st.session_state.get(Keys.USE_IMAGE_SERVER, False)),
            )
        )
        st.session_state[Keys.USE_IMAGE_SERVER] = use_srv
        srv_url = getattr(
            st.sidebar, "text_input", lambda *a, **k: ""
        )(
            "Image server URL",
            value=str(st.session_state.get(Keys.IMAGE_SERVER_URL, "")),
        )
        st.session_state[Keys.IMAGE_SERVER_URL] = srv_url
        try:
            import flux_local as _fl  # local import for tests

            _uis = getattr(_fl, "use_image_server", None)
            if callable(_uis):
                _uis(use_srv, srv_url)
            # Optional health check
            if use_srv and srv_url:
                import json as _json
                import urllib.request as _url

                try:
                    with _url.urlopen(srv_url.rstrip("/") + "/health", timeout=2) as r:  # nosec
                        ok = bool(_json.loads(r.read().decode("utf-8")).get("ok"))
                    st.sidebar.write(f"Image server: {'ok' if ok else 'unavailable'}")
                except Exception:
                    st.sidebar.write("Image server: unavailable")
        except Exception:
            pass
    except Exception:
        pass
    # Size/steps/guidance
    try:
        from ui_controls import build_size_controls  # local import avoids cycles

        width, height, steps, guidance, apply_clicked = build_size_controls(st, lstate)
    except Exception:
        width = getattr(lstate, "width", 512)
        height = getattr(lstate, "height", 512)
        steps = 6
        guidance = 0.0
        apply_clicked = False
    # Model selection (UI only; app will call set_model(selected))
    _model_sel = getattr(st.sidebar, "selectbox", None)
    if callable(_model_sel):
        try:
            selected_model = _model_sel("Model", MODEL_CHOICES, index=0)
        except Exception:
            selected_model = DEFAULT_MODEL
    else:
        selected_model = DEFAULT_MODEL
    # Effective guidance (read-only): clamp to 0 for turbo models.
    try:
        eff_guidance = 0.0 if isinstance(selected_model, str) and "turbo" in selected_model else float(guidance)
        st.session_state[Keys.GUIDANCE_EFF] = eff_guidance
        try:
            st.sidebar.write(f"Effective guidance: {eff_guidance:.2f}")
        except Exception:
            pass
    except Exception:
        pass
    return selected_model, int(width), int(height), int(steps), float(guidance), bool(apply_clicked)
