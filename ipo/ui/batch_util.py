from __future__ import annotations

from typing import Any


def ensure_model_ready() -> None:
    """Ensure a decode model is loaded before any image generation."""
    try:
        from ipo.infra.pipeline_local import CURRENT_MODEL_ID, set_model  # type: ignore
        if CURRENT_MODEL_ID is None:
            from ipo.infra.constants import DEFAULT_MODEL
            set_model(DEFAULT_MODEL)
    except Exception:
        pass


def prep_render_counters(st: Any) -> None:
    """Bump simple counters/nonces used to keep Streamlit keys stable."""
    try:
        st_globals = globals()
        st_globals["GLOBAL_RENDER_COUNTER"] = int(st_globals.get("GLOBAL_RENDER_COUNTER", 0)) + 1
    except Exception:
        globals()["GLOBAL_RENDER_COUNTER"] = 1
    try:
        st.session_state["render_count"] = int(st.session_state.get("render_count", 0)) + 1
    except Exception:
        pass


def save_and_print(prompt: str, feat, label: float, img, st):
    """Persist a labeled sample and emit the exact same CLI/sidebar lines.

    Returns (row_idx, save_dir, message) to keep caller behavior unchanged.
    """
    from ipo.core import persistence as p
    try:
        row_idx = p.append_sample(prompt, feat, float(label), img)
    except Exception:
        row_idx = None
    try:
        save_dir = getattr(p, "data_root_for_prompt", lambda pr: "data")(prompt)
    except Exception:
        save_dir = "data"
    msg = (
        f"Saved sample #{row_idx} → {save_dir}/{row_idx:06d}"
        if row_idx is not None
        else "Saved sample #n/a"
    )
    try:
        print(f"[data] saved row={row_idx if row_idx is not None else 'n/a'}")
    except Exception:
        pass
    try:
        st.sidebar.write(msg)
    except Exception:
        pass
    return row_idx, save_dir, msg


def set_batch_item(st: Any, idx: int, zi) -> None:
    """Set `cur_batch[idx] = zi` and clear its label safely.

    Keeps logging minimal and avoids exceptions from missing structures.
    """
    try:
        zs = getattr(st.session_state, "cur_batch", None) or []
        if not zs:
            return
        i = int(idx) % len(zs)
        try:
            zs[i] = zi
            st.session_state.cur_batch = zs
        except Exception:
            pass
        try:
            st.session_state.cur_labels[i] = None
        except Exception:
            pass
        try:
            print(f"[batch] replace_at idx={i}")
        except Exception:
            pass
    except Exception:
        pass
    try:
        st.session_state["render_nonce"] = int(st.session_state.get("render_nonce", 0)) + 1
        try:
            import secrets as __sec
            st.session_state["render_salt"] = int(__sec.randbits(32))
        except Exception:
            import time as __t
            st.session_state["render_salt"] = int(__t.time() * 1e9)
    except Exception:
        pass


def read_curation_params(st: Any, Keys, default_steps: int = 10, default_lr_mu: float = 0.3):
    """Return (vm_choice, use_xgb, steps, lr_mu, trust_r) from session_state.

    Kept tiny to reduce batch_ui complexity; defaults are deterministic.
    """
    try:
        vm_choice = str(st.session_state.get(Keys.VM_CHOICE) or "")
    except Exception:
        vm_choice = ""
    use_xgb = vm_choice == "XGBoost"
    try:
        steps = int(st.session_state.get(Keys.ITER_STEPS, default_steps))
    except Exception:
        steps = default_steps
    try:
        lr_mu = float(st.session_state.get(Keys.LR_MU_UI, default_lr_mu))
    except Exception:
        lr_mu = default_lr_mu
    try:
        trust = st.session_state.get(Keys.TRUST_R, None)
        trust_r = float(trust) if (trust is not None and float(trust) > 0.0) else None
    except Exception:
        trust_r = None
    return vm_choice, use_xgb, steps, lr_mu, trust_r


def cooldown_recent(st: Any, Keys) -> bool:
    """Return True if a recent train timestamp exists within min interval."""
    try:
        from datetime import datetime, timezone
        last_at = st.session_state.get(Keys.LAST_TRAIN_AT)
        min_wait = float(st.session_state.get("min_train_interval_s", 0.0) or 0.0)
        if last_at and min_wait > 0.0:
            try:
                dt = datetime.fromisoformat(last_at)
                return (datetime.now(timezone.utc) - dt).total_seconds() < min_wait
            except Exception:
                return False
    except Exception:
        return False
    return False
