"""Consolidated labeling module - handles Good/Bad button clicks and data persistence."""
from __future__ import annotations

from typing import Any

import numpy as np


def label_image(st: Any, idx: int, label: int, z: np.ndarray, img: Any = None) -> None:
    """Label an image and save to disk. Single entry point for all labeling."""
    from ipo.core import persistence as p
    from ipo.infra.constants import Keys
    from ipo.core.latent_logic import z_from_prompt

    lstate = st.session_state.lstate
    prompt = st.session_state.prompt
    z_p = z_from_prompt(lstate, prompt)
    feat = (z - z_p).reshape(1, -1)

    # Save to disk
    try:
        row_idx = p.append_sample(prompt, feat, float(label), img)
        print(f"[label] saved idx={idx} label={label:+d} row={row_idx}")
    except Exception as e:
        print(f"[label] save error: {e}")
        row_idx = None

    # Update UI state
    st.session_state.cur_labels[idx] = label
    _update_rows_display(st, Keys)


def _update_rows_display(st: Any, Keys) -> None:
    """Update the ROWS_DISPLAY counter from disk."""
    try:
        from ipo.core.persistence import get_dataset_for_prompt_or_session
        prompt = st.session_state.get('prompt', '')
        X, y = get_dataset_for_prompt_or_session(prompt, st.session_state)
        rows = int(getattr(X, 'shape', (0,))[0]) if X is not None else 0
        st.session_state[Keys.ROWS_DISPLAY] = str(rows)
    except Exception:
        pass
