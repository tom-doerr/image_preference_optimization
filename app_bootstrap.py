from __future__ import annotations

from typing import Any


def prompt_first_bootstrap(st: Any, lstate: Any, base_prompt: str) -> None:
    """Initialize prompt-related session fields without decoding on import.

    - Ensures `Keys.IMAGES` exists (set to (None, None)) so sidebar/status code
      can render deterministically before the first decode.
    - Ensures `prompt_image` placeholder exists and is None.
    Minimal and import-safe.
    """
    try:
        from constants import Keys

        if Keys.IMAGES not in st.session_state:
            st.session_state[Keys.IMAGES] = (None, None)
    except Exception:
        # Fallback to legacy names used in some tests
        if "images" not in st.session_state:
            st.session_state.images = (None, None)
    try:
        if "prompt_image" not in st.session_state:
            st.session_state.prompt_image = None
    except Exception:
        pass

