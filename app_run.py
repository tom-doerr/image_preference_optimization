from __future__ import annotations

from typing import Any


def run_app(st: Any, vm_choice: str, selected_gen_mode: str | None, async_queue_mode: bool) -> None:
    """Dispatch to the selected generation mode; thin wrapper to keep app.py small."""
    if selected_gen_mode == "Upload latents":
        from app_api import run_upload_mode as _run_upload

        _run_upload(st)
        return
    from modes import run_mode

    run_mode(bool(async_queue_mode))


def generate_pair(st: Any, base_prompt: str) -> None:
    """Thin wrapper so tests can call app.generate_pair() via app → app_run.

    Delegates to app_api.generate_pair(st, base_prompt).
    """
    from app_api import generate_pair as _gen

    _gen(st, base_prompt)


def _queue_fill_up_to(st: Any) -> None:
    """Thin wrapper for queue fill to keep app.py lean."""
    from app_api import _queue_fill_up_to as _fill

    _fill(st)
