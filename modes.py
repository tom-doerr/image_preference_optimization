from __future__ import annotations

def run_mode(async_queue_mode: bool) -> None:
    """Dispatch to the selected generation mode.

    Keeps app.py thin by centralizing the one branch we have.
    """
    if async_queue_mode:
        from queue_ui import run_queue_mode
        run_queue_mode()
    else:
        from batch_ui import run_batch_mode
        run_batch_mode()
