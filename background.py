from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from typing import Tuple, Any
import time as _time


_EXECUTOR = None


def get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        # Single worker to avoid contention with CUDA + Streamlit reruns.
        # Attach Streamlit ScriptRunContext to the worker when available to
        # suppress "missing ScriptRunContext" warnings in threaded tasks.
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx  # type: ignore
            import threading as _th  # local import to keep module light

            def _init():
                try:
                    ctx = get_script_run_ctx()
                    if ctx is not None:
                        add_script_run_ctx(_th.current_thread())
                except Exception:
                    pass

            _EXECUTOR = ThreadPoolExecutor(max_workers=1, initializer=_init)
        except Exception:
            _EXECUTOR = ThreadPoolExecutor(max_workers=1)
    return _EXECUTOR


def reset_executor() -> None:
    global _EXECUTOR
    _EXECUTOR = None


def schedule_decode_latents(prompt: str, latents: Any, width: int, height: int, steps: int, guidance: float) -> Future[Any]:
    from flux_local import generate_flux_image_latents  # lazy import for tests
    ex = get_executor()
    return ex.submit(generate_flux_image_latents, prompt, latents, width, height, steps, guidance)


def schedule_decode_pair(prompt: str,
                         lat_a: Any,
                         lat_b: Any,
                         width: int,
                         height: int,
                         steps: int,
                         guidance: float) -> Future[Tuple[Any, Any]]:
    from flux_local import generate_flux_image_latents  # lazy import
    ex = get_executor()

    def _work() -> Tuple[object, object]:
        img_a = generate_flux_image_latents(prompt, lat_a, width, height, steps, guidance)
        img_b = generate_flux_image_latents(prompt, lat_b, width, height, steps, guidance)
        return img_a, img_b

    return ex.submit(_work)


def result_or_sync_after(fut: Any,
                         started_at: float | None,
                         timeout_s: float,
                         sync_callable) -> tuple[Any | None, Any]:
    """Return future.result() if done; otherwise, if elapsed > timeout_s,
    run sync_callable() and return its result, wrapping it in a resolved Future.

    Returns (result_or_none, future_out). Minimal helper to avoid duplicated
    timeout logic in the UI code.
    """
    try:
        if fut is not None and fut.done():
            return fut.result(), fut
    except Exception:
        pass
    try:
        if started_at is not None and (_time.time() - float(started_at)) > float(timeout_s):
            res = sync_callable()
            from concurrent.futures import Future  # lazy import
            nf = Future()
            nf.set_result(res)
            return res, nf
    except Exception:
        pass
    return None, fut
