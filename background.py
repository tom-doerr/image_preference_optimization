from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from typing import Tuple, Any
import time as _time


_EXECUTOR = None
_EXECUTOR_TRAIN = None


def get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        # Decode worker pool: 2 workers to overlap host work and UI rendering.
        _EXECUTOR = ThreadPoolExecutor(max_workers=2)
        try:
            print("[background] created decode executor max_workers=2")
        except Exception:
            pass
    return _EXECUTOR


def get_train_executor() -> ThreadPoolExecutor:
    """Separate single-worker pool for training tasks to avoid blocking decodes."""
    global _EXECUTOR_TRAIN
    if _EXECUTOR_TRAIN is None:
        _EXECUTOR_TRAIN = ThreadPoolExecutor(max_workers=1)
    return _EXECUTOR_TRAIN


def reset_executor() -> None:
    global _EXECUTOR, _EXECUTOR_TRAIN
    _EXECUTOR = None
    _EXECUTOR_TRAIN = None


def schedule_decode_latents(
    prompt: str, latents: Any, width: int, height: int, steps: int, guidance: float
) -> Future[Any]:
    from flux_local import generate_flux_image_latents  # lazy import for tests

    ex = get_executor()
    try:
        print(
            f"[background] submit latents decode steps={steps} size={width}x{height}"
        )
    except Exception:
        pass
    return ex.submit(
        generate_flux_image_latents, prompt, latents, width, height, steps, guidance
    )


def schedule_decode_pair(
    prompt: str,
    lat_a: Any,
    lat_b: Any,
    width: int,
    height: int,
    steps: int,
    guidance: float,
) -> Future[Tuple[Any, Any]]:
    from flux_local import generate_flux_image_latents  # lazy import

    ex = get_executor()

    def _work() -> Tuple[object, object]:
        img_a = generate_flux_image_latents(
            prompt, lat_a, width, height, steps, guidance
        )
        img_b = generate_flux_image_latents(
            prompt, lat_b, width, height, steps, guidance
        )
        return img_a, img_b

    try:
        print(
            f"[background] submit pair decode steps={steps} size={width}x{height}"
        )
    except Exception:
        pass
    return ex.submit(_work)


def result_or_sync_after(
    fut: Any, started_at: float | None, timeout_s: float, sync_callable
) -> tuple[Any | None, Any]:
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
        if started_at is not None and (_time.time() - float(started_at)) > float(
            timeout_s
        ):
            res = sync_callable()
            from concurrent.futures import Future  # lazy import

            nf = Future()
            nf.set_result(res)
            return res, nf
    except Exception:
        pass
    return None, fut
