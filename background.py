from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from typing import Tuple, Any


_EXECUTOR = None


def get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        # Keep minimal but allow modest parallelism to avoid UI stalls in async queue.
        _EXECUTOR = ThreadPoolExecutor(max_workers=2)
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
