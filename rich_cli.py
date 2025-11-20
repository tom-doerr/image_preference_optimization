from __future__ import annotations

import builtins
import re
from typing import Callable


_ENABLED = False


def enable_color_print() -> None:
    """Colorize our bracketed CLI tags using rich, if available.

    Tags like [pipe]/[perf]/[batch]/[latent]/[train]/[queue]/[xgb]/[warn]/[error]
    are highlighted. If rich is unavailable, this is a no‑op.
    """
    global _ENABLED
    if _ENABLED:
        return
    try:
        from rich.console import Console
    except Exception:
        return  # keep prints plain if rich is not installed

    console = Console(force_terminal=True)

    style_map = {
        "pipe": "bold cyan",
        "perf": "green",
        "batch": "magenta",
        "latent": "yellow",
        "train": "bold green",
        "mode": "blue",
        "queue": "bright_magenta",
        "xgb": "bright_yellow",
        "data": "cyan",
        "warn": "bold yellow",
        "error": "bold red",
        "autorun": "cyan",
        "prompt": "cyan",
    }

    orig_print: Callable[..., None] = builtins.print

    tag_re = re.compile(r"^\[(?P<tag>[a-zA-Z0-9_]+)\] ?")

    def _cprint(*args, **kwargs):  # type: ignore[no-redef]
        try:
            msg = " ".join(str(a) for a in args)
            lines = msg.splitlines() or [""]
            for i, ln in enumerate(lines):
                m = tag_re.match(ln)
                if not m:
                    console.print(ln, highlight=False)
                    continue
                tag = m.group("tag").lower()
                style = style_map.get(tag)
                if style:
                    console.print(f"[{style}]{ln}[/]", highlight=False)
                else:
                    console.print(ln, highlight=False)
        except Exception:
            # If anything goes wrong, fall back to the original print
            orig_print(*args, **kwargs)

    builtins.print = _cprint
    _ENABLED = True
