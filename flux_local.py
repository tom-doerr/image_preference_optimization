# Smart proxy to underlying implementation so attribute assignment on this
# module (e.g., test stubs for _run_pipe) affects the real implementation.
import importlib as _il
_IMPL = _il.import_module("ipo.infra.flux_local")

def __getattr__(name):  # noqa: D401
    return getattr(_IMPL, name)

def __setattr__(name, value):  # noqa: D401
    # Forward test monkeypatches to the implementation module
    setattr(_IMPL, name, value)
