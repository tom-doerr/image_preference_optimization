import sys
import types


def test_background_module_removed_sync_paths_only():
    # 199a: background helpers removed; module may not exist
    if "background" in sys.modules:
        del sys.modules["background"]
    try:
        __import__("background")
        mod_present = True
    except Exception:
        mod_present = False
    assert mod_present is False
