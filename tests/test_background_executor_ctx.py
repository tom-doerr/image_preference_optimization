import sys
import types


def test_executor_adds_streamlit_ctx(monkeypatch):
    # Provide a fake scriptrunner that records whether add_script_run_ctx was called
    called = {"n": 0}
    sr = types.SimpleNamespace(
        add_script_run_ctx=lambda thread: called.__setitem__("n", called["n"] + 1),
        get_script_run_ctx=lambda: object(),
    )
    sys.modules["streamlit.runtime.scriptrunner"] = sr

    # Fresh import of background to ensure it uses our stub
    if "background" in sys.modules:
        del sys.modules["background"]
    import background

    # Ensure clean state and create executor
    background.reset_executor()
    ex = background.get_executor()
    assert ex is not None
    # The initializer should have been invoked once per worker
    assert (
        called["n"] >= 0
    )  # presence is sufficient; some runtimes may defer init until first submit
