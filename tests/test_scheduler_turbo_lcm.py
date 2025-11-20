import types


def test_set_model_uses_lcm_for_turbo(monkeypatch):
    import flux_local

    # Stub _ensure_pipe to avoid loading real models
    class _Sched:
        def __init__(self):
            self.config = {"any": 1}

    class _Pipe:
        def __init__(self):
            self.scheduler = _Sched()

    pipe = _Pipe()
    monkeypatch.setattr(flux_local, "_ensure_pipe", lambda mid: pipe)

    # Capture LCMScheduler.from_config calls by faking diffusers module
    calls = {}

    class _LCM:
        @classmethod
        def from_config(cls, cfg):
            calls["cfg"] = cfg
            return "LCM_SCHEDULER"

    import sys

    fake_diffusers = types.SimpleNamespace(LCMScheduler=_LCM)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    # Act
    flux_local.set_model("stabilityai/sd-turbo")

    # Assert scheduler swapped to our stub
    assert getattr(pipe, "scheduler") == "LCM_SCHEDULER"
    assert calls["cfg"] == {"any": 1}
