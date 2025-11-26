"""Test labeling flow: button keys stable, data saves correctly."""
import os
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("IPO_DATA_ROOT", str(tmp_path / "data"))
    yield tmp_path / "data"


def test_button_key_stable():
    """Button keys must NOT change across reruns."""
    from ipo.ui.batch_buttons import _button_key

    class FakeSt:
        session_state = {"render_count": 5, "render_nonce": 10}

    key1 = _button_key(FakeSt(), "good", nonce=99, idx=2)
    FakeSt.session_state["render_count"] = 6
    key2 = _button_key(FakeSt(), "good", nonce=99, idx=2)
    assert key1 == key2, f"Key changed: {key1} != {key2}"


def test_append_saves_correct_dim(isolated_data):
    """append_dataset_row should save sample.npz with correct d."""
    from ipo.core.persistence import append_dataset_row, data_root_for_prompt

    prompt = "test labeling"
    feat = np.random.randn(1, 16384).astype(float)
    row_idx = append_dataset_row(prompt, feat, 1.0)
    assert row_idx == 1
    root = data_root_for_prompt(prompt)
    path = os.path.join(root, "000001", "sample.npz")
    assert os.path.exists(path)
    with np.load(path) as z:
        assert z["X"].shape == (1, 16384)


def test_xgb_train_needs_both_labels():
    """XGB training requires both +1 and -1 labels."""
    from ipo.ui.sidebar.panels import _xgb_train_controls

    class FakeLstate:
        X = None
        y = None

    class FakeSt:
        session_state = {}

    Xd = np.random.randn(5, 100)
    yd = np.ones(5)  # all +1, no -1
    _xgb_train_controls(FakeSt, FakeLstate(), Xd, yd)
