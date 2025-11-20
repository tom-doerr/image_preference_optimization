import os
import sys
import shutil
import unittest
import types
import numpy as np

from tests.helpers.st_streamlit import stub_basic


class DummyXGB:
    def __init__(self, y):
        self.y = np.asarray(y)

    def predict_proba(self, X):
        p = (self.y > 0).astype(float)
        return np.stack([1 - p, p], axis=1)

    def save_model(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("dummy")


class TestXgbCli(unittest.TestCase):
    def setUp(self):
        # Minimal streamlit stub for persistence imports
        st = stub_basic()
        sys.modules["streamlit"] = st

    def test_trains_and_saves_model(self):
        prompt = "cli prompt test"
        # Seed folder dataset via append
        from persistence import append_dataset_row
        append_dataset_row(prompt, np.array([[1.0, 0.0]]), +1.0)
        append_dataset_row(prompt, np.array([[0.0, 1.0]]), -1.0)

        # Stub xgb_value.fit_xgb_classifier
        stub_mod = types.SimpleNamespace()
        stub_mod.fit_xgb_classifier = lambda X, y, n_estimators=50, max_depth=3: DummyXGB(y)
        sys.modules["xgb_value"] = stub_mod

        from xgb_cli import train_xgb_for_prompt, _prompt_hash

        out_dir = ".tmp_cli_models"
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir, exist_ok=True)
        out_path = train_xgb_for_prompt(prompt, save_model=True, model_dir=out_dir)
        self.assertTrue(out_path.endswith(f"xgb_model_{_prompt_hash(prompt)}.bin"))
        self.assertTrue(os.path.exists(out_path))

        # Cleanup
        shutil.rmtree(out_dir)


if __name__ == "__main__":
    unittest.main()
