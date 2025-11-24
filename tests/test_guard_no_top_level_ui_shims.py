import os
import unittest


class TestNoTopLevelUiShims(unittest.TestCase):
    def test_no_root_level_ui_shims(self):
        for fname in ("ui.py", "ui_sidebar.py", "ui_controls.py", "batch_ui.py"):
            self.assertFalse(os.path.exists(fname), f"unexpected file at repo root: {fname}")

    def test_no_reintroduced_ui_controls_in_package(self):
        self.assertFalse(
            os.path.exists(os.path.join("ipo", "ui", "ui_controls.py")),
            "ipo/ui/ui_controls.py should not exist (helpers live in ui_sidebar)",
        )

    def test_ui_facade_imports_from_ui_sidebar(self):
        path = os.path.join("ipo", "ui", "ui.py")
        self.assertTrue(os.path.exists(path), "ipo/ui/ui.py facade missing")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("from .ui_sidebar import", src, "ui facade should re-export from ui_sidebar")


if __name__ == "__main__":
    unittest.main()
