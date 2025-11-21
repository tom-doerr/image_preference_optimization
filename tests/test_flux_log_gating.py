import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO


class TestFluxLogGating(unittest.TestCase):
    def setUp(self):
        # Remove any previously imported module to reset module‑level state
        sys.modules.pop("flux_local", None)

    def test_logs_gated_default(self):
        # Ensure env clean
        import os

        os.environ.pop("FLUX_LOCAL_MODEL", None)
        os.environ["LOG_VERBOSITY"] = "0"
        import flux_local  # type: ignore

        buf = StringIO()
        with redirect_stdout(buf):
            mid = flux_local._get_model_id()  # type: ignore[attr-defined]
        out = buf.getvalue()
        self.assertEqual(mid, "stabilityai/sd-turbo")
        self.assertNotIn("[pipe]", out)

    def test_logs_visible_when_verbose(self):
        import os

        os.environ.pop("FLUX_LOCAL_MODEL", None)
        os.environ["LOG_VERBOSITY"] = "1"
        import flux_local  # type: ignore

        buf = StringIO()
        with redirect_stdout(buf):
            mid = flux_local._get_model_id()  # type: ignore[attr-defined]
        out = buf.getvalue()
        self.assertEqual(mid, "stabilityai/sd-turbo")
        self.assertIn("[pipe]", out)
