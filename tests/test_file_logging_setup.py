import os
import sys
import logging
import tempfile
import unittest


class TestFileLoggingSetup(unittest.TestCase):
    def tearDown(self):
        for name in ("helpers", "ipo.infra.util", "persistence", "ipo.core.persistence"):
            sys.modules.pop(name, None)

    def test_enable_file_logging_creates_and_writes(self):
        # Use a temp file path
        fd, path = tempfile.mkstemp(prefix="ipo_log_", suffix=".log")
        os.close(fd)
        try:
            os.environ["IPO_LOG_FILE"] = path
            from ipo.infra.util import enable_file_logging

            used = enable_file_logging()
            self.assertEqual(used, path)

            # Write via the shared 'ipo' logger
            logger = logging.getLogger("ipo")
            logger.info("[test] hello log")

            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            self.assertIn("hello log", txt)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
