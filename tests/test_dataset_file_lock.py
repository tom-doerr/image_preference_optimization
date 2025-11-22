import os
import shutil
import tempfile
import multiprocessing as mp
import unittest


def _worker(root: str, prompt: str, n: int) -> None:
    os.environ["IPO_DATA_ROOT"] = root
    import numpy as np
    import importlib
    from ipo.core import persistence as p

    importlib.reload(p)
    for k in range(n):
        feat = np.zeros((1, 4), dtype=float) + k
        p.append_dataset_row(prompt, feat, +1.0)


class TestDatasetFileLock(unittest.TestCase):
    def test_multi_process_appends_unique(self):
        root = tempfile.mkdtemp(prefix="ipo_lock_test_")
        prompt = "file-lock-prompt"
        try:
            os.environ["IPO_DATA_ROOT"] = root
            import importlib
            from ipo.core import persistence as p

            importlib.reload(p)
            # Clean any existing
            droot = p.data_root_for_prompt(prompt)
            os.makedirs(droot, exist_ok=True)
            for name in os.listdir(droot):
                d = os.path.join(droot, name)
                try:
                    if os.path.isdir(d):
                        shutil.rmtree(d)
                except Exception:
                    pass
            # Spawn 4 processes × 5 appends each
            procs = [mp.Process(target=_worker, args=(root, prompt, 5)) for _ in range(4)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join(15)
            # Verify rows == 20
            rows = p.dataset_rows_for_prompt(prompt)
            self.assertEqual(rows, 20)
        finally:
            try:
                shutil.rmtree(root)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
