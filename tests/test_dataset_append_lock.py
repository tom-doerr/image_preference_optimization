import os
import threading
import unittest
import numpy as np


class TestDatasetAppendLock(unittest.TestCase):
    def test_concurrent_appends_produce_unique_rows(self):
        import persistence as p

        prompt = "append-lock-test"
        root = p.data_root_for_prompt(prompt)
        os.makedirs(root, exist_ok=True)
        # Clean any existing rows for this prompt in the test scratch
        for name in os.listdir(root):
            d = os.path.join(root, name)
            try:
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        os.remove(os.path.join(d, f))
                    os.rmdir(d)
            except Exception:
                pass
        n_threads = 20
        idxs = []
        lock = threading.Lock()

        def _do_append(k):
            # One 1x4 feature row per append
            feat = np.zeros((1, 4), dtype=float) + k
            i = p.append_dataset_row(prompt, feat, +1.0)
            with lock:
                idxs.append(i)

        ths = [threading.Thread(target=_do_append, args=(i,)) for i in range(n_threads)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        # All indexes should be unique and count should match directory rows
        self.assertEqual(len(set(idxs)), n_threads)
        self.assertEqual(p.dataset_rows_for_prompt(prompt), n_threads)


if __name__ == "__main__":
    unittest.main()

