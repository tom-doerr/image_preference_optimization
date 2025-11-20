import io
import hashlib
import numpy as np
import unittest

from persistence import state_path_for_prompt, export_state_bytes, read_metadata
from latent_opt import init_latent_state, save_state


class TestPersistenceHelpers(unittest.TestCase):
    def test_state_path_for_prompt_sha(self):
        p = 'hello world'
        h = hashlib.sha1(p.encode('utf-8')).hexdigest()[:10]
        self.assertEqual(state_path_for_prompt(p), f'latent_state_{h}.npz')

    def test_export_state_bytes_schema(self):
        st = init_latent_state(d=4, seed=0)
        data = export_state_bytes(st, 'prompt X')
        arr = np.load(io.BytesIO(data))
        self.assertIn('prompt', arr.files)
        self.assertIn('created_at', arr.files)
        self.assertIn('app_version', arr.files)
        self.assertEqual(arr['prompt'].item(), 'prompt X')

    def test_read_metadata(self):
        p = 'meta test'
        path = state_path_for_prompt(p)
        st = init_latent_state(d=4, seed=0)
        save_state(st, path)
        # add meta
        with np.load(path) as data:
            items = {k: data[k] for k in data.files}
        items['created_at'] = np.array('2025-11-13T00:00:00+00:00')
        items['app_version'] = np.array('0.1.0')
        buf = io.BytesIO()
        np.savez_compressed(buf, **items)
        with open(path, 'wb') as f:
            f.write(buf.getvalue())
        meta = read_metadata(path)
        self.assertEqual(meta['app_version'], '0.1.0')
        self.assertIn('2025-11-13', meta['created_at'])


if __name__ == '__main__':
    unittest.main()
