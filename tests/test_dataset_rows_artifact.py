import sys
import types
import unittest


class TestDatasetRowsArtifact(unittest.TestCase):
    def test_rows_metric_includes_spinner(self):
        from tests.helpers.st_streamlit import stub_with_writes
        st, writes = stub_with_writes()
        sys.modules['streamlit'] = st

        # Stub flux + persistence to avoid heavy imports and control rows
        fl = types.ModuleType('flux_local')
        fl.generate_flux_image_latents = lambda *a, **k: 'ok-image'
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules['flux_local'] = fl

        p = types.ModuleType('persistence')
        p.state_path_for_prompt = lambda prompt: 'latent_state_dummy.npz'
        p.dataset_rows_for_prompt = lambda prompt: 7
        p.get_dataset_for_prompt_or_session = lambda *a, **k: (None, None)
        p.dataset_stats_for_prompt = lambda prompt: {"rows": 7, "pos": 4, "neg": 3, "d": 0, "recent_labels": [1,-1,1]}
        p.export_state_bytes = lambda *a, **k: b''
        p.read_metadata = lambda path: {"app_version": "0", "created_at": "", "prompt": "p"}
        sys.modules['persistence'] = p

        if 'app' in sys.modules:
            del sys.modules['app']
        import app  # noqa: F401

        out = "\n".join(writes)
        # Sidebar stubs format metrics as "label: value"; value should include spinner char
        lines = [ln for ln in out.splitlines() if ln.startswith('Dataset rows:')]
        self.assertTrue(lines, 'no Dataset rows line found')
        row_line = lines[-1]
        # Accept any of the simple spinner characters
        self.assertRegex(row_line, r"Dataset rows:\s*7\s+[\|/\\-]")


if __name__ == '__main__':
    unittest.main()
