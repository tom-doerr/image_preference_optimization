import sys
import types
import unittest
from tests.helpers.st_streamlit import stub_basic


class TestTrainToggleNoTrain(unittest.TestCase):
    def test_disable_training_skips_refit(self):
        st = stub_basic()
        st.session_state.prompt = 'train-toggle-off'
        st.session_state.train_on_new_data = False
        # Minimal flux stub
        fl = types.ModuleType('flux_local')
        fl.set_model = lambda *a, **k: None
        fl.generate_flux_image_latents = lambda *a, **k: 'img'
        sys.modules['flux_local'] = fl
        sys.modules['streamlit'] = st
        # Import batch helpers
        import batch_ui as bu
        from latent_state import init_latent_state
        st.session_state.lstate = init_latent_state()
        bu._curation_init_batch()
        z = st.session_state.cur_batch[0]
        # Append data (saves to folders)
        bu._curation_add(1, z, img=None)
        # Stub value_model to detect calls
        vm = types.ModuleType('value_model')
        calls = {'fit': 0, 'ensure': 0}
        vm.fit_value_model = lambda *a, **k: calls.__setitem__('fit', calls['fit'] + 1)
        vm.ensure_fitted = lambda *a, **k: calls.__setitem__('ensure', calls['ensure'] + 1)
        sys.modules['value_model'] = vm
        # Attempt refit (should be skipped)
        bu._refit_from_dataset_keep_batch()
        self.assertEqual(calls['fit'], 0)
        self.assertEqual(calls['ensure'], 0)
        # No last_train_at set
        self.assertFalse(bool(st.session_state.get('last_train_at')))


if __name__ == '__main__':
    unittest.main()

