import sys
import types
import unittest
import numpy as np


class DummyImg:
    def __init__(self):
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[0, 0, 0] = 255  # non-uniform so mean-subtraction leaves signal
        self._arr = arr

    def convert(self, *_):
        return self

    def resize(self, *_):
        return self

    def __array__(self, *_):
        return self._arr


class UploadInterpolateTest(unittest.TestCase):
    def tearDown(self):
        for m in (
            "app",
            "streamlit",
            "persistence",
            "latent_logic",
            "flux_local",
            "value_scorer",
            "batch_ui",
            "PIL",
            "PIL.Image",
            "latent_opt",
            "modes",
        ):
            sys.modules.pop(m, None)

    def test_interpolation_blends_prompt_and_upload(self):
        from tests.helpers import st_streamlit

        st, writes = st_streamlit.stub_with_writes()
        st.session_state.prompt = "interp-test"
        st.session_state.cur_batch_nonce = 2
        st.session_state.steps = 3
        st.session_state.guidance_eff = 0.0
        st.session_state.dataset_y = []
        st.session_state.dataset_X = []
        st.session_state.lstate = types.SimpleNamespace(
            width=64, height=64, d=4 * (64 // 8) * (64 // 8), sigma=1.0, rng=np.random.default_rng(0), w=np.zeros(4), step=0
        )
        st.session_state.lz_pair = (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))

        # Streamlit sidebar controls
        st.sidebar.selectbox = lambda label, opts, index=0: ("Upload latents" if "Generation" in label else opts[index])
        st.sidebar.file_uploader = lambda *a, **k: [DummyImg()]
        st.sidebar.slider = lambda *a, **k: 0.5 if "Interpolate" in a[0] else k.get("value", 1.0)
        st.sidebar.checkbox = lambda *a, **k: False

        # Buttons: only Good returns True
        def _btn(label, *a, **k):
            return "Good" in label

        st.button = _btn
        st.sidebar.button = _btn

        # Stubs
        p = types.ModuleType("persistence")
        p.state_path_for_prompt = lambda prompt: "latent_state_dummy.npz"
        p.dataset_rows_for_prompt = lambda prompt: 0
        p.dataset_stats_for_prompt = lambda prompt: {"rows": 0, "pos": 0, "neg": 0, "d": st.session_state.lstate.d, "recent_labels": []}
        p.export_state_bytes = lambda state, prompt: b""
        p.read_metadata = lambda path: {"app_version": None, "created_at": None, "prompt": None}
        p.get_dataset_for_prompt_or_session = lambda prompt, ss: (None, None)
        p.append_dataset_row = lambda *a, **k: 0
        p.save_sample_image = lambda *a, **k: None
        sys.modules["persistence"] = p

        ll = types.ModuleType("latent_logic")
        ll.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        ll.z_to_latents = lambda z, lstate: z.reshape(1, 4, lstate.height // 8, lstate.width // 8)
        ll.sample_z_xgb_hill = lambda *a, **k: np.zeros(st.session_state.lstate.d)
        ll.propose_latent_pair_ridge = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        ll.propose_pair_prompt_anchor = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        ll.propose_pair_prompt_anchor_iterative = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        ll.propose_pair_prompt_anchor_linesearch = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        ll.update_latent_ridge = lambda *a, **k: None
        sys.modules["latent_logic"] = ll

        vs = types.ModuleType("value_scorer")
        vs.get_value_scorer_with_status = lambda *a, **k: (lambda f: 0.0, "ok")
        vs.get_value_scorer = lambda *a, **k: (lambda f: 0.0)
        sys.modules["value_scorer"] = vs

        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **k: "img"
        fl.set_model = lambda *a, **k: None
        fl.get_last_call = lambda: {}
        sys.modules["flux_local"] = fl

        # Capture added z
        added = []
        bu = types.ModuleType("batch_ui")
        bu._lstate_and_prompt = lambda: (st.session_state.lstate, st.session_state.prompt)
        def _add(lbl, z, img=None):
            added.append(z.copy())
        bu._curation_add = _add
        bu._curation_train_and_next = lambda: None
        bu._curation_init_batch = lambda: None
        bu._curation_new_batch = lambda: None
        bu._curation_replace_at = lambda idx: None
        bu._refit_from_dataset_keep_batch = lambda: None
        bu._render_batch_ui = lambda: None
        bu.run_batch_mode = lambda: None
        sys.modules["batch_ui"] = bu

        # Fake PIL.Image.open
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")
        pil_image.open = lambda *a, **k: DummyImg()
        pil_image.Image = DummyImg
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image

        # latent_opt stub
        lo = types.ModuleType("latent_opt")
        lo.init_latent_state = lambda *a, **k: st.session_state.lstate
        lo.load_state = lambda *a, **k: st.session_state.lstate
        lo.loads_state = lambda *a, **k: st.session_state.lstate
        lo.save_state = lambda *a, **k: None
        lo.state_summary = lambda *a, **k: {"d": st.session_state.lstate.d, "width": 64, "height": 64, "step": 0, "sigma": 1.0, "mu_norm": 0.0, "w_norm": 0.0, "pairs_logged": 0, "choices_logged": 0}
        lo.z_from_prompt = lambda lstate, prompt: np.zeros(lstate.d)
        lo.propose_next_pair = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        lo.z_to_latents = lambda z, lstate: z.reshape(1, 4, lstate.height // 8, lstate.width // 8)
        lo.update_latent_ridge = lambda *a, **k: None
        lo.propose_latent_pair_ridge = lambda *a, **k: (np.zeros(st.session_state.lstate.d), np.zeros(st.session_state.lstate.d))
        sys.modules["latent_opt"] = lo

        # modes stub
        md = types.ModuleType("modes")
        md.run_mode = lambda *a, **k: None
        sys.modules["modes"] = md

        sys.modules["streamlit"] = st
        import app  # noqa: F401

        self.assertEqual(len(added), 1)
        self.assertFalse(np.allclose(added[0], 0.0))


if __name__ == "__main__":
    unittest.main()
