import sys
import types
import unittest
import hashlib
import os


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def state_path_for_prompt(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"latent_state_{h}.npz"


def stub_streamlit_manage_states(initial_prompt, select_prompt, click_switch):
    st = types.ModuleType("streamlit")
    st.session_state = Session()
    st.session_state.recent_prompts = [initial_prompt, select_prompt]
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.number_input = lambda *_, value=None, **__: value
    st.image = lambda *_, **__: None

    def slider(label, *args, **kwargs):
        return kwargs.get("value", args[2] if len(args) >= 3 else 1.0)

    st.slider = slider

    def text_input(label, value=""):
        return initial_prompt

    st.text_input = text_input

    def button(label, *a, **kw):
        return label == "Switch prompt" and click_switch

    st.button = button

    class Sidebar:
        @staticmethod
        def selectbox(label, options, *args, **kwargs):
            if "Recent prompts" in label:
                # Return the desired selection if present
                return select_prompt if select_prompt in options else options[0]
            return "ridge" if "Approach" in label else "stabilityai/sd-turbo"

        @staticmethod
        def header(*_, **__):
            return None

        @staticmethod
        def subheader(*_, **__):
            return None

        @staticmethod
        def download_button(*_, **__):
            return None

        @staticmethod
        def file_uploader(*_, **__):
            return None

        @staticmethod
        def button(label, *_, **__):
            return label == "Switch prompt"

        @staticmethod
        def slider(*_, **__):
            return 1.0

        @staticmethod
        def text_input(*_, **__):
            return ""

        @staticmethod
        def checkbox(*_, **__):
            return False

        @staticmethod
        def write(x):
            return None

    st.sidebar = Sidebar()

    class Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    st.columns = lambda n: (Col(), Col())
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    return st


class TestManageStatesSwitch(unittest.TestCase):
    def test_switch_recent_prompt_loads_state(self):
        # Prepare two prompt states on disk
        from latent_opt import init_latent_state, save_state

        stA = init_latent_state(seed=0)
        stA.step = 3
        pathA = state_path_for_prompt("prompt A")
        save_state(stA, pathA)
        stB = init_latent_state(seed=0)
        stB.step = 0
        pathB = state_path_for_prompt("prompt B")
        save_state(stB, pathB)

        # First import with prompt B; then switch to A via Manage states
        if "app" in sys.modules:
            del sys.modules["app"]
        sys.modules["streamlit"] = stub_streamlit_manage_states(
            "prompt B", "prompt A", click_switch=True
        )
        fl = types.ModuleType("flux_local")
        fl.generate_flux_image_latents = lambda *a, **kw: "ok-image"
        fl.set_model = lambda *a, **kw: None
        sys.modules["flux_local"] = fl
        import app

        # After switch we expect prompt A's path
        self.assertEqual(app.st.session_state.state_path, pathA)

        # Cleanup created files
        for p in (pathA, pathB):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
