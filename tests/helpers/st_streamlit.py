import types


class Session(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def _common(st):
    st.set_page_config = lambda **_: None
    st.title = lambda *_, **__: None
    st.caption = lambda *_, **__: None
    st.subheader = lambda *_, **__: None
    st.text_input = lambda *_, value="": value
    st.number_input = lambda *_, value=None, **__: value

    def _slider(*args, **kwargs):
        # Support both slider(label, min, max, value, step, ...)
        # and slider(..., value=...), returning the configured default.
        if 'value' in kwargs:
            return kwargs['value']
        # args: (label, min, max, value, [step, ...])
        if len(args) >= 4:
            return args[3]
        if len(args) >= 3:
            return args[2]
        return None

    st.slider = _slider
    st.image = lambda *_, **__: None
    st.write = lambda *_, **__: None
    st.experimental_rerun = lambda: None
    st.toast = lambda *_, **__: None
    class Sidebar:
        selectbox = staticmethod(lambda *a, **k: 'stabilityai/sd-turbo')
        header = staticmethod(lambda *a, **k: None)
        subheader = staticmethod(lambda *a, **k: None)
        download_button = staticmethod(lambda *a, **k: None)
        file_uploader = staticmethod(lambda *a, **k: None)
        @staticmethod
        def text_input(*args, value="", **kwargs):
            # Delegate to top-level text_input so tests can patch it.
            return st.text_input(*args, value=value, **kwargs)
        checkbox = staticmethod(lambda *a, **k: k.get("value", False))
        button = staticmethod(lambda *a, **k: False)
    st.sidebar = Sidebar()
    class Col:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: tuple(Col() for _ in range(max(1, int(n or 1))))
    return st


def stub_basic(pre_images=False):
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    if pre_images:
        st.session_state.images = ('ok-image', 'ok-image')
        st.session_state.mu_image = 'ok-image'
    _common(st)
    st.button = lambda *_, **__: False
    # Capture sidebar writes so tests that rely on text dumping (e.g., Latent dim)
    # can see them.
    st.sidebar_writes = []
    st.sidebar.write = lambda *a, **k: st.sidebar_writes.append(str(a[0]) if a else "")
    return st


def stub_with_writes(pre_images=False):
    writes = []
    st = stub_basic(pre_images=pre_images)
    st.sidebar.write = lambda *a, **k: writes.append(str(a[0]) if a else "")
    st.sidebar.metric = lambda label, value, **k: writes.append(f"{label}: {value}")
    return st, writes


def stub_click_button(label_to_click: str):
    st = stub_basic(pre_images=False)
    def _btn(label, *a, **k):
        return label == label_to_click
    st.button = _btn
    return st


def stub_with_main_writes(pre_images=False):
    writes = []
    st = stub_basic(pre_images=pre_images)
    def _w(*a, **k):
        if a:
            writes.append(str(a[0]))
        return None
    st.write = _w
    return st, writes


def stub_capture_images():
    images = []
    st = types.ModuleType('streamlit')
    st.session_state = Session()
    _common(st)
    def _image(*a, **k):
        images.append(k.get('caption') or '')
        return None
    st.image = _image
    st.button = lambda *_, **__: False
    st.sidebar.write = lambda *a, **k: None
    return st, images
