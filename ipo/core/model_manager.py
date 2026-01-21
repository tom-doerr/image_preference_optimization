"""Unified model management with backend abstraction and caching."""
from abc import ABC, abstractmethod
from typing import Optional


class ModelBackend(ABC):
    """Abstract interface for model backends."""

    @abstractmethod
    def get_current_model(self) -> Optional[str]:
        """Return currently loaded model ID."""
        ...

    @abstractmethod
    def ensure_model(self, model_id: str) -> None:
        """Load model if not already loaded."""
        ...


class LocalBackend(ModelBackend):
    """Local pipeline backend."""

    def get_current_model(self) -> Optional[str]:
        from ipo.infra.pipeline_local import CURRENT_MODEL_ID
        return CURRENT_MODEL_ID

    def ensure_model(self, model_id: str) -> None:
        from ipo.infra.pipeline_local import CURRENT_MODEL_ID, set_model
        if CURRENT_MODEL_ID != model_id:
            print(f"[ModelManager] Local: {CURRENT_MODEL_ID} -> {model_id}")
            set_model(model_id)


class ServerBackend(ModelBackend):
    """Remote generation server backend."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._client = None
        self._synced_model: Optional[str] = None

    @property
    def client(self):
        if self._client is None:
            from ipo.server.gen_client import GenerationClient
            self._client = GenerationClient(self.base_url)
        return self._client

    def get_current_model(self) -> Optional[str]:
        return self._synced_model

    def ensure_model(self, model_id: str) -> None:
        if self._synced_model != model_id:
            print(f"[ModelManager] Server: {self._synced_model} -> {model_id}")
            self.client.set_model(model_id)
            self._synced_model = model_id


class ModelManager:
    """Singleton manager for model selection and backend routing."""

    _local: Optional[LocalBackend] = None
    _servers: dict = {}

    @classmethod
    def get_selected_model(cls) -> str:
        """Read user's model selection from session state."""
        import streamlit as st
        from ipo.infra.constants import DEFAULT_MODEL, Keys
        return st.session_state.get(Keys.SELECTED_MODEL) or DEFAULT_MODEL

    @classmethod
    def get_backend(cls) -> ModelBackend:
        """Get backend based on current generation mode."""
        import streamlit as st
        from ipo.infra.constants import DEFAULT_GEN_MODE, DEFAULT_SERVER_URL, Keys
        mode = st.session_state.get(Keys.GEN_MODE) or DEFAULT_GEN_MODE
        if mode == "server":
            url = st.session_state.get(Keys.GEN_SERVER_URL) or DEFAULT_SERVER_URL
            if url not in cls._servers:
                cls._servers[url] = ServerBackend(url)
            return cls._servers[url]
        if cls._local is None:
            cls._local = LocalBackend()
        return cls._local

    @classmethod
    def ensure_ready(cls) -> None:
        """Call before any generation. Safe to call multiple times."""
        backend = cls.get_backend()
        backend.ensure_model(cls.get_selected_model())

    @classmethod
    def generate(cls, prompt: str, width=512, height=512, steps=4, guidance=0.0, seed=42):
        """Generate image from prompt, routing through appropriate backend."""
        cls.ensure_ready()
        backend = cls.get_backend()
        if isinstance(backend, ServerBackend):
            return backend.client.generate(prompt, None, "text", width, height, steps, guidance, seed, 1.0)
        from ipo.infra.pipeline_local import generate
        return generate(prompt, width, height, steps, guidance, seed)
