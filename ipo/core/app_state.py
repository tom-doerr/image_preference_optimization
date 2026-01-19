"""Typed application state to replace raw session_state access."""
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
from ipo.core.latent_state import LatentState


@dataclass
class AppState:
    """Typed state container."""
    lstate: Optional[LatentState] = None
    prompt: str = ""
    vm_choice: str = "Gaussian"
    xgb_n_estimators: int = 50
    xgb_max_depth: int = 8
    xgb_optim_mode: str = "Hill"
    gauss_temp: float = 1.0
    reg_lambda: float = 1000.0
    trust_r: float = 200.0
    iter_steps: int = 100
    iter_eta: float = 10.0
    sample_mode: str = "AvgGood"
    batch_size: int = 48
    steps: int = 6
    noise_seed: int = 42
    # Dataset
    dataset_X: Optional[np.ndarray] = None
    dataset_y: Optional[np.ndarray] = None
    # Model cache
    xgb_model: Any = None
    gauss_mu: Optional[np.ndarray] = None
    gauss_sigma: Optional[np.ndarray] = None
