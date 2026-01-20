from dataclasses import dataclass

APP_VERSION = "0.1.0"

SAFE_EXC = (AttributeError, ImportError, TypeError, ValueError, KeyError, FileNotFoundError)
DEFAULT_PROMPT = "latex, neon punk city, women with short hair, standing in the rain"
# Model is hardcoded to sd-turbo; no selector/choices remain.

# 7 GB VRAM profile clamps
SMALL_VRAM_MAX_WIDTH = 448
SMALL_VRAM_MAX_HEIGHT = 448
SMALL_VRAM_MAX_STEPS = 12

# Minimal config to avoid scattered literals


@dataclass(frozen=True)
class Config:
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512
    DEFAULT_STEPS: int = 6
    DEFAULT_GUIDANCE: float = 3.5


# UI/generation tuning
DECODE_TIMEOUT_S = 3.0

# UI defaults (centralized)
DEFAULT_ITER_STEPS = 100
DEFAULT_ITER_ETA = 10.0
DEFAULT_XGB_OPTIM_MODE = "Hill"
DEFAULT_SPACE_MODE = "PooledEmbed"  # "Latent", "PromptEmbed", or "PooledEmbed"
DEFAULT_CURATION_SIZE = 48
DEFAULT_INFERENCE_BATCH = 1  # images generated per forward pass

# Generation mode and model selection
DEFAULT_MODEL = "sd-turbo"
DEFAULT_GEN_MODE = "local"  # "local" or "server"
DEFAULT_SERVER_URL = "http://gen-server:8580"
# MODEL_OPTIONS imported from model_registry for single source of truth
from ipo.infra.model_registry import MODEL_OPTIONS


class Keys:
    REG_LAMBDA = "reg_lambda"
    ITER_STEPS = "iter_steps"
    ITER_ETA = "iter_eta"
    XGB_CACHE = "xgb_cache"
    LAST_TRAIN_AT = "last_train_at"
    VM_CHOICE = "vm_choice"
    TRUST_R = "trust_r"
    # Common app keys (queue removed)
    PROMPT = "prompt"
    STATE_PATH = "state_path"
    CURATION_SIZE = "curation_size"
    INFERENCE_BATCH = "inference_batch"
    STEPS = "steps"
    GUIDANCE = "guidance"
    GUIDANCE_EFF = "guidance_eff"
    XGB_N_ESTIMATORS = "xgb_n_estimators"
    XGB_MAX_DEPTH = "xgb_max_depth"
    XGB_OPTIM_MODE = "xgb_optim_mode"
    SAMPLE_MODE = "sample_mode"
    REGEN_ALL = "regen_all"
    CURATION_FORM_MODE = "curation_form_mode"
    CURATION_NONCE = "curation_nonce"
    DATASET_Y = "dataset_y"
    DATASET_X = "dataset_X"
    TRAIN_ON_NEW_DATA = "train_on_new_data"
    IMAGES = "images"
    # UI computed display values
    ROWS_DISPLAY = "rows_display"
    IMAGES_PER_ROW = "images_per_row"
    XGB_MOMENTUM = "xgb_momentum"
    SPACE_MODE = "space_mode"  # "Latent" or "PromptEmbed"
    NOISE_SEED = "noise_seed"
    DELTA_SCALE = "delta_scale"
    GAUSS_TEMP = "gauss_temp"
    # Generation mode
    GEN_MODE = "gen_mode"
    GEN_SERVER_URL = "gen_server_url"
    SELECTED_MODEL = "selected_model"
    # CLIP mode keys
    CLIP_IMAGES = "clip_images"
    CLIP_EMBEDS = "clip_embeds"
    CLIP_W = "clip_w"
    CLIP_MAX = "clip_max"
