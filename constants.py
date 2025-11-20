from dataclasses import dataclass

APP_VERSION = "0.1.0"
DEFAULT_PROMPT = "neon punk city, women with short hair, standing in the rain"
DEFAULT_MODEL = "stabilityai/sd-turbo"
MODEL_CHOICES = [
    DEFAULT_MODEL,
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
]

# 7 GB VRAM profile clamps
SMALL_VRAM_MAX_WIDTH = 448
SMALL_VRAM_MAX_HEIGHT = 448
SMALL_VRAM_MAX_STEPS = 12

# Minimal config to avoid scattered literals


@dataclass(frozen=True)
class Config:
    DEFAULT_WIDTH: int = 448
    DEFAULT_HEIGHT: int = 448
    DEFAULT_STEPS: int = 6
    DEFAULT_GUIDANCE: float = 3.5

# UI/generation tuning
DECODE_TIMEOUT_S = 3.0

# UI defaults (centralized)
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.5
DEFAULT_TRUST_R = 2.5
DEFAULT_LR_MU = 0.3
DEFAULT_GAMMA_ORTH = 0.2
DEFAULT_ITER_STEPS = 10
DEFAULT_QUEUE_SIZE = 6
DEFAULT_BATCH_SIZE = 25

# Scoring defaults (avoid scattered literals)
DISTANCEHILL_GAMMA = 0.5
COSINEHILL_BETA = 5.0


class Keys:
    REG_LAMBDA = "reg_lambda"
    ITER_STEPS = "iter_steps"
    ITER_ETA = "iter_eta"
    XGB_TRAIN_ASYNC = "xgb_train_async"
    RIDGE_TRAIN_ASYNC = "ridge_train_async"
    XGB_CACHE = "xgb_cache"
    XGB_FIT_FUTURE = "xgb_fit_future"
    RIDGE_FIT_FUTURE = "ridge_fit_future"
    XGB_TRAIN_STATUS = "xgb_train_status"
    LAST_TRAIN_AT = "last_train_at"
    LAST_TRAIN_MS = "last_train_ms"
    VM_CHOICE = "vm_choice"
    TRUST_R = "trust_r"
    LR_MU_UI = "lr_mu_ui"
    DATASET_DIM_MISMATCH = "dataset_dim_mismatch"
    CV_CACHE = "cv_cache"
    CV_LAST_AT = "cv_last_at"
    # Common app/queue keys
    PROMPT = "prompt"
    STATE_PATH = "state_path"
    VM_TRAIN_CHOICE = "vm_train_choice"
    QUEUE = "queue"
    QUEUE_SIZE = "queue_size"
    BATCH_SIZE = "batch_size"
    STEPS = "steps"
    GUIDANCE = "guidance"
    GUIDANCE_EFF = "guidance_eff"
    SIDEBAR_COMPACT = "sidebar_compact"
    USE_FRAGMENTS = "use_fragments"
    USE_IMAGE_SERVER = "use_image_server"
    IMAGE_SERVER_URL = "image_server_url"
    DEBUG_LOGS = "debug_logs"
    DEBUG_TAIL_LINES = "debug_tail_lines"
    XGB_N_ESTIMATORS = "xgb_n_estimators"
    XGB_MAX_DEPTH = "xgb_max_depth"
    XGB_CV_FOLDS = "xgb_cv_folds"
