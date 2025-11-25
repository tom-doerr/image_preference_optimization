import sys

# Map legacy module names to new package modules to keep tests working
legacy_to_new = {
    "batch_ui": "ipo.ui.batch_ui",
    "constants": "ipo.infra.constants",
    "latent_logic": "ipo.core.latent_logic",
    "latent_state": "ipo.core.latent_state",
    "value_model": "ipo.core.value_model",
    "value_scorer": "ipo.core.value_scorer",
    "flux_local": "ipo.infra.flux_local",
    "env_info": "ipo.infra.env_info",
    "app_bootstrap": "ipo.ui.app_bootstrap",
}

for legacy, new_name in legacy_to_new.items():
    try:
        if legacy not in sys.modules:
            sys.modules[legacy] = __import__(new_name, fromlist=["*"])
    except Exception:
        pass

