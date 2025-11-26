# Minimal shim to preserve old import paths in tests
# Maps legacy flux_local to the current local pipeline wrapper.
from ipo.infra.pipeline_local import *  # noqa: F401,F403

