Development Guide (minimal)

Package layout
- ipo/ui
  - ui_sidebar.py — single source for sidebar panels and small UI helpers.
  - ui.py — thin facade re‑exporting selected ui_sidebar helpers used by tests.
  - batch_ui.py — batch curation flow (init/render/label/save/train buttons).
  - app_bootstrap.py — early sidebar emissions and lightweight bootstrap.
  - app_api.py — small app‑level wrappers kept for test compatibility.
- ipo/core
  - latent_logic.py, latent_state.py — math/state for latents and proposals.
  - value_model.py, value_scorer.py, xgb_value.py — training + scorers.
  - persistence.py, metrics.py — disk I/O (folder‑only) and metrics.
- ipo/infra
  - flux_local.py — Diffusers pipeline wrapper (CUDA‑only).
  - flux_utils.py, util.py, constants.py, env_info.py — small infra helpers.

Import style
- UI helpers: from ipo.ui.ui_sidebar import render_sidebar_tail, build_batch_controls
- Thin facade (kept for tests): from ipo.ui.ui import render_pair_sidebar
- Core: from ipo.core.persistence import append_dataset_row, dataset_rows_for_prompt
- Flux (tests can monkey‑patch): import flux_local as fl  # proxy to ipo.infra.flux_local

Training contract (simplified)
- Sync‑only fits: XGBoost trains when you click “Train XGBoost now (sync)”.
- Single scorer rule: captions use [XGB] when xgb_cache exists; else [Ridge] if ‖w‖>0; else n/a.
- Dataset is per‑prompt and per‑dim; dim mismatches are ignored.

Logging
- LOG_VERBOSITY=0 (default) for quiet CI; raise to 1 for “[scorer] …” lines.

Run
- App: streamlit run app.py
- Tests: python -m unittest discover -s tests -p 'test_*.py'

