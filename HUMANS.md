Q: Why is XGBoost still “unavailable” even though there is data?
- The app trains XGB only when you click “Train XGBoost now (sync)”. We removed auto‑fit on reruns to simplify.
- Data is per‑prompt and per latent dimension. If you change prompt or resolution, you’re in a new folder/hash; old rows don’t count.
- Rows saved at a different dim are filtered (e.g., d=9216 vs d=25600). You’ll see a note in the sidebar when that happens.
- XGB needs at least two classes (+1/−1). Single‑class datasets won’t produce a usable model.
- After a sync fit succeeds, the scorer becomes ready only if `session_state.xgb_cache` is populated. We set it to `{model, n}` and print `[xgb] using cached model rows=N`.

Q: What should I expect in the UI?
- Before training: captions show `Value: n/a`; sidebar line `XGBoost model rows: 0 (status: unavailable)`.
- After saving mixed labels and pressing the sync fit button: captions switch to `Value: … [XGB]` and the status becomes `ok` with `rows=N`.
- Ridge values show only when explicitly selected and ‖w‖>0; otherwise `n/a`.

Q: What’s the fastest way to debug “no values”?
- Confirm dataset counters for the current prompt > 0 and contain both classes.
- Click “Train XGBoost now (sync)”; watch for `[xgb] train start rows=N` and then `[xgb] using cached model rows=N` in the CLI.
- Ensure log verbosity ≥1 if you want `[scorer] …` lines.

Q: Why don’t we auto‑fit on rerun anymore?
- It caused surprising status flaps under Streamlit reruns. Sync‑only keeps behavior deterministic and reduced code/LOC.

Nov 22, 2025 — 229c follow‑up
- You asked to remove top‑level shims (e.g., ui_sidebar.py, ui_controls.py). There are no top‑level files by those names now. The only copies live under `ipo/ui/` (`ipo/ui/ui_sidebar.py`, `ipo/ui/ui_controls.py`). Tests import those package paths, so no deletion was needed at the repo root.

Nov 24, 2025 — XGB auto‑fit on selection
- Implemented: when Value model = XGBoost, the app auto‑fits XGB synchronously during sidebar render if a usable dataset is present and cache is stale. It updates `session_state.xgb_cache = {model, n}` and logs `[xgb] train start …/train done …`. Ridge still fits on every call to keep `w` fresh.
- Added focused test `tests/test_xgb_autofit_when_selected.py` that stubs heavy deps and asserts the cache is populated after render with an in‑memory dataset.
- Question: keep auto‑fit “on render if stale” or restrict to “on selection change only”? Current behavior is the former, with a cache guard to avoid repeated training.

Nov 24, 2025 — Maintainability review questions
- Do you want me to delete the root-level re-export shims (`value_model.py`, `xgb_value.py`, `constants.py`, `batch_ui.py`) once all imports are clean, and add a guard test to keep them from coming back?
- Is it acceptable to extract one more helper `compute_train_results_lines(...)` so `render_sidebar_tail` just formats/prints? This will reduce LOC and make sidebar string tests steadier.
- Should we gate all noncritical logs under `LOG_VERBOSITY` (0 by default) to keep CI quieter? We’ll still print critical errors.
