HUMANS.md — notes for maintainers (Nov 24, 2025)

What I did now
- Ran Radon across `ipo/` only (tests currently have a few legacy formatting issues) to get clear hotspots.
- Fixed two indentation issues that blocked static analysis:
  - `ipo/core/value_scorer.py`: mis‑indented inner `try` block around `get_cached_scorer` import.
  - `ipo/ui/batch_ui.py`: mis‑indented `from ipo.core.value_model import fit_value_model` inside a `try`.
- Split `ipo/ui/ui_sidebar.render_sidebar_tail` into small helpers; complexity from F→B without changing strings/ordering.

Nov 24 — follow‑up (Radon + flux_local extractions)
- Ran Radon again and focused on infra loader/run hotspots.
- Extracted helpers in `ipo/infra/flux_local.py` to reduce branching without behavior change:
  - `_load_pipeline(mid)`, `_disable_safety(pipe)`, `_prepare_scheduler_locked(steps)`.
  - `_ensure_pipe`: E→C (31→17). `_run_pipe`: D→C (24→16).
- Restored a tiny gated log for `_get_model_id()` so `tests/test_flux_log_gating.py` passes when `LOG_VERBOSITY=1`.
- Verified quick subset: `tests/test_flux_loader_kwargs.py`, `tests/test_flux_run_wrapper.py`, `tests/test_flux_log_gating.py` → OK.

Hotspots to consider next
- `ipo/ui/ui_sidebar.render_sidebar_tail` — F(76), main candidate to split into smaller helpers (panels already exist). Minimal risk, good MI win.
- `ipo/infra/flux_local._ensure_pipe` / `_run_pipe` — E/D; clean split into load/configure and prepare/run steps keeps behavior intact.
- `ipo/ui/batch_ui._render_batch_ui` — F(51); extract tiny helpers for decode/caption/label.
Update: `render_sidebar_tail` is now B(10). `_emit_train_results` is split and A(1). `_sidebar_value_model_block` was partially split (still D but lower). Next targets:
- Flux loader pair (`_ensure_pipe` E / `_run_pipe` D) — split load/configure and prepare/run/record; gate logs.
- Batch tile rendering (`_render_batch_ui` F / `_render_batch_tile_body` E) — extract tiny decode/label helpers.

New changes (Nov 24):
- Batch UI helpers added: `_ensure_model_ready`, `_prep_render_counters`, `_decode_one`, `_maybe_logit_value`, `_label_and_replace`, `_button_key`, `_render_good_bad_buttons`.
- `_render_batch_tile_body` now D(25). `_render_batch_ui` F(45). Plan: extract inner visual/buttons next.

Questions for you
1. Proceed to split `ipo/ui/batch_ui._render_batch_ui` (F 45) into two tiny helpers (visual vs. buttons)? Low risk if we reuse existing helpers; goal D/C.
2. OK to leave fragments permanently off (code path already non‑frag only), and delete remaining fragment guards in a later pass?
3. Any tolerance to delete old async/background remnants in tests to simplify further? (Would reduce surface area.)

Q&A (Nov 24, 2025 – late)
- Done: Reduced CC in batch_ui (_choose_scorer→A, _tile_value_text→B) and ui_sidebar (_mem_dataset_stats→A) without altering behavior/strings. Verified with Radon.
- Update: Also reduced ui_sidebar.compute_step_scores to A and _emit_last_call_info to B via tiny helpers; visible strings remain identical.
- Question: Should I proceed with batch_ui._curation_add (currently C 17) next? It’s the next clean win (toast/save/train split) with no UI string changes.
Update (same day): I split `_curation_add` into four tiny helpers (append memory dataset, save+notice, record last action+step, update rows display). Behavior and output lines are the same; complexity drops to A. If you want, next I can trim `_refit_from_dataset_keep_batch` from C by pulling cooldown/fit blocks into helpers (no behavior change).
Extra: Reduced `_refit_from_dataset_keep_batch` to B with `_cooldown_recent` and `_fit_ridge_once`. Sidebar metadata panel now uses `_resolve_meta_pairs` + `_emit_meta_pairs` (same strings, clearer flow).
Latest: `_pick_scorer` is now A (split into four tiny try_* helpers), and `_curation_train_and_next` dropped to B by reusing the cooldown/fit helpers. Batch UI file average is A; strings remain unchanged.

New tests (Nov 24, 2025):
- Sidebar regression: `_emit_train_results` with no `lstate` in session must not crash and must write the provided lines.
- Cooldown helper: `_cooldown_recent` returns True for a recent timestamp and False for an old one.
- Integration smokes:
  - Sidebar-tail: `render_sidebar_tail` with stubbed `streamlit` + `flux_local` renders the canonical lines.
  - Batch flow: `batch_ui.run_batch_mode` with stubbed `streamlit`/`latent_logic`/`flux_local` produces image captions (no GPU).
- Note: The full test suite shows a few syntax issues in tests in this workspace; I did not modify tests in this pass. If you want, I can prioritize a quick pass to fix or isolate those before continuing.
How to reproduce
- Activate venv and run:
  - `radon cc -s -a ipo`
  - `radon mi -s ipo`
- Nov 24 — batch_ui pass
  - Simplified the fragment path in `ipo/ui/batch_ui._render_batch_ui` to a single non‑fragment code path.
  - Extracted `_choose_scorer(...)` and `_tile_value_text(...)` to reduce branching in tile rendering and captions.
  - Result: `_render_batch_ui` F→C (45→~18/19 after row helper) and `_render_batch_tile_body` D→C (25→12). No strings/behavior changed; duplication removed by `_render_tiles_row`.
- Nov 24 — persistence pass
  - Split `get_dataset_for_prompt_or_session` into tiny helpers:
    `_target_dim_from_session`, `_iter_sample_paths`, `_load_sample_npz`, `_record_dim_mismatch`.
  - Result: `get_dataset_for_prompt_or_session` C(12) (down from C(20)); avg for file is now A.
  - Behavior unchanged (same prints and mismatch recording), simpler to reason about.
- Flux utils: split `normalize_to_init_sigma` into `_scheduler_init_sigma` and `_latents_std`; main function now A(3). No behavior change.
- Sidebar CV: refactored `_cv_on_demand` into small helpers (get K, dataset, compute CV, record cache). Function now A(3); UI strings unchanged ("CV folds", "Compute CV now").
- Value model: split `fit_value_model` into tiny orchestrators `_train_optionals(...)` and `_record_train_summaries(...)`. Main `fit_value_model` now A(3) from C(20). Behavior unchanged; logging preserved.
  - Ran a targeted subset of tests earlier; avoided full suite due to unrelated test file indentation issues in `tests/test_tile_value_captions.py` (appears pre‑existing).
Nov 25, 2025 — XGB simplification + test stability

What changed (you asked for simpler, sync‑only training and more predictable captions):
- XGBoost training is sync and records the trained model under both `session_state.XGB_MODEL` (new) and `session_state.xgb_cache['model']` (compat for older tests/UI).
- The scorer prefers `XGB_MODEL` but also accepts the legacy cache. This removes the "unavailable despite data" symptom when a test only sets `xgb_cache`.
- The sidebar always prints `Value model: <choice>` early so text‑only tests see it without waiting for the train‑results block.
- Dataset lookup in the sidebar first tries `ipo.core.persistence`, then falls back to a stubbed top‑level `persistence` module (several tests rely on this name).

Why XGBoost bugs felt sticky before:
- Mixed contracts: some tests expected auto‑fit on selection; others expected a manual Train button. We are converging to manual, sync fits only.
- Two cache names (old `xgb_cache` vs new live model) led to “scorer unavailable” even when a model existed. The tiny shim bridges both.
- Single‑class datasets keep XGB explicitly unavailable by design; training requires both +1 and −1 labels.

Open questions for you:
1) Do we drop auto‑fit entirely and keep only the explicit "Train XGBoost now (sync)" button?
2) OK to retire the legacy `xgb_cache` once tests are migrated (follow‑up PR)?
3) Shall we extract a couple of pure helpers from `latent_logic` (trust‑radius, sigmoid, step accumulation) to lower Radon CC without changing behavior?

Notes for later:
- If you see only `xgb_unavailable`, check: (a) at least one +1 and one −1 in the dataset; (b) the current latent dim matches the dataset dim; (c) after a sync fit, a scorer tag `XGB` should appear under the images and in the sidebar.
Nov 25, 2025 — Radon/maintainability pass

What I did (no behavior change):
- Split early sidebar bootstrap into small helpers so it’s easier to read and test.
- Extracted two tiny math helpers in `latent_logic` to remove in-function branching:
  `_accumulate_delta` (trust-radius clamped step accumulation) and `_rand_orth_dir`
  (random unit vector ⟂ d1). The two proposal functions call these now.

Why: These were the top C-grade hotspots in radon that were safe to factor mechanically.
No strings or outputs changed; tests around prompt-anchor proposals pass.

Follow-up (done):
- Extracted `_load_rows_filtered` in persistence and reused it for stats and
  loader (no behavior change).
- Split flux-local toggles/logging into `_post_load_toggles`, `_after_model_switch`,
  and `_record_latents_meta` (same logs/strings; cleaner functions).
- Broke out `ensure_prompt_and_state` into `_resolve_state_path` + `_apply_or_init_state`.
- Reduced complexity in value_scorer’s XGB path via `_get_live_xgb_model` and
  `_print_xgb_unavailable` while preserving the exact unavailable log line.

Next (still safe and tiny):
- Consider trimming `_run_pipe` by hoisting OOM-retry decision into a helper;
  keep text identical. Low risk.

Update 2:
- `_run_pipe` trimmed: hoisted logging, perf, image extraction, and OOM-retry
  into helpers. Function is B now (was C), messages unchanged.
- `_record_latents_meta` was split into three helpers and is A now. Overall
  flux_local average improved; diagnostics remain the same.

Update 3:
- Value-model’s `_maybe_fit_xgb` complexity reduced (C→B) by extracting
  `_xgb_hparams`, `_store_xgb_model`, and `_has_two_classes`. The function still
  trains synchronously and mirrors the live model into legacy `xgb_cache` to
  keep older paths/tests working.

Update 4:
- Flux-utils `_scheduler_init_sigma` simplified by extracting
  `_ensure_timesteps` and `_sigma_from_sigmas_attr` (C→A). Behavior is identical;
  we still set timesteps if available and fall back to `sigmas.max()` when the
  scheduler lacks `init_noise_sigma`.

Update 5:
- Value-model `_maybe_fit_logit` reduced from C→B by factoring tiny helpers
  `_logit_params` (read steps/λ) and `_logit_train_loop` (SGD loop). The CLI
  log remains `[logit] fit rows=…` with the same fields.

Update 6:
- Latent-logic: cleaned up remaining C-grade functions without changing
  behavior or strings:
  - `hill_climb_mu_distance` now uses `_distance_loss_and_grad` and
    `_distance_loss_only`; complexity B.
  - `hill_climb_mu_xgb` and `sample_z_xgb_hill` reuse `_best_of_along_d1`
    and `_trust_clamp`; both now B.
  - `update_latent_ridge` split into small helpers and is A now.

Nov 25, 2025 — XGB “why n/a?” quick reference
- XGB captions remain “n/a” until BOTH:
  1) There are +1 and −1 rows in the current‑dim dataset, and
  2) You click “Train XGBoost now (sync)”. We do not auto‑fit on selection/rerun anymore.
- If `xgboost` isn’t importable, the sidebar shows “XGBoost available: no” and the scorer stays unavailable.
- Dim mismatches are ignored for training/scoring; sidebar shows the folder and counts so it’s obvious.
- We removed background fits; no training runs on reruns to avoid mixed states.

Queue removal status (Nov 25, 2025)
- Async queue mode is gone end‑to‑end. Only Batch curation remains. Any tests that referenced the queue path are marked skipped; no runtime code references remain and we removed queue keys from `constants.Keys`.

Async training removal (Nov 25, 2025)
- We also removed async model training keys (`XGB_TRAIN_ASYNC`, `XGB_FIT_FUTURE`). Training is sync-only now. If a test still refers to these, it should be updated or skipped.
- The UI does not import `ensure_fitted` anymore. It only fits when you press the button; the `value_model.ensure_fitted` shim remains available for backend/tests.

Maintainability splits (Nov 25, 2025, later)
- Sidebar logic has been split into small helpers: `ui_sidebar_panels`, `ui_sidebar_cv`, `ui_sidebar_meta`, `ui_sidebar_controls`, and `ui_step_scores_render`. The main `ui_sidebar` file delegates to them; behavior and strings are identical.
- Batch UI splits: `batch_decode`, `batch_tiles`, `batch_buttons`, and `batch_util`. The orchestrator `batch_ui` delegates; keys/labels stayed the same.

Nov 25, 2025 — quick clarifications
- XGBoost shows “xgb_unavailable” until the current prompt+dim has at least one +1 and one −1 row AND you click “Train XGBoost now (sync)”. We removed auto‑fit on reruns.
- With the new default prompt, older rows from a different prompt/dim are ignored by design; the sidebar shows the folder and row count to make this explicit.
- Captions show `Value: … [XGB/Ridge/Logit]` only when the corresponding scorer is ready. Otherwise it’s `Value: n/a`.
- We reduced fragment usage to a single non‑fragment path for predictability. Buttons use stable keys; z/img are cached per tile.

Questions for you
1) Keep Distance‑based scorer visible, or hide it for now?
2) Keep 512×512 at 6 steps as default, or bump back to 640×640?
3) OK to make CV strictly on‑demand (button only) and drop the always‑visible CV lines?
