Q (Nov 20, 2025): Why is training still blocking the UI?

A: Because Ridge training was still synchronous and solved a large d×d system (d is the full latent dimension; e.g., 12,544–16,384 for 448–512px). Even when XGBoost training runs in the background (`xgb_train_async=True`), the Ridge fit executed first on the render thread and stalled the page. I switched `latent_logic.ridge_fit` to the dual form `w = X^T (XX^T + λI)^{-1} y`, which makes the solve n×n (n = dataset rows) and removes that stall.

Notes:
- CV already used the dual form and capped rows; training now matches this and stays fast.
- If you still see stalls with very large n, we can either (1) cap rows behind a small “Fast ridge” toggle, or (2) make Ridge async like XGBoost via the existing `background` executor.

Questions for you:
1) When you noticed blocking, which value model was selected (Ridge or XGBoost)?
2) What resolution were you using (so we can infer d), and roughly how many dataset rows were present?
3) Do you want Ridge fits to run fully in the background as well (trade‑off: w updates land a moment later)?

Update (Nov 20, 2025): CV is now on-demand
- We added a “Compute CV now” button and cache both Ridge and XGBoost CV in session state; the sidebar shows the last result and timestamp. This removes CV work from the render loop.
- If you need CV to run automatically after each label, say so and we can add a small toggle.

New (Nov 20, 2025): Ridge async toggle
- Added a tiny policy flag `ridge_train_async` (default False). When set to True (via sidebar checkbox or by setting `st.session_state.ridge_train_async = True`), Ridge fits run in the background using the existing executor. UI stays responsive even for very large d.
- Status isn’t shown separately; you can inspect `st.session_state.ridge_fit_future` if you need to know when it finished. Kept minimal to avoid extra UI.

Open questions (architecture):
1) OK to add a tiny lock around `lstate.w` writes when Ridge async is on? It’s minimal and removes any race risk.
2) Should we unify training status under a single `train_status` dict keyed by model (`{"ridge": "running|ok", "xgb": …}`) instead of separate keys?
3) Do you want CV cache entries keyed by a small fingerprint (rows, λ, xgb params) to avoid confusion when hyperparams change between CV runs?
- Architecture questions (Nov 20, 2025):
1) Is it acceptable to add a tiny lock around `lstate` updates when Ridge async is on (to avoid any chance of races), or do you prefer we keep it minimal and only revisit if we see issues?
2) Do you want all modules to switch to the `ipo` logger now (batch_ui/queue_ui/app) or keep prints for a while and converge gradually?
3) Would you like a simple “decode service” interface (local vs image server) to formalize the boundary, or should we keep direct `flux_local` calls for now?
4) Is it okay to move additional sidebar pieces into `ui_sidebar.py` to slim down `app.py` further, or do you prefer to defer until the next UI pass?
Race when saving many (Nov 20, 2025):
- Cause: dataset row directories were picked by scanning and then creating with `exist_ok=True`, so two fast saves could occasionally choose the same next index.
- Fix: we now create the directory atomically with `os.mkdir(sample_dir)` in a small incrementing loop until success. Folder names remain numeric (`000001`, `000002`, …), and this avoids collisions without adding heavy locking.
Q (Nov 20, 2025): Please show Train score in the sidebar.

A: It is shown in two places: (1) a small metric row near the top next to “Dataset rows”, and (2) inside the “Train results” expander. With no usable rows yet, it displays “n/a” until at least one labeled pair exists (and dims match the current latent size).

Quick answers (Nov 20, 2025, this pass)
- Why no per-step scores? In Ridge, ‖w‖≈0 until you’ve labeled at least one pair; in XGBoost, the scorer is `xgb_unavailable` until a model is trained. Then the sidebar shows “Step scores: …”.
- Why does the next image in Async take long to appear? The queue shows a single visible item and blocks on `future.result()` for that item. The background pool is single-worker by design to avoid CUDA contention. Decode time (≈1.6–2.1 s at 512px, 6 steps) explains the wait. See mitigations above.
- “StreamlitDuplicateElementKey … good_1_10_24”: We now include `(render_nonce, batch_nonce, idx, seq)` in button keys (see `batch_ui.py` around key generation). This prevents rare collisions during fragment re-execution.
- Sidebar fragment error: Writing to `st.sidebar` inside a fragment is unsupported. We now compute the rows metric inside a fragment and render it outside in `ui_sidebar_extra.render_rows_and_last_action`.
- “FLUX_LOCAL_MODEL not set”: Guarded by calling `set_model(DEFAULT_MODEL)` once from the sidebar tail and again from `batch_ui` before any decode, so we don’t hit the env lookup path in `flux_local`.

Refactor notes (Nov 20, 2025):
- I split the sidebar tail into `ui_sidebar.py` and Upload mode into `upload_ui.py`. A tiny `img_latents.py` holds `image_to_z` and `app.image_to_z` now wraps it to preserve imports in existing tests.
- This is structural only; UI strings and behavior remain the same. A new smoke test ensures the sidebar tail renders under our Streamlit stubs.
Debug logs (Nov 20, 2025):
- Sidebar checkbox “Debug logs (set level DEBUG)” toggles the ipo logger level and shows the last N lines of `ipo.debug.log` in a small expander. Default N=200; you can change it inline.

Clarifications (Nov 20, 2025):
- Do you want `value_model` to stop touching `st.session_state` entirely (we can return a tiny status dict instead), or is the current minimal coupling acceptable?
- OK to keep the decode boundary simple (direct `flux_local` calls) and only introduce a `DecodeBackend` shim if/when we wire the image server by default?
- Is gating CV strictly “on button click” acceptable for now, or should we add a “run CV after each label” toggle?

Update (Nov 20, 2025, later):
- Ridge async is now ON by default (`ridge_train_async=True`) to prevent blocking fits; turn it off in the sidebar if you want strictly synchronous updates.
- Per‑step scores: the sidebar block “Step scores: …” appears when a scorer is usable. Conditions:
  - Ridge: requires a non‑zero `w` (at least one labeled pair). With ‖w‖≈0, it shows “Step scores: n/a”.
  - XGBoost: requires a trained XGB model (`xgb_cache.model`); otherwise status is `xgb_unavailable` and “n/a” is expected. Use “Compute CV now” or label a few to trigger training.
- Duplicate Good/Bad keys: fixed by adding a tiny per‑render sequence and batch nonce into keys; also reset the sequence each render to keep keys bounded.

Architecture notes (my take):
- Keep all model training funnelled through `value_model.train_and_record(...)`; no direct fits from UI components. Status lives in `Keys.XGB_TRAIN_STATUS` and `Keys.LAST_TRAIN_*`.
- Treat `lstate.w` as shared mutable state; when Ridge async is enabled we swap‑assign under a tiny lock to avoid races.
- Sidebar extras are now rendered via `ui_sidebar.render_sidebar_extras(...)` so `app.py` stays lean and order is deterministic.
- Keys constants reduce typo bugs; continue the grep‑driven sweep, but don’t churn test stubs that intentionally rely on simple strings.

Questions back to you:
1) Do you want Ridge fits always async, or prefer the previous synchronous default?
2) Shall we expose a small “Fast ridge” mode (subsample dims or rows) for very high‑res runs?
3) OK to remove remaining inline sidebar extras from `app.py` entirely now that `ui_sidebar` covers them?

Note (Nov 20, 2025):
- Dataset rows metric now runs in a fragment with a 1s autorefresh; this limits reruns to that small block. It prefers live in‑session rows over disk and shows a tiny spinner to indicate liveness.

Fix (Nov 20, 2025): IndentationError in value_model.py
- The import/logger init block had inconsistent indentation in the earlier file on your box. I normalized the block and verified with `python -m py_compile value_model.py app.py batch_ui.py queue_ui.py` (no errors). If you still see the error, please restart the Streamlit process to clear old bytecode and reload the module.
Q (Nov 20, 2025): Why does the next image in Async mode take long to appear?

A: Two main reasons in our current design:
- Single shared background pool: background.get_executor() is a single-worker ThreadPool used for both decodes and training. After a label, _queue_label() calls _curation_train_and_next(), which schedules training (Ridge/XGB) onto that same pool before the queue refills. While training runs, decode futures wait in line.
- Blocking wait in queue UI: the visible item calls future.result(), so the fragment blocks until that decode completes (and until any earlier queued jobs finish).

Contributing factors: large width/height or higher steps, per-sample image/NPZ writes, and XGBoost training when enabled. PIPE calls are serialized by a lock (good), so overlapping decode tasks won’t speed up image readiness either.

Quick mitigations (no code):
- Temporarily turn off async training (uncheck “Train Ridge async” / “Train XGBoost async”), or set a positive “Min seconds between trains” so decode gets priority.
- Lower Steps or size for Async mode, or use Batch mode when you want instant next images.
- Watch the Debug logs; [queue] and [perf] lines make any wait cause obvious.

Proposed minimal fixes (pick one):
- 141a: Reorder the queue label path to refill/schedule the next decode before scheduling training (keeps one pool; improves responsiveness).
- 141b: Use a separate executor for training vs decode (CPU-only training runs in parallel; PIPE stays serialized by PIPE_LOCK). Very small change.
- 141c: In queue UI, if future.done() is False, show “decoding…” and return without blocking; the fragment will render the image on the next rerun.

Update (Nov 20, 2025): Ridge status visibility
- The Train results section now shows “Ridge training: running/ok/idle” by checking the async future (`Keys.RIDGE_FIT_FUTURE`) and the current ‖w‖. No extra switches; purely informational.
Per‑state lock for w (Nov 20, 2025):
- We moved the global lock to `lstate.w_lock` so multiple prompts/states don’t contend. Async Ridge fits use this lock to assign `w` atomically per state.

Keys consistency (Nov 20, 2025):
- Added constants for `use_random_anchor`, `images`, and `mu_image` (`Keys.USE_RANDOM_ANCHOR`, `Keys.IMAGES`, `Keys.MU_IMAGE`) and started using them in the app. This keeps session_state access uniform and reduces typo risk.

Sidebar fragment (Nov 20, 2025):
- Streamlit requires writing to the sidebar from inside a fragment to be wrapped in `with st.sidebar:`. The “Dataset rows” auto‑refresh block now runs the fragment inside that context (when available) and falls back gracefully in test stubs.

New (Nov 20, 2025): Persistent label breadcrumb
- After each label, besides the toast, we now write `Saved sample #<n>` into the sidebar as a small persistent line so actions remain visible after reruns.

New (Nov 20, 2025): Last action (timed)
- A compact `Last action: …` line appears at the top of the sidebar for ~6 seconds after any toast (Good/Bad or Save). It reads from `Keys.LAST_ACTION_TEXT`/`Keys.LAST_ACTION_TS` and disappears automatically.

New (Nov 20, 2025): Rows (disk)
- The sidebar now shows “Rows (disk)” next to “Dataset rows” so you can distinguish persisted rows from the live in‑session count (which may update slightly ahead of the on‑disk figure before the next tick).
Q (Nov 20, 2025): what model?

A: stabilityai/sd-turbo (the app’s DEFAULT_MODEL). The sidebar shows it under “Model & decode settings → Model”. Internally, `flux_local.CURRENT_MODEL_ID` reflects the active model after `set_model(...)`. We also added a small guard in `batch_ui` that auto-loads the default model if none is loaded yet, to avoid env lookups.
New (Nov 20, 2025): More CLI prints
- After each label: `[data] saved sample #<n> label=±1 rows_disk=<d> rows_live=<l>`
- Batch clicks: `[batch] click good/bad item=i`; decode timing lines already exist.
- Async queue: `[queue] waiting for decode of item 0…` before blocking.
- Ridge async: `[ridge] scheduled async fit` when submitting background work.
- Prompt embeds: `[pipe] prompt embeds cache: hit/miss` for encode cache visibility.
- Scorer lines: per tile we print `[scorer] tile=i vm=<Value model> v=<score>` (and for the queue item `[scorer] queue vm=<…> v=<…>`) whenever a scorer is available. This is noisy but helps correlate tile decisions with the current value model.
Change (Nov 20, 2025): Split value scorer
- `get_value_scorer_with_status(...)` is now a small dispatcher that calls:
  - `_build_ridge_scorer(lstate)`
  - `_build_xgb_scorer(vm_choice, lstate, prompt, session_state)`
  - `_build_dist_scorer('dist'|'cos', ...)`
- No behavior change; just clearer structure. If you have custom tests that patch the old function, they will continue to work because the public name and return contract are unchanged.
Lint note (Nov 20, 2025):
- I ran `ruff check` (ruff 0.13.1) on the repo; it reported “All checks passed!”. No code changes were needed.
New subpage (Nov 20, 2025): Image match
- Use the Streamlit sidebar “Pages” → “03_image_match” to open it.
- Upload an image; choose size/steps/guidance; click “Step” to iteratively improve a latent so the decoded image matches the target (pixel MSE). “Auto ×5” runs 5 steps. Original and last attempt are displayed side by side with the current MSE.
- Notes: This is a simple NES-like hill climb; it tries a few random directions per step and keeps the best. It’s GPU-heavy and not guaranteed to converge; keep steps small and width/height reasonable (e.g., 384–512) for responsiveness.

Refactor (Nov 20, 2025 — sidebar/modes)
- Extracted “Mode & value model” into `ui_sidebar_modes.render_modes_and_value_model`; app delegates.
- Extracted “Model & decode settings” into `ui_sidebar_extra.render_model_decode_settings`; app delegates.
- Sidebar tail rendering lives in `ui_sidebar.render_sidebar_tail`.
- Upload subpage lives in `upload_ui.run_upload_mode`; image→latent util in `img_latents.image_to_z` (wrapped by `app.image_to_z` for back‑compat).
- Added smoke `tests/smoke/test_smoke_sidebar_tail.py` to validate the refactored sidebar path with our Streamlit stubs.
- Writer robustness: when a test swaps out `st.sidebar`, we install best‑effort `write()`/`metric()` shims that append to `st.sidebar_writes`. If you want, I can add an extra hook to route those to the specific external `writes` list used by one CV expander test — say the word, and I’ll wire it up.
Q (Nov 20, 2025): “make the cli show hum of dataset rows” — clarify “hum”?

- Unsure if you meant “hum” as a spinner/vibe line, or “sum”/“live count”. I added a tiny spinner next to the live rows metric in the sidebar (`|/-\` cycling) and a log line `[rows] live=<L> disk=<D> disp=<text>`. If you want a CLI-only ticker (stdout) instead, say so and I’ll add a `--cli` mode to `rich_cli.py` to print it at 1 Hz.

Answer: Training blocking and async queue latency

- Root cause: training and decode used the same single-worker executor. A long ridge/XGB fit could completely serialize all queued decodes, so “next image” in async mode waited for training to finish.
- Fix: added a separate `background.get_train_executor()` and switched value model fits to use it. Decodes stay on the default pool; both pools are single‑worker to be CUDA-friendly. This keeps UI snappy while still avoiding heavy contention.

Answer: Per-step scores showing “n/a”

- In XGBoost mode, step scores are hidden until an XGB model exists (explicitly no Ridge fallback by design). If you want Ridge‑based step scores while XGB is still training/unavailable, say the word and I’ll add an opt‑in toggle.

Answer: FLUX_LOCAL_MODEL error

- `flux_local` now defaults to `DEFAULT_MODEL` (`stabilityai/sd-turbo`) when `FLUX_LOCAL_MODEL` is unset and logs the choice. Batch UI also calls `set_model` early.
Refactor summary (Nov 20, 2025):
- Moved sidebar control assembly to `app_main.build_controls` and state reset/pair init to `app_state` to slim `app.py` and keep responsibilities clear.
- Centralized sidebar tail writes (Train score, CV, Last train, Value scorer) in `ui_sidebar` so tests don’t rely on expanders.
- Split executors: decodes vs training to keep UI responsive.
- Queue alias: we mirror `Keys.QUEUE` into a plain `'queue'` for tests that read the legacy key.
- Default batch size is now 4 (from 25). Tests reference `constants.DEFAULT_BATCH_SIZE` instead of a literal.

Open questions for you:
- Do you want app.py strictly under 400 lines? If yes, I’ll move the remaining run/dispatch/footer glue to `app_main.run()` and leave only re‑exports in `app.py`.
- OK to add a tiny `app._force_render_sidebar_tail()` helper for tests that re-import with different Streamlit stubs without clearing `sys.modules['app']`?
Updates (Nov 20, 2025 — refactor + async)

- App split finalized: controls/dispatch/state moved into small modules (`app_main`, `app_run`, `app_state`, `app_api`). `app.py` is now under 400 lines.
- Flux default model: if `FLUX_LOCAL_MODEL` is unset we now default to `stabilityai/sd-turbo` in `flux_local` (with a tiny unit test).
- Async training: Ridge and XGBoost fits now run on background executors by default. Futures are exposed in session state (`ridge_fit_future`, `xgb_fit_future`), and we set `xgb_last_updated_rows/lam` in the async path so the sidebar can reflect progress immediately.
- Non‑blocking threshold: simplified executors (no Streamlit context wiring) to keep `fit_value_model(..., ridge_train_async=True)` returning in <150 ms in tests.
- Rows metric: implemented a safe auto‑refresh that computes in a fragment and writes to the sidebar outside the fragment (fixes the `st.sidebar`/fragment error).
- Duplicate keys: batch buttons now include a render nonce + batch nonce + index + seq to avoid `StreamlitDuplicateElementKey` crashes.

Notes for you
- If “Train results”/CV still feels heavy, I can gate CV behind a “Compute CV now” button (131b) and cache the last result/timestamp.
- If you want a per‑tile “scorer: …” line in the CLI (153c), I can leave it on by default or hide behind `IPO_LOG_LEVEL=DEBUG`.
Notes (Nov 20, 2025 — refactor follow‑up)
- Tests now write datasets to a per‑run temp root (IPO_DATA_ROOT) for deterministic row counts. Real app writes remain under `data/`.
- If a test asserts sidebar strings like “Step scores: n/a” without importing the app first, the line may be missing because no UI rendered yet. In the full suite, import order typically renders the sidebar early. If you run such a test alone, import `app` first or I can adjust the test helper to render the sidebar tail when a Streamlit stub is detected.
- Queue labels shrink the list immediately and write “Queue remaining: N” to the sidebar for quick confirmation.
Q&A (Nov 20, 2025)
- Why is training still blocking the UI?
  - Ridge and XGBoost now default to async (`ridge_train_async=True`, `xgb_train_async=True`). UI stalls can still happen if: (1) a one‑shot ensure‑fit runs synchronously when a usable dataset appears for XGB, (2) you manually triggered CV (Compute CV now), or (3) the background executors are unavailable and we fall back to a minimal sync fit. Logs show `[ridge] scheduled async fit` or `[xgb] train … (async submit)` when non‑blocking.
- Why no per‑step scores?
  - We show `Step scores: n/a` until a usable scorer exists. That’s when `‖w‖>0` (Ridge) or an XGBoost model is cached. Once available, the sidebar lists a compact `Step scores: …` line and a few step metrics.
- Duplicate element key error in Batch?
  - Fixed: button keys include `render_nonce`, `cur_batch_nonce`, and a small `btn_seq` counter. Example: `good_{render}_{batch}_{idx}_{seq}`. This prevents `StreamlitDuplicateElementKey` even under fragments.
- Sidebar + fragments error?
  - Streamlit forbids writing to `st.sidebar` inside a fragment. The auto‑refreshing rows metric now computes in a fragment and renders outside in `ui_sidebar_extra.render_rows_and_last_action`.
- “Dataset rows” not increasing?
  - The metric now counts only rows that match the current latent dim; old 512px data is ignored when you switch to 640px, so new labels start from 0 and tick up. We still show “Rows (disk)” for the persisted, dim-matched count. A tiny spinner next to the live metric updates once per second.
- No toast after labeling?
  - Fixed: we emit a toast on label and also write a persistent `Saved sample #n` line in the sidebar. The last toast text appears as a compact `Last action` line for a few seconds.
- Next image in async mode is slow — why?
  - Decoding runs on a single worker by default to avoid CUDA contention; expect ~1.6–2.1s per 512×512 Turbo image at 6 steps. To speed up: reduce width/height/steps, or raise the executor worker count (riskier on CUDA). Model warmup is automatic (we call `set_model(DEFAULT_MODEL)` at import).
- Which model is used by default?
  - `stabilityai/sd-turbo`. `flux_local` guards ensure the model is set on first decode even under fragments.

Operational hints
- If you see many background saves at once, we write per‑row files atomically; large bursts can still stress storage. If needed, we can add a simple file lock or queue the writes explicitly.
- Logs: use `IPO_LOG_LEVEL=INFO|DEBUG`. We also print `[rows] live=… disk=…` and concise `[train-summary] …` lines after fits.

Reminder
- If you type exactly `blueberries` in chat, I will reply with it reversed: `seirrebeulb`.
Notes (Nov 20, 2025, late)
- Refactor: unified training via `value_model.train_and_record`, fixed XGBoost async scheduling, and set async defaults (`xgb_train_async`, `ridge_train_async`) early.
- Sidebar rows now updates via a fragment tick but writes to the sidebar outside the fragment (avoids Streamlit’s sidebar/fragment error). CLI logs `[rows] live=… disk=…` each tick.
- Tests: the runner drops `app`/UI/backend modules before each test so per‑test Streamlit stubs take effect. If you see a test relying on prior import order, favor the `scripts/run_tests.sh` path.
- Next: stabilize import‑time sidebar lines (Value model/Train score/Step scores/XGB active) and ensure XGB one‑shot fit happens before writing Train score when only in‑memory X/y is present.
Nov 20, 2025 — Notes for humans

- Why is training still blocking the UI?
  - Ridge/XGBoost fits now submit to a dedicated single‑worker train executor (`background.get_train_executor`). UI paths only set small flags and return. If you see blocking, it’s likely a synchronous path was hit (e.g., ensure‑fit for first model, or when session_state is a plain dict in tests). Toggle `ridge_train_async=True` (default) to avoid sync ridge fits.

- Why no per‑step/value scores?
  - The sidebar shows “Step scores: n/a” until Ridge has a non‑zero `w`, or an XGBoost model is cached. We added an opportunistic `ensure_fitted()` in the sidebar tail so, when a dataset exists, we train once and then the scores appear. If your dataset is empty or `w≈0`, “n/a” is expected. Tests: `tests/test_step_scores_visible_with_weights.py` (renders when `w≠0`) and `tests/test_step_scores_count_matches_iter_steps.py` (score count honors the optimization step slider).
- Why no toast on queue labels?
  - Queue Accept/Reject now fire `st.toast` with “Accepted (+1)” / “Rejected (-1)” and still log “Saved sample #n” via the batch path. Tests: `tests/test_queue_toast_label.py` (accept) and `tests/test_queue_toast_reject.py` (reject) stub `st.toast` and assert the message is emitted.
- Rows in CLI?
  - After every label we log `[rows] live=<len(session_dataset_y)> disk=<rows_on_disk>`; the print no longer crashes on numpy arrays. Test: `tests/test_rows_cli_print.py` captures stdout and asserts the line.
- XGBoost scorer availability?
  - We now also honor a cached scorer from `xgb_value.get_cached_scorer(prompt, session_state)` when no model is stored in `xgb_cache`. Test: `tests/test_xgb_scorer_available.py` stubs a cached scorer and asserts status=`ok`.
  - We also cover the normal cached-model path: `tests/test_xgb_scorer_model_cache.py` stubs `score_xgb_proba` with a tiny bias and asserts status=`ok` and that the scorer runs.
- Save path visibility
  - Toasts and sidebar breadcrumb now include the save directory: `Saved sample #<n> → data/<hash>/<row>` so you can see exactly where the sample went. Test: `tests/test_saved_path_in_toast.py`.
- Black/blank images after a while?
  - If latents collapse or init sigma mismatches the scheduler, SD-turbo can output near-black frames. Check the sidebar Debug panel for `latents_std` near 0 and `init_sigma`; rerun `scripts/sanity_decode.py` at your current size/steps to confirm. Also, if you changed resolution (e.g., 640px → d=25600) and the dataset is empty/dim-mismatched (`Rows` shows 0), `w` stays ~0 and proposals may hover near the anchor, sometimes yielding flat decodes. Try: lower res to 512, step count 6, ensure `FLUX_LOCAL_MODEL` is set, and watch the Debug checkbox for latents_std.
- Dim mismatch warning
  - Sidebar now shows: “Dataset recorded at d=<old> (ignored); current latent dim d=<new>” when persisted data doesn’t match the active resolution. Test: `tests/test_sidebar_dim_mismatch_warning.py`.
- Sample images saved alongside data
  - `save_sample_image` writes `data/<hash>/<row>/image.png`; covered by `tests/test_save_sample_image_writes_png.py` which asserts the PNG is created at the expected path.
- Value per image
  - Each batch tile caption now includes `Value: …` (or n/a). Test: `tests/test_batch_value_caption.py` stubs a scorer and asserts the caption contains the value.
- Queue value per image
  - Async queue items now show the predicted value in the caption. Test: `tests/test_async_queue_value_caption.py`.
- Scheduler steps guard
  - If `steps=None` is passed to latents decode we default to 20 and still call `set_timesteps`; test: `tests/test_flux_latents_steps_default.py` (stubs PIPE and asserts timesteps are set).
- Scheduler timesteps coverage
  - `_run_pipe` now has explicit coverage to ensure timesteps and `_step_index` are set before calling the pipe. Test: `tests/test_run_pipe_sets_timesteps.py`.
- Latents stats coverage
  - Added `tests/test_last_call_logs_latents_stats.py` to assert `LAST_CALL` captures `latents_std`/`latents_mean` after a decode (stubbed pipe).
- Turbo guidance clamp logging
  - `tests/test_guidance_turbo_clamp_last_call.py` checks that guidance is clamped to 0.0 for turbo models and recorded in `LAST_CALL.guidance`.
- CLI detail logging
  - `_run_pipe` now prints a detailed line (`model=… steps=… size=… guidance=… latents_std=… init_sigma=…`). Test: `tests/test_run_pipe_cli_detail.py` captures stdout and asserts the fields.
- CLI prints
  - `_run_pipe` now logs `[pipe] set_timesteps steps=<n> device=cuda` when timesteps are set, to help diagnose scheduler state.
- Async queue value captions: queue images now render `Item i • Value: …` just like batch tiles. Test: `tests/test_async_queue_value_caption.py`.

- Streamlit DuplicateElementKey errors (e.g., `good_1_10_24`).
  - Batch buttons now derive keys from a render nonce + batch nonce + a per‑render sequence, keeping keys unique even under fragments/reruns.

- Meta-tensor loader error (Nov 20, 2025, later):
  - `_ensure_pipe` now always treats a NotImplementedError from `.to('cuda')` as a trigger to reload with `device_map='cuda'` (requires `low_cpu_mem_usage=True`). If that also fails (meta/device_map), we reload on CPU with `low_cpu_mem_usage=True` and then `.to('cuda')`. This should eliminate the “Cannot copy out of meta tensor” crash seen on app start.
  - Added tests for the second meta path and device_map ValueError so these fallbacks stay covered.
- Turbo guidance brownness:
  - Forced guidance_scale to 0.0 whenever the model id contains “turbo” (text and latents paths). Prevents CFG from flattening colors on sd‑turbo. Test: `tests/test_guidance_turbo_zero.py`.
  - Also skip prompt-embed caching when effective guidance is 0 to avoid generating unconditional CFG pairs that wash out color; falls back to plain prompt strings.
  - Latents path is covered too; we clamp guidance and send the prompt string there as well.
  - Added a CLI sanity check `python scripts/sanity_decode.py` (defaults: sd-turbo, 512px, steps=6). Fails (exit 1) if image std < 30 to catch flat/brown outputs quickly.
  - Sidebar now shows “Effective guidance” (read-only). Turbo models display 0.00 so you can see when clamping is active. Test: `tests/test_sidebar_effective_guidance.py`.
  - To check if brown is turbo-only, run once with SD‑1.5 without code changes: `SANITY_MODEL=runwayml/stable-diffusion-v1-5 python scripts/sanity_decode.py` (uses your GPU and may download weights).

- Sidebar fragment error.
  - We compute the rows metric inside a fragment but write to `st.sidebar` outside of it. Never call `st.sidebar` inside a function wrapped with `st.fragment`.

- FLUX_LOCAL_MODEL missing / meta‑tensor error.
  - We set the model at import to `DEFAULT_MODEL` and fall back to that in `flux_local._get_model_id()` when the env var is missing. For the meta‑tensor `NotImplementedError`, `_ensure_pipe` now reloads with `device_map='cuda'` and `low_cpu_mem_usage=True` (required by diffusers when dispatching with a device map).

- Async queue feels slow.
  - Decodes serialize through a pipeline lock for stability. Queue items are scheduled in the background; the first visible item blocks on `future.result()`. We now run the decode executor with 2 workers to overlap scheduling/host work. Increase `QUEUE_SIZE` so subsequent items are decoded by the time you label the first. Keeping steps small (6) and resolution moderate (512–640) helps.

- “Dataset rows don’t increase” or missing toast.
  - Each label appends to in‑memory X/y and the on‑disk dataset; we write “Saved sample #n” to the sidebar and a toast (when available). The top “Dataset rows” metric auto‑refreshes once per second.

- Debugging help.
  - Turn on the “Debug” checkbox in the sidebar to see last pipeline stats and a tiny tail of `ipo.debug.log` (line count adjustable). Set `IPO_LOG_LEVEL=DEBUG` in the shell for more verbose logs.

- App organization
  - `app.py` is now a thin orchestrator (<400 lines). Generation helpers and mode dispatch live in `app_run.py`. Tests that call `app.generate_pair()` or `app._queue_fill_up_to()` still work via tiny wrappers that delegate to `app_run`.
  - Prompt-first bootstrap (no decode on import): `app_bootstrap.prompt_first_bootstrap` sets placeholders (`images=(None,None)`, `prompt_image=None`) during import. First real decodes happen only when you click or when a mode runs. This keeps import fast and avoids unexpected GPU work.
