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
