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
