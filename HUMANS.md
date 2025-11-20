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
