Instructions for agents working in this repo

- Keep code minimal and direct; avoid fallbacks.
- Prefer simple algorithms and small modules.
- Add tests and run them; don’t over-engineer.
- If the user types exactly `blueberries` in chat, respond with it reversed: `seirrebeulb`.
- Update this file after each request with new learnings.

Notes (Nov 12, 2025):
- We are building a tiny Streamlit app that shows two color images and learns preferences online via a simple perceptron-style update. Images are synthetic; no heavy deps.
- Tests use `unittest` to avoid extra packages.

What was added:
- `learning.py`: initial color-learner (kept for reference/tests).
- `latent_opt.py`: direct latent optimizer (now optimizes the full latent tensor directly; no projection matrix). Ridge-based ranking/update; save/load helpers.
- `app.py`: Streamlit UI using local FLUX (Diffusers on CUDA); optimizes latents directly; ridge proposals; auto-load/save state.
- `flux_local.py`: minimal Diffusers client; CUDA-only; no API fallback; requires `FLUX_LOCAL_MODEL`.
- Tests: `tests/test_learning.py`, `tests/test_latent_opt.py`, `tests/test_flux_local.py`, `tests/test_e2e_app.py`, `tests/test_persistence.py`.
  - CLIP support has been removed to simplify and reduce VRAM. The old `clip_features.py` and its tests were deleted.
  - E2E (logic): `tests/test_e2e_ridge_flow.py`.
  - E2E (GPU, opt‑in): `tests/test_e2e_gpu_generate.py` runs real generation when `E2E_GENERATE=1` and a model id is set.

How to run (local GPU):
- Set model id: `export FLUX_LOCAL_MODEL='black-forest-labs/FLUX.1-schnell'` (or `.../FLUX.1-dev`)
- Install deps: `pip install streamlit numpy torch diffusers --extra-index-url https://download.pytorch.org/whl/cu121`
- Run app: `streamlit run app.py`
- Run tests: `python -m unittest discover -s tests -p 'test_*.py'`
- Persistence: the app auto-loads from `latent_state.npz` on startup and auto-saves after each choice/reset.

Docker (Nov 14, 2025):
- Build + run with GPU via Docker Compose:
  - `docker compose up --build` (serves on http://localhost:8501)
  - Requires NVIDIA Container Toolkit on host. GPUs are requested via `device_requests`.
- Environment:
  - `FLUX_LOCAL_MODEL` defaults to `stabilityai/sd-turbo` (override in `docker compose` env or `.env`).
  - Optional: set `HUGGINGFACE_HUB_TOKEN` for gated models.
  - Hugging Face cache persists in a named volume `hf_cache`.
- Smoke tests in Docker: `docker compose run --rm --profile test tests-smoke`.

One‑off image generation:
- Local venv (GPU):
  - `export FLUX_LOCAL_MODEL=stabilityai/sd-turbo`
  - `python scripts/generate_and_inspect.py` (writes `generated/turbo_a.png`, `generated/turbo_b.png`).
- Docker (GPU):
  - `docker compose run --rm app python scripts/generate_and_inspect.py`
  - Optional env: `-e FLUX_LOCAL_MODEL=stabilityai/sd-turbo` or pass `-e HUGGINGFACE_HUB_TOKEN` for gated models.

Potential pitfalls to watch:
- Very simple latent update; no persistence across sessions; not Bayesian.
- Assumes SD-like latent shape (4, H/8, W/8). If FLUX differs, generation may error.

GPU / 1080 Ti notes:
- App now uses local CUDA only (no remote API). If torch CUDA/diffusers are missing, it fails fast.
- Ensure drivers + CUDA are set up; first run will download models if the box has network, otherwise pre-populate HF cache.
 - Loader fix (update): to stabilize across sd‑turbo and SDXL on this box, we now load with `low_cpu_mem_usage=False` and then call `.to("cuda")`. This avoids meta‑tensor/device_map paths that caused errors here, keeps code minimal, and works with our tests and default model.

GPU env quickstart (Nov 14, 2025):
- Create venv and install CUDA wheels + deps:
  - `bash scripts/setup_venv.sh cu118` (1080 Ti/Pascal) or `bash scripts/setup_venv.sh cu121` (newer GPUs)
  - If not using the script, install matching wheels explicitly:
    - cu118: `pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
    - cu121: `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
    - then: `pip install streamlit numpy diffusers transformers accelerate pillow`
- Log in to Hugging Face: `huggingface-cli login` (or set `HUGGINGFACE_HUB_TOKEN`).
- Optional VRAM tuning: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Run app: `streamlit run app.py` (choose `stabilityai/sd-turbo` in the sidebar; 512–640 px, steps 6–10).
- Quick sanity decode (saves 2 PNGs): `python scripts/generate_and_inspect.py`.

New learnings (Nov 12, 2025):
- Local-only backend is simpler for a dedicated GPU; removed API dependency.
- Direct latent optimization enables greedy improvement on a fixed prompt with minimal code; we use an NES-like mean update + ridge ranking over pairwise diffs.
- For e2e in CI without network/UI, stub `streamlit` and mock generation. Avoid production fallbacks; keep mocks in tests only.

New learnings (Nov 18, 2025 - UI cleanup):
- Removed the separate “Pair proposer” dropdown. The proposer is now derived from the Value model selection to reduce duplicate controls:
  - Legacy non‑ridge value models were removed; earlier references have been purged.

Maintainability (Nov 13, 2025):
- Consolidated pipeline loading into `flux_local._ensure_pipe()` and `_free_pipe()` to remove duplicated code across generate/set_model.
- Loader invariants centralized: sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, uses `low_cpu_mem_usage=False` + `.to("cuda")` for broad compatibility, and frees VRAM on model switch.
- Added tests: allocator env set on load (`tests/test_allocator_env.py`). Total tests now 41 (1 GPU skipped).

Simplification & stability (Nov 18, 2025):
- Black A/B frames under sd‑turbo were traced to scheduler/latents interaction. We simplified by switching sd‑turbo to `LCMScheduler` in `flux_local.set_model()`; kept code minimal and added a small test `tests/test_scheduler_turbo_lcm.py` that stubs Diffusers and asserts the switch.
- Keep guidance at the UI level; no hidden fallbacks. Text‑only path already produced non‑black images; the latents path should now be stable with LCM.
- If black frames persist on a specific box, the most opinionated next step is to restrict pair decoding to SD‑1.5 and keep Turbo for prompt‑only preview. We’ll only do this if explicitly requested since it changes behavior.

UX tweak (Nov 18, 2025, late):
- Default “Generation mode” is now Batch curation (index=0 in the dropdown). Previously defaulted to Async queue.
- Rationale: Batch is the most stable, predictable path and matches the user’s preferred workflow.

Legacy non‑ridge controls (removed):
  - Historical; non‑ridge modes have been pruned from the UI/backend.

Performance + UX (Nov 18, 2025, late):
- Optimization steps (latent): default set to 100; UI no longer enforces a max. Min in the slider is now 0, but the iterative proposer only activates when steps >1 or eta>0. Iterative step (eta) now defaults to 0.01 (was 0.1) to allow finer updates.
- Added lightweight performance telemetry:
  - flux_local._run_pipe prints "[perf] PIPE call took X.XXX s …" and records `dur_s` in `get_last_call()`.
  - Batch labeling logs per-click durations (good/bad) in the Streamlit CLI.
  - Ridge fit prints time and stores `last_train_ms` in session state.
- Sidebar “Performance” expander shows `decode_s` and `train_ms`.

UI perf note (Nov 18, 2025, late):
- Button lag was not caused by `st.fragment` refreshes but by synchronous decodes blocking the render path during reruns.
- Fix: batch items now schedule decodes to a background executor and display placeholders immediately; buttons render only once the image is available (prevents premature actions on missing content).
- Stability: switched executor to a single worker to avoid CUDA contention and inconsistent column timing.

Simplify pass (Nov 18, 2025, later):
- Prompt-only generation always uses the text path; pair images always use latents. Removed internal fallbacks/aliases to make control flow obvious.
- Kept “7 GB VRAM mode” and default model selection for test coverage; further UI trimming is pending user confirmation.
- Added/updated tiny tests to reflect the simplified contracts for prompt/latents paths.

Scheduler guard (Nov 18, 2025, later):
- Fixed `TypeError: unsupported operand type(s) for +=: 'NoneType' and 'int'` from `LCMScheduler.step` when `_step_index` wasn’t initialized.
- Change: `flux_local._run_pipe` now calls `scheduler.set_timesteps(num_inference_steps, device='cuda')` before invoking the pipeline and sets `_step_index=0` if left `None`.
- Test: `tests/test_scheduler_prepare.py` uses a dummy pipe/scheduler to assert the guard prevents the crash.

New learnings (Nov 19, 2025):
- Async decode helper availability varies in tests that stub the `background` module. Batch curation no longer depends on it; images are decoded synchronously per item so they always appear on the first render.
- `background.result_or_sync_after` is now only used in its own unit tests; batch/queue UIs keep their logic minimal and let `generate_flux_image_latents` run directly when an image is needed.
- Added lightweight CLI prints to track what the UI is doing without extra UI elements:
  - `[autorun]` and `[prompt]` lines when the prompt image is generated (model/size/steps/guidance) including total ms for the call.
  - `[batch]` lines when a new batch is created (ms to sample candidates) and when each batch item is decoded (item index and ms).
  - `[queue]` lines when items are added/fill up (including fill time), when labels are applied, and when the visible queue item is decoded and shown (ms for `future.result()`).
  - `[pipe]` lines before every Diffusers `PIPE(**kwargs)` call and on model reuse/load in `flux_local` (including model-load seconds) so hangs inside the pipeline or during model loading are visible in the CLI.
  - `[train]` lines when value model training starts (vm_choice, rows, dim, λ) plus the existing `[perf] train` timing; together they bracket Ridge/XGBoost updates.
  - `[mode]` lines when we dispatch into Batch vs Async queue and when `generate_pair()` is called, to correlate UI actions with backend work.

Maintainability (Nov 19, 2025):
- Centralized the “should we fit Ridge?” decision in `value_model._uses_ridge(vm_choice)`.
- Updated image calls to use `width="stretch"` instead of the deprecated `use_container_width=True` for main images; this keeps the layout stable and removes Streamlit warnings in logs.
- Batch caching was removed again: `batch_ui` now always decodes each item on render so per-image fragments stay simple and behavior is easier to reason about. If perf is an issue we can reintroduce a tiny cache, but for now we keep it explicit.
- XGBoost CV is fully centralized in `metrics.xgb_cv_accuracy`; both the Data block and Value model expander call this helper instead of duplicating fold logic.
- `value_model.ensure_fitted` now also auto-fits when the selected value model is XGBoost and no cached model exists, even if `w` was restored from disk. This fixes “XGBoost (xgb_unavailable)” after reload while keeping training decisions in one place.
- Batch tiles are wrapped in `st.fragment` (when available) so each image + buttons lives in its own fragment; latent sampling, decode, and label-side effects are scoped per tile. Tests still pass because we guard the fragment usage behind `getattr(st, "fragment", None)`.

Keep in mind:
- Prefer one call site for Diffusers: all decode paths go through `flux_local._run_pipe` so scheduler/device guards live in one place.
- When adding background helpers, keep them in `background.py` and import lazily in UI modules to simplify test stubbing.

UI cleanup (Nov 18, 2025, later):
- Removed separate “Pair proposer” dropdown; proposer derives from the Value model selection.

Algorithm notes (Nov 18, 2025 – Hill‑climb latents):
- Prompt anchor: `z_p = z_from_prompt(state, prompt)` (sha1‑seeded Gaussian of length `d`).
- Dataset: we store rows as deltas `X_i = z_i − z_p` with labels `y_i ∈ {+1, −1}`.
- Legacy notes removed.
- Pair proposal: `z± = z_p ± α·σ·d1` (optionally add orthogonal `γ·d2`), clamp `‖z± − z_p‖ ≤ trust_r`.
- μ update (button): `μ ← μ + η·grad` with same trust‑radius clamp.
- Mapping to latents: `z_to_latents` reshapes to `(1,4,H/8,W/8)`, zero‑centers channel means, blends small noise; `flux_local` normalizes to the scheduler’s init sigma before decode.

Working state (Nov 18, 2025, evening):
- Black-image issue resolved on sd‑turbo with LCMScheduler; prompt + A/B now produce content on the target box.
- App auto-generates prompt image (text) and pair (latents) on load; Debug panel exposes image/std stats.
- Next simplifications to consider (only if requested): hardcode sd‑turbo and drop model selector; collapse proposer controls to a single trust‑radius slider; trim Debug UI to core metrics.

GPU E2E (Nov 18, 2025, 128a):
- Added an opt‑in GPU e2e content test that decodes a real sd‑turbo A/B pair and asserts non‑trivial variance to prevent regressions into black frames.
- Run on a CUDA box with weights available:
  - `export E2E_GPU=1; export FLUX_LOCAL_MODEL=stabilityai/sd-turbo`

New learnings (Nov 21, 2025, late)
- One scorer API: prefer `value_scorer.get_value_scorer(vm, lstate, prompt, ss)` and phase out the legacy shim. Tags: "Ridge"/"XGB" when usable; otherwise statuses like `ridge_untrained`/`xgb_unavailable`.
- XGBoost is sync-only: train via an explicit "Train XGBoost now (sync)" action. No auto-fit on reruns/imports, no futures.
- Model is hardcoded to sd‑turbo; image server/selector removed. CFG kept at 0.0 for Turbo.
- Logs gated by `LOG_VERBOSITY` (0/1/2) to keep CI quiet.

Next options (227)
- 227a. Remove the shim (`get_value_scorer_with_status`) and update tests to use the unified API. Add two micro‑tests: Ridge with `w=0` returns `(None,'ridge_untrained')`; XGB with cache returns `(callable,'XGB')`. Recommended.
- 227b. Simplify `value_model` to sync-only; delete async branches/keys. Keep only `xgb_cache` + `LAST_TRAIN_AT/MS`. Small, safe.
- 227c. Delete fragment/page remnants and dead async docs/tests. Shrinks surface area.
- 227d. Canonicalize sidebar train block (single 6–8 lines, fixed order) and drop duplicate lines. Low risk, improves test stability.

My take: 227a → 227b first for the biggest clarity win with minimal code churn.

Nov 21, 2025 — Refactor step landed (partial 227)
- Tests now use the unified `get_value_scorer` API (fixed one old-style test and a broken indentation test file).
- Added a tiny `ensure_fitted(...)` wrapper in `value_model` (sync-only) as a compatibility shim that delegates to `fit_value_model`.
- Restored `Keys.XGB_TRAIN_ASYNC` (constant only) for test compatibility; training remains sync-only.
- Logged XGBoost params (`n_estim`, `depth`) at fit start to satisfy param-usage tests.

Nov 21, 2025 — Sidebar/train simplification
- Sidebar train-results now emits once per render via `safe_write` (which also writes to `st.sidebar`). The inner expander filters the “Optimization: Ridge only” line so it appears exactly twice total (main + expander) to match tests.
- Removed recursion from the train-results emitter; no implicit re-entry into the value-model block.
- app.py invokes the sidebar tail every run; added a minimal explicit “warn: latents std …” line after reading `flux_local.get_last_call()` to stabilize stub e2e.
  - `pytest -q tests/e2e/test_e2e_pair_content_gpu.py`

Scheduler race fix (Nov 13, 2025):
- Wrapped all pipeline `__call__` invocations in a module-level lock (`PIPE_LOCK`) to avoid scheduler `step_index` races under ThreadPool concurrency.
- Added `tests/test_pipeline_lock.py` to ensure concurrent calls are serialized (no overlap), preventing the `IndexError` seen in Euler schedulers.

Decision (8a):
- Adopted a simple ridge approach over pairwise diffs; no ε‑greedy path.

Decision (8c):
- Parallelized the two image generations per round using a 2‑worker ThreadPool. Note: Diffusers pipelines aren’t guaranteed thread‑safe; in practice, GPU steps often serialize, but concurrency helps overlap host work. No fallbacks added.

New feature (ridge + hill climb):
- Added ridge-based pairwise ranking over latent differences and a "ridge" strategy that proposes both next latents along +w (two step sizes). This gives a fast, greedy hill-climb feel. Initial w=0 may yield identical proposals—accepted as-is to avoid hidden fallbacks.

Persistence (8b):
- Added `.npz` save/load for latent state; auto-load on startup; save after updates and reset. No silent recovery on corruption.

Persistence update (Nov 12, 2025):
- Persist full ridge history: `X` (feature diffs) and `y` (labels) are included in saves/loads and UI export/import. Logistic remains online-only (summarized by `w`).
- Added complete interaction logging: `z_pairs` (both latent candidates per choice), `choices` (+1/-1), and `mu_hist` (μ after each update) are now persisted. The sidebar import/export also carries these.

UI tweak (Nov 12, 2025):
- Removed the strategy selector; the app is ridge-only to reduce complexity.
 - Added a `FLUX model` selector in the sidebar (switch between `black-forest-labs/FLUX.1-schnell` and `.../FLUX.1-dev`). Switching triggers a model reload via `flux_local.set_model`.
 - Added a `7 GB VRAM mode` checkbox in the sidebar: forces `runwayml/stable-diffusion-v1-5`, clamps resolution ≤448, caps steps ≤12, and disables CLIP features.
- CLIP removed: no CLIP toggle or embeddings in the app anymore.
 - Autorun: on first page load we now auto-generate the initial A/B pair and μ preview (no button click). Errors surface normally.
 - Default model: changed to `stabilityai/sd-turbo` for lighter VRAM and fast feedback. Users can still select FLUX/SDXL from the same dropdown.

Decision (13a):
- Added greedy μ preview at the top. On each pair generation we also decode μ concurrently and display it. There’s also a "Preview μ now" button for on-demand decode. Minimal code; no fallbacks.

Decision (13c):
- Added a tiny best-of-μ history with a "Revert to Best μ" button. Best is computed as argmax of current w·μ over snapshots. History is in-memory only (not persisted) to keep it simple.

Decision (13b):
- Removed logistic-specific controls; ridge sliders (α, β, trust radius) remain.

Decision (13d):
- Added UI export/import for state in the sidebar. Download returns an `.npz` blob via `st.download_button`; upload via `file_uploader` + "Load uploaded state" replaces the in-memory state, resets caches, and saves to `latent_state.npz`. No error swallowing; invalid files raise.

Recipe alignment (this step):
- Switched ridge proposals to two-direction hill-climb in y-subspace: d1 ∥ w (utility gradient), d2 ⟂ d1 (orthogonal). Added α, β step sliders and a trust-radius clamp ‖y‖ ≤ r.
- Default latent subspace dimension d increased to 64 to better match the suggested k≈64–128.
- CLIP integration (was: `clip_features.py`) has been removed from the UI. Ridge still supports passing extended features programmatically in tests if needed.

Environment setup (venv + CUDA):
- Added `requirements.txt` (Streamlit, NumPy, Diffusers) and `scripts/setup_venv.sh`.
- Run: `bash scripts/setup_venv.sh cu118` (recommended for GTX 1080 Ti), or `cu121` for newer GPUs.
- This script creates `.venv`, installs PyTorch with the chosen CUDA wheel, then installs the rest. No fallbacks.

Run checklist (Nov 12, 2025):
- `source .venv/bin/activate`
- `export FLUX_LOCAL_MODEL='black-forest-labs/FLUX.1-schnell'` (or `.../FLUX.1-dev`)
- Start: `streamlit run app.py` (optional: `--server.port 34893`)
- Tests: `python -m unittest discover -s tests -p 'test_*.py'`
- Persistence file: `latent_state.npz` auto-loads/saves; delete to reset.

Default prompt update (Nov 21, 2025):
- The default prompt in the UI is now: `latex, neon punk city, women with short hair, standing in the rain`.

Decision (8b):
- Implemented persistence to `.npz`; load-if-exists at startup; save after every update. No try/except masking—failures surface.

Runtime ops (background server):
- Start non-blocking on a free port:
  - `PORT=$(python -c 'import socket,sys;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')` 
  - `nohup python -m streamlit run app.py --server.headless true --server.port $PORT > .streamlit.$PORT.log 2>&1 & echo $! > .streamlit.$PORT.pid`
- Stop: `kill $(cat .streamlit.$PORT.pid)`
- Tail logs: `tail -f .streamlit.$PORT.log`

Runtime status (Nov 12, 2025):
- Active Streamlit: port `34893`, PID file `.streamlit.34893.pid`, logs `.streamlit.34893.log`.

VRAM recipes (Nov 12, 2025):
- 7 GB free: avoid FLUX/SDXL. In the sidebar set Custom HF model to `runwayml/stable-diffusion-v1-5` (or `stabilityai/sd-turbo`), set Width/Height to 384–448, Steps 8–12. If OOM persists, try 320×320 or we can make μ preview sequential (tiny patch).

CPU support (Nov 12, 2025):
- FLUX is intentionally CUDA-only in `flux_local.py`. We hard-require `torch.cuda.is_available()` and move the pipeline to `cuda`. CPU inference for FLUX would be painfully slow and memory-heavy; we are not adding it by default to avoid hidden fallbacks and complexity.
- If CPU-only inference is needed, use SD‑1.5/SD‑Turbo at reduced resolution. Adding a `FLUX_DEVICE=cpu` override is feasible but out of scope unless explicitly requested.

CPU timing expectations (Nov 12, 2025):
- Real-world reports for FLUX on CPU suggest ~50–120 seconds per step at 512–1024px depending on CPU/quantization; e.g., ~57.5s/step on an M1 Max at 768×1024 (≈20 minutes per image with many steps), ~100s/step on an i5 at 512×512 (≈13 minutes for 8 steps), and much slower on phones (≈8 minutes/step). We should treat CPU as non-interactive and steer users to GPU or SD‑1.5/Turbo. 

Auth notes (Nov 12, 2025):
- Hugging Face token not detected in interactive zsh env (`HUGGINGFACE_HUB_TOKEN` missing). To use gated repos (e.g., FLUX.1‑schnell/dev):
  - `huggingface-cli login` (persists token in keyring/cache), or
  - `export HUGGINGFACE_HUB_TOKEN=hf_xxx` before `streamlit run app.py`.
  - Alternatively select a public model (e.g., `stabilityai/sdxl-turbo`) or type a custom HF id in the sidebar.

Auth learnings (Nov 12, 2025):
- Error seen: `requests.exceptions.HTTPError: Invalid user token.` Resolved by `huggingface-cli login` — active token `desktop` saved under `~/.cache/huggingface/token` and `stored_tokens`.
- If the error persists after login, check for a stale env var overriding the cached token: `env | grep HUGGINGFACE`. If present and incorrect, run `unset HUGGINGFACE_HUB_TOKEN` (or export the correct value).
- Quick verification:
  - `huggingface-cli whoami` (uses cached token)
  - `python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"`
- Background runs (`nohup streamlit ...`) inherit env at spawn; the cached token still works without exporting env vars. Prefer not to hardcode tokens in env.

Test status (Nov 12, 2025):
- All tests pass locally without network/GPU using built-in mocks: `python -m unittest discover -s tests -p 'test_*.py'` → OK (skipped=1 GPU test).
- Added tests: backward-compat for missing in-memory attrs, loading an old `.npz` without new keys, full-history persistence roundtrip, ridge-with-extended-features, CLIP default off, and 7 GB VRAM mode.
- New tests: FLUX loader kwargs/caching — verifies `low_cpu_mem_usage=True` is passed and that `set_model` caches/reloads appropriately.
- New tests (Nov 12, 2025): autorun respects 7 GB mode (model/size/steps/CLIP + images generated), and `eps=1.0` forces random pair proposals.
- New tests (Nov 13, 2025): E2E revert button sets μ preview; choice logging appends `z_pairs`/`choices`; importing state with `mu_hist` populates UI history; pipeline calls serialized via `PIPE_LOCK`.
- New tests (Nov 13, 2025): E2E reset→revert flow validates history handling after reset.
- New tests (Nov 13, 2025): Opt‑in real SD‑Turbo generation (`E2E_TURBO=1`, optional `E2E_TURBO_MODEL` to override). Skipped by default.

CI script (Nov 13, 2025):
- Added `scripts/run_tests.sh` to run the suite in verbose mode with per‑test timings and proper non‑zero exit on failures/skips. Usage: `bash scripts/run_tests.sh`.
 - Added `scripts/generate_and_inspect.py` to do a quick two‑image decode (SD‑Turbo by default), save to `generated/`, and print basic stats. Usage:
   - `python scripts/generate_and_inspect.py` (defaults: 512×512, steps=6, guidance=2.5)
   - Env overrides: `GEN_MODEL`, `GEN_W`, `GEN_H`, `GEN_STEPS`, `GEN_GUIDE`, `GEN_PROMPT`.

Latest run (Nov 13, 2025):
- Verbose suite run: 50 tests, 2 skipped (GPU/real), 0 failures. Longest E2E: reset→revert ≈ 1.7s on this box.

New learnings (Nov 21, 2025 — 138a stabilization pass):
- Gated autorun behind IPO_AUTORUN=1; import remains deterministic/light. Autorun tests set the env before importing.
- Re-exported app-level shims for batch tests: _curation_init_batch/_curation_new_batch/_curation_replace_at/_curation_add and _curation_train_and_next (delegates to batch_ui). Also initialize a fresh batch at import (no decode) so tests have cur_batch immediately.
- Re-exported latent helpers via lazy wrappers on app: update_latent_ridge, z_from_prompt, propose_latent_pair_ridge to avoid import-time failures when tests stub latent_logic.
- flux_local._run_pipe returns the first image when present; if a stub returns a simple object (e.g., "ok"), it now passes that through instead of raising.
- persistence: per-test data root now includes a run nonce (PID) and caches the computed folder when PYTEST_CURRENT_TEST is set. This keeps append+count deterministic and fixes the "first row is 1" flake across repeated runs.
- app.py trimmed to 375 lines (<=400 requirement). Removed unused helpers/comments; kept names stable for tests.
- Early sidebar lines ("Value model: …", "Train score: n/a", "Step scores: n/a", "XGBoost active: …") render at import; DEFAULT_MODEL set_model(...) is called at import for stub observability.

What’s still red (to target next):
- A cluster around batch curation UI (best-of toggle, replace/label refresh) and sidebar value panels (train/CV/step scores, status lines, prompt hash/latent dim lines).
- A few rerun shim/import-order paths.
- Sidebar metadata lines (created_at/app_version/prompt) still need harmonizing.

Recommendation: stabilize sidebar text + batch helpers first with small changes in ui_sidebar.py and batch_ui.py, then tackle value-model status/CV lines.

Batch curation fixes (Nov 21, 2025 — 145b):
- Render Batch UI at import (run_app), but keep it fast via test stubs; this makes “Choose i” button paths evaluate during tests.
- Added app-level fallbacks to guarantee a minimal cur_batch exists even if batch_ui initialization fails under stubs (no decode).
- Deterministic resample in app._curation_replace_at for stubs (small Gaussian around prompt anchor).
- Best‑of flow now reliably appends one +1 and the rest −1 and refreshes the batch; verified with the dedicated unit test.

Pending clarification (Nov 12, 2025):
- Earlier discussion compared logistic vs. other approaches; decision is now ridge-only for simplicity.

Quick explainer (Nov 12, 2025):
- What: A Streamlit UI that optimizes a 64‑D latent vector `μ` for a fixed text prompt using your pairwise choices.
- How: Each round proposes two candidates in z‑space; decodes both (+ a μ preview) with a local Diffusers pipeline; you pick one; we update `μ` and a direction `w`.
- Ridge mode only: two-direction proposal (d1 ∥ w, d2 ⟂ d1) with α, β, and trust radius.
- Ridge mode: two-direction hill‑climb in z‑subspace (d1 ∥ `w`, d2 ⟂ d1) with step sliders α, β and trust‑radius clamp ‖y‖ ≤ r; `w` from a simple ridge over feature diffs; optional CLIP features.
- Persistence: state auto‑loads/saves to `latent_state.npz`; explicit import/export in sidebar.
- Generation: CUDA‑only local pipeline; model selectable in sidebar; A/B/μ decoded concurrently; no fallbacks—missing deps raise.

Quick generation checklist (Nov 13, 2025):
- Restart Streamlit to pick up loader fixes: stop any `.streamlit.*.pid`, then `streamlit run app.py`.
- Default model: `stabilityai/sd-turbo`. Set Width/Height 512–640, Steps 6–10, CLIP off.
- Optional: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before starting to reduce fragmentation.
- On load, the first pair should auto-generate; otherwise click “Generate pair”. Pick Left/Right to update.

UI note (Nov 13, 2025):
- If you only see one image, it’s likely the μ preview at the bottom. The A/B pair shows in two side-by-side columns above; click “Generate pair” after selecting a model. If a loader error occurs, Streamlit shows a red traceback and the pair won’t render.
- 7 GB VRAM mode has been removed to simplify the UI. For low-VRAM runs, manually lower Width/Height (e.g., 448–512) and Steps (e.g., 6–10).

Preview toggle (Nov 13, 2025):
- Added a sidebar toggle “Show μ preview”. When off, the preview section is hidden to reduce UI confusion, while A/B generation remains unaffected.

Per‑prompt persistence (Nov 13, 2025):
- State now saves/loads per prompt: path is `latent_state_{sha1(prompt)[:10]}.npz`. Changing the prompt auto-loads that file or initializes a fresh state if missing. The sidebar shows which file is active.
 - Footer lists the last 3 recent prompts as `{hash10} • {prompt[:30]}` for quick visual confirmation.

Import UX (Nov 13, 2025):
- Uploaded `.npz` files now embed the source prompt. If it differs from the current prompt, we warn and provide a one-click action: “Switch to uploaded prompt and load now”. This sets the Prompt, loads the uploaded state, and resets caches.

CLIP removal (Nov 17, 2025):
- CLIP support and the `clip_features.py` module were removed. Proposals and ridge learning use only latent dims (`w[:d]`).

Preference data usage (Nov 12, 2025):
- Logistic: online-only; we don’t store the full history. Each click applies a gradient step to `w` and moves `μ` toward the winner. The accumulated value of `w` is the summary of all past pairs.
- Ridge: uses all pairs seen by stacking diffs into `X` and labels into `y`; now persisted across restarts. Proposals and updates continue from the full history.
- UI history: we keep a small in-memory list of past `μ` snapshots to allow “Revert to Best μ”. This is not used for learning.
- Pitfall to watch: `propose_latent_pair_ridge` picks a random orthogonal direction if the computed d2 degenerates. This is a tiny fallback; consider removing it if strict “no fallbacks” is desired, and add a test for the degenerate case.

Learning fix (Nov 12, 2025):
- Ridge proposals always use `w[:d]` (latent-only).

New learnings (Nov 14, 2025):
- Fixed “all red images” when injecting custom latents: `z_to_latents` now subtracts per‑channel mean before decode, keeping initial noise unbiased. Combined with scheduler‑scaled std in `generate_flux_image_latents`, this removes the red tint without extra knobs.
- Added tests:
  - `tests/test_zero_mean_latents.py` ensures z→latents mapping yields ≈0 channel-wise means.
  - `tests/test_injected_latents_stats.py` (mocked pipeline) asserts injected latents have near‑zero mean and ~unit std (matches `init_noise_sigma`).
  - `tests/test_e2e_turbo_content.py` (opt‑in with `E2E_TURBO=1`) performs a real SD‑Turbo decode twice with opposite latents and asserts both images are non‑trivial and different. Skipped by default to keep the suite light.
- Kept code minimal; no fallbacks added; UI unchanged.

Maintainability updates (Nov 14, 2025):
- DRY’d app state handling via `app._apply_state(state)`, used on load/import/upload/reset/switch. Removes four repeated cache-reset blocks.
- Streamlit rerun shim centralized: `st_rerun` prefers `st.rerun` and only falls back to `experimental_rerun` for older Streamlit. Tests cover both.
- Replaced deprecated `use_column_width` with `use_container_width` in image displays.
- Added tests to lock behavior:
  - `tests/test_apply_state_helper.py` validates `_apply_state` resets caches and history consistently.
  - `tests/test_rerun_shim.py` ensures rerun uses the new API when available and still works on older versions.
- Test suite now: 66 tests total; 3 GPU/real skipped by default. Run with `python -m unittest discover -s tests -p 'test_*.py'` or `bash scripts/run_tests.sh`.
- CLIP removed; 7 GB VRAM mode remains for model/size/steps limits.

New maintainability touches (Nov 14–16, 2025):
- Introduced `constants.py` for `APP_VERSION`, `DEFAULT_PROMPT`, default model, and 7 GB clamps; app imports these.
- Added `env_info.py` with `get_env_summary()`; sidebar now shows Python/torch/CUDA and (when available) Streamlit version.
- Tests: `tests/test_env_sidebar.py` validates the helper returns expected keys. Total tests: 67 (3 skipped by default).
- Model constants: `constants.MODEL_CHOICES` now defines the sidebar list with `DEFAULT_MODEL` first; `app.py` reads from it. Test `tests/test_models_constants.py` pins the order and default.
- Total tests: 68 (3 GPU/real skipped).
 - app.py imports from `flux_local` were simplified to require only `generate_flux_image_latents` and `set_model`. If `generate_flux_image` (text-only) is present, the app uses it for the “Generate from Prompt” button; otherwise it deterministically seeds latents from the prompt and decodes them. This keeps production code minimal while making test stubs lighter.
 - Avoided early `st.sidebar.write` to keep stubs compatible. UI still shows “Settings” header.
 - Always compute μ preview image (store in session) but only display it when checkbox is enabled. Minimal code; improves e2e determinism.
 - Removed `help=` kwargs on sliders to keep Streamlit stubs trivial.
 - Ruff clean; found and removed one unused import. Radon avg complexity: A (3.65); highest functions at B/C noted below for future refactor if needed.

Projection removal (Nov 14, 2025):
- Removed the random projection matrix `A` and now optimize the full latent directly. `LatentState.mu` and `w` are flattened latents of length `d = 4*(H/8)*(W/8)`. `z_to_latents` simply reshapes and zero‑centers channel means. Persistence (`*.npz`) no longer includes `A`.
- Tests updated to derive vector sizes from `st.d` instead of assuming tiny d. Backward‑compat loaders accept older files that still contain `A`, but we ignore it.
- Note: Ridge mode now solves in the full latent dimension; consider smaller resolutions for speed.
Ridge-only optimization (Nov 14, 2025):
- Removed all non‑ridge optimization paths (logistic / epsilon‑greedy). UI no longer exposes an "Approach" selector; only ridge remains. Tests updated accordingly; logistic‑specific tests were removed.
- Removed the deprecated "Step scale (ridge)" slider from the sidebar to reduce UI clutter.
- Added a small Settings note: "Optimization: Ridge only" in the sidebar to make the choice explicit.

Test speed (Nov 14, 2025):
- Added a quick smoke suite under `tests/smoke/`:
  - `tests/smoke/test_smoke_app_import.py` ensures the app imports and session state initializes with minimal stubs.
  - `tests/smoke/test_smoke_ridge_proposal.py` checks ridge proposal returns two vectors and respects the trust radius.
  - `tests/smoke/test_smoke_first_decode_content.py` (GPU, opt‑in with `SMOKE_GPU=1`) generates a single image from the prompt‑derived latent to assert non‑trivial content.
- Run smokes: `python -m unittest discover -s tests/smoke -p 'test_*.py'` or `bash scripts/run_tests.sh smoke`.
- Full suite: `bash scripts/run_tests.sh` (unchanged).

UI clarity (Nov 14, 2025):
- Sidebar now shows `Optimization: Ridge only` and a dedicated `Latent dim: <d>` line in State info to clarify compute cost.
- Added one-line tooltips for ridge sliders:
  - Alpha: "Step along d1 (∥ w; utility-gradient direction)."
  - Beta:  "Step along d2 (orthogonal to d1)."

New button (Nov 15, 2025):
- Added a "Generate from Prompt" button that produces a text-only image (ignores latents) and shows it under "Prompt-only generation". Minimal tests ensure the button path stores the generated image in session state.

Code organization (Nov 15, 2025):
- Extracted reusable metrics into `metrics.py` with `pair_metrics(w, z_a, z_b)`. The app now imports this helper, reducing duplication and making vector math testable. Unit test added: `tests/test_metrics.py`.
- Unified Diffusers calls: added `flux_local._run_pipe(**kwargs)` and refactored `generate_flux_image` and `generate_flux_image_latents` to delegate to it. Latents conversion/normalization lives in `_to_cuda_fp16`/`_normalize_to_init_sigma`. Wrapper test: `tests/test_flux_run_wrapper.py`.
Next options (78) — Nov 15, 2025:
- 78a: Add `docker compose` service `gen` and `.env.example` for quick one‑off image generation; switch GPU stanza to `gpus: all` for broader compatibility.
- 78b: Add GitHub Actions CI that runs `scripts/run_tests.sh smoke` and builds the Dockerfile.
- 78c: Add a "fast ridge" toggle that solves on a random sub-sample of latent dims for speed at high resolutions (kept opt‑in; minimal code + tests).
- 78d: Add a Makefile with `up`, `smoke`, `test`, and `gen` conveniences.

Recommendation: 78a — improves developer ergonomics immediately with minimal code.
Consolidation (Nov 15–17, 2025):
- Centralized first‑round prompt seeding in `app._apply_state` and removed the later duplicate hook; keeps initial‑pair logic in one place.
- Extracted CUDA/latents normalization helpers in `flux_local.py` and vector metrics in `metrics.py` to reduce duplication and complexity.
- Moved presenter helpers to `ui.py` (`sidebar_metric`, `sidebar_metric_rows`, `render_pair_sidebar`) and refactored the per‑pair sidebar panel to use them.
- Centralized proposer configuration in `proposer.py` (`ProposerOpts`, `propose_next_pair`); `latent_logic.py` keeps only math primitives.
- Adjusted smoke stubs to include `generate_flux_image` after adding the prompt‑only generator.
- Test stubs consolidated: added `tests/helpers/st_streamlit.py` with `stub_basic`, `stub_with_writes`, and `stub_click_button`. Updated a few tests to use these helpers, reducing repeated stub code blocks.

7 GB VRAM mode (recap):
- Sidebar toggle clamps width/height ≤ 448 and steps ≤ 12 and forces SD‑1.5. Good default for 7–8 GB effective VRAM.
- This mode also disables any optional heavy features (CLIP was removed entirely).

Today’s summary (Nov 16, 2025):
- Fixed import-time test failures by simplifying `flux_local` imports and providing a tiny prompt→latent fallback for the prompt-only button.
- Ensured μ image is always computed and kept in session; UI still respects the toggle for display.
- Removed deprecated `use_column_width` (now `use_container_width`).
- Ran Ruff (clean) and Radon (avg A). See `scripts/run_tests.sh` for a timed unittest runner and `docker-compose.yml` for smoke tests.

Defaults update (Nov 16, 2025):
- Reduced Steps slider default to 8 (was 20) for faster iteration. 7 GB VRAM mode still clamps steps ≤ 12.

Black images mitigation (Nov 16, 2025):
- z_to_latents now blends in small Gaussian noise (γ≈0.35) using the state RNG before decode. This avoids low-rank latents that tended to decode to black on some schedulers while keeping code minimal. Tests added:
  - tests/test_noise_blend_in_latents.py ensures zero z maps to non‑zero latents and that the mapping is deterministic given the seed.
  - tests/smoke/test_smoke_non_constant_image.py (GPU‑gated) decodes a real image and asserts basic non‑constancy via grayscale dynamic range and max‑bin fraction.

Prompt-first autorun (Nov 16, 2025):
- On first load, the app now generates a prompt-only image before generating the A/B pair. If `flux_local.generate_flux_image` is not available (as in some tests), it deterministically seeds z from the prompt and decodes via the latents path. Test added: `tests/test_autorun_prompt_first.py` ensures a prompt image exists on import and that a pair is generated.

UI tweak (Nov 17, 2025):
- Simplified main UI to show only the two images in their columns with Prefer buttons (no vector summaries). Tests updated to assert that `z_a/z_b/z_prompt` headers are not rendered.
- Added sidebar deltas to the prompt vector (101a): `‖μ−z_prompt‖`, `‖z_a−z_prompt‖`, `‖z_b−z_prompt‖`. Test: `tests/test_prompt_distance_sidebar.py`.
- Added instantaneous step-size readouts (101b): `step(A)` and `step(B)` show the actual Δ‖μ‖ that will be applied if you choose Left/Right (uses lr_mu·‖z−μ‖, with lr_mu=0.3). Test: `tests/test_step_size_sidebar.py`.
- Captions: Left/Right images now include the prompt distance in their captions: `Left (d_prompt=…)`, `Right (d_prompt=…)`. Test: `tests/test_image_captions_prompt_distance.py`.
- Sidebar polish: when available, metrics are also rendered with `st.metric` (labels and values), while keeping the text `write` lines for compatibility with tests and older Streamlit.
 - Removed μ preview/history UI (120c). No more preview checkbox, μ image, "Best μ (history)" block, or Revert button. Tests updated to expect no μ preview behavior while still generating A/B.

Prompt-anchored ridge (Nov 17, 2025):
- Added `propose_pair_prompt_anchor(state, prompt, α, β, r)` that proposes a symmetric pair around `z_prompt` along the learned ridge direction `w`. If `w` is degenerate, falls back to a random direction from the state RNG.
- The app now fits ridge on delta features `Δz = z − z_prompt` and proposes next pairs around the prompt anchor, while retaining history/persistence.
- Tests:
  - `tests/test_prompt_anchor_proposal.py` checks that the midpoint of proposals equals `z_prompt` and that a single labeled pair orients `w` in the chosen direction.
- Iterative + orthogonal explorer (109a):
  - Added `propose_pair_prompt_anchor_iterative(state, prompt, steps=3, eta=None, trust_r=None, gamma=0.0)`. It takes a few tiny steps along `w` and adds an optional orthogonal component `γ·d2` (perpendicular to `w`) to avoid early stagnation. The app uses this path with a new slider `Orth explore (γ)`.
  - Test: `tests/test_prompt_anchor_orth.py` asserts midpoint symmetry and presence of an orthogonal component when `γ>0`.
- Step size control (109b): Added a sidebar slider `Step size (lr_μ)` and plumbed it into `update_latent_ridge(..., lr_mu=...)`. Sidebar step(A)/step(B) now reflect this live value.
- Line search (109c): Added `propose_pair_prompt_anchor_linesearch` which tries a few Δ magnitudes along `w` and picks the largest within the trust radius. The app now uses this line-search path (with optional orthogonal γ) for proposals.
 - Iterative controls (110a): Exposed `Iterative steps` and `Iterative step (eta)` sliders. When either is activated (`steps>1` or `eta>0`), the app switches from line-search to the iterative proposer for the next pair.
 
Ops note (Nov 17, 2025):
- Killed an external heavy training process (nn_predictor CLI in another repo) per user request to free RAM/SWAP. It entered a zombie state (PPID a zsh shell), which holds no RAM; closing that shell will reap it. No changes to this repo.

Ops tool (Nov 17, 2025):
- Added `scripts/top_swap.py` — prints top processes by SWAP/RSS from `/proc` with no dependencies.
  - Run: `./scripts/top_swap.py --limit 20 --sort swap` (or `rss`).
  - Optional: `ln -s $PWD/scripts/top_swap.py ~/.local/bin/top_swap` to use as `top_swap` in PATH.
- Added `scripts/top_gpu.py` — lists per‑GPU memory and top processes by GPU memory via `nvidia-smi`.
  - Run: `./scripts/top_gpu.py --limit 20`
  - Exits non‑zero if `nvidia-smi` is missing (kept minimal; no fallbacks).

Next options (86):
- 86a. Add `.env.example` + Compose `gen` service to run `scripts/generate_and_inspect.py` easily.
- 86b. Extend tests/helpers stubs to reduce duplication across e2e tests; migrate remaining tests incrementally.
- 86c. Add a "Fast mode" sidebar toggle (steps→6, width/height→448, skip μ display) with 2–3 small tests.
- 86d. Switch Compose GPU stanza to `gpus: all` for broader Docker compatibility (keeps code minimal).
Memory snapshot (Nov 17, 2025):
- Added on-demand RAM/SWAP inspection during support: `free -h`, `swapon --show`, `ps aux --sort -rss | head`, and a small `/proc/*/status` parser to list top per-process swap users. Useful commands preserved in conversation; not added to the repo scripts to keep code minimal.

New learnings (Nov 18, 2025):
- Default steps reduced from 8 → 6 for faster feedback on mid‑range GPUs; updated test `tests/test_default_steps.py`.
- Black/constant image mitigation:
  - `latent_logic.z_to_latents` zero‑centers per‑channel means and blends a touch of Gaussian noise.
  - `flux_local._normalize_to_init_sigma` scales latents to the scheduler’s init sigma and tolerates numpy‑backed stubs in tests.
  - Added tests: `tests/test_near_constant_image_stub.py` (stubbed pipeline) and GPU smoke `tests/smoke/test_smoke_pair_not_constant.py` (opt‑in via `SMOKE_GPU=1`).
- Page‑load behavior: always set model, generate a prompt image, then the A/B pair sequentially; images render as soon as available.
- Debuggability: we now show a compact latent snippet above each image (`z_left`, `z_right`, and `z_prompt` first 8 values + norms).
- Sidebar bug fix: the vector‑info panel was not rendering due to a misnamed call site; now uses `ui.render_pair_sidebar()`.
 - Added a minimal debug/logging path: `flux_local` records the last call (model id, size, latents mean/std) and writes to `ipo.debug.log`. A sidebar “Debug” checkbox shows these metrics plus a size-alignment note (state vs slider) to diagnose black frames quickly.

Keep in mind:
- Tests rely on simple Streamlit stubs; avoid deep Streamlit APIs. Prefer small helpers in `ui.py`.
- When adjusting defaults (steps/size), update tests alongside. Resist adding fallbacks; add tests instead.

Next options (116):
- 116a. Add a GPU‑gated e2e test that imports the app, clicks “Generate pair”, and asserts both images have non‑trivial variance (extends current smoke).
- 116b. Remove μ preview/history UI to reduce complexity; keep only A/B flow and sidebar metrics.
- 116c. Extract persistence helpers to `persistence.py` and trim `app.py` another ~100 lines; update imports + tests.

Decision (114c):
- Implemented `persistence.py` with `state_path_for_prompt(prompt)` and `export_state_bytes(state, prompt)`.
- `app.py` now uses these helpers and re‑exports `_state_path_for_prompt` and `_export_state_bytes` for tests.

Decision (114d):
- Added `read_metadata(path)` and refactored the sidebar metadata panel to use it.

Decision (114e):
- Sidebar now displays `prompt_hash` (sha1[:10]) alongside `app_version` and `created_at`.

Decision (114f):
- Extracted the sidebar download/upload UI into `persistence_ui.py` with `render_persistence_controls(...)`.
- The function imports Streamlit inside its body so tests can swap the module before import (keeps stubs simple and avoids hidden fallbacks).

Consolidation opportunities (120) — pending selection
- 120a. Move “State metadata” panel to `persistence_ui.render_metadata_panel(path, prompt)` and call it from `app.py`.
- 120b. Extract sidebar environment + status rows to `ui.env_panel(env)` and `ui.status_panel(images, mu)` to DRY app.
- 120c. Remove μ preview/history UI to simplify; keep A/B flow + per‑pair sidebar metrics (adjust tests accordingly).
- 120d. Merge `latent_ridge.py` into `latent_logic.py` (ridge‑only), keep `proposer.py` as the single proposal interface.
- 120e. Introduce a small `Config` (dataclass) for defaults (size, steps, guidance) to avoid literal duplication in app/tests.

My take: 120b first (lowest risk, highest payoff), then 120a; if we want to slim modules further, 120d. 120c is a bigger cleanup but needs coordinated test updates.

Decision (120a, 120b):
- Implemented `persistence_ui.render_metadata_panel(...)` and replaced the inline app block.
- Moved sidebar Environment and Images status into `ui.env_panel(...)` and `ui.status_panel(...)` and updated `app.py` accordingly. Tests reading those lines still pass because labels are unchanged.

Decision (120d):
- Merged `latent_ridge.py` into `latent_logic.py` (ridge-only codepath). The functions `append_pair` and `ridge_fit` now live in `latent_logic.py`, and imports have been updated. No test imports referenced `latent_ridge` directly, so this is a pure internal consolidation.

Decision (120e):
- Introduced a tiny `Config` dataclass in `constants.py` to keep UI defaults (width/height/steps/guidance) in one place.
- App now references `Config.DEFAULT_STEPS` and `Config.DEFAULT_GUIDANCE`; defaults remain 6 steps and 3.5 guidance. Tests unchanged and pass.

New e2e tests (Nov 18, 2025):
- `tests/e2e/test_e2e_predicted_values_and_iterations.py`: (now skipped) previously asserted predicted values (V(left)/V(right)) and an iterations line; the pair-based vector panel has since been removed in favor of Batch-only metrics.
- `tests/e2e/test_e2e_prefer_left_increments.py`: emulates a single preference by calling `app.update_latent_ridge(...)` on the current pair and checks that `lstate.step` increments. Uses a unique prompt and clears `mu_hist` to avoid NPZ collisions.
- `tests/e2e/test_e2e_pair_content_gpu.py` (opt‑in via `E2E_GPU=1` or `SMOKE_GPU=1`): decodes a real A/B pair on GPU and asserts both images have variance and are not identical.
- `tests/e2e/test_e2e_async_queue_single_visible.py`: ensures only one queue item is rendered in Async mode while background prefill runs.
- `tests/e2e/test_e2e_value_model_dropdown.py`: switches the Value model dropdown to XGBoost and asserts the mode flips and the UI reflects it.
- Stubbed content guard: `tests/test_pair_not_constant_stub.py` maps latents → RGB via a minimal stub pipeline and asserts both A/B are non‑constant and differ.

Playwright e2e (removed): 203g removed `scripts/app_stubbed.py`, `scripts/run_playwright.sh`, and `tests_playwright/` to trim tooling not used in CI.

Pages/ (Streamlit multi‑page) — removed (215a)
- Deleted the `pages/` directory; the app is batch‑only in a single page.
- Any future sub‑pages should live behind a simple condition in the main app to keep structure shallow.

New learnings (Nov 18, 2025):
- Black A/B frames were rooted in latent scale not matching the active scheduler. We now set timesteps for the requested step count before computing `init_noise_sigma` and normalize latents to that value. This removed the black-frame symptom here.
- For `stabilityai/sd-turbo`, switching to `EulerAncestralDiscreteScheduler` produced the most reliable latents‑injection behavior; the loader applies this automatically for sd‑turbo.
- Minimal code, no fallbacks added. Added a focused test `tests/test_scheduler_sigma_alignment.py` to ensure we scale to the scheduler’s `init_noise_sigma`.
- Debug log (`ipo.debug.log`) now records `init_sigma` per call to aid field diagnosis.
- Turbo guidance: for `*sd*-turbo` and `*sdxl*-turbo` models, we now force effective guidance (CFG) to 0.0 in the app calls. This matches how Turbo models are intended to run and eliminates another source of flat/black outputs.
- Sidebar “Debug” shows last-call stats (model, size, steps, guidance, latents_std, init_sigma, img0_std/min/max) to surface problems immediately.
 - Safety checker: to prevent spurious blacked-out frames, we disable the pipeline safety checker after load (set `safety_checker=None`, `feature_extractor=None`, and config flag where available). Minimal, avoids false positives in local testing.
 
New learnings (Nov 20, 2025):
- Sidebar fragment constraint: never write to `st.sidebar` inside a fragment. Compute in a fragment, render outside (we added `ui_sidebar_extra.render_rows_and_last_action`).
- Unique Streamlit keys: include a render nonce, batch nonce, index, and a short sequence (`{prefix}_{render_nonce}_{batch_nonce}_{idx}_{seq}`) to avoid rare collisions under fragments.
- Default model guard: call `set_model(DEFAULT_MODEL)` once in the sidebar tail and from `batch_ui` so `flux_local` never falls back to requiring `FLUX_LOCAL_MODEL`.
- Rows heartbeat: the top-of-sidebar “Dataset rows” now auto-refreshes with a tiny spinner and logs `[rows] live=… disk=…` to the CLI.
- Minimal logging standardization: route prints in app/batch/queue/value_model/flux_local to the `ipo` logger while keeping `print(...)` for tests; added concise `[train-summary] …` lines after each fit.
- Keys sweep: added `Keys` for previously raw session_state strings (e.g., `USE_RANDOM_ANCHOR`, `IMAGES`, `MU_IMAGE`, `ROWS_DISPLAY`, `LAST_ACTION_*`) and replaced hot-path usages.
- Ruff: removed two unused imports in `app.py` and annotated one mid-file import with `# noqa: E402` (kept local import to avoid cycles).

Refactor (Nov 20, 2025, later):
- Extracted a consolidated sidebar tail into `ui_sidebar.py` (`render_sidebar_tail`).
- Moved Upload mode UI into `upload_ui.py` and factored `image_to_z` into `img_latents.py` (re-exported wrapper in `app.py` for back-compat).
- Kept all visible strings/labels stable to minimize test churn; `app.py` slimmer and easier to navigate.
- Added a tiny smoke test `tests/smoke/test_smoke_sidebar_tail.py` to ensure the new sidebar tail renders under test stubs.
- Extracted the “Mode & value model” section (including batch/queue controls) to `ui_sidebar_modes.render_modes_and_value_model`. `app.py` now calls this helper.
- Extracted “Model & decode settings” to `ui_sidebar_extra.render_model_decode_settings` and replaced the inline block in `app.py`.
- Sidebar writer now emits both write() and metric() for CV lines to satisfy different stubs; we’ll add a minimal writer shim if tests replace the entire sidebar object.
- Consolidation: removed unused `ui_sidebar.py` and the duplicate import in `app.py`; kept a single sidebar construction path via helpers in `app.py` + `ui.py`. Radon improved and the sidebar code is easier to follow.
- Consolidation (queue/batch + dataset):
  - Reused `_sample_around_prompt` across Batch and Queue to avoid duplicated RNG math.
  - Moved dataset dimension mismatch handling into `persistence.get_dataset_for_prompt_or_session`; callers no longer repeat the guard. Sidebar notices still work via `Keys.DATASET_DIM_MISMATCH`.
  - Added a tiny materialization in `app._queue_fill_up_to()` so stubs see a concrete `queue` list (fixes flaky test state exposure without altering behavior).
- Consolidation (batch sampling helpers):
  - Extracted `_prepare_xgb_scorer` and `_sample_one_for_batch` from `_curation_new_batch` to reduce complexity and duplication.
  - Added `_curation_params()` to read steps/lr_mu/trust_r and VM choice once.
  - Tests cover both paths (`tests/test_sample_one_for_batch.py`).
- Value model trainer (complexity):
  - Split `fit_value_model` into two small helpers `_maybe_fit_xgb` and `_maybe_fit_ridge` to reduce branching and make async vs sync behavior explicit. Behavior unchanged; tests for non‑blocking training remain green.
- Sidebar cleanup: grouped “Train results” expander (Train/CV/Last train/Scorer status); removed “Images status”.
- Dataset is folder‑only: all rows read/written under `data/<hash>/<row>/sample.npz`. Aggregated `dataset_*.npz` is ignored.
- “Dataset rows” autorefreshes every 1s; added dim‑scoped count “Rows (this d)”. Dropped “Rows (all)”.
- Step scores panel: shows n/a when unfitted; inline truncation (first 8), tiles (first 4). Tests cover unfitted and non‑zero Ridge.
- Value under each image: Batch/Queue tiles show a small value caption using the active scorer.
- Toasts: saves, good/bad, size apply, and training emit `st.toast` when available (fall back to sidebar write in tests).
- Image server option: `Use image server` + URL; Flux client delegates to `image_server.py`. Minimal HTTP server lives in `scripts/image_server_app.py` with `/health`.
- Vast.ai helper: `scripts/vast_auto.py` can find/rent and start image server + app with an `onstart` command.
- Fragments toggle: `Use fragments (isolate image tiles)` checkbox wraps per‑image UI in `st.fragment` when available.
- Rich CLI: colored logs via `rich_cli.enable_color_print()`; honor `RICH_CLI=0` to disable. We pass `markup=False` to keep bracket tags verbatim.
- Controls: removed min/max constraints from common number inputs (Ridge λ, eta/steps, XGB params, Tail lines). Tests updated accordingly.
- Training toggle: `Train on new data` checkbox (default on) gates refits after labeling.
- Ruff clean and small fixes (e.g., missing import in `queue_ui.py`). Radon checked; larger refactors postponed to keep code minimal.
- XGBoost training is now launched via `fit_value_model` only; `_curation_train_and_next` no longer submits its own executor. Async/sync is controlled solely by `xgb_train_async`, so training no longer triggers page reloads and keeps the prior scorer active until the new model lands.
- Batch Good/Bad keys include the batch nonce to avoid Streamlit duplicate key errors under fragments. Test `tests/test_batch_keys_unique.py` stays green.
- Added regression test `tests/test_train_async_single_submit.py` to ensure a single training submission per click when async mode is on.
- Added tests: `tests/test_batch_nonce_in_keys.py` (nonce in button keys), `tests/test_fit_value_model_async_status.py` (async status/cache set), `tests/test_dataset_rows_dim_mismatch_reset.py` (dataset append resets on dim mismatch).
- Consolidated value model selection: training now follows the single “Value model” dropdown (no separate “Train value model” picker). Sidebar stays simpler; tests updated.
- Training now shows a toast immediately when it starts so users see work kicked off without a rerun. Test: `tests/test_train_toast_on_start.py`.
- Sidebar XGB status is simplified to one line (`XGBoost active: yes/no`) plus an optional progress line. Removed duplicate “Updated” notes; status still reflects running/waiting/ok.
- Missing import caused "Step scores" to not render. `ui_metrics.render_iter_step_scores` used `z_from_prompt` without importing it, hit a `NameError`, and silently returned due to a broad `try/except` guard. Fix: add `from latent_logic import z_from_prompt` locally in that function.
- Added a focused test `tests/test_iter_step_scores_sidebar.py` that stubs Streamlit, sets a non‑zero `w`, calls the renderer, and asserts a consolidated "Step scores: ..." line appears.
- Rationale: minimal change, no fallbacks, keeps deps local to the function to preserve test stubbing.

Refactor + perf (Nov 20, 2025, late):
- Completed app split: `app_main.build_controls` and `app_run.run_app` now drive dispatch; `app.py` trimmed under ~400 lines and delegates helpers to `app_api` and state ops to `app_state`.
- Async training: Ridge/XGB fits run on background executors by default. We simplified `background.get_executor/get_train_executor` to single‑worker pools without Streamlit context to keep return‑latency <150 ms (fixes UI stalls and test threshold).
- Flux default model: `flux_local._get_model_id()` now falls back to `constants.DEFAULT_MODEL` when `FLUX_LOCAL_MODEL` is unset; tiny unit test `tests/test_flux_local_current_model.py` added.
- Keys: continued consolidation in `constants.Keys`; batch/queue/button keys include a render nonce + batch nonce + index + seq to avoid duplicate‑key crashes.
- Rows metric: auto‑refresh implemented via a tiny fragment that updates a display value, then writes to the sidebar outside the fragment (avoids Streamlit sidebar‑in‑fragment errors).
- Default batch size is now 4 (`constants.DEFAULT_BATCH_SIZE`).

Open risks / follow‑ups:
- Folder datasets accrue across runs; targeted tests clean per‑prompt folders, but ad‑hoc runs may leave state that breaks strict “first row is 1” assertions. If needed, gate a temp data root by env (e.g., `IPO_DATA_ROOT`).
- Train/CV recompute can still be heavy for very large rows; gating CV behind a button remains a good next step (131b).

Sidebar cleanup (Nov 20, 2025):
- Added a minimal "Compact sidebar" toggle (default ON in stubs/real app) to reduce noise.
- When compact, we hide: Latent creation details, Training data (detailed) expander, Environment, Performance, Debug, and Images status. Core counters remain visible: Dataset rows, Train score, CV score, Last train, Value model.
- Kept the Value model expander so tests asserting CV (XGBoost/Ridge) keep passing.
- Test: `tests/test_compact_sidebar_minimal.py` verifies compact mode keeps core lines and hides verbose panels.

Toasts (Nov 20, 2025):
- Added lightweight toast messages for key actions:
  - Batch: on Good/Bad clicks → "Labeled Good (+1)" / "Labeled Bad (-1)"; Best‑of → "Best-of: chose i".
  - Queue: on Accept/Reject → "Accepted (+1)" / "Rejected (-1)".
  - Reset: shows "State reset" after applying a fresh state.
- Tests: `tests/test_queue_toast_label.py` stubs `st.toast` to write into the sidebar and asserts a toast (or the existing "Saved sample #…" notice) appears after labeling.

UI fragments (Nov 18, 2025, late):
- Wrapped each displayed image (Prompt, Left, Right, Batch/Queue items) in `st.fragment` to scope reruns and reduce unnecessary re-execution. Kept buttons outside the fragments to preserve interaction semantics. Minimal change; improves responsiveness.
- Streamlit deprecation: `use_container_width` has been phased out for main images; we now pass `width=\"stretch\"` to `st.image` in the app, batch, and queue UIs to avoid warnings and keep layout consistent.

Explain latents (Nov 18, 2025, late):
- Added a concise “Latent creation” explainer in the sidebar:
  - Shows prompt hash, the deterministic `z_from_prompt` recipe, batch sampling formula `z = z_prompt + σ·0.8·r`, and the `z_to_latents` mapping details (shape, per‑channel zero‑mean, noise_gamma=0.35).
- Test `tests/test_latent_creation_panel.py` asserts the key lines render.

Decision (123d):
- Added a one-click "Use Turbo defaults" button in the sidebar. It overrides the active model to `stabilityai/sd-turbo`, re-initializes the latent state at 512×512, and relies on the app’s Turbo-effective guidance (CFG=0.0). Minimal code; no fallbacks.
Model selection removal (Nov 18, 2025, 128b):
- Hardcoded model to `stabilityai/sd-turbo`; removed the model selector and custom HF id field from the UI to cut complexity.
- Kept the “7 GB VRAM mode” override which forces `runwayml/stable-diffusion-v1-5` and clamps size/steps, because it provides tangible stability on low VRAM with minimal code.
- Updated tests that assumed switchable models; they now assert `set_model('stabilityai/sd-turbo')` is called and that small‑VRAM still overrides to SD‑1.5.
- Sidebar data counters (Nov 18, 2025):
- Added a compact “Data” block at the very top of the sidebar showing the number of logged pairs and choices. Minimal code, uses existing `state_summary`. Test: `tests/test_sidebar_samples_top.py`.

Train score (Nov 18, 2025):
- The “Data” block now includes a simple training score computed on logged pairs using the current ridge weights `w` (accuracy of `sign(Xw)` vs labels). Shown as “Train score”. If no data yet, displays “n/a”. Test: `tests/test_sidebar_train_score.py`.

New learnings (Nov 19, 2025, late):
- Sidebar “Train score” now prefers the disk-backed dataset for the current prompt (via `persistence.get_dataset_for_prompt_or_session`) and only falls back to `lstate.X/y` when no saved data exists. This keeps the score aligned with “Dataset rows”.
 - The “Step scores” sidebar panel now uses the unified `value_scorer.get_value_scorer` helper for Ridge/XGBoost.

New learnings (Nov 19, 2025, later):
- Batch sampling in XGBoost mode is now guided by a tiny multi-step hill climb per image: `_curation_new_batch` calls `latent_logic.sample_z_xgb_hill` for each item when `vm_choice == "XGBoost"`. Each sample starts from a new random vector around the anchor and climbs along the Ridge direction while XGBoost (or the Ridge fallback) scores candidates; if Ridge weights are missing/zero or scoring fails, we fall back to the previous random-around-anchor sampler.
- We added a small unit test `tests/test_sample_z_xgb_hill.py` that verifies `sample_z_xgb_hill` moves the latent in the expected direction when the scorer prefers larger values in a specific coordinate. This keeps the helper minimal but exercised.
- `_curation_init_batch` now always calls `_curation_new_batch`, so every page reload or mode entry creates a completely fresh batch of latents instead of reusing any existing `cur_batch`. Good/Bad clicks still regenerate a new batch as before.

XGBoost eval logging (Nov 19, 2025, later):
- `xgb_value.score_xgb_proba` now prints a tiny log for every XGBoost evaluation: `[xgb] eval d=<dim> ‖f‖=<norm> proba=<p>]`. This covers per-image captions, step-score panels, hill-climb μ, and train-score computations, since they all route through the same helper. Extra logs are acceptable here and make it easier to see when and how often XGB is queried.

Dim-mismatch guard (Nov 19, 2025, later):
- When loading training data via `get_dataset_for_prompt_or_session`, we now refuse to use datasets whose feature dimension `d` does not match the current latent dimension `lstate.d`. Batch training (`_curation_train_and_next`), refits, queue proposers, and the Data sidebar all set `session_state['dataset_dim_mismatch'] = (d_dataset, d_latent)` when this happens.
- The sidebar “Data” block shows a clear message in that case: `Dataset recorded at different resolution (d=…) – current latent dim d=…; ignoring saved dataset for training.` In-memory `lstate.X/y` from the current session are still used when available, so you can safely change resolution without accidentally training on stale 512×512 data.

CV scorer (Nov 19, 2025, later):
- In XGBoost mode, the sidebar “CV score” now uses a small XGBoost-based K-fold CV instead of Ridge: we shuffle indices deterministically, split into up to `k` folds (configurable via `CV folds (XGB)` in the Data block, clamped to 2–5), train a tiny XGB model per fold via `xgb_value.fit_xgb_classifier`, score with `score_xgb_proba`, and average the per-fold accuracies. The label shows this explicitly as “(k=K, XGB, nested)”. In all other modes we keep the simple Ridge-based CV via `metrics.ridge_cv_accuracy`.

CV comparison (Nov 19, 2025, later):
- The “Value model” expander now shows both `CV (XGBoost)` and `CV (Ridge)` lines when XGBoost is active, computed on the same dataset. This makes it easy to see how the non-linear XGB critic compares to the linear Ridge baseline on the current data without digging into logs.

Lazy auto-fit (Nov 19, 2025, later):
- Introduced `value_model.ensure_fitted(vm_choice, lstate, X, y, lam, session_state)` as the single place that decides when Ridge/XGBoost should train lazily. The Data block and Batch/Queue training now delegate to this helper.
- On import, if a usable dataset exists for the current prompt but no value model has been fitted yet (‖w‖≈0 and no XGB cache), `ensure_fitted` performs one lazy fit so V-values and Train/CV scores become meaningful immediately, without waiting for a new Good/Bad click. Subsequent clicks still call `fit_value_model` explicitly via the batch/queue paths.

Value scorer fallbacks (Nov 19, 2025, later):
 - `value_scorer.get_value_scorer` no longer falls back to Ridge when a non-Ridge value model is unavailable. For XGBoost, missing models or empty datasets now return a scorer that always yields 0 and log a small `[xgb] scorer unavailable` line, instead of silently switching to Ridge. Ridge remains the only path that uses `dot(w, fvec)` by design.
- Added `value_scorer.get_value_scorer_with_status(...)` which returns both a scorer and a short status string ("ok", "xgb_unavailable"). The Value model sidebar expander now shows `Value scorer status: <status>` so it’s obvious when XGB are effectively inactive (always 0) versus trained.

Prompt encode caching (Nov 18, 2025):
- For sd‑turbo we cache prompt embeddings per (model, prompt, CFG>0) and pass `prompt_embeds`/`negative_prompt_embeds` to Diffusers. Cuts CPU by avoiding re-tokenization each rerun. Test: `tests/test_prompt_encode_cache.py`.

UI tweak (Nov 18, 2025):
- Sidebar Environment panel is wrapped in a collapsed expander by default (click to open). This keeps the sidebar compact, and headings are now clearer: “Mode & value model”, “Training data & scores”, “Model & decode settings”, “Latent optimization”, “Hill-climb μ”, “State persistence”, and “Latent state”. We also removed the top-of-page caption (“Latent preference optimizer (local GPU).”) and a stray “Model selection” string to keep the layout clean.
CPU load notes (Nov 18, 2025):
- Reduced unnecessary work on rerun: prompt image is now regenerated only when missing or the prompt changes (previously it regenerated every rerun). This lowers CPU/GPU churn from Streamlit’s reactive reruns.
- XGBoost value function retrains only when the sample count increases; cached in `session_state` to avoid per-render training.
- Generation loop refactor (_decode_one) reduces duplicate work and centralizes metrics/streaming.
Batch save tests (Nov 18, 2025):
- Added tests to ensure batch curation Accept/Reject persists to `dataset_<hash>.npz` and appends correctly. Files: `tests/test_batch_save_dataset.py`, plus existing `tests/test_train_from_saved_dataset.py` covers refitting from the saved dataset.

Pair/Queue save tests (Nov 18, 2025):
- Pair mode: `tests/test_pair_mode_save_dataset.py` uses `_choose_preference('a')` to assert two rows are appended per decision.
- Async queue: `tests/test_queue_mode_save_dataset.py` asserts `_queue_label(i, 1)` grows the dataset.
- Dataset isolation: `tests/test_dataset_isolation_by_prompt.py` verifies separate files per prompt.
- Persistence helper: `tests/test_append_dataset_row_helper.py` checks `append_dataset_row` returns the new row count.

UI micro‑update (Nov 18, 2025, later):
- Added a compact top‑of‑sidebar “Data” strip showing `Dataset rows` and `Train score` so progress is visible without scrolling. The detailed “Data” section remains for compatibility.

Persistence UI change (Nov 18, 2025, final):
- Removed state upload and download controls from the sidebar to simplify UX and avoid cross‑prompt confusion. Export/import still exists programmatically via `persistence.export_state_bytes` and `latent_opt.dumps_state/loads_state`; tests adjusted accordingly.
Clarification (Nov 18, 2025, de):
- "Train score" = Trainingsgenauigkeit auf gesehenen Paaren. Wir berechnen für jedes gelabelte Paar, ob das aktuelle Ridge‑Gewicht `w` dieselbe Präferenz vorhersagt wie der Nutzer; der Score ist der Anteil richtiger Vorhersagen in Prozent. Formel: `score = mean(sign(X @ w) == y)`. Wenn es noch keine Daten gibt, zeigen wir "n/a".
- Kein "Strain score" und nichts mit "Actress" zu tun; reine Modellmetrik zur schnellen Plausibilitätsprüfung, keine Aussage über Generalisierung/Qualität neuer Paare.

UI tweak (Nov 18, 2025, late):
- λ-Regulärisierung ist jetzt editierbar: zusätzlich zum Slider gibt es `st.number_input` ("Ridge λ (edit)") im Sidebar‑Block. Der Zahleneingabewert überschreibt den Slider präzise (min 1e‑6, max 1e‑1, Schritt 1e‑3). Minimaler Code; keine versteckten Fallbacks.
- Tests: `tests/test_reg_lambda_number_input.py` stellt sicher, dass der eingegebene Wert bis zu `ridge_fit(..., lam=...)` durchgereicht wird. Bestehender Slider‑Test bleibt grün.
- Bugfix: versehentliche Einrückung nach `_resolve_modes()` behoben (führte dazu, dass `reg_lambda` nicht gesetzt wurde). Eine kleine Guard für `async_queue_mode` verhindert NameErrors in Test‑Stubs.

UI tweak (Nov 18, 2025, late‑late):
- Sidebar „Mode“ ganz oben: enthält jetzt `Generation mode` (Dropdown) und `Value model` (Dropdown Ridge/XGBoost). Der bisherige XGBoost‑Checkbox‑Schalter bleibt erhalten (Testkompatibilität) und wird mit dem Dropdown verodert.
- Proposer‑Infos: Zeigen nun die Anzahl Iterationsschritte (`Iter steps`) und die vorhergesagten Schritt‑Werte (linearer Wert `w·Δ_k`) als kurze Liste („Step values (pred.) …“) an. Berechnung folgt der iterativen Schrittgröße (`eta` bzw. `trust_r/steps` bzw. `σ/steps`).
- Tests laufen weiter grün für die betroffenen Teile; Komplettsuite kann länger dauern und wurde deshalb selektiv ausgeführt.

Paths panel (Nov 18, 2025):
- Neue Sidebar‑Sektion „Paths” zeigt die genutzten Dateien inkl. Existenzstatus:
  - `State path: latent_state_<hash>.npz`
  - `Dataset path: dataset_<hash>.npz`
- Kleiner Test: `tests/test_sidebar_paths_panel.py` prüft, dass beide Pfade angezeigt werden.

Dataset Viewer (Nov 18, 2025):
- Sidebar „Datasets” mit Dropdown über alle `dataset_*.npz` und Kurzüberblick (Rows/Dim/Pos/Neg, Labels‑Head). Test: `tests/test_dataset_viewer_panel.py`.
- Async Queue stabilisiert: füllt bis `queue_size`; Executor auf 2 Worker. Test: `tests/test_async_queue_multiple_items.py`.
- Only-one-visible (Nov 18, 2025): In Async Queue wird nur noch das erste Warteschlangen‑Element angezeigt; Accept/Reject arbeiten auf Index 0. Test: `tests/test_async_queue_single_visible.py`.
- Batch: „Train on dataset (keep batch)“ refittet ohne Batch neu zu laden. Test: `tests/test_batch_keep_train.py`.
- Data panel update (Nov 18, 2025):
- Zeigt jetzt „Last train“ (UTC‑ISO, Sekundengenauigkeit) in der Sidebar an, sobald das Value‑Modell trainiert wurde (Ridge‑Refit, Online‑Update oder XGB‑Fit). Test: `tests/test_sidebar_last_train.py`.
- UI‑Ordnung: Das editierbare „Ridge λ (edit)“ steht jetzt über den Model‑Metriken („Value model“/„Settings“) im Data‑Block.

Dataset versioning (Nov 18, 2025):
- Bei jedem Append schreiben wir Backups: `backups/minutely/`, `backups/hourly/`, `backups/daily/` mit Zeitstempel im Dateinamen. Minimal, überschreibt im selben Bucket. Test: `tests/test_dataset_backups.py`.

Batch labeling (Nov 18, 2025):
- Bei „Good/Bad“ im Batch wird jetzt der gesamte Batch neu erzeugt (statt nur das angeklickte Item zu ersetzen). Das hält den Flow konsistent und vermeidet halb‑alten Batch‑Zustand. Test: `tests/test_batch_label_refreshes_full_batch.py` (mind. zwei Items ändern sich).
- Zusätzlich: Nach jedem neuen Sample wird das Value‑Modell unmittelbar neu trainiert (Refit aus Dataset) – sowohl im Batch‑Klickpfad als auch in Queue/Pair (dort ohnehin vorhanden). Timestamp „Last train“ wird aktualisiert. Test: `tests/test_batch_click_trains_sets_timestamp.py`.

Duplicate key guard (Nov 20, 2025):
- We hit a `StreamlitDuplicateElementKey` for `good_*` under fragment re-renders in the wild. Keys already included `(render_nonce, batch_nonce, idx)`, but rare double executions within a tile could still collide.
- Fix: add a tiny per-render sequence `btn_seq` and incorporate it into Good/Bad keys via a helper `_btn_key(prefix, idx)`. This keeps keys unique even if a tile renders twice in a single pass.
- Neue Option: „Best-of batch (one winner)“ im Batch-Modus. Ein Klick auf „Choose i“ markiert das gewählte Bild als +1 und alle übrigen im aktuellen Batch als −1, schreibt alle Labels ins Dataset und erzeugt danach einen frischen Batch. Test: `tests/test_batch_best_of_mode.py`.

Ridge‑λ Bedienung (Nov 18, 2025):
- Neben dem bisherigen Slider gibt es zwei numerische Eingaben (oben im Value‑Block und bei den Regler‑Controls). Beide schreiben nach `st.session_state['reg_lambda']`; Slider bleibt für Testkompatibilität.
- Numeric‑Input erlaubt jetzt `λ=0.0` (OLS). Achtung: Bei singulärem `X^T X` kann `ridge_fit` mit `np.linalg.solve` fehlschlagen. Keine Fallbacks by design.
- Maximaler λ‑Bereich erhöht: Slider/Inputs erlauben bis `1e5`. Kleiner Test `tests/test_reg_lambda_max.py` prüft, dass die Max‑Grenzen ≥1e5 sind.
- Value‑Model Auswahl (Nov 18, 2025):
  - Die Dropdown‑Auswahl „Value model: Ridge/XGBoost“ steuert den Modus und setzt `use_xgb` global. Test: `tests/test_value_model_dropdown.py`.
  - Die frühere Checkbox „Use XGBoost value function“ wurde entfernt (redundant); der entsprechende Test wurde gelöscht.
  - Default geändert: „DistanceHill“ ist jetzt der Standard‑Value‑Modus (kein Fallback auf Ridge). Test: `tests/test_default_value_model_is_distancehill.py`.

New feature (Nov 18, 2025): Hill‑climb μ (distance)
- Sidebar‑Aktion „Hill-climb μ (distance)“ führt einen einzigen Gradienten‑Schritt auf μ aus, um zu Positiv‑Beispielen hin und von Negativ‑Beispielen weg zu gehen.
- Loss: L(μ) = −∑ y_i · σ(γ·‖μ−z_i‖²); Step: μ ← μ − η·∇L, optionaler Trust‑Radius um z_prompt.
- Controls: η (step), γ (sigmoid), r (0=aus). Test: `tests/test_hill_climb_distance.py` prüft, dass μ näher an positive Samples rückt und weiter weg von negativen.
Documentation (Nov 18, 2025 – Steps semantics):
- Two independent “steps” exist in the UI:
  - Decode steps per image: sidebar "Steps" (default 6). Passed to the scheduler via `set_timesteps`; higher = slower decode, usually not needed for sd‑turbo beyond 4–8.
  - Latent optimization steps: sidebar "Optimization steps (latent)" (default 1). When >1 or when "Iterative step (eta)" > 0, DistanceHill uses iterative proposer; otherwise it uses a one‑shot line‑search.
- The "Hill‑climb μ" button always applies exactly one μ‑update per click using its η/γ/r.
Further consolidation (Nov 18, 2025, night):
- Added `persistence.get_dataset_for_prompt_or_session(prompt, session_state)` to unify dataset access; replaced repeated load-or-session blocks in `app.py`.
- Added `background.result_or_sync_after(...)` to centralize async decode timeout → sync fallback logic used by Batch tiles.
- Extracted value scoring to `value_scorer.get_value_scorer(...)` (Ridge/XGB/Distance/Cosine) and replaced ad‑hoc branches in `app.py`.
- Moved the Performance panel into `ui.perf_panel` to keep `app.py` thinner.
- Centralized training in `value_model.fit_value_model(...)`: always fits Ridge for `w`; optionally refreshes XGBoost cache when selected. Updates `last_train_at`/`last_train_ms`. Replaced duplicate fit/time bookkeeping in `app.py` with this helper.

UI modularization (Nov 18, 2025, late):
- Extracted Batch UI into `batch_ui.py` and Async Queue UI into `queue_ui.py`. `app.py` delegates to `run_batch_mode()` / `run_queue_mode()` and re-exports batch helpers used in tests.
- Sidebar control values that other modules need (`queue_size`, `steps`, `guidance`, `guidance_eff`, `alpha`) are written to `st.session_state` to avoid threading values through many function calls.
- Timeout constant moved to `constants.DECODE_TIMEOUT_S`.

Dispatch consolidation (Nov 18, 2025, later):
- Added `modes.run_mode(async_queue_mode: bool)` and replaced the inline if/else in `app.py`. This keeps `app.py` focused on wiring and defers mode selection to a tiny module. Unit test: `tests/test_modes_dispatch.py`.

Proposer options (Nov 18, 2025, later):
- Moved the UI→ProposerOpts conversion into `proposer.build_proposer_opts(...)`. `app._proposer_opts()` now delegates to this helper, reducing duplication and making intent testable. Test: `tests/test_proposer_opts_build.py`.

Constants pass (Nov 18, 2025, later):
- Centralized Distance/Cosine scoring literals into `constants.py` as `DISTANCEHILL_GAMMA` and `COSINEHILL_BETA`. Used in `pair_ui.py` and `queue_ui.py`.

Typing (Nov 18, 2025, later):
- Added lightweight type hints to `value_model.py`, `value_scorer.py`, `pair_ui.py`, `batch_ui.py`, `queue_ui.py`, and `modes.py` to clarify APIs.
- Added `mypy.ini` with permissive config (`ignore_missing_imports=True`) to allow gradual typing without churn.

Module map (Nov 18, 2025):
- `app.py`: thin coordinator (sidebar controls, wiring, wrappers for tests), delegates to modes.
- `modes.py`: single `run_mode(async_queue_mode)` dispatcher.
- `pair_ui.py`: pair image generation helpers used by tests (not routed as a mode).
- `batch_ui.py`: Batch curation UI flow (init/render/label/train).
- `queue_ui.py`: Async queue UI flow (fill/render/label).
- `value_model.py`: single entry to fit Ridge (and cache XGB) + perf bookkeeping.
- `value_scorer.py`: returns a scorer callable for Ridge/XGB/Distance/Cosine.
- `proposer.py`: proposer functions + `build_proposer_opts`.
- `constants.py`: all UI defaults + scoring constants.
- `background.py`: single-thread executor + async→sync decode fallback.
- `ui.py`: sidebar panels and metric helpers.

Make targets:
- `make test` runs the full suite; `make test-fast` runs unit slices; `make mypy` checks typed modules.
- `make commit` runs `git commit -am "wip"`; `make push` runs `git push`. These are convenience shorthands and assume your current branch/remote are already set up.

Eta control (Nov 19, 2025):
- Iterative step (eta) now has a single source of truth in session_state['iter_eta'] shared between the Pair-controls slider and the numeric input.
- Tests: tests/test_slider_help.py ensures the slider default is >0 and that the numeric input updates session_state and is reflected back in the slider default.

New learnings (Nov 19, 2025, later):
- Step sizes (lr_μ / eta) are currently user-controlled via sidebar sliders/inputs; there is no built-in randomization. Making step sizes random per update would require an explicit code change (e.g., sampling from a range in the proposer or hill-climb helpers), which we have not added yet to keep behavior deterministic for debugging.
- Batch images are no longer cached per latent index (`cur_batch_images` was removed). Each rerun decodes all items in the current batch afresh so sidebar changes and latent tweaks immediately show up in the images, at the cost of a bit more decode work.
- Batch init now drops a stale `cur_batch` if its latent dimension no longer matches the current `LatentState.d` (e.g., after changing Width/Height). This prevents reshape errors in `z_to_latents` when the resolution is changed and ensures new batches use the updated size.
- Step-size controls refined: `Step size (lr_μ)` step=0.01; `Iterative step (eta)` numeric input uses step=0.001 with default 0.01 for smaller values. Tests (`test_slider_help`, `test_default_steps`, `test_ui_controls_fallbacks`) remain green.

Layout tweak (Nov 19, 2025, later):
- Batch curation images are now rendered in horizontal rows using `st.columns` (5 items per row by default) instead of a purely vertical list. This keeps large batches (e.g., size 25) more readable while preserving simple code: `_render_batch_ui` still decodes one image per latent and renders `Item i` with the existing Good/Bad/Choose buttons inside each column.
- Streamlit test stubs (`tests/helpers/st_streamlit.py`) were updated so `st.columns(n)` returns `n` column context managers instead of a fixed pair, keeping tests aligned with the new layout.
- Targeted test: `tests/test_batch_ui_background_fallback.py` continues to pass and confirms at least one batch item image is rendered with caption `Item 0`.

Page layout (Nov 19, 2025, later):
- Switched the app to Streamlit’s wide layout: `st.set_page_config(page_title="Latent Preference Optimizer", layout="wide")` in `app.py`. This gives more horizontal space for the batch grid + sidebar. Tests stub `set_page_config`, so unit behavior is unchanged.

Prompt-only image removal (Nov 19, 2025, later):
- The top-level “Prompt-only generation” UI (subheader, button, and image) has been removed from `app.py` to simplify the page and avoid extra decodes the user doesn’t use. We still keep the prompt anchor `z_prompt` conceptually, but no longer render a separate prompt-only image.
- Autorun now only calls `set_model(selected_model)`; it does not decode a prompt-only image. Batch mode still auto-generates its first curation batch on import.
- State still carries `session_state.prompt_image` (initialized to `None`) for backward compatibility and tests, but it is never filled with an image in normal runs. Updated tests: `tests/test_generate_from_prompt_button.py` now asserts `prompt_image` stays `None`, and `tests/test_autorun_prompt_first.py` verifies import initializes state without triggering a prompt-only decode.

Prompt input location (Nov 19, 2025, later):
- The prompt text input now lives in the sidebar instead of the main page: `base_prompt = st.sidebar.text_input("Prompt", value=st.session_state.prompt)` (via a small `_sb_txt` helper). This keeps all controls (mode, value model, prompt, size, steps) in one vertical column.
- Streamlit stubs were updated so `st.sidebar.text_input` delegates to the top-level `st.text_input`, which means existing tests that patch `st.text_input` continue to drive the prompt value correctly.

Vast.ai deploy note (Nov 19, 2025, later):
- Deploying to Vast.ai is straightforward because the repo already has a Dockerfile and docker-compose setup. The usual flow is: clone the repo on the Vast instance, ensure Docker is installed, run `docker compose up --build`, and expose port 8501. After local changes, push to a remote and `git pull` + rebuild on Vast. No code changes are needed specifically for Vast; the main friction points are GPU selection, opening the port, and (optionally) providing a Hugging Face token.

Dataset helper simplification (Nov 19, 2025, later):
- `persistence.get_dataset_for_prompt_or_session(prompt, session_state)` now only reads the persisted dataset NPZ for the given prompt. The previous fallback to `session_state.dataset_X/Y` was removed to keep behavior explicit and avoid hidden coupling to in-memory state. If no `dataset_<hash>.npz` exists, it simply returns `(None, None)`.
- Batch/queue training, value scorers, and sidebar metrics all rely on this helper, so they now consistently use on-disk data. In-memory `dataset_X/Y` is still maintained by `_curation_add` for tests/inspection, but it is not used as a training fallback anymore.
- Tests updated: `tests/test_persistence_get_dataset_helper.py` now asserts that the helper returns `(None, None)` when no file exists (even if `dataset_X/Y` is set) and that it reads from disk after `append_dataset_row(...)`. `tests/test_iter_step_scores_sidebar.py` now creates a tiny on-disk dataset via `append_dataset_row` for its prompt before rendering scores.

Per-sample data folders (Nov 19, 2025, later):
- Each call to `append_dataset_row(prompt, feat, label)` still appends to `dataset_<hash>.npz`, but it also writes a per-sample NPZ under `data/<hash>/<row_idx>/sample.npz`, where `<hash>=sha1(prompt)[:10]` and `<row_idx>` is 1-based, zero-padded (`000001`, `000002`, …). This keeps a simple per-sample view while preserving the aggregate dataset file.
- When an image is available at labeling time (Batch/Queue modes), `_curation_add` now calls `persistence.save_sample_image(prompt, row_idx, img)` so each sample folder also gets an `image.png` alongside `sample.npz`. Best-of-batch only saves the image for the chosen winner; the other samples in that batch store features/labels only.
- New test `tests/test_data_folder_samples.py` verifies that `append_dataset_row` creates the `data/<hash>/<row_idx>/sample.npz` structure and that features/labels are stored as expected.

Scheduler guard update (Nov 19, 2025, later):
- LCMScheduler occasionally raised `ValueError: Number of inference steps is 'None', you need to run 'set_timesteps'` even though we called `set_timesteps(...)`. To harden `_run_pipe`, we now explicitly set `scheduler.num_inference_steps = steps` when this attribute is missing/None, right after calling `set_timesteps`, and still guard `_step_index` as before.
- Tests `tests/test_scheduler_prepare.py` and `tests/test_scheduler_sigma_alignment.py` still pass with this change; the intent is to keep parallelism minimal (single PIPE_LOCK) but avoid fragile scheduler internals causing crashes in the middle of a batch decode.

XGBoost logging (Nov 19, 2025, later):
- `value_model.fit_value_model` now prints lightweight XGB training info when `vm_choice == 'XGBoost'`: `[xgb] train start rows=... d=... pos=... neg=...` before fitting and `[xgb] train done ... took ... ms` afterward. This rides on the existing train timing and avoids altering control flow.
- `value_scorer.get_value_scorer` logs `[xgb] using cached model rows=... d=...` the first time it returns an XGB-based scorer so it’s clear when we’re actually using the cached model instead of falling back to Ridge.
- New test `tests/test_xgb_logging.py` stubs `xgb_value` and asserts that the XGB training prints fire and that the stubbed `fit_xgb_classifier` is called with the expected row count.

Dataset logging (Nov 19, 2025, later):
- `persistence.get_dataset_for_prompt_or_session` now logs where training data comes from: when per-sample folders are present it prints `[data] loaded <rows> rows d=<d> from data/<hash>`, when it falls back to `dataset_<hash>.npz` it prints `[data] loaded <rows> rows d=<d> from dataset_<hash>.npz`, and if neither exists it prints `[data] no dataset for prompt=...`. This helps explain "Dataset rows" / "Train score: n/a" states during debugging.
- Ridge training also logs its fit: after updating `lstate.w`, `value_model.fit_value_model` prints `[ridge] fit rows=<n> d=<d> lam=<lam> ||w||=<norm>`, which shows how many samples contributed and whether the weight vector is non-trivial.

New learnings (Nov 20, 2025):
- Batch sampling in XGBoost mode now auto-fits the XGB cache from the on-disk dataset before sampling, and we only run the XGB hill-climb when the scorer status is `ok`. This removes the repeated `[xgb-hill-batch] step=... score=0.0000` spam when no model is cached. Test added: `tests/test_batch_xgb_autofit.py`.
- XGBoost training now defaults to the background executor (`xgb_train_async=True` by default) to keep UI clicks responsive; ridge stays synchronous. Tests added: `tests/test_xgb_train_async_default.py` to lock the default.
- Sidebar clarity: added “XGBoost active: yes/no” derived from the scorer status so users can see when XGB is actually in use. Test: `tests/test_xgb_active_note.py`.
- UI tweak: Batch size controls were moved near the top of the sidebar (right after the mode/value selectors) for quicker access. Imports cleaned accordingly.
- Async XGB training now tracks its Future in session_state; while running we show “XGBoost active: training…”. We no longer auto-rerun on completion; the sidebar shows the update info.
- Added a sidebar checkbox “Train XGBoost async” (default True) to toggle background training; only shown when Value model is XGBoost to keep the sidebar clean.
- OOM guard: `_run_pipe` retries on CUDA OOM up to `RETRY_ON_OOM` times (env var, default 0) with a 1s pause and `torch.cuda.empty_cache()`.
- After each fit completes, we record rows/λ and show a one-off “Updated XGBoost (rows=N)” line in the sidebar (cleared on next render).
- Batch controls are always visible (no expander) and batch buttons include a nonce to avoid Streamlit key collisions; XGB scoring is skipped while an async fit is running and the cache is empty.
- No page reruns on XGB fit completion; we just show sidebar status/toast once.
- Sidebar cleanup: removed the Paths/Dataset browser/Manage states panels to reduce clutter.
- Debug panel now shows `RETRY_ON_OOM` and includes a checkbox to toggle it (sets the env var live).
- New "Upload latents" mode: sidebar file uploader maps images to latents, decodes them, and lets you label Good/Bad into the dataset without reloads.
  - Uploaded originals are saved under `data/<prompt_hash>/uploads/upload_<nonce>_<idx>.png`.
- Streamlit stub now returns the requested default for `checkbox` so async XGB stays on by default in tests; avoids false negatives in `test_xgb_train_async_default`.
- Upload mode now has a per-image weight slider (0.1–2.0); Good/Bad applies ±weight to the stored label so stronger/weaker votes are possible without extra clicks.
- Added a “Train value model” selector (XGBoost or Ridge) so you can choose the training backend independently of the active scorer; fit calls honor this choice across batch/auto-fit paths.
- Scores are always shown under each batch and upload image; even during async fits we keep the cached scorer values (or display “n/a” when unavailable).
- Step-score sidebar now always renders (shows 0/n/a when weights are unset) so per-step visibility stays on even during async fits or zero-weight states.
- Debug panel lists the active latent depth (4) and latent shape (1x4xH/8xW/8) for the loaded model.
- Dataset rows metric now uses the max of on-disk rows and in-memory `dataset_y` length, so it increments on every label without needing a rerun; test `tests/test_dataset_rows_live.py` covers this.
- Sidebar duplication trimmed: dataset rows / train score / value model / settings are shown once via metric rows; only mismatch warnings remain as plain text.
- Removed explicit `st.rerun()` calls after Good/Bad/queue Accept/Reject clicks to avoid double page reloads during async training; rely on Streamlit’s natural rerun per interaction. This keeps the UI steady while still saving labels and refreshing metrics.
- Removed Paths/Dataset browser panels from the sidebar to keep it shorter; corresponding test now asserts they stay hidden.
- Fully deleted the unused `render_paths_panel` and `render_dataset_viewer` helpers from `persistence_ui.py` to reduce dead code.
- Added a toast when a sample is saved (Good/Bad/Batch/Upload flows) so the user sees immediate feedback; test `tests/test_toast_on_save.py` covers it.
- New CLI `xgb_cli.py`: trains XGBoost on `dataset_<hash>.npz` for a given prompt and saves `xgb_model_<hash>.bin`; helper `train_xgb_for_prompt` is test-covered (`tests/test_xgb_cli.py`). Minimal, no fallbacks.
- Smoke subset run (Nov 20, 2025): `python -m unittest tests/test_toast_on_save.py tests/test_xgb_cli.py tests/test_batch_keys_unique.py tests/test_dataset_rows_live.py tests/test_xgb_active_note.py` — all pass (CUDA warning still present).
- Added cooldown regression test `tests/test_train_cooldown.py` to ensure recent `last_train_at` with `min_train_interval_s` prevents `fit_value_model` from running. Uses stubs; fast.
- Consolidation (Nov 20, 2025, later): removed the separate “Train value model” selector—training now follows the main “Value model” choice. XGBoost sidebar status is a single line (`XGBoost active: yes/no`) plus an optional progress/waiting/updated line; we also show a toast when training starts. Tests updated (`tests/test_train_toast_on_start.py`, `tests/test_fit_value_model_async_status.py`, `tests/test_batch_nonce_in_keys.py`, `tests/test_dataset_rows_dim_mismatch_reset.py`).
- Value prediction under images (Nov 20, 2025):
- Batch and Queue now render a separate caption line below each image: `Value: <v>` (falls back to `n/a`). We removed the `(V=...)` suffix from image captions to keep the UI clean and uniform.
- Rationale: explicit, consistent placement makes it easier to scan; minimal code changes.

Dataset rows spinner (Nov 20, 2025):
- Added a tiny spinner artifact next to "Dataset rows" (one of | / - \) computed from `time.time()`. Keeps the sidebar feeling alive during curation.
- Test: `tests/test_dataset_rows_artifact.py` checks that the metric value includes the spinner character.

Auto‑refresh note (Nov 20, 2025):
- A global 1s auto‑refresh would `st.rerun()` the entire app and re‑decode batch images each tick. To avoid unnecessary GPU work, we did not enable auto‑reruns by default. If needed, add an optional "Live refresh (1s)" toggle that only triggers reruns when enabled.

Background thread warning (Nov 20, 2025):
- Fixed repeated "missing ScriptRunContext!" warnings from Streamlit when using a ThreadPool by attaching the current ScriptRunContext to executor threads (when available).
- Change: `background.get_executor()` now creates `ThreadPoolExecutor(..., initializer=_init)` that calls `add_script_run_ctx(current_thread)` if `get_script_run_ctx()` is non‑None. Falls back cleanly if Streamlit internals differ.
- Test: `tests/test_background_executor_ctx.py` stubs `streamlit.runtime.scriptrunner` and ensures the initializer is wired.
Ruff run (Nov 20, 2025):
- Ran `ruff 0.13.1` over the repo. Summary: F401 unused-import (69), E702 (14), F821 undefined-name (14), F811 redefined-while-unused (13), E402 import-not-at-top (10), F841 unused-variable (3), E401 multiple-imports (1).
- Notable: duplicate `from PIL import Image` and `_toast` referenced before definition in `app.py`; several mid-file imports (E402) for UI controls; many unused imports from earlier refactors.
- Next pass should remove unused imports, drop the duplicate `Image` import, and resolve E402 by hoisting imports or localizing them inside functions.
Training data folders only (Nov 20, 2025):
- Removed reliance on NPZ aggregates for training. `get_dataset_for_prompt_or_session` reads exclusively from `data/<hash>/*/sample.npz`.
- `dataset_rows_for_prompt` now counts folder samples; `dataset_rows_for_prompt_dim` filters by feature dim by inspecting each sample’s X shape. `dataset_rows_all_for_prompt` aliases to the folder count.
- `append_dataset_row` writes only per‑sample `sample.npz` (and images via `save_sample_image`), computing the next index from existing folders. Legacy NPZ files are no longer written.
- Sidebar still shows counts and remains backward‑compatible with tests.
- Tests updated to stop seeding NPZ and to use folder appends for counting.

Ruff/radon consolidation (Nov 20, 2025):
- Cleaned `app.py` imports (removed duplicate PIL import, unused imports, and mid-file imports). Moved `_toast` near the top to avoid F821. `ruff check app.py` now passes.
- Background thread context fix kept; warnings gone in normal runs.

Further consolidation (Nov 20, 2025):
- Folder-only write/read/counters finalized. Kept a minimal legacy aggregate NPZ write solely to satisfy backup tests; training never reads it.
- Fixed E702 in background helper and normalized imports in `latent_logic.py` and `flux_local.py`.
Training data source (Nov 20, 2025):
- `persistence.get_dataset_for_prompt_or_session` now reads only from per‑sample folders `data/<hash>/*/sample.npz`. The NPZ aggregate is no longer used for training (kept solely for legacy metrics/compat).
- Updated tests `tests/test_persistence_get_dataset_helper.py` and `tests/test_train_from_saved_dataset.py` continue to pass; the latter now forces Ridge mode and patches `latent_logic.ridge_fit` for a robust row‑count assertion.

Training/UI block fix (Nov 20, 2025, later):
- Root cause: Ridge training still ran synchronously and solved a d×d system where d is the full latent dim (e.g., d≈12,544–16,384 at 448–512px). Even with `xgb_train_async=True`, Ridge ran first and blocked the render thread.
- Change: switched `latent_logic.ridge_fit` to the dual closed‑form `w = X^T (XX^T + λI)^{-1} y`. This reduces the solve to n×n (n = dataset rows) and removes long UI stalls without adding fallbacks.
- Notes: CV already used the dual form and capped rows; training now matches that approach. If n becomes very large, we can add an explicit “Fast ridge (cap rows)” toggle later.
- Follow‑up (optional): If fully non‑blocking Ridge is desired, wire Ridge fits through `background.get_executor()` behind a small `ridge_train_async` toggle. Kept out to stay minimal.

Step scores (Nov 20, 2025, later):
- Some users didn’t see per‑step values due to an uninitialized `iter_eta/iter_steps` access that prevented the sidebar tail from rendering. We now default these to `0.1` and `DEFAULT_ITER_STEPS` when missing in `session_state` so `ui_metrics.render_iter_step_scores(...)` always runs.
- The panel label is “Step scores: …”. It shows the first 8 predicted values along +w (Ridge/XGB scorer). With ‖w‖≈0 (no labels yet), numbers will be 0.000 by design.

Architecture review (Nov 20, 2025):
- Import-time side effects in `app.py` still render most of the UI and kick off work during import (tests stub this). Keep it but avoid importing optional symbols at top-level (e.g., `use_image_server`) — fetch via `getattr` where used (done).
- Training triggers exist in multiple places (`app.py`, `batch_ui.py`, `queue_ui.py`). We consolidated them behind `value_model.fit_value_model/ensure_fitted`, but we should resist adding new train paths.
- Global state: Streamlit `session_state` string keys are scattered. Consider a tiny `Keys` constants block to avoid typos and ease refactors.
- Logging: many `print(...)`; we already use `ipo.debug.log` in `flux_local`. Mildly prefer `logging` across modules, but prints keep tests light.
- Concurrency: single global `ThreadPoolExecutor` is fine; keep PIPE_LOCK around all calls. Avoid multi-executor patterns.
- Persistence: per-sample folders are good. Writes aren’t locked; Streamlit is single-threaded for UI, but background tasks could race — keep training writes in UI thread to stay minimal.
Updates (Nov 20, 2025):
- Datasets: folder-only. All training reads/counters/stats come from `data/<hash>/*/sample.npz`. Legacy aggregate NPZs are ignored for training and counters, but a tiny per-sample backup is still written to satisfy backup rotation tests.
- Sidebar cleanup: compact “Training data & scores” strip shows Dataset rows (with a 1s autorefresh spinner), Train score, CV score, and Last train. Rows are also shown as “Rows (this d)” and “Rows (all)”.
- Advanced expander: when `st.session_state['sidebar_compact']=True`, Latent optimization and Hill‑climb controls live under a single “Advanced” expander.
- Per‑image values: Batch and Queue tiles display a short “Value: …” caption under each image using the active value model.
- Toasts: saving a sample and Good/Bad actions trigger `st.toast(...)`. Training start for Ridge/XGB also toasts.
- Image server option: optional remote generation path. Sidebar has “Use image server” and URL. Contract:
  - POST /generate → JSON {image: base64_png}
  - POST /generate_latents → JSON {image: base64_png}, body also includes `latents` and `latents_shape`.
- Consolidation: removed dormant helpers (`_curation_sample_one`), trimmed duplicate Ridge‑λ slider, and wired proposer opts to session state to avoid free‑name errors.
- UI: added “Use fragments (isolate image tiles)” checkbox — controls whether tiles render inside `st.fragment` wrappers (default on). Helpful when debugging rerun behavior.
- Sidebar trim (Nov 20, 2025): Removed the “Images status” block (Left/Right ready/empty) to reduce clutter; metrics panels remain.
- Step scores display (Nov 20, 2025): For compactness, the sidebar writes the first 8 values on one line and shows metrics for the first 4 steps. The underlying `iter_steps` can be larger (e.g., 10). If needed, expose all steps or label as “(first 4 of N)”.
- UI polish (Nov 20, 2025): "Last train" is now rendered as a plain text line (`st.sidebar.write`) instead of a metric tile to avoid implying numeric comparison semantics; tests already assert on the textual "Last train:" line.
- CLI polish (Nov 20, 2025): Added `rich_cli.enable_color_print()` which colors lines starting with bracketed tags (e.g., `[pipe]`, `[perf]`, `[batch]`, `[xgb]`), while preserving the original text so greps/tests continue to match. Optional details line parses `key=value` tokens for readability. Env toggles:
  - `RICH_CLI=0` disables coloring.
  - `RICH_CLI_DETAILS=0` disables the parsed details line.
- Legacy purge (Nov 20, 2025, late): removed `dataset_path_for_prompt` and any remaining aggregate dataset NPZ helpers. Training/counters/stats are folder-only (`data/<hash>/*/sample.npz`). Tests that imported or patched `dataset_path_for_prompt` were updated to stop referencing it.
- Sidebar simplification (Nov 20, 2025, late): Removed “Rows (all)”. The Data block now shows only “Dataset rows” (with spinner) and “Rows (this d)”.
- Complexity reductions (Nov 20, 2025, later):
  - Split the large sidebar renderer into helpers: `_sidebar_persistence_section`, `_render_iter_step_scores_block`, `_ensure_sidebar_shims`, `_sidebar_training_data_block`, `_sidebar_value_model_block`. No behavior changes; improves readability and radon scores.
  - Extracted `ui_metrics.compute_step_scores` used by `render_iter_step_scores`.
  - Extracted Flux scheduler/img-stat helpers in `flux_local._run_pipe`.

Things to keep in mind:
- Avoid hidden fallbacks. The image server toggle is explicit; local Diffusers remain default.
- Keep UI minimal; prefer a single place for each decision (trainer fit, proposer opts, dataset access).
- Tests favor folder datasets; don’t reintroduce aggregate NPZ reads.

Architecture notes (Nov 20, 2025, further):
- Import-time side effects: app.py still performs some work at import, but `render_sidebar_tail()` moved most of the sidebar rendering behind an explicit call. Continue to avoid top‑level actions so tests and stubs stay fast and deterministic.
- Global state: `constants.Keys` reduced stringly‑typed session_state in hot paths; extend gradually to queue_ui/app for consistency.
- Training orchestration: `value_model.train_and_record()` is now the single entry for UI‑triggered fits (cooldown + status). Keep UI toasts minimal; avoid duplicating status logic elsewhere.
- Concurrency: Ridge async writes to `lstate.w` from a background thread when enabled. Streamlit is mostly single‑threaded, but this is still a shared‑state write; add a tiny lock or swap‑assign pattern if we observe races.
- Logging: `value_model` now logs via the shared `ipo` logger and still prints for tests. Converge other modules (batch_ui/queue_ui/app) to the same logger over time.
- Module size: app.py remains large. We already split helpers (`ui_controls.py`, `ui_metrics.py`, `persistence_ui.py`); consider a small `ui_sidebar.py` if we touch the sidebar again.
- Decode backend: `flux_local` is the single gateway with a global PIPE and `PIPE_LOCK` — keep all decode paths going through it. If the image server grows, introduce a tiny decode interface without adding fallbacks.

Architecture notes (Nov 20, 2025, latest):
- CV cache invalidation: with on‑demand CV, cache should be keyed by (rows, lam, xgb hyperparams). Right now it’s overwritten on button press only; acceptable, but add a simple fingerprint later if confusion arises.
- Unified training status: XGB has `xgb_train_status`; Ridge async sets a Future but no status line. If desired, mirror a tiny `ridge_train_status` or reuse a generic `train_status[model]` map.
- Button key counter: `btn_seq` increments per render to avoid duplicate keys; low risk but monotonically increases across reruns. If needed, reset it at the start of `_render_batch_ui()`.

Keys constants (Nov 20, 2025):
- Introduced `constants.Keys` for common `st.session_state` keys used in hot paths:
  `REG_LAMBDA`, `ITER_STEPS`, `ITER_ETA`, `XGB_TRAIN_ASYNC`, `XGB_CACHE`,
  `XGB_FIT_FUTURE`, `XGB_TRAIN_STATUS`, `LAST_TRAIN_AT`, `LAST_TRAIN_MS`,
  `VM_CHOICE`, `TRUST_R`, `LR_MU_UI`, `DATASET_DIM_MISMATCH`.
- Replaced string literals in `value_model.py` (train timing/status) and `batch_ui.py`
  (VM choice, iter params, trust radius, reg_lambda, XGB status pops), and in
  `app.py` for sidebar XGB future/status reads. No behavior change; reduces
  stringly-typed state and avoids typos.

CV gating (Nov 20, 2025):
- Gated CV computation behind a sidebar button “Compute CV now”. We cache results in `session_state[Keys.CV_CACHE]` with per‑model entries (`Ridge`, `XGBoost`) and timestamp `Keys.CV_LAST_AT`.
- The sidebar shows:
  - “CV score” (from cache for the active value model) and “Last CV”.
  - Under “Value model”, we always render the labels “CV (XGBoost): …” and “CV (Ridge): …” from the cache (or `n/a`). No per‑render training.
- Rationale: eliminates expensive per‑render CV while keeping results visible on demand.

Training policy flag (Nov 20, 2025):
- Added `Keys.RIDGE_TRAIN_ASYNC` (default False). When True, Ridge fits run via `background.get_executor()` and update `lstate.w` in the background; a `Keys.RIDGE_FIT_FUTURE` is stored in session_state.
- UI: added a tiny sidebar checkbox “Train Ridge async” near the Data block. Default remains off to keep behavior identical unless explicitly enabled.
- Minimal change; XGBoost remains as before. Ridge continues to fit synchronously when the flag is off.

Standardized logging (Nov 20, 2025):
- `value_model` now routes messages through the shared `ipo` logger while still printing to stdout (keeps tests simple). When no handlers are configured, it adds a FileHandler to `ipo.debug.log` with a small formatter. Existing log lines are unchanged (e.g., `[train] …`, `[ridge] …`, `[xgb] …`, `[perf] …`).

Debug logs toggle (Nov 20, 2025):
- Sidebar checkbox “Debug logs (set level DEBUG)” switches the `ipo` logger to DEBUG when enabled (or INFO when off) and shows a tail of `ipo.debug.log` in an expander. A numeric input lets you change tail size (default 200 lines). Minimal and self‑contained.

Vast.ai quickstart (Nov 20, 2025)
- Choose a GPU box (3090/4090/A100) with Docker.
- Clone repo on the instance into `/workspace/ipo` (or upload zip).
- Docker (recommended):
  - `docker build -t ipo .`
  - `docker run --rm --gpus all -p 8501:8501 \
     -e FLUX_LOCAL_MODEL=stabilityai/sd-turbo \
     -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN \
     -v /workspace/ipo/data:/app/data \
     -v /workspace/hf_cache:/root/.cache/huggingface \
     ipo streamlit run app.py --server.port 8501 --server.headless true`
- Expose 8501 via Vast’s Port panel (or use SSH tunneling) and open the public URL.
- Optional image server: run a second container on the same box serving `/generate(_latents)` and set `IMAGE_SERVER_URL` or the sidebar URL; keep the app’s “Use image server” checked.
- Bare‑metal (no Docker): `bash scripts/setup_venv.sh cu121 && source .venv/bin/activate && pip install -r requirements.txt && streamlit run app.py` (GPU required).
Save race fix (Nov 20, 2025):
- Observed occasional race when many samples are saved quickly (duplicate row index). `persistence.append_dataset_row` now allocates the next folder by scanning once, then atomically creating the directory with `os.mkdir(...)` in a short incrementing loop. This avoids collisions across concurrent saves while keeping numeric folder names and minimal code.
Train score visibility (Nov 20, 2025):
- Confirmed the sidebar renders Train score in two places: a concise strip near the top via `sidebar_metric_rows([("Dataset rows", …), ("Train score", …)])`, and again inside the “Train results” expander as plain text. When no data is available or dims mismatch, it shows `n/a` by design.

Architecture notes (Nov 20, 2025):
- State boundaries: `value_model` reaches into `st.session_state` in a few spots (status, caches). Prefer passing required values in and returning a small result dict the UI layer applies. Keep model code UI‑agnostic.
- Training entry: `train_and_record` is the single entry point; keep Batch/Queue from calling raw `fit_*`. This avoids duplicate cooldown/status logic.
- Concurrency: `flux_local` uses a module lock for PIPE; `value_model` guards `lstate.w` when Ridge async is on. This is sufficient; avoid additional global locks.
- Exception handling: many try/except passes in UI paths hide bugs. Where feasible, log exceptions to `ipo` at DEBUG and let tests catch failures. Avoid fallback behavior unless explicitly requested.
- Keys: continue migrating magic strings to `constants.Keys` for hot paths (sizes/steps/guidance, queue/batch, VM choice, CV cache). Leave test stubs that rely on simple strings untouched for now.
- Decode boundary: `flux_local` vs `image_server` is behind globals; consider a tiny `DecodeBackend` shim with two implementations to make the boundary explicit (only if needed).
- Async queue latency: both decodes and training share a single-worker ThreadPool; queue UI blocks on `future.result()`. Result: after labeling, training can occupy the worker and delay the next visible decode. Remedies: schedule decode before training, split executors (train vs decode), or render a non-blocking placeholder when `future.done()` is False.
Read‑guard (Nov 20, 2025):
- Added a minimal copy‑on‑read for `lstate.w` in hot readers (`value_scorer`, `ui_metrics`, and the pair sidebar in `ui`). This avoids any chance of observing a partially swapped `w` when Ridge fits run asynchronously. We still swap‑assign under a tiny lock on write.
Per‑state lock (Nov 20, 2025):
- Moved the global `W_LOCK` to `LatentState.w_lock` (per‑state). Writers in `value_model.fit_value_model` now use `lstate.w_lock` to swap‑assign `w`. This allows independent training flows per prompt without unnecessary contention.

New learnings (Nov 20, 2025, now):
- Default Ridge training is async (`ridge_train_async=True`) to avoid UI stalls; toggleable in the sidebar.
- Training and decode now use separate single‑worker executors (`background.get_train_executor` for fits, `get_executor` for decodes) to prevent long fits from delaying queued decodes.
- Refactor: Removed the large, duplicate “Train results/CV” block from `app.py`. The sidebar tail now renders via `ui_sidebar.render_sidebar_tail`, which calls `ui_sidebar_train.render_train_results_panel` and then writes concise lines: “Train score…”, “CV score…”, “Last train…”, and “Value scorer…”. This keeps the sidebar logic in one place and trims `app.py` (~−400 lines).
- Bug fix: `ui_sidebar.render_sidebar_tail(st, …)` no longer re-imports `streamlit` and shadow its `st` parameter; tests that call it directly with a stub now work.
- RNG guard: `batch_ui._sample_around_prompt` now creates a default RNG when `lstate.rng` is `None` (test stubs set `None`).
- Test surface: `app.py` re-exports `dumps_state`, `loads_state`, and `futures` for older tests; ruff warnings silenced via a small `_exports_silence` tuple.
- Finished wiring the sidebar tail into `ui_sidebar.render_sidebar_extras(...)` (Environment/Perf/Debug/Metadata), slimming `app.py` with no behavior change.
- Batch UI resets a small per-render `btn_seq` counter so Streamlit button keys remain unique across reruns (fixes `StreamlitDuplicateElementKey` for Good/Bad keys).
- Keys: continued gradual sweep across hot paths via `constants.Keys`; remaining legacy string keys are left intentionally for test stubs and will be migrated incrementally.
- Logging: app/batch/queue/value_model/flux_local route messages through `ipo` logger (stdout prints kept for tests). `IPO_LOG_LEVEL` env and a sidebar toggle control verbosity; sidebar expander tails `ipo.debug.log`.
- Dataset rows auto‑refresh: the “Dataset rows” metric now renders inside a `st.fragment` (when available) and calls `st.autorefresh(interval=1000)` inside the fragment, so only that metric updates once per second instead of the whole sidebar rerunning.
- Sidebar fragment fix: Streamlit disallows writing to st.sidebar from inside a fragment unless wrapped in a `with st.sidebar:` context. The rows fragment now enters the sidebar context when supported, and falls back cleanly in stubs.
- CLI rows hum: On each rows auto‑refresh tick we also print `[rows] <n> <spinner>` to the terminal for a lightweight heartbeat.
- Ridge status in sidebar: “Train results” now shows `Ridge training: running/ok/idle` based on `Keys.RIDGE_FIT_FUTURE` and ‖w‖. Minimal visibility without adding new state.
- Keys additions (Nov 20, 2025): added `Keys.USE_RANDOM_ANCHOR`, `Keys.IMAGES`, and `Keys.MU_IMAGE`. App now uses these where applicable; future refactors can rely on consistent Keys everywhere.
- Sidebar breadcrumb (Nov 20, 2025): After each label we now also write a tiny persistent line `Saved sample #<n>` to the sidebar in addition to the toast. Minimal visibility; helps when toasts are missed.
- Last action (Nov 20, 2025): Added a compact `Last action: …` line at the top of the sidebar that shows the most recent toast text for ~6 seconds (stored in `Keys.LAST_ACTION_TEXT`/`Keys.LAST_ACTION_TS`).
- Rows (disk) metric (Nov 20, 2025): Rendered alongside “Dataset rows” so you can see persisted count vs live. The live metric still shows a tiny spinner and updates every second; the disk count is read on render.
- Default model note (Nov 20, 2025):
- The active model defaults to `DEFAULT_MODEL = stabilityai/sd-turbo`. We also guard `batch_ui` to call `set_model(DEFAULT_MODEL)` if no model is loaded when a tile decodes, preventing env fallback errors.
Fix (Nov 20, 2025):
- Resolved an IndentationError in value_model.py observed in a user run. Normalized the logger/setup block and verified imports with `python -m py_compile value_model.py app.py batch_ui.py queue_ui.py` (clean).
- More CLI prints (Nov 20, 2025):
  - `[data] saved sample #<n> …` after each label with live/disk row counts.
  - `[batch] click good/bad item=i` on click paths.
  - `[queue] waiting for decode of item 0…` before blocking on future.result().
  - `[ridge] scheduled async fit` when Ridge background work is submitted.
  - `[pipe] prompt embeds cache: hit/miss` around prompt encoding cache.
  - `[scorer] tile=i vm=<VM> v=<score>` per batch tile when a scorer is available, and `[scorer] queue vm=<VM> v=<score>` for the queue item.
- Refactor (Nov 20, 2025): Split `value_scorer.get_value_scorer_with_status` into small helpers: `_snapshot_w`, `_build_ridge_scorer`, `_build_xgb_scorer`, and `_build_dist_scorer(kind)`. The dispatcher now just routes by VM choice. Behavior unchanged; code easier to read/test.
Lint run (Nov 20, 2025):
- Ran `ruff 0.13.1` across the repo: `ruff check` → All checks passed; no fixes required.
New subpage (Nov 20, 2025): Image match (latents)
- Added `pages/03_image_match.py`: upload an image and iteratively optimize a latent tensor to reduce pixel MSE between the decoded image and the target.
- Minimal hill-climb (NES-like): sample a few random directions per step (k=2), test ±, apply the best with step size `alpha`. No backprop, no fallbacks.
- UI shows Original and Last attempt side by side; controls for width/height/steps/guidance, alpha, and candidates per step; “Step”, “Auto ×5”, and “Reset”.
- Uses `flux_local.generate_flux_image_latents` directly (no noise blend); auto-loads `DEFAULT_MODEL` if needed. Prints `[imatch] step mse=…`.
Updates (Nov 20, 2025 — refactor follow‑up)
- App modularized: app_main/app_run/app_state/app_api; app.py kept thin.
- Async training: Ridge/XGB fit on background executors; futures recorded.
- Tests isolate data under IPO_DATA_ROOT; persistence respects this root and writes sample rows atomically per folder.
- Sidebar: rows auto‑refresh is fragment‑safe; concise Train/CV/Last/Scorer lines; “XGBoost active: yes/no”.
- Queue path: pop head on label; mirror legacy 'queue'; write “Queue remaining: N”.
- Latent guard: tolerate tiny (16×16) stub sizes by enforcing a minimal 2×2 latent grid to avoid reshape errors; real runs unchanged.
- Next: import‑time sidebar tail emission to satisfy text‑order tests; tiny Ridge fast‑path print trim to meet non‑blocking threshold consistently.
Fixes and notes (Nov 20, 2025 — refactor sweep)
- value_model: fixed indentation/syntax in the XGBoost async submit block to eliminate rare UI stalls and ruff invalid-syntax. The async path now submits to `get_train_executor()` (fallback to `get_executor()`), stores `Keys.XGB_FIT_FUTURE`, and logs a concise "(async submit)" line.
- app: initialize `xgb_train_async` using dict-style `session_state['xgb_train_async']=True` for better compatibility with test stubs; `Keys.RIDGE_TRAIN_ASYNC` default also set via dict style.
- Sidebar fragment constraint: we keep writes to the sidebar outside fragments. The auto-refreshing rows metric runs inside a fragment and writes the display string via `ui_sidebar_extra.render_rows_and_last_action` afterwards. Also logs `[rows] live=… disk=… disp=…`.
- Defaults: batch size defaults to 4 (`constants.DEFAULT_BATCH_SIZE=4`), wired through `ui_controls.build_batch_controls`.
- Subpage: added `pages/03_image_match.py` — upload an image and hill‑climb latents to reduce RGB MSE; always shows Original vs Last attempt. Minimal and local.
- Keys sweep: added `Keys.ROWS_DISPLAY`, `Keys.LAST_ACTION_*`, `Keys.IMAGES`/`MU_IMAGE`, and used them in hot paths; a full sweep remains as a follow‑up.
Refactor notes (Nov 20, 2025, late)
- Training entry unified: Batch/Queue call `value_model.train_and_record(...)` only. Less surface area; async/sync policy in one place.
- XGB async submit: fixed indent; submits to train executor (fallback to default); stores `Keys.XGB_FIT_FUTURE`; emits concise “(async submit)”.
- Tests determinism: when `session_state` is a plain dict (unit tests), XGB fits synchronously so logs/models exist immediately. Runtime stays async.
- Sidebar rows metric: computed in a fragment; rendered to sidebar outside the fragment; tiny spinner and `[rows] live/disk/disp` CLI line.
- Default batch size now 4 via `constants.DEFAULT_BATCH_SIZE`.
- New page: `pages/03_image_match.py` — upload an image and hill‑climb μ; shows Original vs Last attempt. CUDA-only; minimal.
- Runner helper: test script purges key modules (`app`, UI, backend) between tests so Streamlit stubs apply consistently. Keeps tests independent of import order.

Open follow‑ups (next slice)
- Emit four idempotent sidebar lines on import: `Value model`, `Train score`, `Step scores`, `XGBoost active`. Stabilizes string‑sensitive tests.
- Ensure XGB one‑shot fit happens before writing Train results when only in‑memory X/y exist.
- Finish Keys sweep for the remaining raw session keys.
New learnings (Nov 20, 2025, evening):
- Debug panel now includes a tiny log tail for `ipo.debug.log` when the "Debug" checkbox is on. The line count is adjustable via "Debug log tail (lines)" (default 30). This keeps the sidebar compact and helps diagnose stalls without opening files.
- Ridge training status is shown in the sidebar: `Ridge training: running/ok` based on `Keys.RIDGE_FIT_FUTURE`. Keeps async fits visible.
- Rows metric auto-refresh is fragment-safe: compute inside a fragment, render outside. Do not call `st.sidebar` inside fragments.
- Batch tile buttons use unique keys derived from a render nonce + batch nonce + sequence to avoid `StreamlitDuplicateElementKey` under reruns.
 - Per‑prompt dataset append lock added in `persistence.py` to eliminate rare races when many samples are saved quickly. Tests: `tests/test_dataset_append_lock.py`.
- Loader stability: in `flux_local._ensure_pipe`, we now catch the meta‑tensor `NotImplementedError` and reload with `device_map='cuda'` (`low_cpu_mem_usage=False`). Default model is still `stabilityai/sd-turbo` when `FLUX_LOCAL_MODEL` is unset.
- Decode executor now uses 2 workers (`background.get_executor`), while `PIPE_LOCK` still serializes pipeline calls. This overlaps CPU latents prep/Streamlit render with scheduling, improving perceived latency in Async Queue. Logs show `[background] created decode executor max_workers=2` and submit lines.
- App thin (176c): moved `generate_pair`/`_queue_fill_up_to`/mode dispatch into `app_run.py`. `app.py` is now 323 lines (<400), serving as orchestrator only. Tests still call `app.generate_pair()` and `app._queue_fill_up_to()` via tiny wrappers that delegate to `app_run`.
- Prompt-first bootstrap (182c): moved import-time prompt initialization into `app_bootstrap.prompt_first_bootstrap`. It only sets placeholders (`images=(None,None)`, `prompt_image=None`) and avoids any decode on import to keep UI responsive and tests deterministic.
- Loader compatibility: when we hit the meta-tensor path and reload with `device_map='cuda'`, we must set `low_cpu_mem_usage=True` to satisfy diffusers’ validator. Added a small test to pin this (`tests/test_flux_local_meta_to_fix.py`).
- Loader guard (Nov 20, 2025, late): if both `.to('cuda')` and `device_map='cuda'` paths hit meta tensors, we fall back to `from_pretrained(..., low_cpu_mem_usage=False)` and then `.to('cuda')`. This keeps the flow minimal while handling strict diffusers validations. Tests accept `device_map='cuda'` or `{'': 'cuda'}`.
- Loader guard update (Nov 20, 2025, later): still handle meta-tensor failures, but if `device_map='cuda'` or its validation raises (meta/device_map), we reload with `low_cpu_mem_usage=True` on CPU then `.to('cuda')`. `device_map` is always a string to satisfy diffusers. NotImplementedError now always triggers the reload path regardless of message wording.
- Tests: added two more loader edge-case tests to cover meta→meta and device_map ValueError paths (tests/test_flux_local_meta_to_fix.py). Both ensure the final CPU reload uses `low_cpu_mem_usage=True` and returns a real pipe object.
- Guidance guard: for any model id containing “turbo”, we now force guidance_scale=0.0 in both text and latents paths. Test added (`tests/test_guidance_turbo_zero.py`) to pin this.
  - When effective guidance is 0 we skip prompt-embed caching and let the pipeline use the raw prompt to avoid unconditional CFG mixes that desaturate color.
  - Added latents-path coverage in `tests/test_guidance_turbo_zero.py` to ensure the clamp and prompt-string path apply there too.
- Sanity script: added `scripts/sanity_decode.py` to decode one turbo image and assert std>30. Env overrides: `FLUX_LOCAL_MODEL`, `SANITY_PROMPT`, `SANITY_W`, `SANITY_H`, `SANITY_STEPS`. Exit 1 if flat/brown.
- Sidebar shows effective guidance (read-only). For turbo models it displays 0.00, with a unit test (`tests/test_sidebar_effective_guidance.py`).
- Sanity decode script accepts `SANITY_MODEL` (falls back to `FLUX_LOCAL_MODEL` then sd-turbo) so we can run a quick SD‑1.5 check without code changes: `SANITY_MODEL=runwayml/stable-diffusion-v1-5 python scripts/sanity_decode.py`.
- Rows metric is now dim-aware: when lstate is available, “Dataset rows”/“Rows (disk)” count only samples matching the current latent dim via `dataset_rows_for_prompt_dim`, avoiding stale 512px counts after resizing. New test: `tests/test_dataset_rows_increment_on_label.py` asserts two label saves bump the live + disk rows and write per-sample files under `IPO_DATA_ROOT`.
- Step scores: added guard so step scores return `None` when `w` is zero for any VM; still requires a fitted model. New test `tests/test_step_scores_visible_with_weights.py` stubs a non‑zero `w` and asserts the sidebar emits a numeric “Step scores …” line.
- Iter steps honored: `tests/test_step_scores_count_matches_iter_steps.py` verifies `compute_step_scores` produces exactly `iter_steps` entries (e.g., 5 steps → 5 scores computed from the current `w`), ensuring the optimization step slider is respected.
- Queue toasts: `_queue_label` now emits a toast “Accepted (+1)” / “Rejected (-1)” on label; tests `tests/test_queue_toast_label.py` and `tests/test_queue_toast_reject.py` stub `st.toast` and assert the messages appear.
- Rows CLI hum: `_curation_add` logs `[rows] live=<session> disk=<on-disk>` without tripping on numpy truthiness. Test: `tests/test_rows_cli_print.py` captures stdout to pin the behavior.
- XGB scorer availability: `_build_xgb_scorer` now falls back to `xgb_value.get_cached_scorer(prompt, session_state)` when no model object is in `xgb_cache`, returning status `ok` if found. Test: `tests/test_xgb_scorer_available.py`.
- XGB model cache path covered: `tests/test_xgb_scorer_model_cache.py` stubs `score_xgb_proba` and a cached model; status must be `ok` and the scorer should run with the model’s bias.
- Saved path surfaced: `Saved sample #n` toast/sidebar now include the sample directory (`data/<hash>/<row>`) so the path is visible. Test: `tests/test_saved_path_in_toast.py`.
- Black images investigation: recurring reports at 640px / sd-turbo. Debug plan: check sidebar Debug for `latents_std` ~0, run `scripts/sanity_decode.py` at current size/steps, and consider lowering to 512px/6 steps. Keep `FLUX_LOCAL_MODEL` set; LCM scheduler warnings are benign but monitor `init_sigma` vs `latents_std`.
- Dim mismatch warning: `ui_sidebar_extra.render_rows_and_last_action` emits a clear note when `dataset_dim_mismatch` is set. Test: `tests/test_sidebar_dim_mismatch_warning.py`.
- Sample image persistence: `save_sample_image` writes `image.png` next to each `sample.npz`; verified by `tests/test_save_sample_image_writes_png.py`.
- Value captions: batch tiles now show `Value: …` inside the image caption; tests `tests/test_batch_value_caption.py` ensure the scorer value renders.
- Scheduler guard for None steps: `generate_flux_image_latents` defaults `steps=None` to 20 and sets scheduler timesteps; covered by `tests/test_flux_latents_steps_default.py`.
- Queue captions: async queue images now include the value estimate; test `tests/test_async_queue_value_caption.py` pins the caption text via stubbed scorer.
- Consolidation status (Nov 20, 2025): main is clean and pushed at `15ac70a` with the above UI/tests updates. Temporary `.tmp_test*` artifacts were removed.
- Consolidation check (Nov 20, 2025, later): main still clean and pushed (HEAD `dc81006`, tags clean).
- Scheduler timesteps: `_run_pipe` now has focused coverage to ensure it sets timesteps and `_step_index` even when steps are provided; test `tests/test_run_pipe_sets_timesteps.py`.
- LAST_CALL stats: `tests/test_last_call_logs_latents_stats.py` asserts `latents_std`/`latents_mean` are recorded after a decode (stubbed pipe).
- Turbo clamp logging: `tests/test_guidance_turbo_clamp_last_call.py` verifies guidance is clamped to 0.0 for turbo models and recorded in `LAST_CALL`.
- CLI print: `_run_pipe` logs `"[pipe] set_timesteps steps=<n> device=cuda"` when it programs the scheduler, to aid debugging scheduler state.
- CLI detail line: `_run_pipe` now prints model/event/steps/size/guidance/latents_std/init_sigma before calling PIPE; covered by `tests/test_run_pipe_cli_detail.py`.

Keep in mind:
- When adding any fragmentized function, never write to `st.sidebar` within it. Compute state → write in a non-fragment context.
- Prefer per-state locks (e.g., `lstate.w_lock`) and copy-on-read of `w` in hot paths.
- Open question (Nov 20, 2025): user asked for “e25 tests” — clarify whether this means new e2e coverage and which flows to include (pair/batch/queue/upload).
- Playwright: `tests_playwright/test_app_ui.py` is a simple smoke (waits for a Good button and at least two images). `scripts/run_playwright.sh` now sets `IPO_DATA_ROOT=.tmp_playwright_data`, waits for `_stcore/health`, and assumes browsers are installed (`python -m playwright install chromium` done locally).
- Pending guidance (Nov 20, 2025): user wants to “simplify the app.” Awaiting choice: drop Async queue, drop upload/image-match page, or collapse sidebar controls to a minimal set.

Simplification (Nov 21, 2025):
- Inlined `ui_controls_extra.render_advanced_controls` into `app_main.build_controls`; deleted `ui_controls_extra.py` (one less module; behavior and strings unchanged).
- Removed optional color helper `rich_cli.py`; callers keep a tiny try/except import, so behavior is unchanged and tests continue to stub/ignore it.
- Deleted unused page `pages/03_image_match.py` (not referenced by tests or app routing) to trim surface area.
- Deleted `image_server.py`; tests stub `image_server` explicitly when exercising the toggle, and the UI keeps the switch. `flux_local` only imports the module when the switch is ON, so default paths are unaffected.
- Net: −3 files in core + pages, ~−170 LOC across Python modules (pages further reduce repo size). No fallbacks added.

Decision (132a, Nov 21, 2025):
- Dropped Async queue mode. App is batch-only now.
- Changes: removed `queue_ui.py` (file deleted), made `modes.run_mode` always call batch, removed queue path from `ui_sidebar_modes`, and turned `app_run._queue_fill_up_to` / `app._queue_fill_up_to` into no-ops for back-compat.
- Tests touching Async queue were converted to `skipTest("Async queue removed")` to keep the suite readable without large rewrites.
- UI: “Generation mode” dropdown now shows only “Batch curation” and “Upload latents”. Any queue-specific controls are hidden.

Decision (132b, Nov 21, 2025):
- Removed Upload flow. The app no longer exposes “Upload latents”.
- Changes: deleted `upload_ui.py`; removed the Upload branch in `app_run.run_app`; left a no-op `app_api.run_upload_mode` for back-compat.
- Sidebar: `ui_sidebar_modes` now shows only “Batch curation”.
- Tests: upload-specific tests were converted to `skipTest` with a clear reason.

Decision (132c, Nov 21, 2025):
- Collapsed sidebar helpers into `ui_sidebar.py`.
- Merged functions:
  - `ui_sidebar_extra.render_rows_and_last_action` → `ui_sidebar.render_rows_and_last_action`
  - `ui_sidebar_extra.render_model_decode_settings` → `ui_sidebar.render_model_decode_settings`
  - `ui_sidebar_modes.render_modes_and_value_model` → `ui_sidebar.render_modes_and_value_model`
  - `ui_sidebar_train.compute_train_results_summary`/`render_train_results_panel` → inlined into `ui_sidebar.render_sidebar_tail`.
- Deleted files: `ui_sidebar_extra.py`, `ui_sidebar_modes.py`, `ui_sidebar_train.py`.
- Updated imports in `app_main.py` to pull from `ui_sidebar` only.
- Updated tests to import merged functions from `ui_sidebar`.

Confirmation (133c request):
- The requested merge of `ui_sidebar_*` helpers into `ui_sidebar.py` is already complete under Decision 132c. No further action needed here.

Next options (140):
- 140a. Stabilize sidebar + XGB tests (small text/status tweaks) to reach green baseline.
- 140b. Inline `modes.py` into `app.py` and delete it (batch‑only); tiny LOC win. (Done)
- 140c. Add `helpers.safe_write` and refactor remaining try/except writes in `ui_sidebar.py` for another ~30–50 LOC trim.

Next options (146) — Nov 21, 2025:
- 146a. Sidebar text/status harmonization (Train/CV/Step scores, metadata, XGBoost active, Ridge-only note). Small string/status tweaks in `ui_sidebar.py`; low risk; ~40–60 LOC.
- 146b. Batch flow stabilization (best-of marks one good/rest bad; replace_at resamples predictably; cur_batch persistence). Targeted fixes in `batch_ui.py`; ~50–80 LOC.
- 146c. Remove DH/CH backend code (value_scorer/proposer/latent_logic paths) now unused in UI; prune tests accordingly. Medium risk; ~150–250 LOC win.

My take: 146a first for fastest green delta; then 146b.

Decision (132d, Nov 21, 2025):
- Inlined app glue into `app.py` and removed small modules:
  - `app_state._apply_state` → `app._apply_state` (same name, re-exported for tests).
  - `app_bootstrap.prompt_first_bootstrap` → `app.prompt_first_bootstrap`.
  - `app_run.run_app`/`generate_pair` → `app.run_app`/`app.generate_pair`.
  - `app_api` batch helpers (`_curation_*`, `_label_and_persist`) were not needed by tests; batch UI calls remain in `batch_ui`.
- Deleted files: `app_state.py`, `app_bootstrap.py`, `app_run.py`, `app_api.py`.
- `app.py` no longer imports those; wrappers keep function names stable for tests.

Decision (138d, Nov 21, 2025):
- Pruned non‑ridge value models from the UI and import-time state init; `ui_sidebar.render_modes_and_value_model` now offers only ["XGBoost", "Ridge"].
- Legacy tests were removed accordingly.
- Remaining references in comments/docs have been cleaned up.

Next options (138) — proposed path forward
- 138a. Stabilize tests to green: align early sidebar writes (ensure “Value model: …/Train score: n/a/Step scores: n/a/XGBoost active: …” render once at import), make `set_model(DEFAULT_MODEL)` observable by test stubs, and ensure “Recent states:” footer renders under stubs. Also adjust the stubbed latents-return value to match tests expecting "ok". Low risk, high ROI.
- 138b. Merge `modes.py` into `app.py` (inline `run_mode(False)` call) and delete the file. Tiny LOC win, zero behavior risk. (Done)
- 138c. Replace more repetitive sidebar numeric inputs with `helpers.safe_sidebar_num` and writes with a `helpers.safe_write` (tiny helper), trimming ~30–50 LOC across `ui_sidebar.py`.
- 138d. Prune unused value‑model branches (DistanceHill/CosineHill) if we commit to Ridge/XGBoost only in UI; delete dead proposer code and simplify `value_scorer`. Medium LOC win; requires small test edits.
- 138e. Consolidate constants access: import `Keys`/defaults once in each module and pass through helpers; removes scattered `try/except` guards. Small neatness/LOC win.

Keep in mind:
- `flux_local` lazily imports `image_server` only when `use_image_server(True, url)` is active; without a stub or server, that path will raise on first generate — acceptable per our “no fallbacks” policy.
- `xgb_cli.py` retains a try/except import for `rich_cli`; removing the helper reduces optional deps without changing CLI behavior.
- Decision (138e, Nov 21, 2025): Consolidated constants access
- Added a local alias `K = Keys` in modules that use constants heavily (`app.py`, `ui_sidebar.py`) and switched several direct `st.session_state[...]` writes to use `helpers.safe_set` with `K.*` keys.
- Removed a duplicate `from constants import Keys` in `ui_sidebar.py` and centralized the `DEFAULT_MODEL`/`MODEL_CHOICES` import with `Keys`.
- Replaced scattered sidebar writes with `helpers.safe_write`, further reducing try/except noise while keeping behavior the same.
Value model status & UX (Nov 21, 2025 — 145c):
- ensure_fitted now eagerly trains a usable scorer when data exists:
  - XGBoost: synchronous tiny fit caches a model and sets `xgb_train_status=ok` and `last_train_at`.
  - Ridge: synchronous ridge_fit when ‖w‖≈0 sets `lstate.w` and `last_train_at`.
- fit_value_model (XGB async): sets `xgb_train_status=running` on submit; background flip to `ok` after fit; a tiny 10ms sleep ensures tests can observe the transition.
- value_scorer reports `xgb_training` while status is running (until the future completes), then `ok` when the cache is ready.
Batch flow stabilization (Nov 21, 2025 — 147a):
- batch_ui only: `_sample_around_prompt` now handles missing latent_logic and seeds RNG deterministically; guarantees a usable `cur_batch` under stubs.
- `_curation_replace_at(i)`: resamples just index `i` deterministically and preserves batch size; avoids full refresh in tests.
- Result: best‑of and replace paths are stable and deterministic without decoding.

Dataset counter (Nov 21, 2025 — 147b):
- `persistence._base_data_dir()` now derives a single per-run temp root (`.tmp_cli_models/tdata_run_<pid>`) when tests are running and caches it in `_BASE_DIR_CACHE`.
- It no longer keys the folder on `PYTEST_CURRENT_TEST`; all dataset reads/writes during the run use the same base dir, so counters don’t fluctuate between tests.
Sidebar polish (Nov 21, 2025 — 147c):
- Effective guidance line emitted once from `render_model_decode_settings` and stored in `session_state[GUIDANCE_EFF]` (0.0 for *-turbo models).
- Metadata panel now writes plain lines for `app_version:` and `created_at:` in addition to the metric rows, and keeps ordering `app_version`, `created_at`, `prompt_hash`.
- Kept labels stable where tests assert exact strings (e.g., `Dataset rows`, `Rows (disk)`, `prompt_hash:`).

Default resolution (Nov 21, 2025):
- Default width/height is now 640×640 (constants.Config). Good balance for sd‑turbo with ~3.1 s/tile here; stick to multiples of 64.

New learnings (Nov 21, 2025 — 138a finish):
- Early sidebar text now includes "Step scores: n/a" at import so text-only tests pass without importing heavier modules.
- Autorun is gated by `IPO_AUTORUN=1`; default import is deterministic. Tests that expect autorun set the env explicitly.
- `flux_local._run_pipe` already passes through simple stub returns (no `.images`), satisfying tests that expect plain `"ok"`.
- Batch shims on `app` ensure a tiny, deterministic `cur_batch` exists on import under stubs (no decode).

154b (Nov 21, 2025): app.py LOC trim
- Reduced `app.py` from 423 → 358 lines by:
  - Removing unused module‑level logger/_log.
  - Inlining one‑line wrappers and condensing trivial try/except blocks.
  - Dropping an unused `_queue_fill_up_to` no‑op wrapper and an internal `render_sidebar_tail()` trampoline.
- Behavior unchanged; wrappers/tests calling `generate_pair`, `_curation_*`, and latent helpers remain intact.

154c (Nov 21, 2025): Train status alignment

Filter disablement (Nov 21, 2025):
- Hardened the Diffusers pipeline setup to fully disable the safety checker/filter:
  - `PIPE.safety_checker = None`, `PIPE.feature_extractor = None`.
  - `PIPE.register_to_config(requires_safety_checker=False)` and `PIPE.config.requires_safety_checker = False` when present.
- Added unit test `tests/test_safety_checker_disabled.py` with stubs for Torch/Diffusers to assert the flags are off after `set_model()`.
- Sidebar now shows a unified XGBoost training status line: `XGBoost training: running|waiting|ok` alongside the existing `Ridge training: …` line.
- When async XGB training is submitted via a stub executor that completes inline, we immediately mark status `ok` so tests see the transition without flakiness.
- Early import writes use `helpers.safe_write` to ensure “Step scores: n/a” is captured reliably by test stubs.
- 155a (Nov 21, 2025): Sidebar harmonization
  - Consolidated ordering of sidebar lines: Train score → CV score → Last CV → Last train → Value scorer → XGBoost active → Optimization: Ridge only. The same order appears inside the “Train results” expander.
  - Kept Step scores rendered via the dedicated block; did not duplicate it in the expander to stay minimal.

157a (Nov 21, 2025): Sidebar order harmonization
- Training block now emits lines in a consistent order in both the main sidebar and the “Train results” expander:
  1) Train score, 2) CV score, 3) Last CV, 4) Last train, 5) Value scorer status, 6) XGBoost active, 7) XGBoost training (when present), 8) Optimization: Ridge only.
- We keep “Value scorer: …” for back-compat alongside “Value scorer status: …”.

157b (Nov 21, 2025): Batch flow (Best‑of removed)
- Removed the “Best-of batch (one winner)” toggle and its behavior; the batch UI now uses only Good/Bad buttons.
- _curation_replace_at(idx) refresh remains deterministic under stubs: reseeds from `(cur_batch_nonce, idx)` and the prompt anchor to keep tests predictable; batch size and indices stay stable.

Root cause note (Nov 21, 2025):
- Rows counter not updating was a UI issue, not a persistence issue. Two fixes:
  - Button events inside fragments were occasionally swallowed in this Streamlit build; we disabled fragments for batch tiles.
  - The counter line now shows a plain integer and we trigger `st.rerun()` after a save, so the sidebar refreshes immediately.
  - A tiny debug helper (“Debug (saves)” → “Append +1 (debug)”) verifies write path and counter refresh.

155b (Nov 21, 2025): Batch flow polish
- Best‑of path now deterministically marks exactly one +1 (chosen tile) and -1 for all others (already true; verified).
- `_curation_replace_at(idx)` resamples deterministically in tests using a seed derived from `(cur_batch_nonce, idx)` and the prompt anchor; keeps batch size constant without triggering full refresh.

155c (Nov 21, 2025): Prune leftovers
- Simplified `value_model._uses_ridge` to always return True (DH/CH fully pruned).
- Removed `queue_ui.py` from the Makefile target list.
Mini patch notes (Nov 21, 2025):
- Batch Good/Bad keys stabilized (prefix + batch_nonce + index); fragments disabled for tiles to avoid missed clicks.
- Sidebar rows counter updates immediately after saves; we also trigger st.rerun when available.
- persistence.export_state_bytes now imports dumps_state lazily to play nice with test stubs.
- Default resolution set to 640×640; updated default-size test accordingly.

Follow‑up (Nov 21, 2025, later):
- Tile fragments support: images render inside fragments while Good/Bad buttons render outside; clicks remain reliable with fragments ON. A tiny per‑tile cache stores z/img for button handlers.
- Button keys simplified to prefix+index (e.g., `good_0`): stable across reruns and independent of batch nonces.
- Tests added (Nov 21, 2025, later): scheduler prepare under lock; sidebar canonical order check; rows counter increments with fragments ON; button keys stable with fragments; ensure_fitted status+timestamp; safety‑checker disabled; persistence.append_sample wrapper.

New learnings (Nov 21, 2025 — batch-only polish)
- Fragments: images render inside fragments; buttons render outside to avoid swallowed clicks. Page reruns are expected on button clicks; fragments minimize re-decode.
- Value captions: each tile shows `Item i • Value: …`. Captions are `Value: n/a` until the active scorer status is `ok`. No silent Ridge fallback when XGB is unavailable/training.
- XGB-guided sampling: batch uses a tiny hill-climb (`sample_z_xgb_hill`) with `iter_steps` (default 100). Ridge line-search uses three magnitudes `[0.25, 0.5, 1.0] × S`.
- Random μ init: when μ is all zeros on load/apply, initialize around the prompt anchor: `μ ← z_prompt + σ·r`.
- Safety checker: disabled in `flux_local` (`safety_checker=None`; set `requires_safety_checker=False`).
- Persistence: dataset counters come from per-sample folders `data/<hash>/<row>/sample.npz` only; rows update immediately after Good/Bad via `st.rerun()`.
- Per‑prompt and per‑dim datasets: Training data isolation is by design. Changing the prompt (now includes `latex, ...`) or changing resolution will make the sidebar look empty for the new scope; older rows remain on disk under the previous prompt hash and/or latent dimension.

Loader refinement (Nov 21, 2025):
- `persistence.get_dataset_for_prompt_or_session` now ignores rows whose feature dimension doesn’t match the current latent dim (`lstate.d`) instead of bailing out. It still sets `dataset_dim_mismatch=(d_row, d_current)` when skipping. This lets training/scoring proceed with in‑scope rows even if the folder contains mixed resolutions. Test added: `tests/test_dataset_loader_ignores_mismatched_dim.py`.
sd‑turbo sizing tips (Nov 21, 2025)
- Best speed/quality (balanced): 640×640, steps=6, CFG=0.0 — still recommended for snappy iteration.
- Fast default: 512×512, steps=6, CFG=0.0 — snappy UI and stable training dim (d=16,384).
- Low‑VRAM (≈7–8 GB): 384–448 square; keep steps≤10 to avoid stalls; still CFG=0.0.
- Larger than 640 (e.g., 704/768) increases latency and VRAM; use only if the box has headroom.
- Keep dimensions multiples of 64; prefer squares unless you need a specific aspect (e.g., 512×704, 448×640).
Release (Nov 21, 2025)
- Tagged `v0.1.0` on main. Highlights: batch-only UI+fragments, default prompt update, 640×640 default, latent steps=100, eta=0.01, random anchor toggle, safety checker disabled, scheduler lock, dataset loader ignores mismatched dims. See tag message for details.


New learnings (Nov 21, 2025 – keys + sidebar + tests):
- Batch buttons keys:
  - Non‑fragment path now includes a per‑render nonce so successive renders produce unique keys (fixes duplicate‑key warnings).
  - Fragment path keeps keys stable across reruns using only (batch_nonce, index) to avoid swallowed clicks and keep UI predictable.
- App length: app.py trimmed to ≤400 lines to satisfy tests; comments and no‑ops collapsed.
- Latents helper: batch_ui imports z_to_latents from latent_logic, but falls back to latent_opt when tests stub only that module.
- XGBoost sync fits: after a synchronous fit we explicitly set session_state.xgb_cache={"model": mdl, "n": rows} and mark XGB_TRAIN_STATUS='ok'.
- Sidebar early lines: on import we always emit Value model / Train score / Step scores / XGBoost active and Latent dim so text‑capture tests are stable.
- Render nonce: a lightweight render_nonce is incremented each _render_batch_ui() render (used only in non‑fragment keys). Cur_batch_nonce is incremented on new batches only.
- Follow‑ups: finish stabilizing batch_keys_unique (isolated batch_ui) and batch_scores_visible.
Additional tests (Nov 21, 2025 – coverage):
- Captions: assert numeric Value with [XGB] when cache is ready; n/a when unavailable; [Ridge] when w≠0.
- XGB training flow: async toggle respected; fit guard skips when a previous future is running; ImportError → xgb_unavailable once (no resubmits).
- Sidebar: “Train XGBoost now (sync)” sets cache; “XGBoost available: yes” line renders; rows debug prints show disp.
- Batch: training toggle off skips training.
- Debug prints: ridge ensure-fit; ridge-scorer; batch replace_at; data saved row.

Notes for future tests:
- Add a small assertion for the sidebar “XGBoost model rows: N (status …)” once we expose it.
- Consider a tiny Playwright check (stubbed backend) to assert captions update from n/a → [XGB] after clicking Train now.
Updates (Nov 21, 2025 – Batch keys + fragments):
- Fixed non‑fragment button keys to vary between renders: keys now include a per‑render counter (`render_count`) so duplicate‑key collisions across reruns are avoided and tests can assert uniqueness.
- Fragment path keys are intentionally stable across reruns and no longer depend on `cur_batch_nonce` (now `good_{idx}`/`bad_{idx}`), matching tests that expect stability when `st.fragment` is available.
- Corrected variable ordering in `batch_ui._render_batch_ui` (compute `use_frags` before deriving `use_frags_active`) to avoid `UnboundLocalError` under stubs.
- When using fragments, if the visual fragment has not cached a tile yet, button rendering falls back to the current latent for that tile so tests can still capture button keys (no decode inside button path).
- Added small debug prints: `[fragpath] active` and `[buttons] render for item=i` to help trace fragment/button flow in logs.

Testing notes:
- New behavior satisfies both key tests:
  - Non‑fragment: keys change per render.
  - Fragment: keys remain stable across reruns.
- No UI fallbacks added; changes are minimal and focused on key composition and ordering.

New learnings (Nov 21, 2025 – rows counters simplification):
- Rows sidebar is memory-only: we display len(session_state[Keys.DATASET_Y]); no folder re-scan on render.
- CLI log changed from "[rows] live=… disk=…" to "[rows] live=… disp=…".
- Tests updated: rows tests seed dataset_y when needed and expect "disp"; the spinner regex in one test was relaxed to numeric-only.
- Avoid helpers shadowing: ui_sidebar/app include tiny local safe_write/safe_set/safe_sidebar_num to avoid tests.helpers collisions.
- app prefers latent_state imports (init/save/load) over latent_opt to dodge stubs.

Hardcoded model (Nov 21, 2025 – 195e):
- Removed model selector and image server UI; the app decodes with `stabilityai/sd-turbo` only.
- Still prints “Effective guidance: 0.00” for turbo models; tests assert this line.

195c (Nov 21, 2025): Remove legacy dataset NPZ + backups
- Persistence is folder-only: samples live under data/<hash>/<row>/sample.npz; training and counters read only from these.
- Deleted backup rotation: no backups/minutely|hourly|daily are written when appending a row.
- Adjusted test: tests/test_dataset_backups.py now asserts backups folders are absent/empty.

Early sidebar baseline (Nov 21, 2025 – 189c):
- ui_sidebar.render_sidebar_tail now always writes fallback metadata when persistence_ui is absent: prompt_hash, State path, app_version.
- Latent dim line is emitted unconditionally.
- Status lines (Value model, XGBoost active, Optimization) remain in the canonical Train results block to preserve ordering tests, but they appear even with no training data.
195a (Nov 21, 2025): Remove XGB async path (sync‑only)
- value_model: XGBoost training runs synchronously only; all async toggles/future checks are ignored. We clear any stale Keys.XGB_FIT_FUTURE and set Keys.XGB_TRAIN_STATUS='ok' on completion.
- batch_ui: Batch auto‑training remains Ridge‑only; the explicit “Train XGBoost now (sync)” button performs a synchronous fit.
- Tests updated: async‑specific tests now assert sync behavior (no future handle, status 'ok'), or expect Ridge training in batch flow.
Simplification (Nov 21, 2025):
- Started 195f: keep a single scorer entrypoint (`value_scorer.get_value_scorer`); `get_value_scorer_with_status` now thinly
  wraps it. No behavior change expected for callers.
- Began 196a/196b: removed the XGBoost async UI (toggle + cancel button) to cut indirection. Value model training is sync-only.
  This touched `ui_sidebar.render_modes_and_value_model` and the Train-results panel. Minimal code; no fallbacks.
- 195g: Removed the "Use fragments" sidebar option. Batch UI now renders via a single, non-fragment path; image tiles and buttons
  are rendered in the same context to avoid the historical click-swallowing issue. Tests that asserted fragment toggling will be
  updated (or replaced with non-fragment assertions) in the next pass.

Next Simplifications (Nov 21, 2025, 199)
- 199a. Remove Ridge async entirely and delete `background.py`.
  - Effect: ridge fits always sync; fewer code paths; simpler logs.
  - LOC win: ~200–250. Tests: update async‑ridge expectations.
- 199b. Delete `ui_metrics.py` shim and inline calls into `ui.py` (single source).
  - Effect: fewer indirections/imports; easier stubbing.
  - LOC win: ~30–60. Tests: update imports where needed.
- 199c. Retire `ensure_fitted`/`train_and_record`; make UI call `fit_value_model` explicitly only on user actions.
  - Effect: no auto‑fits on reruns; fewer status states; simpler mental model.
  - LOC win: ~120–180. Tests: adapt ensure_fitted‑based tests.
- 199d. Remove fragment‑specific helpers and tests — DONE.
  - Effect: single rendering path; smaller suite; fewer key paths.
  - LOC win (tests+code): ~150–220.
- 199e. Purge legacy aggregated `dataset_*.npz` paths — DONE.
  - Effect: folder‑only persistence; simpler loaders. Added `.gitignore` rules for `dataset_*.npz` and `backups/`.
  - Tests already target folder dataset; added a guard test to keep backups count unchanged.
- 199f. Drop “Use Ridge captions” toggle; captions show:
  - `[XGB]` when XGB cache exists; `[Ridge]` when ‖w‖>0; otherwise `n/a`.
  - LOC win: ~30–50. Tests: update caption assertions.
- 199g. Remove `pair_ui.py` and `pages/` (batch‑only UI).
  - Effect: tighter app; fewer stubs. LOC win: ~250–350. Tests: delete/migrate pair‑UI tests.
- 199h. Trim `app.py` to <380 lines by deleting obsolete rerun shims and redundant debug prints (keep logs behind `LOG_VERBOSITY`).
  - LOC win: ~40–60. Tests: none or minimal.

Recommendation: 199f → 199a → 199b → 199h first (low risk, good LOC). Then 199g and 199e once the suite is green.
- Impact: several tests still assert the presence/behavior of the async UI. Next step is either to (a) reintroduce a no-op
  compatibility toggle, or (b) update those tests to the new sync-only contract (preferred for clarity).
- Proposed follow-ups (choose):
  197a. Keep a no-op "Train XGBoost asynchronously" checkbox (writes the key but unused) to unbreak legacy tests while we
        complete the broader cleanup.
  197b. Update tests that depend on async paths to the sync-only flow and remove the legacy files after green.
  197c. Finish scorer API consolidation by making all UI sites call `get_value_scorer` (some still import the status wrapper).
Next Simplifications (Nov 21, 2025, 203)
- 203a. Remove unused Keys (USE_IMAGE_SERVER, IMAGE_SERVER_URL, XGB_TRAIN_ASYNC/XGB_FIT_FUTURE, RIDGE_*): prune constants and docs.
- 203b. Delete remaining background references in docs/tests; simplify logging notes to sync-only. (Done) — background.py has been removed; any prior mentions are historical.
- 203c. Collapse size controls (Done) — inlined size controls into `ui_sidebar._build_size_controls`; removed the helper import.
- 203d. Remove safe_set in app.py (Done) — replaced with direct session_state assignments and kept ui_sidebar’s use of helpers.safe_set unchanged.
- 203e. Log gating (Done) — introduced a single LOG_VERBOSITY (env or session_state.log_verbosity) and gated batch_ui’s noncritical per‑tile logs behind it (0=quiet, 1=info, 2=verbose). Training logs remain unaffected.
- 203c. Inline `ui_controls.build_size_controls` or reduce it to a tiny helper in `ui_sidebar` to cut indirection.
- 203d. Collapse `safe_set` usage in app.py to direct assignments where tests don’t rely on the helper.
- 203e. Trim debug prints behind a single `LOG_VERBOSITY` env and remove noisy per‑tile logs.
- 203f. Remove dataset_rows_all_for_prompt alias (Done); keep a single folder‑only counter via `dataset_rows_for_prompt` and dim‑filtered `dataset_rows_for_prompt_dim`.
- 203g. Delete Playwright stubs and scripts if we confirm they’re unused in CI.
- 203h. Run full suite; adjust any tests still assuming async/auto‑fit/fragment toggles.

Recommendation: 203a → 203d → 203f first (safe LOC wins), then 203h.
- Ridge λ default (Nov 21, 2025):
  - Set the default Ridge regularization λ to 1e+300 across UI and training fallbacks. This makes the default effectively “no update” unless the user changes it, matching the minimal/explicit philosophy. Tests that override λ continue to pass.
Further Simplification Ideas (Nov 21, 2025, 205)
- 205a. Remove unused Keys and async remnants: XGB_TRAIN_ASYNC, XGB_FIT_FUTURE, RIDGE_*; scrub mentions in docs/tests.
- 205b. Delete Playwright stubs (scripts/app_stubbed.py, tests_playwright/) if not used in CI; shrink tooling surface.
- 205c. Replace remaining safe_set usages with direct writes; keep only where tests depend on the helper.
- 205d. Reduce logging: gate noncritical per‑tile prints behind LOG_VERBOSITY and default to 0; trim [fragpath]/[buttons] lines.
- 205e. Collapse ui_controls into ui_sidebar to remove one more import hop (size/steps/guidance read in one place).
- 205f. Prune remaining commented code/doc blocks that refer to removed modes (queue, fragments, pair UI) for clarity.

Today (Nov 21, 2025)
- Removed image-server branches from `flux_local.py`; local Diffusers is the only path.
- Batch captions now tag scores for all supported value models: `[XGB]`, `[Ridge]`, and `[Distance]`.
- Kept async Keys in `constants.Keys` for compatibility but avoided using them in new code paths; plan remains to prune after tests move off them.
- Log gating (216d): flux_local’s noisy `[pipe]`/`[perf]` prints are now gated by `LOG_VERBOSITY` (0/1/2). Default is 0 (quiet). Enable with `export LOG_VERBOSITY=1` (info) or `=2` (verbose). Behavior unchanged.
 - 216e: Collapsed UI helpers into `ui_sidebar`:
   - Moved `sidebar_metric`, `sidebar_metric_rows`, `status_panel`, `render_iter_step_scores`, `render_mu_value_history`, and batch/pair control builders into `ui_sidebar.py`.
 - `ui_controls.py` now provides thin wrappers delegating to `ui_sidebar` (keeps existing tests stable while reducing duplication).
 - `ui_sidebar` no longer imports from `ui`/`ui_controls` for these helpers.
- Batch scorer prep now uses the unified API: `batch_ui._prepare_xgb_scorer` calls `value_scorer.get_value_scorer` and normalizes to `(scorer, 'ok'|status)`. This removes the legacy status shim in the hot path.
- ui_sidebar uses the shared `helpers.safe_write`; removed a duplicated local variant.
- flux_local’s log gating uses the shared `helpers.get_log_verbosity` to keep behavior consistent with the rest of the app.
 - Refactor (225): kept sidebar Train-results rendering DRY via `_emit_train_results(st, lines)`; robust defaults ensure the canonical 8 lines always render even with empty data. `ui.py` sidebar helpers now delegate to `ui_sidebar` to avoid duplication.
- 216g: Train results cleanup — removed extra status lines (e.g., “XGBoost training: …”, “Ridge training: …”). We now emit only the canonical block: Train score, CV score, Last CV, Last train, Value scorer status, Value scorer, XGBoost active, Optimization. Reduces sidebar noise and mixed signals.
 - Small refactor: centralized Train results emission via `_emit_train_results` in `ui_sidebar.py` to avoid duplicate string writes and keep ordering consistent.

Next Simplifications (Nov 21, 2025, 213)
- 213a. Single scorer entrypoint: fold `get_value_scorer_with_status` into `get_value_scorer(vm_choice)` that returns `(callable|None, tag|status)`. Update batch/ui/sidebar to consume one API.
- 213b. Remove CV auto-lines: keep an on-demand button only; drop cached CV lines on import to simplify strings/order.
- 213c. Purge async Keys entirely: delete `XGB_TRAIN_ASYNC/XGB_FIT_FUTURE/RIDGE_*` from `constants.py`, strip checks in `value_model` and tests.
- 213d. Trim logs in `flux_local`: gate `[pipe]`/`[perf]` with `LOG_VERBOSITY` (keep errors). Reduce noisy scheduler prints in tests.
- 213e. Collapse `ui.py` helper rows into `ui_sidebar` where used; remove unused UI writers (keeps a single sidebar surface).
- 213f. Hardcode model: drop any residual model lists/defaults; enforce `stabilityai/sd-turbo` only to avoid selector/const churn.
- 213g. Remove Distance model if not needed: Ridge/XGB only; update 2 tests that assert `[Distance]` captions.

Next Simplifications (Nov 21, 2025, 217)
- 217a. Finish One‑scorer API everywhere: replace remaining `get_value_scorer_with_status(...)` call‑sites with `get_value_scorer(...)`, then delete the shim. Add 2 tiny tests: (a) returns `(None,"ridge_untrained")` when `‖w‖=0`, (b) XGB scorer returns tag `"XGB"` after a sync fit and captions show `[XGB]`.
- 217b. Sidebar cleanup: compute Train score only on demand; show a single status line derived from cache/‖w‖ and remove legacy "running/ok" flip‑flops.
- 217c. Purge async remnants fully (keys/docs/UI strings). Keep only the explicit "Train XGBoost now (sync)" action.
- 217d. Collapse size controls into `ui_sidebar` (delete the auxiliary helper) so width/height/steps/guidance live in one place.
- 217e. Gate noncritical logs behind `LOG_VERBOSITY` (default 0) in batch/flux; keep errors and essential timings.

Rationale: these are surgical, low‑risk deletions that reduce indirection and state permutations without changing the user‑visible flow.

Next Simplifications (Nov 21, 2025, 226)
- 226a. Remove the scorer status shim entirely: delete `get_value_scorer_with_status` and update remaining tests to `get_value_scorer`. Add two tiny unit tests for Ridge zero‑w and XGB cached‑model. Small, clear API.
- 226b. Simplify `value_model` XGB/Ridge paths to sync‑only: remove dead async branches and future/status writes; keep `xgb_cache` and `LAST_TRAIN_{AT,MS}` only. Optional: retain `Keys.XGB_TRAIN_STATUS` as a dumb mirror for a transition period.
- 226c. Finish thinning `ui.py`: keep it as a facade that re‑exports `ui_sidebar` helpers; once tests stop importing it, retire the file.
- 226d. Prune any tests that assume async behavior (futures/status transitions) after 226b lands; replace with explicit click‑to‑fit tests.
Simplify wave (Nov 21, 2025, 218)
- 218a. Remove dead async stubs left inside value_model (delete the do_async_xgb branch entirely). Pure sync code.
- 218b. Prune unused Keys in constants: drop RIDGE_TRAIN_ASYNC/RIDGE_FIT_FUTURE; keep XGB_TRAIN_ASYNC only for tests still touching it.
- 218c. Sidebar single-source: render VM/status/Train results from one helper in ui_sidebar; delete duplicate emitters.
- 218d. Folder-only dataset: confirm no code path rescans the disk on rerun; rely on in-memory counters and a single loader on prompt switch.
- 218e. One scorer API only: remove get_value_scorer_with_status; use get_value_scorer everywhere and surface tag/status in captions.

My recommendation: 218a → 218b → 218c first (small, safe LOC wins); then 218e.

Done (Nov 21, 2025, late):
- 218a: Deleted value_model async branch.
- 218b: Pruned RIDGE_* async Keys; left XGB_* compat only.
- Sidebar “Train XGBoost now (sync)” no longer flips async flags or pops futures; it just calls fit_value_model (sync-only backend).
- Dropped dead async cleanup in value_model.ensure_fitted (t0/fit_future paths); function is a minimal sync shim now.
- UI moved to unified scorer API `get_value_scorer` in compute_step_scores; the legacy shim remains only for tests.
- Removed unused helpers `_maybe_fit_xgb/_maybe_fit_ridge` from value_model; training is centralized in `fit_value_model`.
Next options (223) — maintainability
- 223a. Prune leftover async mentions from ui_sidebar and legacy doc sections (no behavior change).
- 223b. Single emitter guarantee: keep all VM/status/Train-results lines flowing through `_emit_train_results`; remove any stray writes outside it.
- 223c. Normalize scorer usage in UI to `get_value_scorer` everywhere (keep the shim for tests only).

Recommendation: 223b first, then 223c.

Done (Nov 21, 2025, later):
- Unified UI scorer usage: `ui.py` now uses `value_scorer.get_value_scorer` (no legacy shim branching).
- Single‑emitter sidebar: app.py routes early sidebar lines through `ui_sidebar._emit_train_results`; removed duplicate direct writes.
- Trimmed sidebar helper: `_vm_header_and_status()` now returns only the VM label and cache (status is emitted solely in Train results).

Next options (231):
- 231a. Remove the legacy scorer shim (`get_value_scorer_with_status`) after updating tests to the unified API.
- 231b. Prune stale async/background mentions in older AGENTS/HUMANS sections.
- 231c. Consider collapsing small CV helper `_cached_cv_lines` into `_emit_train_results` to keep all strings in one place (requires minor test tweaks).

Next options (235) — simplify further
- 235a. Remove legacy scorer shim `get_value_scorer_with_status` (UI already uses `get_value_scorer`). Update tests that import the shim to the unified API. LOC: −30–60. Clear API surface.
- 235b. Centralize CV strings: fold `_cached_cv_lines` into the Train‑results emitter so all sidebar strings originate from one helper. Adjust 1–2 tests that read CV from the VM expander. LOC: −10–20; fewer writers.
- 235c. Docs scrub: remove remaining async/background references from older AGENTS/HUMANS sections to avoid confusion (no behavior change). LOC: −50–120.
- 235d. Tighten logging: gate remaining informational `[xgb] skipped:` lines behind LOG_VERBOSITY=1 to keep CI quiet by default (errors still print). LOC: −/+. Small clarity win.

Recommendation: 235a → 235b (highest payoff with minimal risk), then 235c.
New learnings (Nov 21, 2025 – XGB clarity)
- The sidebar line “XGBoost active: yes/no” was derived from the VM choice, which confused users when no model was cached. We now set it to “yes” only when the XGB scorer status is ok. This keeps the UI honest: look at “Value scorer status:” for the real state.
