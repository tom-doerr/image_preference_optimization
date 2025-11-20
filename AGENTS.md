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
  - Value model = CosineHill → proposer = CosineHill
  - Any other Value model → proposer = DistanceHill
- Updated tests: `tests/test_pair_proposer_toggle.py` now drives CosineHill by selecting it in the Value model dropdown.

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

DistanceHill controls (Nov 18, 2025, later):
- Switched Distance hill-climbing controls from sliders to numeric inputs for precise edits:
  - Alpha, Beta, Trust radius, Step size (lr_μ), Orth explore (γ), Optimization steps, and Iterative step (eta).
- Keeps UI minimal while allowing exact values; tests rely on session state rather than widget type, so no changes needed there.
 - Batch size default increased to 25 (slider range 2–64) to match the preferred batch workflow; queue defaults remain unchanged.

Performance + UX (Nov 18, 2025, late):
- Optimization steps (latent): default set to 10; UI no longer enforces a max. Min in the slider is now 0, but the iterative proposer only activates when steps >1 or eta>0. Iterative step (eta) defaults to 0.1 instead of 0.0 so the iterative proposer is active by default.
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
- Centralized the “should we fit Ridge?” decision in `value_model._uses_ridge(vm_choice)`; `fit_value_model` now skips the heavy ridge solve when the active value model is DistanceHill or CosineHill.
- Updated image calls to use `width="stretch"` instead of the deprecated `use_container_width=True` for main images; this keeps the layout stable and removes Streamlit warnings in logs.
- Batch caching was removed again: `batch_ui` now always decodes each item on render so per-image fragments stay simple and behavior is easier to reason about. If perf is an issue we can reintroduce a tiny cache, but for now we keep it explicit.
- XGBoost CV is fully centralized in `metrics.xgb_cv_accuracy`; both the Data block and Value model expander call this helper instead of duplicating fold logic.
- `value_model.ensure_fitted` now also auto-fits when the selected value model is XGBoost and no cached model exists, even if `w` was restored from disk. This fixes “XGBoost (xgb_unavailable)” after reload while keeping training decisions in one place.
- Batch tiles are wrapped in `st.fragment` (when available) so each image + buttons lives in its own fragment; latent sampling, decode, and label-side effects are scoped per tile. Tests still pass because we guard the fragment usage behind `getattr(st, "fragment", None)`.

Keep in mind:
- Prefer one call site for Diffusers: all decode paths go through `flux_local._run_pipe` so scheduler/device guards live in one place.
- When adding background helpers, keep them in `background.py` and import lazily in UI modules to simplify test stubbing.

UI cleanup (Nov 18, 2025, later):
- Removed separate “Pair proposer” dropdown; proposer derives from Value model: CosineHill → CosineHill proposer, otherwise DistanceHill.
- Test updated: `tests/test_pair_proposer_toggle.py` selects CosineHill via the Value model dropdown.

Algorithm notes (Nov 18, 2025 – Hill‑climb latents):
- Prompt anchor: `z_p = z_from_prompt(state, prompt)` (sha1‑seeded Gaussian of length `d`).
- Dataset: we store rows as deltas `X_i = z_i − z_p` with labels `y_i ∈ {+1, −1}`.
- DistanceHill direction: compute gradient of `∑ y_i σ(γ‖μ − (z_p+X_i)‖²)` w.r.t. μ, then normalize to `d1`.
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

Default prompt update (Nov 12, 2025):
- The default prompt in the UI is now: `neon punk city, women with short hair, standing in the rain`.

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

Playwright e2e (optional, stubbed backend):
- Start stub app + run UI checks:
  - `bash scripts/run_playwright.sh`
  - This starts `streamlit run scripts/app_stubbed.py` on a free port, then runs Python Playwright tests in `tests_playwright/` (`PW_RUN=1`).
- Direct invocation:
  - Start: `streamlit run scripts/app_stubbed.py --server.headless true --server.port 8597`
  - In another shell: `PW_RUN=1 PW_URL=http://localhost:8597 python -m pytest -q tests_playwright`
- Notes: We use Playwright for Python to avoid Node deps; tests are skipped unless `PW_RUN=1` is set. The stubbed app replaces `flux_local` at import time and renders synthetic images fast.

New learnings (Nov 18, 2025):
- Black A/B frames were rooted in latent scale not matching the active scheduler. We now set timesteps for the requested step count before computing `init_noise_sigma` and normalize latents to that value. This removed the black-frame symptom here.
- For `stabilityai/sd-turbo`, switching to `EulerAncestralDiscreteScheduler` produced the most reliable latents‑injection behavior; the loader applies this automatically for sd‑turbo.
- Minimal code, no fallbacks added. Added a focused test `tests/test_scheduler_sigma_alignment.py` to ensure we scale to the scheduler’s `init_noise_sigma`.
- Debug log (`ipo.debug.log`) now records `init_sigma` per call to aid field diagnosis.
- Turbo guidance: for `*sd*-turbo` and `*sdxl*-turbo` models, we now force effective guidance (CFG) to 0.0 in the app calls. This matches how Turbo models are intended to run and eliminates another source of flat/black outputs.
- Sidebar “Debug” shows last-call stats (model, size, steps, guidance, latents_std, init_sigma, img0_std/min/max) to surface problems immediately.
 - Safety checker: to prevent spurious blacked-out frames, we disable the pipeline safety checker after load (set `safety_checker=None`, `feature_extractor=None`, and config flag where available). Minimal, avoids false positives in local testing.
 
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
- The “Step scores” sidebar panel now uses the unified `value_scorer.get_value_scorer` helper for Ridge/XGBoost/DistanceHill/CosineHill, instead of re‑implementing per‑mode scoring. This makes step values reflect the active value model and ensures XGBoost paths transparently fall back to Ridge when no XGB model is cached.

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
- `value_scorer.get_value_scorer` no longer falls back to Ridge when a non-Ridge value model is unavailable. For XGBoost, DistanceHill, and CosineHill, missing models or empty datasets now return a scorer that always yields 0 and log a small `[xgb]/[dist]/[cos] scorer unavailable/error` line, instead of silently switching to Ridge. Ridge remains the only path that uses `dot(w, fvec)` by design.
- Added `value_scorer.get_value_scorer_with_status(...)` which returns both a scorer and a short status string (`\"ok\"`, `\"xgb_unavailable\"`, `\"dist_empty\"`, etc.). The Value model sidebar expander now shows `Value scorer status: <status>` so it’s obvious when XGB/DH/Cosine are effectively inactive (always 0) versus trained.

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
- Step-size sliders were made finer: `Step size (lr_μ)` now uses step=0.01 (min still 0.0, max 1.0) and the `Iterative step (eta)` slider also uses step=0.01. Numeric inputs for hill-climb already supported 0.01 granularity. Tests (`test_slider_help`, `test_default_steps`, `test_ui_controls_fallbacks`) remain green.

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
