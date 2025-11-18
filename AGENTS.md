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

Maintainability (Nov 13, 2025):
- Consolidated pipeline loading into `flux_local._ensure_pipe()` and `_free_pipe()` to remove duplicated code across generate/set_model.
- Loader invariants centralized: sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, uses `low_cpu_mem_usage=False` + `.to("cuda")` for broad compatibility, and frees VRAM on model switch.
- Added tests: allocator env set on load (`tests/test_allocator_env.py`). Total tests now 41 (1 GPU skipped).

Simplification & stability (Nov 18, 2025):
- Black A/B frames under sd‑turbo were traced to scheduler/latents interaction. We simplified by switching sd‑turbo to `LCMScheduler` in `flux_local.set_model()`; kept code minimal and added a small test `tests/test_scheduler_turbo_lcm.py` that stubs Diffusers and asserts the switch.
- Keep guidance at the UI level; no hidden fallbacks. Text‑only path already produced non‑black images; the latents path should now be stable with LCM.
- If black frames persist on a specific box, the most opinionated next step is to restrict pair decoding to SD‑1.5 and keep Turbo for prompt‑only preview. We’ll only do this if explicitly requested since it changes behavior.

Simplify pass (Nov 18, 2025, later):
- Prompt-only generation always uses the text path; pair images always use latents. Removed internal fallbacks/aliases to make control flow obvious.
- Kept “7 GB VRAM mode” and default model selection for test coverage; further UI trimming is pending user confirmation.
- Added/updated tiny tests to reflect the simplified contracts for prompt/latents paths.

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
- For low-VRAM runs, use “7 GB VRAM mode” or set 512–640px; otherwise one of the generation calls may OOM and the page will error.

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
- `tests/e2e/test_e2e_predicted_values_and_iterations.py`: imports the app with stubs and asserts that predicted values (V(left)/V(right)) and the iterations line render on import.
- `tests/e2e/test_e2e_prefer_left_increments.py`: emulates a single preference by calling `app.update_latent_ridge(...)` on the current pair and checks that `lstate.step` increments. Uses a unique prompt and clears `mu_hist` to avoid NPZ collisions.
- `tests/e2e/test_e2e_pair_content_gpu.py` (opt‑in via `E2E_GPU=1` or `SMOKE_GPU=1`): decodes a real A/B pair on GPU and asserts both images have variance and are not identical.
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

Prompt encode caching (Nov 18, 2025):
- For sd‑turbo we cache prompt embeddings per (model, prompt, CFG>0) and pass `prompt_embeds`/`negative_prompt_embeds` to Diffusers. Cuts CPU by avoiding re-tokenization each rerun. Test: `tests/test_prompt_encode_cache.py`.

UI tweak (Nov 18, 2025):
- Sidebar Environment panel is wrapped in a collapsed expander by default (click to open). This keeps the sidebar compact.
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
