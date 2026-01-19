# Image Preference Optimization

Streamlit app for iterative image generation using human preference feedback.

## Docker

```bash
docker compose up app -d    # Start on port 8577
docker compose logs app     # Check logs
docker compose down         # Stop
docker compose --profile server up  # With generation server
```

Uses NVIDIA CUDA 13.0.1 base image for GB10 GPU compatibility.

## Stack

- Streamlit UI on port 8577
- Generation server on port 8580 (optional)
- PyTorch + Diffusers
- Models: sd-turbo (default), flux-schnell, flux-dev (NF4 quantized)
- File-based storage in `data/`

## Architecture

- `ipo/core/` - Core algorithms (optimizer, latent_state, value_model)
- `ipo/core/model_manager.py` - Unified model management (local/server)
- `ipo/ui/` - Streamlit UI (sampling.py, curation.py, batch_ui.py)
- `ipo/infra/` - Infrastructure (pipeline, constants)
- `ipo/server/` - FastAPI generation server + client

## Inference Batching

Sidebar control "Inference Batch" sets images generated per forward pass.

| Batch | Time (s) | img/s |
|-------|----------|-------|
| 1     | 3.24     | 0.31  |
| 2     | 4.32     | 0.46  |
| 4     | 7.79     | 0.51  |
| 8     | 15.76    | 0.51  |

batch=4 gives ~65% speedup; larger batches don't help.

## Supported Models

| Model | HF ID | Quantization | Guidance |
|-------|-------|--------------|----------|
| sd-turbo | stabilityai/sd-turbo | FP16 | 0.0 |
| flux-schnell | black-forest-labs/FLUX.1-schnell | NF4 | 0.0 |
| flux-dev | black-forest-labs/FLUX.1-dev | NF4 | 3.5 |

Select via sidebar "Model" dropdown. Flux models use bitsandbytes NF4 quantization.

## Model Management

`ModelManager.ensure_ready()` routes to local or server backend based on mode:
- Local: calls `pipeline_local.set_model()`
- Server: calls `gen_client.set_model()` → server `/model` endpoint

Both backends cache loaded models to avoid redundant reloads.

## Development

```bash
pytest tests/ -v              # Run tests (19 tests)
python scripts/benchmark_batch.py   # Benchmark inference batching
python scripts/benchmark_models.py  # Benchmark all models
```
