# Image Preference Optimization

Streamlit app for iterative image generation using human preference feedback.

## Docker

```bash
docker compose up app -d    # Start on port 8577
docker compose logs app     # Check logs
docker compose down         # Stop
```

Uses NVIDIA CUDA 13.0.1 base image for GB10 GPU compatibility.

## Stack

- Streamlit UI on port 8577
- PyTorch + Diffusers (sd-turbo default)
- File-based storage in `data/`
