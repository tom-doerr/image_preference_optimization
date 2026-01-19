#!/usr/bin/env python3
"""Benchmark all supported models for throughput."""
import time
import torch

MODELS = ["sd-turbo", "flux-schnell", "flux-dev"]
RESOLUTIONS = [(512, 512)]
STEPS = {"sd-turbo": 4, "flux-schnell": 4, "flux-dev": 20}


def benchmark_model(model_key, w, h, steps, n_warmup=2, n_runs=5):
    from ipo.infra.flux_pipeline import load_pipeline, generate_image
    print(f"Loading {model_key}...")
    pipe = load_pipeline(model_key)
    for _ in range(n_warmup):
        generate_image(pipe, "warmup", None, "pooled", w, h, steps, 0.0, 42, 0.1)
    times = []
    for i in range(n_runs):
        t0 = time.time()
        generate_image(pipe, "a cat", None, "pooled", w, h, steps, 0.0, 42 + i, 0.1)
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    vram = torch.cuda.max_memory_allocated() / 1e9
    return {"model": model_key, "res": f"{w}x{h}", "steps": steps, "time": avg, "img_s": 1/avg, "vram_gb": vram}


def main():
    results = []
    for model in MODELS:
        for w, h in RESOLUTIONS:
            steps = STEPS.get(model, 4)
            try:
                r = benchmark_model(model, w, h, steps)
                results.append(r)
                print(f"{r['model']}: {r['img_s']:.2f} img/s")
            except Exception as e:
                print(f"{model}: FAILED - {e}")
            torch.cuda.empty_cache()
    print("\n| Model | Steps | img/s | VRAM |")
    print("|-------|-------|-------|------|")
    for r in results:
        print(f"| {r['model']} | {r['steps']} | {r['img_s']:.2f} | {r['vram_gb']:.1f}GB |")


if __name__ == "__main__":
    main()
