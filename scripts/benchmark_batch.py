#!/usr/bin/env python3
"""Benchmark inference batch sizes."""
import time, numpy as np

def main():
    from ipo.infra.pipeline_local import gen_batch_pooled, set_model, get_pooled_embed_dim
    set_model(None)
    dim = get_pooled_embed_dim()
    print(f"embed_dim={dim}")
    for bs in [1, 2, 4, 8]:
        embs = [np.random.randn(dim)*0.1 for _ in range(bs)]
        t0 = time.time()
        gen_batch_pooled(embs, 512, 512, 6, 0.0, 42, "photo", 0.1)
        dt = time.time() - t0
        print(f"batch={bs}: {dt:.2f}s, {bs/dt:.2f} img/s")

if __name__ == "__main__":
    main()
