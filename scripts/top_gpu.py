#!/usr/bin/env python3
"""
top_gpu.py — list processes by GPU memory (MiB) using nvidia-smi.

Usage:
  ./scripts/top_gpu.py [--limit N]

Minimal, no deps. Exits non‑zero if nvidia-smi is unavailable.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess


def _cmd(args: list[str]) -> list[str]:
    out = subprocess.check_output(args, text=True).strip().splitlines()
    return [ln.strip() for ln in out if ln.strip()]


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().split(b"\x00")
            return " ".join(x.decode("utf-8", "ignore") for x in raw if x)
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Top processes by GPU memory (MiB)")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if not shutil.which("nvidia-smi"):
        print("error: nvidia-smi not found")
        return 1

    # Per-process usage across all GPUs
    # fields: pid, process_name, used_memory [MiB], gpu_uuid
    lines = _cmd([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader",
    ])
    by_pid: dict[int, dict] = {}
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            mem = int(parts[2].split()[0])  # "123 MiB" -> 123
        except Exception:
            continue
        rec = by_pid.setdefault(pid, {"mem": 0, "name": parts[1]})
        rec["mem"] += mem
        # prefer /proc cmdline for more detail when available
        cmd = _cmdline(pid)
        if cmd:
            rec["name"] = cmd

    rows = sorted(((pid, rec["mem"], rec["name"]) for pid, rec in by_pid.items()), key=lambda r: r[1], reverse=True)

    # Print per-GPU summary first
    try:
        gsum = _cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"])
        print("Per-GPU memory:")
        for g in gsum:
            idx, name, tot, used = [x.strip() for x in g.split(",")]
            print(f"GPU{idx} {name}  used {used.strip()} / total {tot.strip()}")
        print()
    except Exception:
        pass

    print(f"Top GPU processes (limit {args.limit})")
    print(f"{'PID':>6} {'GPU_MB':>8} CMD/NAME")
    for pid, mem, name in rows[: args.limit]:
        print(f"{pid:6d} {mem:8d} {name[:100]}")
    if not rows:
        print("<no active compute processes>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
