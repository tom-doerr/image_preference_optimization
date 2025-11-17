#!/usr/bin/env python3
"""
top_swap.py — show top processes by swap and RSS usage (Linux /proc only).

Usage:
  ./scripts/top_swap.py [--limit N] [--sort swap|rss]

No dependencies; reads /proc/*/status and /proc/*/cmdline.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass
class Proc:
    pid: int
    name: str
    rss_kb: int
    swap_kb: int
    cmd: str


def _read_status(pid: str) -> tuple[str, int, int]:
    name = "?"
    rss = swap = 0
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1])
    except Exception:
        pass
    return name, rss, swap


def _read_cmd(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().split(b"\x00")
            return " ".join(x.decode("utf-8", "ignore") for x in raw if x)
    except Exception:
        return ""


def collect() -> list[Proc]:
    procs: list[Proc] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        name, rss, swap = _read_status(pid)
        if rss == 0 and swap == 0:
            continue
        cmd = _read_cmd(pid)
        procs.append(Proc(pid=int(pid), name=name, rss_kb=rss, swap_kb=swap, cmd=cmd))
    return procs


def main() -> int:
    ap = argparse.ArgumentParser(description="Top processes by swap/RSS usage")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sort", choices=["swap", "rss"], default="swap")
    args = ap.parse_args()

    rows = collect()
    key = (lambda p: p.swap_kb) if args.sort == "swap" else (lambda p: p.rss_kb)
    rows.sort(key=key, reverse=True)

    print(f"Top by {args.sort.upper()} (limit {args.limit})")
    print(f"{'PID':>6} {'SWAP(MB)':>9} {'RSS(MB)':>8} NAME CMD")
    for p in rows[: args.limit]:
        print(f"{p.pid:6d} {p.swap_kb/1024:9.1f} {p.rss_kb/1024:8.1f} {p.name} {p.cmd[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

