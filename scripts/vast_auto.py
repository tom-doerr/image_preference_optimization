#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request


API = os.getenv("VAST_API", "https://vast.ai/api/v0")
KEY = os.getenv("VAST_API_KEY", "")


def _hdrs() -> dict:
    if not KEY:
        raise SystemExit("VAST_API_KEY env var is required")
    return {"Accept": "application/json", "Authorization": f"Bearer {KEY}"}


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_hdrs())
    with urllib.request.urlopen(req, timeout=60) as r:  # nosec - trusted API base
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, payload: dict) -> dict:
    url = f"{API}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={**_hdrs(), "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:  # nosec
        return json.loads(r.read().decode("utf-8"))


def _put(path: str, payload: dict) -> dict:
    url = f"{API}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={**_hdrs(), "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:  # nosec
        return json.loads(r.read().decode("utf-8"))


def find_offer(min_vram_gb: int = 10, max_dph: float | None = None) -> dict | None:
    # Minimal query: verified, rentable, 1+ direct port; sort by total $/h
    q = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "direct_port_count": {"gte": 1},
        "num_gpus": {"gte": 1},
        "gpu_ram": {"gte": int(min_vram_gb)},
    }
    out = _post("/bundles/", {"query": q, "limit": 20, "order": "dph_total"})
    results = out.get("offers", []) or out.get("results", [])
    if not results:
        return None
    if max_dph is not None:
        results = [
            o
            for o in results
            if float(o.get("dph_total", o.get("dph", 0.0))) <= float(max_dph)
        ]
    return results[0] if results else None


def onstart_cmd(
    repo_url: str, model: str, server_port: int = 8000, app_port: int = 8501
) -> str:
    # Keep it short; install deps, start image server in background, then Streamlit app
    return (
        "bash -lc '"
        "apt-get update && apt-get install -y git && "
        f"git clone {repo_url} /app || true && cd /app && "
        "pip install --upgrade pip && "
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && "
        "pip install -r requirements.txt && pip install transformers accelerate pillow && "
        f"export FLUX_LOCAL_MODEL={model} && "
        f"nohup python scripts/image_server_app.py --port {server_port} > /server.log 2>&1 & "
        f"streamlit run app.py --server.headless true --server.port {app_port} "
        "'"
    )


def rent(
    ask_id: int,
    model: str,
    repo_url: str,
    disk_gb: int = 40,
    server_port: int = 8000,
    app_port: int = 8501,
) -> dict:
    payload = {
        "image": "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime",
        "env": {
            "FLUX_LOCAL_MODEL": model,
            "IMAGE_SERVER_URL": f"http://127.0.0.1:{server_port}",
            # Pass through user token if present
            "HUGGINGFACE_HUB_TOKEN": os.getenv("HUGGINGFACE_HUB_TOKEN", ""),
        },
        "disk": int(disk_gb),
        "args": [],
        "onstart": onstart_cmd(repo_url, model, server_port, app_port),
    }
    return _put(f"/asks/{int(ask_id)}/", payload)


def instances() -> dict:
    return _get("/instances")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  vast_auto.py find [--min_vram 10] [--max_dph 0.8]\n"
            "  vast_auto.py rent <ask_id> [--repo <url>] [--model <hf_id>] [--disk 40] [--srv_port 8000] [--app_port 8501]\n"
            "  vast_auto.py status\n"
            "Env: VAST_API_KEY required; optional HUGGINGFACE_HUB_TOKEN."
        )
        return 0
    cmd = argv[1]
    if cmd == "find":
        mv = int(argv[argv.index("--min_vram") + 1]) if "--min_vram" in argv else 10
        md = float(argv[argv.index("--max_dph") + 1]) if "--max_dph" in argv else None  # type: ignore[assignment]
        off = find_offer(mv, md)
        print(json.dumps(off or {}, indent=2))
        if off:
            print(f"ask_id={off.get('id')} dph_total={off.get('dph_total')}")
        return 0
    if cmd == "rent":
        if len(argv) < 3:
            print("rent requires <ask_id>")
            return 2
        ask_id = int(argv[2])
        repo = (
            argv[argv.index("--repo") + 1]
            if "--repo" in argv
            else os.getenv("IPO_REPO_URL", "https://github.com/")
        )
        model = (
            argv[argv.index("--model") + 1]
            if "--model" in argv
            else os.getenv("FLUX_LOCAL_MODEL", "stabilityai/sd-turbo")
        )
        disk = int(argv[argv.index("--disk") + 1]) if "--disk" in argv else 40
        sp = int(argv[argv.index("--srv_port") + 1]) if "--srv_port" in argv else 8000
        ap = int(argv[argv.index("--app_port") + 1]) if "--app_port" in argv else 8501
        out = rent(ask_id, model, repo, disk, sp, ap)
        print(json.dumps(out, indent=2))
        print(
            "Instance requested. Use 'status' to list instances; expose ports via Vast panel."
        )
        return 0
    if cmd == "status":
        out = instances()
        print(json.dumps(out, indent=2))
        # Tiny hint
        print(
            "Tip: map public ports to 8501 (app) and 8000 (image server) in Vast's ports UI."
        )
        return 0
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":  # pragma: no cover - tiny CLI
    raise SystemExit(main(sys.argv))
