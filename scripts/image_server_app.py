#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from rich_cli import enable_color_print as _enable_color
    _enable_color()
except Exception:
    pass


def _b64_png(img) -> str:
    try:
        from PIL import Image  # type: ignore
        buf = io.BytesIO()
        (img if hasattr(img, "save") else Image.fromarray(img)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:  # pragma: no cover - best-effort encode
        raise RuntimeError(f"encode error: {e}")


def _ensure_model():
    try:
        from flux_local import set_model
        mid = os.getenv("FLUX_LOCAL_MODEL", "stabilityai/sd-turbo")
        set_model(mid)
    except Exception as e:
        raise RuntimeError(f"model load failed: {e}")


class App(BaseHTTPRequestHandler):
    def _json(self):
        try:
            ln = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(ln)
            return json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            return {}

    def _reply(self, code: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802 - stdlib API
        try:
            _ensure_model()
            req = self._json()
            width = int(req.get("width", 512))
            height = int(req.get("height", 512))
            steps = int(req.get("steps", 6))
            guidance = float(req.get("guidance", 0.0))
            prompt = str(req.get("prompt", ""))
            if self.path == "/generate":
                from flux_local import generate_flux_image
                img = generate_flux_image(prompt, width=width, height=height, steps=steps, guidance=guidance)
                return self._reply(200, {"image": _b64_png(img)})
            if self.path == "/generate_latents":
                from flux_local import generate_flux_image_latents
                latents = req.get("latents")
                shape = req.get("latents_shape")
                if latents is None or shape is None:
                    return self._reply(400, {"error": "missing latents/latents_shape"})
                # Minimal: trust shape and reshape
                import numpy as np
                arr = np.array(latents, dtype=float).reshape(tuple(int(x) for x in shape))
                img = generate_flux_image_latents(prompt, arr, width=width, height=height, steps=steps, guidance=guidance)
                return self._reply(200, {"image": _b64_png(img)})
            return self._reply(404, {"error": "unknown path"})
        except Exception as e:  # keep minimal; surface errors plainly
            return self._reply(500, {"error": str(e)})

    def do_GET(self):  # noqa: N802 - stdlib API
        # Minimal health endpoint
        if self.path == "/health":
            return self._reply(200, {"ok": True})
        return self._reply(404, {"error": "unknown path"})


def main() -> None:
    port = int(os.getenv("PORT", os.getenv("IMAGE_SERVER_PORT", "8000")))
    addr = ("0.0.0.0", port)
    httpd = HTTPServer(addr, App)
    print(f"[srv] listening on 0.0.0.0:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
