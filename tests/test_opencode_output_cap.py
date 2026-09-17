"""The opencode seat's output cap, end to end (docs/adr/0002).

opencode sends a fixed `max_tokens` of 32000 unless
OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX is set. These tests prove the cap on
the wire through the installed opencode binary, and prove against OpenRouter's
live endpoint list that every pinned backend accepts it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from quorum.agents.opencode import (
    _OUTPUT_TOKEN_MAX,
    _PROVIDER_ORDER,
    FLASH_MODEL,
    OpenCodeAgent,
    _build_subprocess_env,
    _ensure_sandbox_home,
)


def test_every_capped_model_restricts_routing_to_its_pinned_backends() -> None:
    # A cap is only safe on backends that were checked against it; an unpinned
    # model would let OpenRouter route to any host.
    assert set(_OUTPUT_TOKEN_MAX) <= set(_PROVIDER_ORDER)


@pytest.mark.skipif(shutil.which("opencode") is None, reason="needs opencode")
def test_opencode_sends_the_seat_output_cap_on_the_wire(tmp_path: Path) -> None:
    bodies: list[dict] = []

    class FakeOpenRouter(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[]}')

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            bodies.append(body)
            chunk = {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sandbox = _ensure_sandbox_home(sandbox=tmp_path / "home", model=FLASH_MODEL)
        env = _build_subprocess_env(
            sandbox,
            "sk-fake",
            model=FLASH_MODEL,
            relay_base_url=f"http://127.0.0.1:{server.server_port}/api/v1",
        )
        workdir = tmp_path / "work"
        workdir.mkdir()
        cmd = OpenCodeAgent(model=FLASH_MODEL).build_command(
            prompt="say OK", cwd=str(workdir)
        )
        proc = subprocess.run(
            cmd, env=env, capture_output=True, timeout=120, check=False
        )
    finally:
        server.shutdown()
        server.server_close()

    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr
    chat = [b for b in bodies if "messages" in b]
    assert chat, f"opencode sent no chat request to the fake server: {stderr}"
    for body in chat:
        assert body["max_tokens"] == 256000
        assert body["provider"]["only"] == ["novita", "parasail"]


@pytest.mark.live
def test_every_pinned_backend_accepts_the_output_cap() -> None:
    for model_id, cap in _OUTPUT_TOKEN_MAX.items():
        response = httpx.get(
            f"https://openrouter.ai/api/v1/models/{model_id}/endpoints",
            follow_redirects=True,
            timeout=30,
        )
        response.raise_for_status()
        endpoints = response.json()["data"]["endpoints"]
        limits: dict[str, int] = {}
        for endpoint in endpoints:
            slug = endpoint["tag"].split("/")[0]
            limit = endpoint.get("max_completion_tokens") or 0
            limits[slug] = max(limits.get(slug, 0), limit)
        for backend in _PROVIDER_ORDER[model_id]:
            assert backend in limits, f"{backend} no longer serves {model_id}"
            assert limits[backend] >= cap, (
                f"{backend} caps {model_id} at {limits[backend]}, below {cap}"
            )
