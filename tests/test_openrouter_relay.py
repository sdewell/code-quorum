"""The loopback relay that records which OpenRouter backend served each turn.

Every test runs a fake upstream on 127.0.0.1 and points the relay at it, so
the assertions cover the real socket path opencode will use, not a mocked
transport. Nothing here touches the network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from quorum.agents.openrouter_relay import OpenRouterRelay, ServedTurn

SECRET = "sk-or-v1-never-in-the-ledger"
PROMPT = "review this very secret prompt text"


def _sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


class _FakeUpstream:
    """Minimal HTTP/1.1 server that streams a scripted SSE body. `gate` lets
    a test hold the second chunk until the client has seen the first, which
    is how the no-buffering property is proven without timing."""

    def __init__(self, chunks: list[bytes], *, gate: asyncio.Event | None = None):
        self.chunks = chunks
        self.gate = gate
        self.requests: list[tuple[str, dict[str, str], bytes]] = []
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        request_line, *lines = head.decode().split("\r\n")
        headers = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        body = await reader.readexactly(int(headers.get("content-length", "0")))
        self.requests.append((request_line, headers, body))
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\nX-Upstream: fake\r\n\r\n"
        )
        await writer.drain()
        for i, chunk in enumerate(self.chunks):
            if i == 1 and self.gate is not None:
                await self.gate.wait()
            writer.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            await writer.drain()
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        writer.close()


CHUNK_1 = _sse(
    {
        "id": "gen-abc123",
        "provider": "Novita",
        "model": "deepseek/deepseek-v4.1-flash",
        "choices": [{"delta": {"content": "hel"}, "finish_reason": None}],
    }
)
CHUNK_2 = _sse(
    {
        "id": "gen-abc123",
        "provider": "Novita",
        "model": "deepseek/deepseek-v4.1-flash",
        "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 33},
            "cost": 0.00042,
        },
    }
)
DONE = b"data: [DONE]\n\n"


async def _post(
    relay_base: str, *, stream: bool = True
) -> tuple[httpx.Response, bytes]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        req = client.build_request(
            "POST",
            relay_base + "/chat/completions",
            headers={"Authorization": f"Bearer {SECRET}"},
            json={
                "model": "deepseek/deepseek-v4.1-flash",
                "messages": [{"content": PROMPT}],
            },
        )
        resp = await client.send(req, stream=stream)
        body = await resp.aread()
        await resp.aclose()
        return resp, body


@pytest.mark.asyncio
async def test_relay_passes_bytes_through_and_records_backend(tmp_path: Path) -> None:
    upstream = _FakeUpstream([CHUNK_1, CHUNK_2, DONE])
    up = await upstream.start()
    relay = OpenRouterRelay(upstream=up, ledger_path=tmp_path / "led" / "served.jsonl")
    base = await relay.start()
    try:
        resp, body = await _post(base)
    finally:
        await relay.stop()
        await upstream.stop()

    assert resp.status_code == 200
    assert body == CHUNK_1 + CHUNK_2 + DONE, "the body must reach the client unaltered"
    assert resp.headers["x-upstream"] == "fake", "upstream headers pass through"
    # The forwarded request kept the caller's auth and path, upstream-relative.
    request_line, headers, req_body = upstream.requests[0]
    assert request_line.startswith("POST /api/v1/chat/completions ")
    assert headers["authorization"] == f"Bearer {SECRET}"
    assert json.loads(req_body)["messages"][0]["content"] == PROMPT

    assert len(relay.turns) == 1
    turn = relay.turns[0]
    assert turn.provider == "Novita"
    assert turn.generation_id == "gen-abc123"
    assert turn.model == "deepseek/deepseek-v4.1-flash"
    assert (turn.prompt_tokens, turn.completion_tokens, turn.reasoning_tokens) == (
        120,
        40,
        33,
    )
    assert turn.cost == pytest.approx(0.00042)
    assert turn.finish_reason == "stop"
    assert turn.status == 200
    assert turn.error is None
    assert turn.t_first_byte_s is not None and turn.t_total_s is not None


@pytest.mark.asyncio
async def test_relay_ledger_row_is_metadata_only_and_owner_readable(
    tmp_path: Path,
) -> None:
    # The relay sits between opencode and OpenRouter, so it sees the API key
    # and the reviewed material. The ledger must carry neither: only what is
    # needed to audit the provider pin.
    upstream = _FakeUpstream([CHUNK_1, CHUNK_2, DONE])
    up = await upstream.start()
    ledger = tmp_path / "led" / "served.jsonl"
    relay = OpenRouterRelay(upstream=up, ledger_path=ledger)
    base = await relay.start()
    try:
        await _post(base)
        await _post(base)
    finally:
        await relay.stop()
        await upstream.stop()

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "one JSON line per request, appended"
    raw = ledger.read_text(encoding="utf-8")
    assert SECRET not in raw
    assert PROMPT not in raw
    assert "hel" not in json.loads(lines[0]).values()
    row = json.loads(lines[1])
    assert row["provider"] == "Novita"
    assert row["generation_id"] == "gen-abc123"
    assert row["path"] == "/api/v1/chat/completions"
    assert set(row) == set(ServedTurn.__dataclass_fields__)
    assert (ledger.stat().st_mode & 0o777) == 0o600
    assert (ledger.parent.stat().st_mode & 0o777) == 0o700


@pytest.mark.asyncio
async def test_relay_forwards_each_chunk_as_it_arrives(tmp_path: Path) -> None:
    # No buffering: the client must receive chunk 1 while the upstream is
    # still holding chunk 2 behind the gate. A relay that waits for the whole
    # body would hang this read until the 5s guard fires.
    gate = asyncio.Event()
    upstream = _FakeUpstream([CHUNK_1, CHUNK_2, DONE], gate=gate)
    up = await upstream.start()
    relay = OpenRouterRelay(upstream=up, ledger_path=tmp_path / "served.jsonl")
    base = await relay.start()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            async with client.stream(
                "POST", base + "/chat/completions", json={"messages": []}
            ) as resp:
                it = resp.aiter_raw()
                first = await asyncio.wait_for(it.__anext__(), 5.0)
                assert first.startswith(b"data: "), first
                assert b"gen-abc123" in first
                gate.set()
                rest = b"".join([c async for c in it])
                assert rest.endswith(DONE)
    finally:
        await relay.stop()
        await upstream.stop()


@pytest.mark.asyncio
async def test_relay_reports_unreachable_upstream_as_502(tmp_path: Path) -> None:
    # Pick a port nothing listens on: bind-and-release.
    probe = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()

    relay = OpenRouterRelay(
        upstream=f"http://127.0.0.1:{port}", ledger_path=tmp_path / "served.jsonl"
    )
    base = await relay.start()
    try:
        resp, body = await _post(base, stream=False)
    finally:
        await relay.stop()

    assert resp.status_code == 502
    assert "relay" in json.loads(body)["error"]["message"]
    assert relay.turns[0].status == 502
    assert relay.turns[0].error is not None
    assert relay.turns[0].provider is None


@pytest.mark.asyncio
async def test_relay_parses_non_streaming_json_response(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "id": "gen-xyz",
            "provider": "Parasail",
            "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    ).encode()

    class _JsonUpstream(_FakeUpstream):
        async def _serve(self, reader, writer):  # type: ignore[override]
            head = await reader.readuntil(b"\r\n\r\n")
            n = 0
            for line in head.decode().split("\r\n"):
                if line.lower().startswith("content-length:"):
                    n = int(line.split(":", 1)[1])
            await reader.readexactly(n)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: %d\r\n\r\n%s" % (len(payload), payload)
            )
            await writer.drain()
            writer.close()

    upstream = _JsonUpstream([])
    up = await upstream.start()
    relay = OpenRouterRelay(upstream=up, ledger_path=tmp_path / "served.jsonl")
    base = await relay.start()
    try:
        resp, body = await _post(base, stream=False)
    finally:
        await relay.stop()
        await upstream.stop()

    assert json.loads(body)["id"] == "gen-xyz"
    assert relay.turns[0].provider == "Parasail"
    assert relay.turns[0].generation_id == "gen-xyz"
    assert relay.turns[0].completion_tokens == 2


def test_relay_summary_counts_backends() -> None:
    relay = OpenRouterRelay(
        upstream="http://127.0.0.1:1", ledger_path=Path("/dev/null")
    )
    relay.turns.extend(
        [
            ServedTurn(started_at="t", method="POST", path="/p", provider="Novita"),
            ServedTurn(started_at="t", method="POST", path="/p", provider="Novita"),
            ServedTurn(started_at="t", method="POST", path="/p", provider="Parasail"),
            ServedTurn(
                started_at="t", method="POST", path="/p", provider=None, error="x"
            ),
        ]
    )
    assert relay.summary() == {"Novita": 2, "Parasail": 1, "(none)": 1}


# --- review-fix round (q-review 2026-09-15) --------------------------------


async def _raw_post(base: str) -> bytes:
    """POST with a bare socket and return every byte the relay sends, so a
    test can inspect the HTTP framing itself rather than httpx's view of it."""
    host, port = base.removeprefix("http://").split("/", 1)[0].split(":")
    reader, writer = await asyncio.open_connection(host, int(port))
    body = b'{"messages": []}'
    writer.write(
        b"POST /api/v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n%s"
        % (len(body), body)
    )
    await writer.drain()
    raw = await asyncio.wait_for(reader.read(), 5.0)
    writer.close()
    return raw


@pytest.mark.asyncio
async def test_relay_upstream_drop_midstream_cuts_stream_without_second_status(
    tmp_path: Path,
) -> None:
    # Finding: on an upstream failure after the response head was sent, the
    # relay wrote a second "HTTP/1.1 502" into the open chunked body and
    # overwrote the ledger status. The client must instead see one status
    # line and a stream cut short of the terminating 0-chunk (the truth), and
    # the ledger must keep the real status with the error alongside.
    class _Dropping(_FakeUpstream):
        async def _serve(self, reader, writer):  # type: ignore[override]
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            writer.write(b"%x\r\n%s\r\n" % (len(CHUNK_1), CHUNK_1))
            await writer.drain()
            writer.close()  # no terminating 0-chunk

    upstream = _Dropping([])
    up = await upstream.start()
    relay = OpenRouterRelay(upstream=up, ledger_path=tmp_path / "served.jsonl")
    base = await relay.start()
    try:
        raw = await _raw_post(base)
    finally:
        await relay.stop()
        await upstream.stop()

    assert raw.startswith(b"HTTP/1.1 200 OK\r\n")
    assert raw.count(b"HTTP/1.1 ") == 1, raw
    assert b"502" not in raw
    assert CHUNK_1 in raw, "the chunk that did arrive was forwarded"
    assert not raw.endswith(b"0\r\n\r\n"), "no terminating chunk: the cut is visible"
    turn = relay.turns[0]
    assert turn.status == 200
    assert turn.error is not None and "Error" in turn.error
    assert turn.provider == "Novita"


@pytest.mark.asyncio
async def test_relay_start_bind_failure_leaves_no_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Finding: the httpx client was created before the bind, so a bind
    # failure (the fallback path in opencode.run) leaked it.
    async def _refuse(*_a: object, **_kw: object) -> None:
        raise OSError(48, "address already in use")

    monkeypatch.setattr(asyncio, "start_server", _refuse)
    relay = OpenRouterRelay(upstream="http://127.0.0.1:1", ledger_path=tmp_path / "l")
    with pytest.raises(OSError):
        await relay.start()
    assert relay._client is None
    await relay.stop()  # and stop() after a failed start is harmless


@pytest.mark.asyncio
async def test_relay_forwards_non_ascii_header_octets_verbatim(tmp_path: Path) -> None:
    # Finding: httpx decodes a UTF-8 header value to str; re-encoding it as
    # latin-1 raised and killed the connection. Header octets now pass
    # through untouched.
    class _Utf8Header(_FakeUpstream):
        async def _serve(self, reader, writer):  # type: ignore[override]
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"X-Note: caf\xc3\xa9\r\nTransfer-Encoding: chunked\r\n\r\n"
                b"%x\r\n%s\r\n0\r\n\r\n" % (len(DONE), DONE)
            )
            await writer.drain()
            writer.close()

    upstream = _Utf8Header([])
    up = await upstream.start()
    relay = OpenRouterRelay(upstream=up, ledger_path=tmp_path / "served.jsonl")
    base = await relay.start()
    try:
        raw = await _raw_post(base)
    finally:
        await relay.stop()
        await upstream.stop()

    assert b"X-Note: caf\xc3\xa9\r\n" in raw
    assert raw.endswith(b"0\r\n\r\n")
    assert relay.turns[0].status == 200 and relay.turns[0].error is None
