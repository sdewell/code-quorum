"""Loopback relay between opencode and OpenRouter that records which upstream
backend served each request.

Why a relay: OpenRouter names the serving backend (`provider`) and the
generation id in every response, but opencode's SDK drops both -- they appear
in neither its `--format json` events nor its session database (checked
2026-09-15, opencode 1.18.25), and OpenRouter's per-account activity endpoint
needs a management key and only aggregates by day. The seat pins a provider
order for a reproducible model flavor (docs/adr/0001); without this record
the pin cannot be audited.

What it does: listens on 127.0.0.1 at an ephemeral port for the life of one
seat run, forwards every request to OpenRouter unchanged (auth header
included -- opencode already holds the key), and streams the response back
chunk by chunk as it arrives. While the bytes pass through it reads the
metadata out of the SSE stream and appends one JSON line per request to an
owner-only ledger: backend, generation id, model, token counts, cost, status,
timings. Never the headers, never the prompt, never the answer.

What it must not do: add latency or timeouts. Each chunk is written and
drained the moment it lands; the upstream read timeout is None so a slow
reasoning turn is bounded only by opencode's own chunkTimeout and the seat's
idle guard, exactly as before. If the relay cannot start, the seat runs
direct to OpenRouter (see opencode.run) rather than failing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .base import _ensure_private_dir

logger = logging.getLogger(__name__)

OPENROUTER_UPSTREAM = "https://openrouter.ai"
SERVED_LEDGER_PATH = Path.home() / ".cache" / "code-quorum" / "opencode-served.jsonl"

# Headers that describe one hop, not the message (RFC 9110 s7.6.1), plus the
# ones the relay re-derives: Host and Content-Length are set by httpx for the
# upstream leg; Accept-Encoding is forced to identity so the bytes forwarded
# are the bytes parsed; Content-Encoding is dropped because httpx would have
# decoded anyway.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_REQUEST_DROP = _HOP_BY_HOP | {"host", "content-length", "accept-encoding"}
_RESPONSE_DROP = _HOP_BY_HOP | {"content-length", "content-encoding"}
_RESPONSE_DROP_BYTES = frozenset(name.encode() for name in _RESPONSE_DROP)

# Request head cap (opencode's requests carry the body separately) and the
# cap on a NON-streaming JSON body kept in memory for metadata parsing; a
# larger one is forwarded untouched and simply not parsed.
_MAX_HEAD_BYTES = 256 * 1024
_MAX_JSON_PARSE_BYTES = 4 * 1024 * 1024

# connect bounded; read/write unbounded on purpose (module docstring).
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=None, pool=None)


@dataclass
class ServedTurn:
    """One request through the relay. Every field is metadata; the ledger is
    exactly these fields, so anything added here is published to disk."""

    started_at: str
    method: str
    path: str
    status: int | None = None
    model: str | None = None
    provider: str | None = None
    generation_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None
    finish_reason: str | None = None
    t_first_byte_s: float | None = None
    t_total_s: float | None = None
    error: str | None = None


class _MetadataReader:
    """Pulls OpenRouter response metadata out of the bytes as they stream by.
    Keeps only the current partial SSE line (or, for a non-streaming JSON
    response, the body up to the parse cap); content deltas are looked at
    once for their finish_reason and discarded."""

    def __init__(self, turn: ServedTurn, *, streaming: bool) -> None:
        self.turn = turn
        self.streaming = streaming
        self._pending = b""
        self._body = bytearray()
        self._overflow = False

    def feed(self, chunk: bytes) -> None:
        if self.streaming:
            self._pending += chunk
            *lines, self._pending = self._pending.split(b"\n")
            for line in lines:
                line = line.strip()
                if line.startswith(b"data:"):
                    self._absorb_json(line[5:].strip())
        elif not self._overflow:
            if len(self._body) + len(chunk) > _MAX_JSON_PARSE_BYTES:
                self._overflow = True
                self._body = bytearray()
            else:
                self._body.extend(chunk)

    def finish(self) -> None:
        if self.streaming:
            if self._pending.strip().startswith(b"data:"):
                self._absorb_json(self._pending.strip()[5:].strip())
        elif self._body:
            self._absorb_json(bytes(self._body))

    def _absorb_json(self, payload: bytes) -> None:
        if not payload or payload == b"[DONE]":
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        turn = self.turn
        provider = obj.get("provider")
        if isinstance(provider, str) and provider:
            turn.provider = provider
        gen = obj.get("id")
        if isinstance(gen, str) and gen.startswith("gen-"):
            turn.generation_id = gen
        model = obj.get("model")
        if isinstance(model, str) and model:
            turn.model = model
        for choice in obj.get("choices") or []:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                turn.finish_reason = str(choice["finish_reason"])
        usage = obj.get("usage")
        if isinstance(usage, dict):
            turn.prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
            turn.completion_tokens = _int_or_none(usage.get("completion_tokens"))
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                turn.reasoning_tokens = _int_or_none(details.get("reasoning_tokens"))
            cost = usage.get("cost")
            if isinstance(cost, int | float):
                turn.cost = float(cost)


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _append_ledger(path: Path, turn: ServedTurn) -> None:
    """One JSON line, owner-only file in an owner-only directory. Best-effort:
    a ledger failure is logged and never fails the request it describes."""
    try:
        _ensure_private_dir(path.parent, parents=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(asdict(turn)) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except OSError as e:
        logger.debug("served-backend ledger write skipped: %s", e)


class OpenRouterRelay:
    """See the module docstring. One instance per seat run: start() before
    the opencode subprocess, stop() after it exits."""

    def __init__(
        self,
        *,
        upstream: str = OPENROUTER_UPSTREAM,
        ledger_path: Path = SERVED_LEDGER_PATH,
    ) -> None:
        self._upstream = upstream.rstrip("/")
        self._ledger_path = ledger_path
        self._server: asyncio.AbstractServer | None = None
        self._client: httpx.AsyncClient | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self.base_url: str | None = None
        self.turns: list[ServedTurn] = []

    async def start(self) -> str:
        """Bind 127.0.0.1:0 and return the base URL opencode should use
        (`.../api/v1`). Raises OSError if the loopback bind fails."""
        # Bind first: a failed bind is the fallback path in opencode.run and
        # must leave nothing to close.
        self._server = await asyncio.start_server(
            self._serve, "127.0.0.1", 0, limit=_MAX_HEAD_BYTES
        )
        self._client = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)
        port = self._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/api/v1"
        return self.base_url

    async def stop(self) -> None:
        """Close the listener, let in-flight handlers finish briefly, drop
        the upstream client. Safe to call twice."""
        server, self._server = self._server, None
        if server is not None:
            server.close()
        if self._tasks:
            _done, pending = await asyncio.wait(self._tasks, timeout=5.0)
            for task in pending:
                task.cancel()
            # Let the cancelled handlers unwind (ledger row, socket close)
            # before the client goes away and summary() is read.
            await asyncio.gather(*pending, return_exceptions=True)
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    @property
    def ledger_path(self) -> Path:
        return self._ledger_path

    def summary(self) -> dict[str, int]:
        """Backend name -> request count for this run; `(none)` counts
        requests that ended without a backend (errors, non-chat paths)."""
        return dict(Counter(t.provider or "(none)" for t in self.turns))

    # -- connection handling ------------------------------------------------

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            await self._handle(reader, writer)
        except (
            ConnectionError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            ValueError,  # malformed request line, length, or chunk size
        ):
            pass  # opencode hung up or sent a malformed head; nothing to relay
        finally:
            if task is not None:
                self._tasks.discard(task)
            writer.close()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        request_line, *header_lines = head[:-4].decode("latin-1").split("\r\n")
        method, target, _version = request_line.split(" ", 2)
        headers: list[tuple[str, str]] = []
        for line in header_lines:
            name, _, value = line.partition(":")
            headers.append((name.strip(), value.strip()))
        body = await self._read_body(reader, headers)

        turn = ServedTurn(
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            method=method,
            path=target.split("?", 1)[0],
        )
        started = time.monotonic()
        try:
            await self._relay(turn, started, method, target, headers, body, writer)
        except httpx.HTTPError as e:
            turn.error = f"{type(e).__name__}: {e}"
            if turn.status is None:
                # Nothing sent yet: a clean 502 opencode will retry.
                turn.status = 502
                _write_error(
                    writer,
                    502,
                    f"relay could not reach OpenRouter: {type(e).__name__}",
                )
            # Else the head and part of the chunked body are already out:
            # closing without the terminating 0-chunk tells the client the
            # stream was cut, which is the truth; a 502 written here would
            # land mid-body as garbage.
        except ConnectionError as e:
            turn.error = f"client closed: {type(e).__name__}"
            raise
        finally:
            turn.t_total_s = round(time.monotonic() - started, 3)
            self.turns.append(turn)
            _append_ledger(self._ledger_path, turn)
            try:
                await writer.drain()
            except ConnectionError:
                pass

    async def _read_body(
        self, reader: asyncio.StreamReader, headers: list[tuple[str, str]]
    ) -> bytes:
        length = 0
        chunked = False
        for name, value in headers:
            lname = name.lower()
            if lname == "content-length":
                length = int(value)
            elif lname == "transfer-encoding" and "chunked" in value.lower():
                chunked = True
        if not chunked:
            return await reader.readexactly(length) if length else b""
        parts = bytearray()
        while True:
            size_line = await reader.readuntil(b"\r\n")
            size = int(size_line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                # Trailer section: header lines until an empty line.
                while (await reader.readuntil(b"\r\n")) != b"\r\n":
                    pass
                break
            parts.extend(await reader.readexactly(size))
            await reader.readexactly(2)  # CRLF after the chunk
        return bytes(parts)

    async def _relay(
        self,
        turn: ServedTurn,
        started: float,
        method: str,
        target: str,
        headers: list[tuple[str, str]],
        body: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        assert self._client is not None
        forward = [(k, v) for k, v in headers if k.lower() not in _REQUEST_DROP]
        forward.append(("Accept-Encoding", "identity"))
        async with self._client.stream(
            method, self._upstream + target, headers=forward, content=body
        ) as resp:
            turn.status = resp.status_code
            streaming = "text/event-stream" in resp.headers.get("content-type", "")
            meta = _MetadataReader(turn, streaming=streaming)
            # Forward the header octets as received: httpx decodes non-ASCII
            # header values as UTF-8, so re-encoding its str form could raise.
            head = [f"HTTP/1.1 {resp.status_code} {resp.reason_phrase}".encode()]
            head += [
                k + b": " + v
                for k, v in resp.headers.raw
                if k.lower() not in _RESPONSE_DROP_BYTES
            ]
            head += [b"Transfer-Encoding: chunked", b"Connection: close", b"", b""]
            writer.write(b"\r\n".join(head))
            await writer.drain()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                if turn.t_first_byte_s is None:
                    turn.t_first_byte_s = round(time.monotonic() - started, 3)
                writer.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                await writer.drain()
                meta.feed(chunk)
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            meta.finish()


def _write_error(writer: asyncio.StreamWriter, status: int, message: str) -> None:
    payload = json.dumps({"error": {"code": status, "message": message}}).encode()
    reason = {502: "Bad Gateway"}.get(status, "Error")
    writer.write(
        (
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
        ).encode("latin-1")
        + payload
    )
