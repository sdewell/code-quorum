"""Tests for scripts/unawaited_job_stop_hook.py — the Stop hook that refuses to end
a host turn while a quorum job was started and never collected.

The transcript lines copy the shape Claude Code 2.1.274 writes (a start's result is
a bare `{"job_id": ...}` string; an await's input carries `job_id`) and Codex
0.154.0's structured MCP completion events. Tests run the real `hooks/hooks.json`
Stop command through the plugin launcher, the way the host does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PREFIX = "mcp__plugin_code-quorum_quorum__"
_JOB = "46b3102dfa47405d90a9ed3d43847f53"


def _tool_use(use_id: str, name: str, tool_input: dict[str, object]) -> str:
    block = {"type": "tool_use", "id": use_id, "name": name, "input": tool_input}
    return json.dumps(
        {"type": "assistant", "message": {"role": "assistant", "content": [block]}}
    )


def _tool_result(use_id: str, text: str, timestamp: str) -> str:
    block = {"tool_use_id": use_id, "type": "tool_result", "content": text}
    message = {"role": "user", "content": [block]}
    return json.dumps({"type": "user", "message": message, "timestamp": timestamp})


def _started(timestamp: str | None = None) -> list[str]:
    if timestamp is None:
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return [
        _tool_use("toolu_start", f"{_PREFIX}q_validate_start", {"cwd": "/p"}),
        _tool_result("toolu_start", json.dumps({"job_id": _JOB}), timestamp),
    ]


def _run_stop_hook(
    tmp_path: Path, lines: list[str], *, active: bool = False
) -> subprocess.CompletedProcess[str]:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hooks = json.loads((_ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    [entry] = hooks["hooks"]["Stop"]
    [hook] = entry["hooks"]
    payload = {
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
        "stop_hook_active": active,
    }
    return subprocess.run(
        hook["command"],
        shell=True,
        executable="/bin/sh",
        cwd=tmp_path,
        env={
            "CLAUDE_PLUGIN_ROOT": str(_ROOT),
            "HOME": str(tmp_path),
            "PATH": os.environ["PATH"],
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "UV_PYTHON": sys.executable,
        },
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_stop_is_blocked_while_a_started_job_was_never_awaited(tmp_path: Path) -> None:
    result = _run_stop_hook(tmp_path, _started())

    assert result.returncode == 0, result.stderr
    decision = json.loads(result.stdout)
    assert decision["decision"] == "block"
    assert _JOB in decision["reason"]


def test_stop_is_allowed_once_the_job_was_awaited(tmp_path: Path) -> None:
    lines = [
        *_started(),
        _tool_use("toolu_await", f"{_PREFIX}q_await", {"job_id": _JOB}),
    ]

    result = _run_stop_hook(tmp_path, lines)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_stop_is_allowed_on_the_retry_after_one_block(tmp_path: Path) -> None:
    result = _run_stop_hook(tmp_path, _started(), active=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_a_job_past_the_server_ttl_does_not_block_later_turns(tmp_path: Path) -> None:
    # The server has already dropped it; blocking would only force a doomed await.
    result = _run_stop_hook(tmp_path, _started("2020-01-01T00:00:00.000Z"))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("state", ["pending", "awaited", "expired", "retry"])
def test_codex_stop_collects_only_live_unawaited_jobs(
    tmp_path: Path, state: str
) -> None:
    # Captured from Codex CLI 0.154.0 with a harmless stub MCP server.
    fixture = _ROOT / "tests" / "fixtures" / "codex_quorum_stop.jsonl"
    start, awaited = [json.loads(line) for line in fixture.read_text().splitlines()]
    start["timestamp"] = (
        "2020-01-01T00:00:00Z" if state == "expired" else datetime.now(UTC).isoformat()
    )
    lines = [json.dumps(start)]
    if state == "awaited":
        lines.append(json.dumps(awaited))
    result = _run_stop_hook(tmp_path, lines, active=state == "retry")
    assert result.returncode == 0, result.stderr
    if state == "pending":
        decision = json.loads(result.stdout)
        assert decision["decision"] == "block"
        assert "test-quorum-job" in decision["reason"]
    else:
        assert result.stdout == ""
