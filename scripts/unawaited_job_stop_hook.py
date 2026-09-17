"""Stop hook: keep either host running while a quorum job was never awaited.

A job lives only in the MCP server's memory (`quorum_mcp/jobs.py`). The reaper
cancels and drops it JOB_TTL_SECONDS (30 min) after its start, and a server restart
drops it at once. A host that starts a job, ends its turn, and comes back later gets
"job_id not found (expired, already retrieved, or invalid)" and the work is lost.

Reads either host's session transcript (`transcript_path`), collects job ids from
quorum `q_*_start` results and every `job_id` passed to `q_await`, and blocks the
stop while a started id has no await call. An await that errored or was moved to
the background still counts: the call was made, and a backgrounded await delivers
its result as a task notification. When the harness sets `stop_hook_active` (this
hook already blocked once in this stop sequence) the stop is allowed, so a job that
cannot be awaited never loops within one stop. A start older than the server can
keep a job is ignored, so an abandoned job stops blocking later turns once the
server has dropped it. Any unreadable input allows the stop.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

# quorum_mcp/jobs.py: JOB_TTL_SECONDS plus one REAPER_INTERVAL_SECONDS, the longest a
# job can survive. An older start cannot be collected, so it never blocks.
MAX_JOB_AGE = timedelta(seconds=30 * 60 + 60)

START_TOOL = re.compile(r"^mcp__.*quorum.*__q_[a-z_]+_start$")
AWAIT_TOOL = re.compile(r"^mcp__.*quorum.*__q_await$")
JOB_ID = re.compile(r'"job_id"\s*:\s*"([^"]+)"')


def _result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) for block in content if isinstance(block, dict)
        )
    return ""


def _expired(timestamp: object, now: datetime) -> bool:
    """True only for a parseable timestamp older than MAX_JOB_AGE."""
    if not isinstance(timestamp, str):
        return False
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    if when.tzinfo is None:
        return False
    return now - when > MAX_JOB_AGE


def _tool_blocks(event: dict) -> list:
    """Normalize Codex's structured MCP event to Claude's tool block shape.

    Codex records these events for direct MCP calls and calls through code mode;
    parsing JavaScript source would mistake mentions for actual tool execution.
    """
    payload = event.get("payload")
    if (
        event.get("type") == "event_msg"
        and isinstance(payload, dict)
        and payload.get("type") == "item_completed"
    ):
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "McpToolCall":
            call_id = item.get("id")
            blocks = [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": f"mcp__{item.get('server')}__{item.get('tool')}",
                    "input": item.get("arguments"),
                }
            ]
            result = item.get("result")
            if isinstance(result, dict) and not result.get("isError"):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": result.get("content"),
                    }
                )
            return blocks
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, list) else []


def unawaited_jobs(
    transcript_lines: Iterable[str], now: datetime | None = None
) -> list[str]:
    now = now or datetime.now(UTC)
    start_calls: set[str] = set()
    started: list[str] = []
    awaited: set[str] = set()
    for raw in transcript_lines:
        # Transcripts run to hundreds of megabytes. A start's result line names no tool,
        # only the job id, so a line is parsed if it carries either marker.
        if "q_" not in raw and "job_id" not in raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for block in _tool_blocks(event):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = str(block.get("name") or "")
                if START_TOOL.match(name):
                    start_calls.add(str(block.get("id")))
                elif AWAIT_TOOL.match(name):
                    tool_input = block.get("input")
                    if isinstance(tool_input, dict) and isinstance(
                        tool_input.get("job_id"), str
                    ):
                        awaited.add(tool_input["job_id"])
            elif (
                block.get("type") == "tool_result"
                and block.get("tool_use_id") in start_calls
                and not _expired(event.get("timestamp"), now)
            ):
                match = JOB_ID.search(_result_text(block.get("content")))
                if match and match.group(1) not in started:
                    started.append(match.group(1))
    return [job for job in started if job not in awaited]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if not isinstance(payload, dict) or payload.get("stop_hook_active"):
        return 0
    path = payload.get("transcript_path")
    if not isinstance(path, str) or not path:
        return 0
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            pending = unawaited_jobs(handle)
    except OSError:
        return 0
    if pending:
        reason = (
            "A code-quorum job was started and never collected: "
            + ", ".join(pending)
            + ". The server keeps a job for 30 minutes from its start and loses it on a"
            " restart. Call q_await with each job_id now; an await that errors or moves"
            " to the background still clears this."
        )
        print(json.dumps({"decision": "block", "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
