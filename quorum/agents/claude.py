from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from ..model_config import (
    append_recorded_source_hint,
    probe_cli_version,
    resolve_seat_choice,
)
from .base import (
    Agent,
    AgentResult,
    allowlisted_seat_subprocess_env,
    communicate_or_kill,
    has_usage_limit_diagnostic,
)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"
MODEL_ENV_VAR = "CODE_QUORUM_CLAUDE_MODEL"
EFFORT_ENV_VAR = "CODE_QUORUM_CLAUDE_EFFORT"
VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
CLAUDE_AUTH_RC = 2
CLAUDE_NO_OUTPUT_RC = 125
READ_ONLY_TOOLS = "Read,Glob,Grep"
CLAUDE_SUBSCRIPTION_ENV_VARS = frozenset(
    {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"}
)


def _subscription_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Build the minimal Claude subscription environment."""
    return allowlisted_seat_subprocess_env(
        extra=CLAUDE_SUBSCRIPTION_ENV_VARS, base=base
    )


@dataclass(frozen=True)
class SubscriptionAuthFailure:
    message: str
    unavailable_reason: str
    returncode: int


def _subscription_auth_failure(
    binary: str = "claude",
) -> SubscriptionAuthFailure | None:
    """Return structured failure data unless Claude.ai subscription auth is live."""
    try:
        proc = subprocess.run(
            [binary, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_subscription_env(),
        )
    except FileNotFoundError:
        return SubscriptionAuthFailure(
            f"{binary} not found on PATH", "not installed", 127
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return SubscriptionAuthFailure(
            f"{binary} auth status failed: {exc}", "preflight", 1
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = proc.stderr.strip() or (
            "non-JSON stdout omitted" if proc.stdout else "no diagnostic output"
        )
        if proc.returncode != 0:
            return SubscriptionAuthFailure(
                f"{binary} auth status failed (exit {proc.returncode}): {detail}",
                "preflight",
                proc.returncode,
            )
        return SubscriptionAuthFailure(
            f"{binary} auth status returned invalid JSON: {detail}", "preflight", 1
        )
    if not isinstance(payload, dict):
        return SubscriptionAuthFailure(
            f"{binary} auth status returned a non-object JSON value", "preflight", 1
        )
    if payload.get("loggedIn") is not True:
        return SubscriptionAuthFailure(
            f"{binary} is not logged in; run `claude auth login` in a terminal",
            "authentication",
            CLAUDE_AUTH_RC,
        )
    if payload.get("authMethod") != "claude.ai":
        return SubscriptionAuthFailure(
            (
                f"{binary} is not using a Claude.ai subscription; "
                f"auth method is {payload.get('authMethod')!r}"
            ),
            "authentication",
            CLAUDE_AUTH_RC,
        )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or "no diagnostic output"
        return SubscriptionAuthFailure(
            f"{binary} auth status failed (exit {proc.returncode}): {detail}",
            "preflight",
            proc.returncode,
        )
    return None


def check_subscription_auth(binary: str = "claude") -> str | None:
    """Return None only for a logged-in Claude.ai subscription session."""
    failure = _subscription_auth_failure(binary)
    return failure.message if failure is not None else None


def _parse_output(raw: bytes) -> tuple[str, str, bool]:
    """Return (answer, error, is_error) from Claude's JSON result."""
    try:
        payload: Any = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return "", f"claude returned invalid JSON: {exc}", True
    if not isinstance(payload, dict):
        return "", "claude returned a non-object JSON value", True
    result = payload.get("result")
    if not isinstance(result, str):
        return "", "claude JSON result is missing a string `result` field", True
    if payload.get("is_error") is True:
        return "", result.strip() or "claude reported an unspecified error", True
    return result.strip(), "", False


class ClaudeAgent(Agent):
    """Claude Code subscription seat with an explicit read-only tool surface."""

    name = "claude"
    default_role = "skeptic"

    def __init__(
        self,
        binary: str = "claude",
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        model_source: str = "shipped",
    ) -> None:
        self.binary = binary
        self.model = model
        self.effort = effort
        self.model_source = model_source

    def build_command(self) -> list[str]:
        return [
            self.binary,
            "--safe-mode",
            "--no-session-persistence",
            "--tools",
            READ_ONLY_TOOLS,
            "--permission-mode",
            "dontAsk",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--output-format",
            "json",
            "-p",
        ]

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        start = time.monotonic()
        auth_failure = await asyncio.to_thread(_subscription_auth_failure, self.binary)
        if auth_failure is not None:
            return AgentResult(
                agent=self.name,
                output="",
                error=auth_failure.message,
                returncode=auth_failure.returncode,
                duration_s=time.monotonic() - start,
                unavailable_reason=auth_failure.unavailable_reason,
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                *self.build_command(),
                cwd=cwd,
                env=_subscription_env(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            return AgentResult(
                agent=self.name,
                output="",
                error=f"{self.binary} not found on PATH",
                returncode=127,
                duration_s=time.monotonic() - start,
                unavailable_reason="not installed",
            )

        stdout, stderr = await communicate_or_kill(
            proc, prompt.encode("utf-8"), pgid=proc.pid
        )
        duration = time.monotonic() - start
        output, parse_error, is_error = _parse_output(stdout)
        returncode = proc.returncode or 0
        error = stderr.decode("utf-8", errors="replace").strip()
        synthesized_no_output = False
        if is_error:
            returncode = returncode or 1
            error = parse_error if not error else f"{error}\n{parse_error}"
        elif not output:
            synthesized_no_output = True
            returncode = CLAUDE_NO_OUTPUT_RC
            error = error or "claude exited without an answer"
        if returncode != 0 and has_usage_limit_diagnostic(error):
            unavailable_reason = "usage limit"
        elif synthesized_no_output:
            unavailable_reason = "no output"
        else:
            unavailable_reason = ""
        if returncode not in (0, 127, CLAUDE_AUTH_RC, CLAUDE_NO_OUTPUT_RC):
            error = append_recorded_source_hint(error, self.model, self.model_source)
        return AgentResult(
            agent=self.name,
            output=output,
            error=error,
            returncode=returncode,
            duration_s=duration,
            unavailable_reason=unavailable_reason,
        )


def _claude_version(binary: str = "claude") -> str | None:
    """Best-effort installed claude version, or None if it can't be
    determined. Module seam so tests override it without spawning a real
    subprocess; delegates to model_config.probe_cli_version, the probe+cache
    shared by every seat's CLI version check."""
    return probe_cli_version(binary)


def make_claude_agent() -> ClaudeAgent:
    model, effort, source = resolve_seat_choice(
        "claude",
        model_env_var=MODEL_ENV_VAR,
        effort_env_var=EFFORT_ENV_VAR,
        shipped_model=DEFAULT_MODEL,
        shipped_effort=DEFAULT_EFFORT,
        valid_efforts=VALID_EFFORTS,
        effort_label="Claude effort",
        version_fn=_claude_version,
    )
    return ClaudeAgent(model=model, effort=effort, model_source=source)
