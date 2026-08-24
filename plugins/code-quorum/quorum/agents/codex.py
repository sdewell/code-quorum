import asyncio
import tempfile
import time
from pathlib import Path

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

# Council seat defaults. Unlike every other seat, `codex exec` was invoked with no
# --model, so the seat floated with the ambient ~/.codex config (observed silently
# running GPT-5.3 Spark when a session set it) -- non-deterministic council
# composition. Pin the model AND reasoning effort so a council run is reproducible
# regardless of the developer's codex config. Reasoning effort rides `-c` because
# codex exec has no dedicated flag; the key matches ~/.codex/config.toml's
# `model_reasoning_effort` and the value is TOML-quoted, per `codex exec --help`.
# terra/medium (was sol/xhigh): sol at xhigh routinely blew the council's 600s
# AGENT_TIMEOUT_S on review-sized prompts (observed options-omega, 2026-07-20),
# silently degrading the council to 2 seats. The seat must fit the wall clock.
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_EFFORT = "medium"
MODEL_ENV_VAR = "CODE_QUORUM_CODEX_MODEL"
EFFORT_ENV_VAR = "CODE_QUORUM_CODEX_EFFORT"

# codex's ReasoningEffort enum (grounded from the codex binary's serde variants).
# The effort rides an untyped `-c model_reasoning_effort=` override, so a typo
# would otherwise reach `codex exec` and either error opaquely or silently fall
# back. Validate the env override against this set and fail loud, mirroring
# make_gemini_agent rejecting an unknown backend. Model is deliberately NOT
# whitelisted -- model ids are an open, fast-moving set.
VALID_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "ultra"})


def _subprocess_env() -> dict[str, str]:
    # `--ignore-user-config` still reads authentication from CODEX_HOME. The
    # shared runtime allowlist omits unrelated credential environment variables.
    # HOME remains available for runtime/auth discovery; the documented lack of
    # universal filesystem confinement still applies to files below it.
    return allowlisted_seat_subprocess_env(extra={"CODEX_HOME"})


# Sandbox notice prepended to every prompt. Unlike the gemini/opencode seats,
# codex HAS a shell tool -- but it runs `-s read-only` with escalation denied.
# Without an explicit notice, a seat can spend its turn on denied test and
# escalation requests and return nothing. Mirrors gemini's
# COUNCIL_SYSTEM_INSTRUCTIONS read-only paragraph; codex exec has no separate
# system-prompt channel, so this rides the stdin prompt.
SANDBOX_NOTICE = """\
Your shell runs in a read-only sandbox and sandbox escalation is always \
denied -- do not request it. Never attempt to run test suites, builds, \
installs, or any command that writes files (even to /tmp): they will fail. \
Verify claims by reading code with read-only commands, and where a test run \
would settle a question, recommend the exact command instead of running it. \
You are non-interactive: no one can approve commands or answer follow-ups; \
state assumptions explicitly and always finish with your full answer as text."""


class CodexAgent(Agent):
    name = "codex"
    default_role = "skeptic"

    def __init__(
        self,
        binary: str = "codex",
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        model_source: str = "shipped",
    ):
        self.binary = binary
        self.model = model
        self.effort = effort
        # Set by make_codex_agent from resolve_model's (model, source); a
        # direct construction (tests, other callers) defaults to "shipped".
        # Threaded through so a failed run can point at `quorum setup-models`
        # only when the model actually came from a recorded choice (see
        # run()) -- mirrors the gemini seat's model_source precedent.
        self.model_source = model_source

    def build_prompt(self, prompt: str) -> str:
        """Prepend the read-only sandbox notice; it must precede the whole
        prompt (including the stance prefix) so the constraint frames every
        instruction that follows. Every codex seat runs read-only (no
        production caller constructs otherwise), so this always applies."""
        return f"{SANDBOX_NOTICE}\n\n{prompt}"

    def build_command(self, cwd: str, output_file: str) -> list[str]:
        return [
            self.binary,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--cd",
            cwd,
            "-s",
            "read-only",
            "-m",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.effort}"',
            "-o",
            output_file,
            "-",
        ]

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="quorum-codex-"
        ) as f:
            output_path = f.name

        # Single outer try/finally so cancellation (e.g. council timeout)
        # cannot skip the unlink — otherwise every timed-out codex call
        # leaks /tmp/quorum-codex-*.txt.
        try:
            cmd = self.build_command(cwd=cwd, output_file=output_path)
            start = time.monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=_subprocess_env(),
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
            _, stderr = await communicate_or_kill(
                proc, self.build_prompt(prompt).encode("utf-8"), pgid=proc.pid
            )
            duration = time.monotonic() - start

            try:
                output = Path(output_path).read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                output = ""

            returncode = proc.returncode or 0
            error = stderr.decode("utf-8", errors="replace").strip()
            unavailable_reason = (
                "usage limit"
                if returncode != 0 and has_usage_limit_diagnostic(error)
                else ""
            )
            if returncode not in (0, 127):
                # 127 (binary missing) is definitely not a model problem --
                # the hint would mislead. Every other nonzero exit could be
                # (an unknown model id, among other things), so it gets the
                # conditional pointer.
                error = append_recorded_source_hint(
                    error, self.model, self.model_source
                )
            return AgentResult(
                agent=self.name,
                output=output.strip(),
                error=error,
                returncode=returncode,
                duration_s=duration,
                unavailable_reason=unavailable_reason,
            )
        finally:
            Path(output_path).unlink(missing_ok=True)


def _codex_version(binary: str = "codex") -> str | None:
    """Best-effort installed codex version (e.g. "0.147.0"), or None if it
    can't be determined. Module seam so tests override it without spawning a
    real subprocess; delegates to model_config.probe_cli_version, the
    probe+cache shared by every seat's CLI version check."""
    return probe_cli_version(binary)


def make_codex_agent() -> CodexAgent:
    """Build the codex seat, pinning model + reasoning effort so it never floats
    with the ambient ~/.codex config. Model resolves per_run(none here) > env
    CODE_QUORUM_CODEX_MODEL > a recorded `quorum setup-models` choice > the
    shipped DEFAULT_MODEL. Effort resolves env CODE_QUORUM_CODEX_EFFORT > a
    recorded choice's `effort` > DEFAULT_EFFORT; both the env value and the
    recorded value are validated inside resolve_seat_choice (a disabled
    table's effort skips load-time validation). When a recorded choice
    exists, compares the
    installed codex version against the version recorded at choice time and
    warns once (never fails) on drift -- the recorded model still runs."""
    model, effort, source = resolve_seat_choice(
        "codex",
        model_env_var=MODEL_ENV_VAR,
        effort_env_var=EFFORT_ENV_VAR,
        shipped_model=DEFAULT_MODEL,
        shipped_effort=DEFAULT_EFFORT,
        valid_efforts=VALID_EFFORTS,
        effort_label="codex reasoning effort",
        version_fn=_codex_version,
    )
    return CodexAgent(model=model, effort=effort, model_source=source)
