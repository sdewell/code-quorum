import os
import time
from pathlib import Path

from .base import Agent, AgentResult, has_usage_limit_diagnostic

# Council member persona. The role/stance for a given round is applied to the
# prompt by the orchestrator (apply_role); this is the static, read-only voice.
# Mirrors the opencode council agent: ground answers in the repo via read-only
# tools, never write, never ask follow-ups (the run is non-interactive), and
# always finish with a plain-text answer.
COUNCIL_SYSTEM_INSTRUCTIONS = """\
You are a council member responding to a single prompt. Ground your answer in \
the repository: use your read tools (view_file, find_file, search_directory, \
list_directory) to open any file the prompt refers to before you answer.

Your tools are read-only. You have no write, edit, or shell/terminal tool. Do \
not attempt them, and do not treat their absence as an error -- reach the goal \
by reading files directly.

Never invent or guess file contents, paths, headings, or code. If a file you \
need cannot be read, say so plainly and reason only from what you actually read.

You are running non-interactively: no one can answer follow-up questions. If the \
prompt is ambiguous or refers to something you cannot find, state your \
assumption explicitly and give your best substantive answer -- do not ask \
clarifying questions or wait for input.

Respond with plain text or markdown only. Keep your answer focused and \
substantive, and always finish by writing your full answer as text."""

# Returncode when GEMINI_API_KEY is absent -- distinct from 1 (SDK error) and
# 127 (SDK missing). Mirrors the "2 == no API key" convention in the opencode
# adapter.
GEMINI_NO_KEY_RC = 2

# Returncode when the SDK Python package isn't importable.
GEMINI_NO_SDK_RC = 127

# Returncode when cwd sits under a hidden directory and the widened workspace
# would reach $HOME -- the SDK backend has no sandbox to fence that read
# scope, unlike the seatbelted CLI path's check_sandbox_cwd. Mirrors
# GEMINI_CLI_NO_SANDBOX_RC's value in gemini_cli.py (same class of refusal).
GEMINI_UNSAFE_CWD_RC = 4

# Returncode when the agent exits cleanly but produces no answer text -- the
# model ended its turn with only whitespace. A blank "success" silently
# vanishes at the orchestrator (run_council drops agents whose output is empty)
# and gets the agent dropped from later rounds invisibly; a distinct non-zero
# code keeps the empty turn VISIBLE. Mirrors OPENCODE_NO_OUTPUT_RC.
GEMINI_NO_OUTPUT_RC = 125

# Council default model. The SDK's own default is gemini-3.5-flash, whose
# free-tier quota is 5 requests/min -- an agentic turn (read -> reason ->
# answer) makes several model calls and exhausts it, 429-ing the peer out of
# the round. gemini-3.1-flash-lite has a higher free-tier RPM and keeps the
# peer reliably in the council; quota is per-model, so it draws a separate
# bucket. Override per-instance via GeminiAgent(model=...).
DEFAULT_MODEL = "gemini-3.1-flash-lite"


def _non_hidden_workspace(cwd: str) -> str:
    """Resolve ``cwd`` to a workspace root the Antigravity localharness accepts.

    The harness refuses a hidden (dot-prefixed) directory as a workspace root:
    it drops the URI ("... is hidden: ignore uri") and agent creation fails
    outright, so the seat is dead when the council runs from e.g. ``~/.claude``.
    Hidden *subpaths* under a non-hidden root read fine (verified end-to-end), so
    return the nearest ancestor whose path contains NO hidden segment -- the
    agent then reads ``cwd``'s files via their real (still-hidden) paths under
    that broader, read-only workspace. A ``cwd`` that already has no hidden
    segment is returned unchanged (the common case). The match is lexical (no
    symlink resolution), mirroring how the harness validates the URI it is given.
    """
    parts = Path(os.path.abspath(cwd)).parts
    first_hidden = next(
        (i for i, p in enumerate(parts) if p.startswith(".")),
        None,
    )
    if first_hidden is None:
        return cwd
    return str(Path(*parts[:first_hidden]))


class GeminiAgent(Agent):
    """Council peer backed by the Google Antigravity Python SDK
    (``google-antigravity``), running Gemini models read-only.

    The legacy ``gemini`` CLI's free OAuth tier was discontinued (migrated to
    Antigravity); the SDK replaces the CLI shell-out. The SDK's documented
    read-only *default* does NOT hold for the local connection (it enables all
    tools but ``run_command``), so we pin the read-only tool allowlist
    explicitly -- the same posture as the other council peers (codex
    ``-s read-only``, opencode permission-locked). Auth is a Gemini API key
    (``GEMINI_API_KEY``); the SDK has no OAuth path.
    """

    name = "gemini"
    default_role = "architect"

    def __init__(self, model: str = DEFAULT_MODEL):
        # Defaults to DEFAULT_MODEL (gemini-3.1-flash-lite); pass an explicit
        # model id to override.
        self.model = model

    def build_config(self, cwd: str, api_key: str):
        """Construct the read-only LocalAgentConfig for one run. Pure (no I/O):
        the read-only tool allowlist and the caller's repo as the workspace are
        the safety contract, asserted by tests. When ``cwd`` is (or sits under) a
        hidden directory, the workspace is widened to the nearest non-hidden
        ancestor -- the harness rejects a hidden workspace root outright; see
        ``_non_hidden_workspace``. Callers must reject a cwd whose widened
        workspace would reach $HOME before calling this (see ``run``); this
        method stays pure and does not itself refuse that case."""
        from google.antigravity import (
            BuiltinTools,
            CapabilitiesConfig,
            LocalAgentConfig,
        )

        return LocalAgentConfig(
            system_instructions=COUNCIL_SYSTEM_INSTRUCTIONS,
            workspaces=[_non_hidden_workspace(cwd)],
            capabilities=CapabilitiesConfig(enabled_tools=BuiltinTools.read_only()),
            api_key=api_key,
            model=self.model,
        )

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        start = time.monotonic()
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return AgentResult(
                agent=self.name,
                output="",
                error=(
                    "GEMINI_API_KEY not set. Export it (e.g. in ~/.zshrc.local) "
                    "so the council process inherits it; get a key from "
                    "https://aistudio.google.com/apikey"
                ),
                returncode=GEMINI_NO_KEY_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="authentication",
            )

        workspace = _non_hidden_workspace(cwd)
        home = Path.home().resolve()
        try:
            home.relative_to(Path(workspace).resolve())
        except ValueError:
            pass
        else:
            if workspace != cwd:
                cause = (
                    f"cwd {cwd} sits under a hidden directory, so the "
                    f"Antigravity workspace would widen to {workspace} -- "
                    "the whole home directory."
                )
            else:
                cause = (
                    f"cwd {cwd} is the home directory or one of its "
                    "ancestors, so the Antigravity workspace would be the "
                    "whole home directory."
                )
            return AgentResult(
                agent=self.name,
                output="",
                error=(
                    f"{cause} The SDK backend has no sandbox to fence that. "
                    "Run from a repository under a non-hidden project path, "
                    "or use the default agy CLI backend on macOS."
                ),
                returncode=GEMINI_UNSAFE_CWD_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="sandbox",
            )

        try:
            from google.antigravity import Agent as AntigravityAgent
        except ImportError as exc:
            return AgentResult(
                agent=self.name,
                output="",
                error=f"google-antigravity SDK not importable: {exc}",
                returncode=GEMINI_NO_SDK_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="not installed",
            )

        # `except Exception` deliberately does NOT catch asyncio.CancelledError
        # (a BaseException): when the council's wall-clock cap fires, the
        # cancellation must propagate through the async context manager's
        # __aexit__ (which tears down the SDK runtime) and out of run() so
        # wait_for resolves the timeout. Catching it would swallow the council's
        # timeout and leak the runtime.
        try:
            config = self.build_config(cwd=cwd, api_key=api_key)
            chunks: list[str] = []
            async with AntigravityAgent(config) as agent:
                response = await agent.chat(prompt)
                async for token in response:
                    chunks.append(token)
            output = "".join(chunks).strip()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return AgentResult(
                agent=self.name,
                output="",
                error=error,
                returncode=1,
                duration_s=time.monotonic() - start,
                unavailable_reason=(
                    "usage limit" if has_usage_limit_diagnostic(error) else ""
                ),
            )

        duration = time.monotonic() - start
        if not output:
            return AgentResult(
                agent=self.name,
                output="",
                error=(
                    "gemini produced no answer: the model ended its turn "
                    "without writing a final response (empty/whitespace-only "
                    "output)."
                ),
                returncode=GEMINI_NO_OUTPUT_RC,
                duration_s=duration,
                unavailable_reason="no output",
            )
        return AgentResult(
            agent=self.name,
            output=output,
            returncode=0,
            duration_s=duration,
        )
