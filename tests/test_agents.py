import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from quorum import model_config as mc
from quorum.agents import (
    AgentResult,
    CodexAgent,
    GeminiAgent,
    OpenCodeAgent,
)
from quorum.agents import codex as codex_mod
from quorum.agents import opencode as opencode_mod
from quorum.agents.base import (
    RESEARCH_PROVIDER_ENV_VARS,
    allowlisted_seat_subprocess_env,
    has_usage_limit_diagnostic,
)
from quorum.agents.codex import make_codex_agent
from quorum.agents.opencode import (
    COUNCIL_AGENT_MD,
    DEFAULT_SMALL_MODEL,
    OPENCODE_IDLE_TIMEOUT_S,
    OPENROUTER_CHUNK_TIMEOUT_MS,
    _build_opencode_config,
    _build_subprocess_env,
    _capture_reason,
    _debug_enabled,
    _ensure_sandbox_home,
    _extract_text_and_error,
    _maybe_capture_debug,
    _npm_cache_needs_prune,
    _prune_npm_cache,
    _recover_text_from_db,
    _resolve_openrouter_key,
    _session_id_from_stdout,
    make_opencode_agent,
)


def test_codex_build_command_shape() -> None:
    cmd = CodexAgent().build_command(cwd="/tmp/x", output_file="/tmp/out.txt")
    assert cmd[0] == "codex"
    assert "exec" in cmd
    assert "--skip-git-repo-check" in cmd
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert cmd[cmd.index("--cd") + 1] == "/tmp/x"
    assert cmd[cmd.index("-s") + 1] == "read-only"
    assert cmd[cmd.index("-o") + 1] == "/tmp/out.txt"
    assert cmd[-1] == "-"


def test_codex_build_command_pins_model_and_effort() -> None:
    # The codex seat must pin its model + reasoning effort so it never floats with
    # the ambient ~/.codex config (which silently pulled in GPT-5.3 Spark once).
    cmd = CodexAgent().build_command(cwd="/tmp/x", output_file="/tmp/o")
    assert cmd[cmd.index("-m") + 1] == "gpt-5.6-terra"
    assert "-c" in cmd
    assert 'model_reasoning_effort="medium"' in cmd
    assert cmd[-1] == "-"  # stdin marker stays last


def test_codex_build_command_model_and_effort_override() -> None:
    cmd = CodexAgent(model="gpt-6", effort="low").build_command(
        cwd="/x", output_file="/o"
    )
    assert cmd[cmd.index("-m") + 1] == "gpt-6"
    assert 'model_reasoning_effort="low"' in cmd


def test_codex_build_prompt_prepends_sandbox_notice() -> None:
    # The codex seat HAS a shell tool but runs read-only with escalation
    # denied; without an up-front notice it discovers the wall by hitting it
    # (observed: a review seat spent its run trying to pytest, got denied,
    # and returned nothing -- a silent 2-agent council). The notice must land
    # before the stance/role prefix, i.e. before the whole prompt.
    prompt = "Approach this as a skeptic.\n\nReview the diff."
    built = CodexAgent().build_prompt(prompt)
    assert built.endswith(prompt)
    notice = built[: -len(prompt)]
    assert "read-only" in notice
    assert "escalation" in notice
    # The behavioral core: never run tests/builds, recommend instead.
    assert "test" in notice
    assert "recommend" in notice


@pytest.mark.parametrize(
    "diagnostic",
    [
        "ERROR: You've hit your usage limit.",
        "You’ve hit your usage limit.",
        "You've hit your limit · resets 5pm",
        "429 RESOURCE_EXHAUSTED",
        "rate limit exceeded",
    ],
)
def test_usage_limit_diagnostic_accepts_provider_phrasing(diagnostic: str) -> None:
    assert has_usage_limit_diagnostic(diagnostic)


def test_usage_limit_diagnostic_ignores_earlier_echoed_prompt_text() -> None:
    error = "ERROR: You've hit your usage limit.\n" + "prompt echo\n" * 10

    assert not has_usage_limit_diagnostic(error)


def test_usage_limit_diagnostic_tolerates_trailing_stack_lines() -> None:
    error = "ERROR: You've hit your usage limit.\n" + "at stack frame\n" * 6

    assert has_usage_limit_diagnostic(error)


def test_usage_limit_diagnostic_rejects_echoed_diff_literal_at_tail() -> None:
    error = '+        "rate limit exceeded"\n+        "quota exhaustion"'

    assert not has_usage_limit_diagnostic(error)


# --- run(): recorded-source failure hint (item 4) ----------------------------


class _FakeCodexProc:
    def __init__(self, returncode: int) -> None:
        self.pid = 5252
        self.returncode = returncode


@pytest.mark.asyncio
async def test_codex_run_recorded_source_failure_gets_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_spawn(*_a: object, **_kw: object) -> _FakeCodexProc:
        return _FakeCodexProc(returncode=1)

    async def _fake_communicate(
        proc: object, stdin: bytes | None = None, *, pgid: int
    ) -> tuple[bytes, bytes]:
        return (b"", b"unknown model id")

    monkeypatch.setattr(codex_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(codex_mod, "communicate_or_kill", _fake_communicate)

    result = await CodexAgent(model_source="recorded").run(prompt="x", cwd="/tmp")

    assert result.returncode == 1
    assert "quorum setup-models" in result.error
    assert codex_mod.DEFAULT_MODEL in result.error


@pytest.mark.asyncio
async def test_codex_run_strips_research_provider_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    for name in RESEARCH_PROVIDER_ENV_VARS:
        monkeypatch.setenv(name, "must-not-reach-codex")
    for name in ("ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "unrelated-secret")
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex-auth")

    async def _fake_spawn(*_a: object, **kwargs: object) -> _FakeCodexProc:
        captured.update(kwargs)
        return _FakeCodexProc(returncode=1)

    async def _fake_communicate(
        proc: object, stdin: bytes | None = None, *, pgid: int
    ) -> tuple[bytes, bytes]:
        return (b"", b"review failed")

    monkeypatch.setattr(codex_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(codex_mod, "communicate_or_kill", _fake_communicate)

    await CodexAgent().run(prompt="x", cwd="/tmp")

    assert isinstance(captured["env"], dict)
    env = cast(dict[str, str], captured["env"])
    assert not set(RESEARCH_PROVIDER_ENV_VARS) & env.keys()
    assert "ANTHROPIC_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "OPENROUTER_API_KEY" not in env
    assert env["CODEX_HOME"] == "/tmp/codex-auth"
    assert "PATH" in env


def test_seat_allowlist_preserves_base_and_rejects_research_extras() -> None:
    env = allowlisted_seat_subprocess_env(
        extra={"CLAUDE_CONFIG_DIR", "GH_TOKEN"},
        base={
            "PATH": "/usr/bin",
            "CLAUDE_CONFIG_DIR": "/tmp/alternate-claude-config",
            "GH_TOKEN": "must-not-reach-seat",
            "UNRELATED_SECRET": "must-not-reach-seat",
        },
    )

    assert env == {
        "PATH": "/usr/bin",
        "CLAUDE_CONFIG_DIR": "/tmp/alternate-claude-config",
    }


@pytest.mark.asyncio
async def test_codex_run_shipped_source_failure_gets_no_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_spawn(*_a: object, **_kw: object) -> _FakeCodexProc:
        return _FakeCodexProc(returncode=1)

    async def _fake_communicate(
        proc: object, stdin: bytes | None = None, *, pgid: int
    ) -> tuple[bytes, bytes]:
        return (b"", b"unknown model id")

    monkeypatch.setattr(codex_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(codex_mod, "communicate_or_kill", _fake_communicate)

    result = await CodexAgent().run(prompt="x", cwd="/tmp")  # default: "shipped"

    assert result.returncode == 1
    assert "quorum setup-models" not in result.error


@pytest.mark.asyncio
async def test_codex_run_classifies_usage_limit_before_error_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_spawn(*_a: object, **_kw: object) -> _FakeCodexProc:
        return _FakeCodexProc(returncode=1)

    async def _fake_communicate(
        proc: object, stdin: bytes | None = None, *, pgid: int
    ) -> tuple[bytes, bytes]:
        return (b"", b"ERROR: You've hit your usage limit.")

    monkeypatch.setattr(codex_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(codex_mod, "communicate_or_kill", _fake_communicate)

    result = await CodexAgent(model_source="recorded").run(prompt="x", cwd="/tmp")

    assert result.unavailable_reason == "usage limit"


@pytest.mark.asyncio
async def test_codex_run_recorded_source_missing_binary_gets_no_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rc 127 (binary missing) is definitely not a model problem -- gated
    # away from the hint even when the model came from a recorded choice.
    async def _missing(*_a: object, **_kw: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(codex_mod.asyncio, "create_subprocess_exec", _missing)

    result = await CodexAgent(model_source="recorded").run(prompt="x", cwd="/tmp")

    assert result.returncode == 127
    assert result.unavailable_reason == "not installed"
    assert "quorum setup-models" not in result.error


def test_gemini_build_config_read_only() -> None:
    from google.antigravity import BuiltinTools

    cfg = GeminiAgent().build_config(cwd="/repo/proj", api_key="test-key")
    assert cfg.workspaces == ["/repo/proj"]
    assert cfg.api_key == "test-key"
    # The read-only tool allowlist is the council safety contract: reads are
    # enabled, writes/edits/shell are not. The SDK's documented read-only
    # *default* does not hold for the local connection, so we pin it explicitly.
    assert cfg.capabilities.enabled_tools == BuiltinTools.read_only()
    assert BuiltinTools.VIEW_FILE in cfg.capabilities.enabled_tools
    for write_tool in (
        BuiltinTools.CREATE_FILE,
        BuiltinTools.EDIT_FILE,
        BuiltinTools.RUN_COMMAND,
    ):
        assert write_tool not in cfg.capabilities.enabled_tools


def test_gemini_build_config_resolves_hidden_cwd_root() -> None:
    # Antigravity's localharness refuses a hidden (dot-prefixed) directory as a
    # workspace root -- it drops the URI ("... is hidden: ignore uri") and agent
    # creation fails outright -- so the seat is dead when the council runs from
    # e.g. ~/.claude. Hidden *subpaths* under a non-hidden root read fine
    # (verified), so the workspace resolves to the nearest ancestor with no
    # hidden segment; the agent still reads cwd's files via their real paths.
    cfg = GeminiAgent().build_config(cwd="/Users/sd/.claude", api_key="k")
    assert cfg.workspaces == ["/Users/sd"]


def test_gemini_build_config_resolves_hidden_ancestor() -> None:
    # A non-hidden leaf under a hidden ancestor still carries a hidden segment;
    # walk up past the FIRST hidden segment, not just the basename.
    cfg = GeminiAgent().build_config(cwd="/Users/sd/.config/foo/bar", api_key="k")
    assert cfg.workspaces == ["/Users/sd"]


def test_gemini_build_config_nonhidden_cwd_unchanged() -> None:
    # Regression guard: the common (non-hidden) case is passed through verbatim.
    cfg = GeminiAgent().build_config(cwd="/Users/sd/Code/proj", api_key="k")
    assert cfg.workspaces == ["/Users/sd/Code/proj"]


def test_gemini_build_config_model_default_flash_lite() -> None:
    # The council pins gemini-3.1-flash-lite rather than the SDK default
    # (gemini-3.5-flash), whose 5 req/min free-tier cap an agentic turn
    # exhausts. See DEFAULT_MODEL in gemini.py.
    cfg = GeminiAgent().build_config(cwd="/r", api_key="k")
    assert cfg.model == "gemini-3.1-flash-lite"


def test_gemini_build_config_model_override() -> None:
    cfg = GeminiAgent(model="gemini-3-pro").build_config(cwd="/r", api_key="k")
    assert cfg.model == "gemini-3-pro"


def test_gemini_run_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = asyncio.run(GeminiAgent().run(prompt="x", cwd="/tmp"))
    assert result.returncode == 2
    assert "GEMINI_API_KEY" in result.error
    assert result.output == ""


def test_gemini_run_refuses_hidden_cwd_that_widens_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A cwd under ~/.codex/agent-worktrees is hidden -- _non_hidden_workspace
    # widens it to $HOME, and the SDK backend has no sandbox to fence that.
    # run() must refuse before the SDK is ever imported/called.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cwd = tmp_path / ".codex" / "agent-worktrees" / "o" / "r"
    cwd.mkdir(parents=True)

    import sys

    monkeypatch.setitem(sys.modules, "google.antigravity", None)

    result = asyncio.run(GeminiAgent().run(prompt="x", cwd=str(cwd)))

    assert result.returncode != 0
    assert str(tmp_path) in result.error
    assert "home directory" in result.error
    assert result.output == ""


def test_gemini_run_nonhidden_cwd_does_not_hit_home_widening_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A non-hidden cwd must not trip the new guard. Stand in a fake
    # google.antigravity module so this stays offline: the fake Agent raises
    # immediately on entry, standing in for "fails later at the SDK stage".
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cwd = tmp_path / "Code" / "proj"
    cwd.mkdir(parents=True)

    import sys
    import types

    class _FakeBuiltinTools:
        VIEW_FILE = "view_file"
        CREATE_FILE = "create_file"
        EDIT_FILE = "edit_file"
        RUN_COMMAND = "run_command"

        @staticmethod
        def read_only():
            return frozenset({"view_file"})

    class _FakeCapabilitiesConfig:
        def __init__(self, enabled_tools):
            self.enabled_tools = enabled_tools

    class _FakeLocalAgentConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _FakeAgent:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            raise RuntimeError("fake SDK stage reached -- no network involved")

        async def __aexit__(self, *_exc_info):
            return False

    fake_module = types.ModuleType("google.antigravity")
    for attr, value in {
        "Agent": _FakeAgent,
        "BuiltinTools": _FakeBuiltinTools,
        "CapabilitiesConfig": _FakeCapabilitiesConfig,
        "LocalAgentConfig": _FakeLocalAgentConfig,
    }.items():
        setattr(fake_module, attr, value)
    monkeypatch.setitem(sys.modules, "google.antigravity", fake_module)

    result = asyncio.run(GeminiAgent().run(prompt="x", cwd=str(cwd)))

    assert "home directory" not in result.error
    assert "fake SDK stage reached" in result.error


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="live test: requires GEMINI_API_KEY in the environment",
)
def test_gemini_run_live_reads_workspace_read_only(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("SECRET_MARKER_42\nsecond\n", encoding="utf-8")
    result = asyncio.run(
        GeminiAgent().run(
            prompt=(
                "Read marker.txt in the workspace and reply with ONLY its first line."
            ),
            cwd=str(tmp_path),
        )
    )
    assert result.returncode == 0, result.error
    assert "SECRET_MARKER_42" in result.output
    # read-only: the agent must not have created anything in the workspace
    assert sorted(p.name for p in tmp_path.iterdir()) == ["marker.txt"]


def test_agent_default_roles() -> None:
    assert CodexAgent.default_role == "skeptic"
    assert GeminiAgent.default_role == "architect"


def test_agent_result_has_role_field_defaulting_neutral() -> None:
    r = AgentResult(agent="codex", output="x")
    assert r.role == "neutral"


# --- OpenCodeAgent ----------------------------------------------------------


def test_opencode_default_role() -> None:
    assert OpenCodeAgent.default_role == "neutral"


def test_opencode_build_command_shape() -> None:
    cmd = OpenCodeAgent().build_command(prompt="hello world", cwd="/repo/proj")
    assert cmd[0] == "opencode"
    assert cmd[1] == "run"
    assert cmd[cmd.index("--agent") + 1] == "council"
    assert cmd[cmd.index("--format") + 1] == "json"
    assert cmd[cmd.index("--dir") + 1] == "/repo/proj"
    assert cmd[-1] == "hello world"


def test_opencode_build_command_default_model_is_v41_flash() -> None:
    cmd = OpenCodeAgent().build_command(prompt="x", cwd="/repo")
    assert cmd[cmd.index("-m") + 1] == "openrouter/deepseek/deepseek-v4.1-flash"


def test_opencode_build_command_with_model_override() -> None:
    cmd = OpenCodeAgent(model="openrouter/deepseek/deepseek-v4").build_command(
        prompt="x", cwd="/repo"
    )
    assert cmd[cmd.index("-m") + 1] == "openrouter/deepseek/deepseek-v4"


def test_opencode_build_command_dir_is_user_cwd() -> None:
    # opencode is a read-only peer in the user's repo: --dir points at the
    # caller's cwd so it can read the project files a task references (e.g.
    # a README the brainstorm consults). Project-config shadowing is blocked
    # via OPENCODE_DISABLE_PROJECT_CONFIG in the subprocess env (verified
    # not to disable file reads), not by hiding the repo from opencode.
    cmd = OpenCodeAgent().build_command(prompt="x", cwd="/repo/proj")
    assert cmd[cmd.index("--dir") + 1] == "/repo/proj"


def test_opencode_ensure_sandbox_writes_council_md(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    result = _ensure_sandbox_home(sandbox)
    assert result == sandbox
    assert sandbox.stat().st_mode & 0o777 == 0o700
    council = sandbox / ".config" / "opencode" / "agents" / "council.md"
    assert council.read_text(encoding="utf-8") == COUNCIL_AGENT_MD


def test_opencode_ensure_sandbox_is_idempotent(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    _ensure_sandbox_home(sandbox)
    _ensure_sandbox_home(sandbox)  # second call must not raise
    council = sandbox / ".config" / "opencode" / "agents" / "council.md"
    assert council.exists()


def test_opencode_ensure_sandbox_writes_no_opencode_jsonc(tmp_path: Path) -> None:
    # Sandbox must NOT carry opencode.jsonc — that's what triggers the
    # instructions[] auto-load we're sandboxing to avoid.
    sandbox = tmp_path / "sandbox"
    _ensure_sandbox_home(sandbox)
    assert not (sandbox / ".config" / "opencode" / "opencode.jsonc").exists()
    assert not (sandbox / ".config" / "opencode" / "AGENTS.md").exists()


def test_build_opencode_config_openrouter_pins_throughput() -> None:
    # OpenRouter "lowest latency" account routing ranks providers by
    # first-token latency and parks reasoning models on a throughput-poor
    # backend (Parasail: 394s / 324s-to-first-content). Throughput-sorted
    # routing keeps the council off it. The model id is keyed WITHOUT the
    # "openrouter/" provider prefix (opencode's per-provider model id).
    cfg = _build_opencode_config("openrouter/deepseek/deepseek-v4-pro")
    assert cfg is not None
    assert cfg["$schema"] == "https://opencode.ai/config.json"
    model_cfg = cfg["provider"]["openrouter"]["models"]["deepseek/deepseek-v4-pro"]
    assert model_cfg["options"]["provider"] == {"sort": "throughput"}


def test_build_opencode_config_v41_flash_pins_provider_order() -> None:
    # Reproducible flavor: both backends serve V4.1 Flash at fp8 and pass the
    # account's zero-data-retention policy (DeepSeek's own endpoint does not).
    # Novita first -- in the 2026-09-15 probe (docs/adr/0001) Parasail was
    # rate-limited upstream on half its requests and 3-8x slower. Fallbacks
    # stay on so an unavailable backend degrades to the next, same quant, and
    # `only` keeps the fallback inside the pinned list: without it OpenRouter
    # proceeds past the order to any host, including ones below the output cap
    # (docs/adr/0002).
    cfg = _build_opencode_config("openrouter/deepseek/deepseek-v4.1-flash")
    assert cfg is not None
    model_cfg = cfg["provider"]["openrouter"]["models"]["deepseek/deepseek-v4.1-flash"]
    assert model_cfg["options"]["provider"] == {
        "order": ["novita", "parasail"],
        "only": ["novita", "parasail"],
        "allow_fallbacks": True,
    }


def test_build_opencode_config_v41_flash_pins_high_reasoning_effort() -> None:
    # opencode forwards `options.reasoning` verbatim as OpenRouter's
    # `reasoning` request object (captured on the wire 2026-09-15). S chose
    # high on 2026-09-17 (docs/adr/0003): ADR 0001 picked medium partly to
    # stay under opencode's then-fixed 32K max_tokens, and ADR 0002 raised
    # that cap to 256K, so the guard reason is gone.
    cfg = _build_opencode_config("openrouter/deepseek/deepseek-v4.1-flash")
    assert cfg is not None
    model_cfg = cfg["provider"]["openrouter"]["models"]["deepseek/deepseek-v4.1-flash"]
    assert model_cfg["options"]["reasoning"] == {"effort": "high"}


def test_build_opencode_config_unmeasured_model_sends_no_reasoning_effort() -> None:
    # The effort knob is per-model evidence, not a default: on V4 Flash the
    # same parameter was accepted and ignored (seat-eval 2026-08-27 probe).
    cfg = _build_opencode_config("openrouter/deepseek/deepseek-v4-pro")
    assert cfg is not None
    model_cfg = cfg["provider"]["openrouter"]["models"]["deepseek/deepseek-v4-pro"]
    assert "reasoning" not in model_cfg["options"]


def test_build_opencode_config_arms_opencodes_own_stream_watchdog() -> None:
    # opencode ships wrapSSE but only arms it when a provider sets chunkTimeout,
    # and there is no default on the openai-compatible path OpenRouter uses
    # (anomalyco/opencode#37580). Unarmed, a silently dropped SSE stream hangs
    # with the connection still open: nothing throws, so opencode's own
    # error-driven retries never fire and our idle timer burns its whole window
    # before killing a seat we then refuse to retry. Diagnosed 2026-07-29.
    cfg = _build_opencode_config("openrouter/deepseek/deepseek-v4-pro")
    assert cfg is not None
    ms = cfg["provider"]["openrouter"]["options"]["chunkTimeout"]
    assert ms == OPENROUTER_CHUNK_TIMEOUT_MS
    # MILLISECONDS here, seconds in our own constants -- the units differ across
    # that boundary, so pin the conversion, not just the number.
    assert ms == pytest.approx(90.0 * 1000)
    # opencode must notice a dead stream BEFORE our backstop does, or arming it
    # changes nothing about how long a hung seat costs the round.
    assert ms / 1000 < OPENCODE_IDLE_TIMEOUT_S


def test_build_opencode_config_pins_small_model() -> None:
    # opencode auto-selects a "small model" for its auxiliary calls (session
    # title, summarize, classify) when small_model is unset, matching provider
    # model ids against its embedded /\b(nano|flash|lite|mini|haiku|small|fast)\b/
    # regex -- which lands on Claude Haiku 4.5 in the OpenRouter-only sandbox,
    # billing an unintended Anthropic call per council run (OpenRouter activity
    # 2026-06-22: App=OpenCode, anthropic/claude-haiku-4.5, 1254->11 tokens,
    # served via Amazon Bedrock as OpenRouter's upstream). Pin small_model so
    # auxiliary calls bill a single known model, not whatever the regex picks.
    cfg = _build_opencode_config("openrouter/deepseek/deepseek-v4-pro")
    assert cfg is not None
    assert cfg["small_model"] == DEFAULT_SMALL_MODEL
    # The whole point is to keep opencode off the Haiku auto-pick.
    assert "haiku" not in cfg["small_model"].lower()
    assert "anthropic" not in cfg["small_model"].lower()


def test_build_opencode_config_non_openrouter_returns_none() -> None:
    # Provider routing is an OpenRouter concept; a non-OpenRouter model has
    # nothing to pin, so no config is emitted (no file gets written).
    assert _build_opencode_config("ollama/llama3") is None


def test_opencode_ensure_sandbox_writes_throughput_config_for_model(
    tmp_path: Path,
) -> None:
    # With a model, the sandbox carries opencode.json pinning throughput
    # routing — AND still writes council.md (the two are independent).
    sandbox = tmp_path / "sandbox"
    _ensure_sandbox_home(sandbox, model="openrouter/deepseek/deepseek-v4-pro")
    council = sandbox / ".config" / "opencode" / "agents" / "council.md"
    assert council.read_text(encoding="utf-8") == COUNCIL_AGENT_MD
    config_path = sandbox / ".config" / "opencode" / "opencode.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    routing = cfg["provider"]["openrouter"]["models"]["deepseek/deepseek-v4-pro"][
        "options"
    ]["provider"]
    assert routing == {"sort": "throughput"}
    # The written config also pins small_model so opencode's auxiliary
    # title/summarize calls don't auto-select Haiku (see config test above).
    assert cfg["small_model"] == DEFAULT_SMALL_MODEL


def test_opencode_ensure_sandbox_no_model_writes_no_config(tmp_path: Path) -> None:
    # The model-less call (test/utility harnesses) keeps the original
    # behaviour: council.md only, no opencode.json.
    sandbox = tmp_path / "sandbox"
    _ensure_sandbox_home(sandbox)
    assert not (sandbox / ".config" / "opencode" / "opencode.json").exists()


def _make_npm_cache(sandbox: Path, size_bytes: int) -> Path:
    """Create sandbox/.npm/_cacache/content-v2 holding a single blob of the
    given size, mimicking npm's content-addressed cache layout."""
    blob_dir = sandbox / ".npm" / "_cacache" / "content-v2" / "sha512" / "ab" / "cd"
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / "deadbeef").write_bytes(b"\0" * size_bytes)
    return sandbox / ".npm"


def test_npm_cache_needs_prune_missing_dir_is_false(tmp_path: Path) -> None:
    # No .npm cache yet (fresh sandbox) -> nothing to prune, no spurious work.
    assert _npm_cache_needs_prune(tmp_path, threshold=100) is False


def test_npm_cache_needs_prune_under_threshold_is_false(tmp_path: Path) -> None:
    _make_npm_cache(tmp_path, size_bytes=50)
    assert _npm_cache_needs_prune(tmp_path, threshold=100) is False


def test_npm_cache_needs_prune_over_threshold_is_true(tmp_path: Path) -> None:
    _make_npm_cache(tmp_path, size_bytes=200)
    assert _npm_cache_needs_prune(tmp_path, threshold=100) is True


class _FakeVerifyProc:
    """asyncio subprocess stand-in for `npm cache verify`: records nothing,
    just exits 0 on wait()."""

    def __init__(self) -> None:
        self.returncode = 0

    async def wait(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_prune_npm_cache_skips_when_under_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under threshold: the common path must NOT shell out to npm at all.
    _make_npm_cache(tmp_path, size_bytes=50)
    spawned: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def _spy_spawn(*cmd: str, **kwargs: object) -> _FakeVerifyProc:
        spawned.append((cmd, kwargs))
        return _FakeVerifyProc()

    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _spy_spawn)
    ran = await _prune_npm_cache(tmp_path, threshold=100)
    assert ran is False
    assert spawned == [], "npm must not be invoked under the size threshold"


@pytest.mark.asyncio
async def test_prune_npm_cache_runs_verify_when_over_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Over threshold: invoke `npm cache verify` targeting the sandbox's own
    # cache dir (--cache), never the user's global ~/.npm.
    npm_dir = _make_npm_cache(tmp_path, size_bytes=200)
    monkeypatch.setenv("DATABASE_URL", "postgres://secret")
    spawned: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def _spy_spawn(*cmd: str, **kwargs: object) -> _FakeVerifyProc:
        spawned.append((cmd, kwargs))
        return _FakeVerifyProc()

    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _spy_spawn)
    ran = await _prune_npm_cache(tmp_path, threshold=100)
    assert ran is True
    assert len(spawned) == 1
    cmd, kwargs = spawned[0]
    assert cmd[0] == "npm"
    assert cmd[1:3] == ("cache", "verify")
    assert cmd[cmd.index("--cache") + 1] == str(npm_dir)
    assert isinstance(kwargs["env"], dict)
    env = cast(dict[str, str], kwargs["env"])
    assert env["HOME"] == str(tmp_path)
    assert "DATABASE_URL" not in env


@pytest.mark.asyncio
async def test_prune_npm_cache_swallows_missing_npm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # npm absent from PATH: prune is best-effort and must never break a run.
    _make_npm_cache(tmp_path, size_bytes=200)

    async def _missing(*_a: object, **_kw: object) -> _FakeVerifyProc:
        raise FileNotFoundError("npm")

    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _missing)
    # Must not raise.
    ran = await _prune_npm_cache(tmp_path, threshold=100)
    assert ran is False


def test_council_md_instructs_headless_no_clarifying_questions() -> None:
    # Headless `opencode run` has stdin=DEVNULL: a clarifying question can
    # never be answered, and a mismatched/ambiguous prompt otherwise spirals
    # into ever-longer reasoning. The agent must assume-and-answer instead.
    # Collapse hand-wrap whitespace so the assertions don't couple to line
    # breaks (the source is manually wrapped at ~78 cols).
    low = " ".join(COUNCIL_AGENT_MD.lower().split())
    assert "non-interactiv" in low
    assert "do not ask" in low
    assert "assum" in low  # "assume" / "assumption"


def test_opencode_extract_text_concatenates_text_events() -> None:
    stdout = (
        b'{"type":"step_start","part":{"type":"step-start"}}\n'
        b'{"type":"text","part":{"type":"text","text":"hello "}}\n'
        b'{"type":"text","part":{"type":"text","text":"world"}}\n'
    )
    text, error = _extract_text_and_error(stdout)
    assert text == "hello world"
    assert error == ""


def test_opencode_extract_text_ignores_non_text_events() -> None:
    stdout = (
        b'{"type":"step_start","part":{}}\n'
        b'{"type":"reasoning","part":{"text":"thinking out loud"}}\n'
        b'{"type":"text","part":{"type":"text","text":"final"}}\n'
    )
    text, error = _extract_text_and_error(stdout)
    assert text == "final"
    assert error == ""


def test_opencode_extract_text_handles_empty_stdout() -> None:
    assert _extract_text_and_error(b"") == ("", "")


def test_opencode_extract_text_skips_malformed_json_lines() -> None:
    stdout = (
        b"not json\n"
        b'{"type":"text","part":{"type":"text","text":"ok"}}\n'
        b"another bad line\n"
    )
    text, _ = _extract_text_and_error(stdout)
    assert text == "ok"


def test_opencode_extract_collects_error_events_string_form() -> None:
    stdout = (
        b'{"type":"text","part":{"type":"text","text":"partial "}}\n'
        b'{"type":"error","error":"invalid api key"}\n'
    )
    text, error = _extract_text_and_error(stdout)
    assert text == "partial"
    assert error == "invalid api key"


def test_opencode_extract_collects_error_events_dict_form() -> None:
    stdout = (
        b'{"type":"error","error":{"name":"AuthError","message":"401 from provider"}}\n'
    )
    _, error = _extract_text_and_error(stdout)
    assert "401 from provider" in error


def test_opencode_extract_collects_error_events_in_part() -> None:
    stdout = b'{"type":"error","part":{"message":"agent not found: bogus"}}\n'
    _, error = _extract_text_and_error(stdout)
    assert error == "agent not found: bogus"


def test_opencode_extract_prefers_nested_data_message() -> None:
    # opencode serializes a NamedError as {name, data:{message}}. The bare
    # `name` ("UnknownError") is generic; the actionable text lives in
    # data.message (e.g. the doom-loop rule denial that aborts headless runs,
    # 2026-06-08). Surface the nested message so the council shows WHY, not
    # just "UnknownError".
    stdout = (
        b'{"type":"error","error":{"name":"UnknownError",'
        b'"data":{"message":"rule prevents this tool call"}}}\n'
    )
    _, error = _extract_text_and_error(stdout)
    assert error == "rule prevents this tool call"


def test_opencode_extract_falls_back_when_data_message_unusable() -> None:
    # A non-string (or blank) data.message must NOT suppress a usable top-level
    # message/name (codex review, 2026-06-08): extraction prefers the first
    # non-empty STRING among data.message, message, name. Here data.message is
    # a dict, so it falls through to the name instead of dumping the raw dict.
    stdout = (
        b'{"type":"error","error":{"name":"AuthError",'
        b'"data":{"message":{"unexpected":"shape"}}}}\n'
    )
    _, error = _extract_text_and_error(stdout)
    assert error == "AuthError"


def test_opencode_resolve_key_reads_environ() -> None:
    environ = {"OPENROUTER_API_KEY": "sk-from-shell"}
    assert _resolve_openrouter_key(environ=environ) == "sk-from-shell"


def test_opencode_resolve_key_returns_none_when_missing() -> None:
    assert _resolve_openrouter_key(environ={}) is None


def test_opencode_resolve_key_treats_whitespace_as_missing() -> None:
    environ = {"OPENROUTER_API_KEY": "   "}  # whitespace only
    assert _resolve_openrouter_key(environ=environ) is None


class _FakeProc:
    """Minimal stand-in for an asyncio subprocess so run() can be exercised
    without spawning opencode. Supports both the streaming path (idle stub)
    and the legacy communicate path (pre-change code)."""

    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = 0

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        return (b"", b"")


@pytest.mark.asyncio
async def test_opencode_run_reports_idle_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When opencode's stream stalls (no output for the idle window),
    # communicate_lines_or_kill returns idle_timed_out=True. run() must turn
    # that into a distinct, non-zero result that names the stall -- a dropped
    # or hung job must be VISIBLE, not a silent empty success. 124 is the
    # conventional "command timed out" exit code (matches GNU timeout).
    async def _fake_spawn(*_a: object, **_kw: object) -> _FakeProc:
        return _FakeProc()

    async def _stalled_lines(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        return (b"", b"", True)

    async def _noidle_communicate(
        proc: object, *_a: object, **_kw: object
    ) -> tuple[bytes, bytes]:
        return (b"", b"")

    monkeypatch.setattr(opencode_mod, "_resolve_openrouter_key", lambda: "fake-key")
    monkeypatch.setattr(opencode_mod, "_ensure_sandbox_home", lambda *a, **k: tmp_path)
    monkeypatch.setattr(opencode_mod, "_build_subprocess_env", lambda *a, **k: {})
    # run() now writes a raw-stream capture on failure; keep it off the real
    # ~/.cache during tests (an idle stall is a capture-worthy failure).
    monkeypatch.setattr(opencode_mod, "_DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    # Pre-change run() calls communicate_or_kill (no idle signal); post-change
    # it calls communicate_lines_or_kill. Stub both so the test pins behaviour
    # across the refactor.
    monkeypatch.setattr(
        opencode_mod, "communicate_or_kill", _noidle_communicate, raising=False
    )
    monkeypatch.setattr(
        opencode_mod, "communicate_lines_or_kill", _stalled_lines, raising=False
    )

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert result.returncode == 124, (
        "an idle stall must map to a distinct non-zero returncode (124), got "
        f"{result.returncode}"
    )
    assert result.returncode == opencode_mod.OPENCODE_IDLE_TIMEOUT_RC
    low = result.error.lower()
    assert "no output" in low or "stall" in low, (
        f"idle-timeout error must name the stall; got: {result.error!r}"
    )
    assert "240" in result.error, "error should state the idle window that elapsed"
    assert opencode_mod.OPENCODE_IDLE_TIMEOUT_S == 240.0


# --- empty-output retry (intermittent upstream empty completion) ------------
# Evidence (2026-06-22): deepseek-v4-pro via OpenRouter twice emitted only a
# `step_start` then ended the turn with no content (~37s, clean exit), which
# run() surfaces as OPENCODE_NO_OUTPUT_RC and the council then drops from later
# rounds. A single re-invocation recovers most of these; we retry ONLY the
# empty-output case (not an idle stall, which already burned its window, nor a
# genuine error, which won't fix itself).


def _empty_output_stub(calls: list[int]):
    """communicate_lines_or_kill stub: record each call, return empty stdout
    (no text/error events) with idle_timed_out=False -> empty_output path."""

    async def _lines(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        calls.append(1)
        return (b"", b"", False)

    return _lines


def _patch_opencode_run_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lines_stub: object
) -> None:
    async def _fake_spawn(*_a: object, **_kw: object) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(opencode_mod, "_resolve_openrouter_key", lambda: "fake-key")
    monkeypatch.setattr(opencode_mod, "_ensure_sandbox_home", lambda *a, **k: tmp_path)
    monkeypatch.setattr(opencode_mod, "_build_subprocess_env", lambda *a, **k: {})
    monkeypatch.setattr(opencode_mod, "_DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(
        opencode_mod, "communicate_lines_or_kill", lines_stub, raising=False
    )


@pytest.mark.asyncio
async def test_opencode_run_launches_pinned_model_with_output_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The seat, not just _build_subprocess_env, must hand opencode the cap
    # (docs/adr/0002): run() with the real env builder, capture the spawn env.
    envs: list[dict[str, str]] = []

    async def _spawn(*_a: object, **kw: object) -> _FakeProc:
        envs.append(cast(dict[str, str], kw["env"]))
        return _FakeProc()

    _patch_opencode_run_seams(monkeypatch, tmp_path, _empty_output_stub([]))
    monkeypatch.setattr(opencode_mod, "_build_subprocess_env", _build_subprocess_env)
    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setenv("CODE_QUORUM_OPENCODE_RELAY", "0")

    await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert envs
    assert {e.get("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX") for e in envs} == {"256000"}


@pytest.mark.asyncio
async def test_opencode_run_retries_empty_output_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # First attempt returns an empty turn; the retry returns a real text event.
    # run() must return the successful (rc 0) result, not the empty one.
    calls: list[int] = []
    text_event = b'{"type":"text","part":{"text":"recovered answer"}}\n'

    async def _lines(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        calls.append(1)
        return (b"", b"", False) if len(calls) == 1 else (text_event, b"", False)

    _patch_opencode_run_seams(monkeypatch, tmp_path, _lines)

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert len(calls) == 2, "an empty turn must trigger exactly one retry"
    assert result.returncode == 0, (
        f"retry should recover to success, got {result.returncode}"
    )
    assert result.output == "recovered answer"
    assert opencode_mod.OPENCODE_EMPTY_OUTPUT_RETRIES == 1


@pytest.mark.asyncio
async def test_opencode_run_repairs_and_prunes_existing_debug_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir(mode=0o755)
    for index in range(opencode_mod.OPENCODE_DEBUG_MAX_FILES + 2):
        path = debug_dir / f"opencode-empty_output-{index:02d}.txt"
        path.write_text("old", encoding="utf-8")
        os.utime(path, (index + 1, index + 1))

    async def _success(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        return (b'{"type":"text","part":{"text":"ok"}}\n', b"", False)

    _patch_opencode_run_seams(monkeypatch, tmp_path, _success)

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert result.returncode == 0
    assert debug_dir.stat().st_mode & 0o777 == 0o700
    assert (
        len(list(debug_dir.glob("opencode-*.txt")))
        == opencode_mod.OPENCODE_DEBUG_MAX_FILES
    )


@pytest.mark.asyncio
async def test_opencode_run_gives_up_after_retry_when_still_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both attempts empty: run() must stop after 1 retry (bounded) and surface
    # the empty turn as OPENCODE_NO_OUTPUT_RC, not loop forever.
    calls: list[int] = []
    _patch_opencode_run_seams(monkeypatch, tmp_path, _empty_output_stub(calls))

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert len(calls) == 1 + opencode_mod.OPENCODE_EMPTY_OUTPUT_RETRIES == 2
    assert result.returncode == opencode_mod.OPENCODE_NO_OUTPUT_RC


@pytest.mark.asyncio
async def test_opencode_run_does_not_retry_idle_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An idle stall already burned the full idle window; retrying would double
    # the wall-clock for a likely-hung stream. It must NOT be retried.
    calls: list[int] = []

    async def _lines(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        calls.append(1)
        return (b"", b"", True)

    _patch_opencode_run_seams(monkeypatch, tmp_path, _lines)

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert len(calls) == 1, "idle stall must not be retried"
    assert result.returncode == opencode_mod.OPENCODE_IDLE_TIMEOUT_RC


# --- run(): recorded-source failure hint (item 4) ----------------------------


@pytest.mark.asyncio
async def test_opencode_run_recorded_source_generic_failure_gets_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _error_lines(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        return (b'{"type":"error","error":"invalid api key"}\n', b"", False)

    _patch_opencode_run_seams(monkeypatch, tmp_path, _error_lines)

    result = await OpenCodeAgent(model_source="recorded").run(prompt="x", cwd="/tmp")

    assert result.returncode == 1
    assert "quorum setup-models" in result.error
    assert opencode_mod.DEFAULT_MODEL in result.error


@pytest.mark.asyncio
async def test_opencode_run_shipped_source_generic_failure_gets_no_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _error_lines(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        return (b'{"type":"error","error":"invalid api key"}\n', b"", False)

    _patch_opencode_run_seams(monkeypatch, tmp_path, _error_lines)

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")  # default: "shipped"

    assert result.returncode == 1
    assert "quorum setup-models" not in result.error


@pytest.mark.asyncio
async def test_opencode_run_recorded_source_idle_timeout_gets_no_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rc 124 (idle stall) is definitely not a model problem -- gated away
    # from the hint even when the model came from a recorded choice.
    async def _lines(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        return (b"", b"", True)

    _patch_opencode_run_seams(monkeypatch, tmp_path, _lines)

    result = await OpenCodeAgent(model_source="recorded").run(prompt="x", cwd="/tmp")

    assert result.returncode == opencode_mod.OPENCODE_IDLE_TIMEOUT_RC
    assert "quorum setup-models" not in result.error


@pytest.mark.asyncio
async def test_opencode_run_recorded_source_no_api_key_gets_no_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rc 2 (no API key) is definitely not a model problem -- gated away
    # from the hint even when the model came from a recorded choice.
    monkeypatch.setattr(opencode_mod, "_resolve_openrouter_key", lambda: None)

    result = await OpenCodeAgent(model_source="recorded").run(prompt="x", cwd="/tmp")

    assert result.returncode == 2
    assert "quorum setup-models" not in result.error
    assert result.error == (
        "OPENROUTER_API_KEY not found. Export it in your shell environment "
        "(get a key at https://openrouter.ai)."
    )


def test_opencode_build_subprocess_env_pins_home_and_key(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    env = _build_subprocess_env(
        sandbox,
        "sk-or-v1-test",
        base={
            "PATH": "/usr/bin",
            "ALL_PROXY": "socks5://proxy.test:1080",
            "NODE_EXTRA_CA_CERTS": "/certs/company.pem",
        },
    )
    assert env["HOME"] == str(sandbox)
    assert env["OPENROUTER_API_KEY"] == "sk-or-v1-test"
    assert env["PATH"] == "/usr/bin"
    assert env["ALL_PROXY"] == "socks5://proxy.test:1080"
    assert env["NODE_EXTRA_CA_CERTS"] == "/certs/company.pem"


def test_opencode_subprocess_env_raises_output_cap_for_pinned_model(
    tmp_path: Path,
) -> None:
    # opencode sends a fixed max_tokens of 32000; only this env var raises it
    # (per-model limit.output does not). Wire capture 2026-09-17, docs/adr/0002.
    env = _build_subprocess_env(
        tmp_path, "k", base={}, model="openrouter/deepseek/deepseek-v4.1-flash"
    )
    assert env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "256000"
    unpinned = _build_subprocess_env(
        tmp_path, "k", base={}, model="openrouter/deepseek/deepseek-v4-pro"
    )
    assert "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX" not in unpinned


def test_opencode_subprocess_env_strips_unrelated_secret_vars(tmp_path: Path) -> None:
    research_env = {
        name: "research-provider-secret" for name in RESEARCH_PROVIDER_ENV_VARS
    }
    env = _build_subprocess_env(
        tmp_path,
        "sk-from-resolver",
        base={
            "PATH": "/bin",
            "GH_TOKEN": "gh-secret",
            "GITHUB_TOKEN": "github-secret",
            "GEMINI_API_KEY": "gemini-secret",
            "OPENAI_API_KEY": "openai-secret",
            "OPENROUTER_API_KEY": "shell-openrouter-secret",
            "DATABASE_URL": "postgres://secret",
            "SAFE_FLAG": "ok",
            **research_env,
        },
    )

    assert "SAFE_FLAG" not in env
    assert "DATABASE_URL" not in env
    assert env["OPENROUTER_API_KEY"] == "sk-from-resolver"
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        assert key not in env
    assert not set(RESEARCH_PROVIDER_ENV_VARS) & env.keys()


def test_opencode_build_subprocess_env_strips_xdg_config(tmp_path: Path) -> None:
    base = {
        "PATH": "/usr/bin",
        "XDG_CONFIG_HOME": "/usr/me/.config",
        "XDG_DATA_HOME": "/usr/me/.local/share",
        "XDG_CACHE_HOME": "/usr/me/.cache",
        "XDG_STATE_HOME": "/usr/me/.local/state",
    }
    env = _build_subprocess_env(tmp_path / "sandbox", "k", base=base)
    for var in (
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
    ):
        assert var not in env, f"{var} must be stripped from sandbox env"


def test_opencode_build_subprocess_env_strips_opencode_prefix(tmp_path: Path) -> None:
    base = {
        "PATH": "/usr/bin",
        "OPENCODE_CONFIG": "/somewhere/else/opencode.jsonc",
        "OPENCODE_HOME": "/somewhere/else",
        "OPENCODE_DIR": "/somewhere/else",
    }
    env = _build_subprocess_env(tmp_path / "sandbox", "k", base=base)
    for var in ("OPENCODE_CONFIG", "OPENCODE_HOME", "OPENCODE_DIR"):
        assert var not in env, f"{var} must be stripped from sandbox env"


def test_opencode_council_md_allows_read_only_tools() -> None:
    # opencode is a read-only council peer: it may read/glob/grep to ground
    # its reasoning in the repo, but must not edit, write, or run shell
    # commands. Enforcement is via `permission:` (the documented key);
    # `tools:` is treated as non-enforcing options by opencode 1.15.
    assert "permission:" in COUNCIL_AGENT_MD
    # read uses the object form (see the .env-deny test); glob/list stay flat.
    # grep is intentionally NOT here -- see test_opencode_council_md_denies_grep.
    assert "read:" in COUNCIL_AGENT_MD
    for tool in ("glob", "list"):
        assert f"{tool}: allow" in COUNCIL_AGENT_MD
    for tool in ("bash", "edit", "write", "webfetch", "task", "skill"):
        assert f"{tool}: deny" in COUNCIL_AGENT_MD


def test_opencode_council_md_denies_grep() -> None:
    # grep matches on the search PATTERN (not a file path) and scans the whole
    # tree, so .env contents can't be path-denied for grep the way they are
    # for read (codex review P1 follow-up). Verified 2026-06-02: grep returned
    # a planted .env canary. grep is therefore denied outright; read+glob+list
    # still let the council ground itself in the repo.
    assert "grep: deny" in COUNCIL_AGENT_MD
    assert "grep: allow" not in COUNCIL_AGENT_MD


def test_opencode_council_md_denies_env_file_reads() -> None:
    # Codex review P1 (2026-06-02): a flat `read: allow` scalar REPLACES
    # opencode's shipped default read object, which denies *.env / *.env.*.
    # Since council read-contents are sent to OpenRouter, that would expose
    # repo secrets. The MD must use read's object form to restore the deny.
    assert '"*.env": deny' in COUNCIL_AGENT_MD
    assert '"*.env.*": deny' in COUNCIL_AGENT_MD


def test_opencode_council_md_has_wildcard_deny() -> None:
    # Omitted permissions remain allowed by opencode default. The catch-all
    # "*: deny" closes that gap. Specific rules re-allow the read-only tools
    # above it because a specific rule overrides a wildcard. The behavior was
    # verified against opencode 1.15.13.
    assert '"*": deny' in COUNCIL_AGENT_MD


def test_opencode_council_md_allows_doom_loop() -> None:
    # opencode's doom-loop guard (processor.ts, DOOM_LOOP_THRESHOLD=3) fires
    # when the last 3 tool calls are byte-identical. Kimi K2.6 intermittently
    # gets stuck re-issuing an identical call and trips it (verified against
    # opencode source, 2026-06-08). The guard's permission defaults to `ask`,
    # which is unanswerable under headless `opencode run` (stdin=DEVNULL) --
    # opencode then raises a fatal UnknownError and the run exits 1 with empty
    # output. `deny` aborts the same way; only `allow` lets the run complete.
    # The 900s AGENT_TIMEOUT_S (council.py) is the real backstop.
    assert "doom_loop: allow" in COUNCIL_AGENT_MD


def test_opencode_council_md_prose_invites_read_only_grounding() -> None:
    # Kimi-specific guidance (codex review follow-up, 2026-06-02): state the
    # tool reality so the model uses Read instead of trying grep, getting
    # blocked, and confabulating. A/B-tested: current prose fabricated denied
    # .env contents in 1/2 runs; this prose was grounded in 2/2. The prose
    # must (a) not claim "no filesystem", (b) direct the Read tool, (c) name
    # grep as unavailable, and (d) forbid inventing file contents. Normalize
    # whitespace so line wraps don't hide a phrase.
    normalized = " ".join(COUNCIL_AGENT_MD.lower().split())
    assert "no filesystem" not in normalized
    assert "read tool" in normalized
    assert "grep" in normalized
    assert "invent" in normalized or "guess" in normalized


def test_opencode_build_subprocess_env_sets_project_disable_flags(
    tmp_path: Path,
) -> None:
    # Project-config discovery must be explicitly disabled. Otherwise, a
    # target-repo opencode.json(c) shadows the sandboxed quorum. These flags
    # survive the OPENCODE_* strip.
    env = _build_subprocess_env(tmp_path / "sandbox", "k", base={"PATH": "/usr/bin"})
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert env["OPENCODE_PURE"] == "1"


def test_opencode_build_subprocess_env_disable_flags_overwrite_caller(
    tmp_path: Path,
) -> None:
    # Even if the caller exported OPENCODE_DISABLE_PROJECT_CONFIG=0, we
    # must force it to 1 -- the sandbox guarantee can't be opted out of.
    base = {
        "PATH": "/usr/bin",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "0",
        "OPENCODE_PURE": "false",
    }
    env = _build_subprocess_env(tmp_path / "sandbox", "k", base=base)
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert env["OPENCODE_PURE"] == "1"


def test_opencode_ensure_sandbox_wipes_stale_agents_md(tmp_path: Path) -> None:
    # A stale AGENTS.md from a prior code-quorum version persists across runs.
    # opencode loads it automatically, which defeats the sandbox.
    # _ensure_sandbox_home must remove it.
    sandbox = tmp_path / "sandbox"
    config_dir = sandbox / ".config" / "opencode"
    config_dir.mkdir(parents=True)
    (config_dir / "AGENTS.md").write_text("STALE INSTRUCTIONS", encoding="utf-8")
    (config_dir / "opencode.jsonc").write_text(
        '{"instructions":["leaked"]}', encoding="utf-8"
    )
    _ensure_sandbox_home(sandbox)
    assert not (config_dir / "AGENTS.md").exists()
    assert not (config_dir / "opencode.jsonc").exists()
    # And council.md is freshly written
    assert (config_dir / "agents" / "council.md").exists()


def test_opencode_ensure_sandbox_wipes_stale_other_agents(tmp_path: Path) -> None:
    # Pre-existing agents/foo.md must also be cleared so a user-injected
    # agent can't replace `council` if it had the same name in a prior
    # version of our COUNCIL_AGENT_MD layout.
    sandbox = tmp_path / "sandbox"
    agents_dir = sandbox / ".config" / "opencode" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "stale-agent.md").write_text("STALE", encoding="utf-8")
    _ensure_sandbox_home(sandbox)
    assert not (agents_dir / "stale-agent.md").exists()
    assert (agents_dir / "council.md").exists()


def test_opencode_ensure_sandbox_preserves_runtime_dirs(tmp_path: Path) -> None:
    # The wipe should ONLY touch .config/opencode. Runtime dirs
    # (.local/share, .cache, .local/state) hold opencode's session/log
    # DBs and must persist across calls so sessions aren't lost.
    sandbox = tmp_path / "sandbox"
    data_dir = sandbox / ".local" / "share" / "opencode"
    data_dir.mkdir(parents=True)
    (data_dir / "session.db").write_text("KEEPME", encoding="utf-8")
    _ensure_sandbox_home(sandbox)
    assert (data_dir / "session.db").read_text(encoding="utf-8") == "KEEPME"


def test_opencode_ensure_sandbox_wipes_legacy_opencode_dir(tmp_path: Path) -> None:
    # opencode 1.15 scans both $HOME/.config/opencode and the legacy
    # $HOME/.opencode layout. A stale .opencode/agents/
    # council.md must be cleared too, or it shadows our sandboxed agent.
    sandbox = tmp_path / "sandbox"
    legacy_dir = sandbox / ".opencode"
    legacy_agents = legacy_dir / "agents"
    legacy_agents.mkdir(parents=True)
    (legacy_agents / "council.md").write_text("STALE LEGACY", encoding="utf-8")
    (legacy_dir / "opencode.jsonc").write_text(
        '{"instructions":["leaked"]}', encoding="utf-8"
    )
    _ensure_sandbox_home(sandbox)
    assert not legacy_dir.exists(), ".opencode legacy dir must be wiped"
    # And the .config/opencode council.md is in its expected place
    assert (sandbox / ".config" / "opencode" / "agents" / "council.md").exists()


# --- opencode debug capture (intermittent silent empty-output) --------------
# Evidence (2026-06): opencode/openrouter intermittently returns an empty turn
# with returncode 0 -- the model reasons/calls tools but emits no final `text`
# event, so `_extract_text_and_error` yields "". format_rounds then renders a
# blank body (no [exit N]) and run_council drops the agent from later rounds,
# so the orchestrator silently says nothing about opencode. The raw stdout is
# the only thing that can tell us WHY the text is empty, but run() discards it.
# These cover the capture that preserves it for the next intermittent failure.


def test_capture_reason_flags_empty_output() -> None:
    assert (
        _capture_reason(text="", returncode=0, idle_timed_out=False) == "empty_output"
    )


def test_capture_reason_flags_nonzero_exit() -> None:
    reason = _capture_reason(text="some text", returncode=1, idle_timed_out=False)
    assert reason == "nonzero_exit"


def test_capture_reason_flags_idle_timeout_even_with_partial_text() -> None:
    # idle stall wins: a partial answer captured before the stall is still a
    # failure worth the raw dump.
    reason = _capture_reason(text="partial", returncode=124, idle_timed_out=True)
    assert reason == "idle_timeout"


def test_capture_reason_none_on_real_success() -> None:
    assert (
        _capture_reason(text="a real answer", returncode=0, idle_timed_out=False)
        is None
    )


def test_debug_enabled_default_on() -> None:
    assert _debug_enabled({}) is True


def test_debug_enabled_off_when_explicitly_disabled() -> None:
    for val in ("0", "false", "no", "off", "  0 "):
        assert _debug_enabled({"CODE_QUORUM_OPENCODE_DEBUG": val}) is False


def test_maybe_capture_debug_writes_file_on_empty_output(tmp_path: Path) -> None:
    # Formatting is no longer separately callable (_format_debug_capture was
    # folded into _maybe_capture_debug), so this is now also the coverage for
    # what was test_format_debug_capture_preserves_raw_streams_and_meta:
    # reason tag, model, and both raw streams verbatim in the written file.
    tmp_path.chmod(0o755)
    path = _maybe_capture_debug(
        cmd=["opencode", "run", "PRIVATE PROMPT"],
        model="openrouter/deepseek/deepseek-v4-pro",
        cwd="/x",
        duration_s=1.0,
        returncode=0,
        idle_timed_out=False,
        stdout=b'{"type":"reasoning"}\n{"type":"step-finish"}\n',
        stderr=b"some stderr noise",
        text="",
        json_error="",
        proc_pid=4242,
        debug_dir=tmp_path,
        environ={},
        stamp="STAMP",
    )
    assert path is not None
    assert path.exists()
    assert path.parent == tmp_path
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    content = path.read_text(encoding="utf-8")
    assert "PRIVATE PROMPT" not in content
    assert "<prompt omitted>" in content
    assert "reasoning" in content
    # raw streams must be preserved verbatim -- that's the whole point
    assert "step-finish" in content
    assert "some stderr noise" in content
    assert "empty_output" in content
    assert "deepseek-v4-pro" in content


def test_maybe_capture_debug_prunes_oldest_capture(tmp_path: Path) -> None:
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    oldest = debug_dir / "opencode-empty_output-oldest.txt"
    for index in range(opencode_mod.OPENCODE_DEBUG_MAX_FILES):
        path = debug_dir / f"opencode-empty_output-{index:02d}.txt"
        path.write_text("old", encoding="utf-8")
        os.utime(path, (index + 1, index + 1))
        if index == 0:
            oldest = path

    created = _maybe_capture_debug(
        cmd=["opencode", "run", "prompt"],
        model="m",
        cwd="/x",
        duration_s=1.0,
        returncode=1,
        idle_timed_out=False,
        stdout=b"raw",
        stderr=b"failure",
        text="partial",
        json_error="",
        proc_pid=1,
        debug_dir=debug_dir,
        environ={},
        stamp="new",
    )

    assert created is not None and created.exists()
    assert not oldest.exists()
    assert (
        len(list(debug_dir.glob("opencode-*.txt")))
        == opencode_mod.OPENCODE_DEBUG_MAX_FILES
    )


def test_maybe_capture_debug_skips_on_success(tmp_path: Path) -> None:
    path = _maybe_capture_debug(
        cmd=["opencode"],
        model="m",
        cwd="/x",
        duration_s=1.0,
        returncode=0,
        idle_timed_out=False,
        stdout=b"...",
        stderr=b"",
        text="a real answer",
        json_error="",
        proc_pid=1,
        debug_dir=tmp_path,
        environ={},
        stamp="STAMP",
    )
    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_maybe_capture_debug_writes_nothing_when_disabled(tmp_path: Path) -> None:
    path = _maybe_capture_debug(
        cmd=["opencode"],
        model="m",
        cwd="/x",
        duration_s=1.0,
        returncode=0,
        idle_timed_out=False,
        stdout=b"...",
        stderr=b"",
        text="",
        json_error="",
        proc_pid=1,
        debug_dir=tmp_path,
        environ={"CODE_QUORUM_OPENCODE_DEBUG": "0"},
        stamp="STAMP",
    )
    assert path is None
    assert list(tmp_path.iterdir()) == []


# --- empty-turn surfacing (no longer a silent success) ----------------------


def test_capture_reason_prefers_empty_over_nonzero() -> None:
    # After surfacing, an empty turn carries a non-zero rc; the capture must
    # still label it "empty_output" (the informative tag), not "nonzero_exit".
    from quorum.agents.opencode import OPENCODE_NO_OUTPUT_RC

    reason = _capture_reason(
        text="", returncode=OPENCODE_NO_OUTPUT_RC, idle_timed_out=False
    )
    assert reason == "empty_output"


def test_council_md_demands_written_answer_not_tool_call() -> None:
    # The exact failure: model ended its turn after tool calls with only
    # whitespace. The agent definition must insist on a final written answer.
    low = COUNCIL_AGENT_MD.lower()
    assert "written" in low or "do not end" in low or "never reply with only" in low


@pytest.mark.asyncio
async def test_opencode_run_surfaces_empty_output_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # opencode exited cleanly (rc 0) but the model produced only whitespace
    # text -> extracted output is "". This used to render as a silent blank
    # (rc 0) and get the agent dropped from later rounds invisibly. It must now
    # be a distinct, visible failure.
    class _EmptyProc:
        def __init__(self) -> None:
            self.pid = 4243
            self.returncode = 0

    async def _spawn(*_a: object, **_kw: object) -> _EmptyProc:
        return _EmptyProc()

    async def _whitespace_lines(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        # clean EOF (not idle), assistant emitted only whitespace text
        return (b'{"type":"text","part":{"text":"  "}}\n', b"", False)

    monkeypatch.setattr(opencode_mod, "_resolve_openrouter_key", lambda: "fake-key")
    monkeypatch.setattr(opencode_mod, "_ensure_sandbox_home", lambda *a, **k: tmp_path)
    monkeypatch.setattr(opencode_mod, "_build_subprocess_env", lambda *a, **k: {})
    monkeypatch.setattr(opencode_mod, "_DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setattr(
        opencode_mod, "communicate_lines_or_kill", _whitespace_lines, raising=False
    )

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert result.output == "", "whitespace-only text strips to empty"
    assert result.returncode == opencode_mod.OPENCODE_NO_OUTPUT_RC
    assert result.returncode != 0, "an empty turn must be a visible failure"
    assert "no" in result.error.lower() and result.error, (
        f"empty turn must carry an explanatory error; got {result.error!r}"
    )


# --- opencode.db text recovery (opencode #31435: run --format json drops the
# final text/step_finish events from stdout while persisting the full answer) ---


def _write_opencode_db(
    sandbox_home: Path,
    session_id: str,
    *,
    assistant_texts: list[str],
    user_text: str = "the prompt",
    include_noise: bool = True,
) -> Path:
    """Build a minimal opencode.db (message+part tables, as opencode 1.16
    persists) for the recovery tests."""
    db_path = sandbox_home / ".local" / "share" / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
            "time_created INTEGER, time_updated INTEGER, data TEXT);"
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, "
            "session_id TEXT, time_created INTEGER, time_updated INTEGER, data TEXT);"
        )
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?,?)",
            ("msg_user", session_id, 1, 1, json.dumps({"role": "user"})),
        )
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?,?)",
            (
                "prt_user",
                "msg_user",
                session_id,
                1,
                1,
                json.dumps({"type": "text", "text": user_text}),
            ),
        )
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?,?)",
            ("msg_asst", session_id, 2, 2, json.dumps({"role": "assistant"})),
        )
        for i, txt in enumerate(assistant_texts):
            conn.execute(
                "INSERT INTO part VALUES (?,?,?,?,?,?)",
                (
                    f"prt_a{i}",
                    "msg_asst",
                    session_id,
                    10 + i,
                    10 + i,
                    json.dumps({"type": "text", "text": txt}),
                ),
            )
        if include_noise:
            conn.execute(
                "INSERT INTO part VALUES (?,?,?,?,?,?)",
                (
                    "prt_reason",
                    "msg_asst",
                    session_id,
                    5,
                    5,
                    json.dumps({"type": "reasoning", "text": "internal thoughts"}),
                ),
            )
            conn.execute(
                "INSERT INTO part VALUES (?,?,?,?,?,?)",
                (
                    "prt_tool",
                    "msg_asst",
                    session_id,
                    6,
                    6,
                    json.dumps({"type": "tool", "tool": "read"}),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_opencode_session_id_from_stdout_returns_first() -> None:
    stdout = (
        b'{"type":"step_start","sessionID":"ses_abc","part":{}}\n'
        b'{"type":"text","sessionID":"ses_abc","part":{"type":"text","text":"hi"}}\n'
    )
    assert _session_id_from_stdout(stdout) == "ses_abc"


def test_opencode_session_id_from_stdout_none_when_absent() -> None:
    stdout = b'{"type":"text","part":{"type":"text","text":"hi"}}\n'
    assert _session_id_from_stdout(stdout) is None


def test_opencode_session_id_from_stdout_skips_malformed() -> None:
    stdout = b'not json\n{"type":"step_start","sessionID":"ses_z","part":{}}\n'
    assert _session_id_from_stdout(stdout) == "ses_z"


def test_opencode_recover_text_from_db_concatenates_assistant_text(
    tmp_path: Path,
) -> None:
    _write_opencode_db(tmp_path, "ses_1", assistant_texts=["Hello ", "world"])
    assert _recover_text_from_db(tmp_path, "ses_1") == "Hello world"


def test_opencode_recover_text_from_db_ignores_user_and_non_text(
    tmp_path: Path,
) -> None:
    _write_opencode_db(
        tmp_path, "ses_1", assistant_texts=["answer"], user_text="PROMPT"
    )
    recovered = _recover_text_from_db(tmp_path, "ses_1")
    assert recovered == "answer"
    assert recovered is not None
    assert "PROMPT" not in recovered and "internal thoughts" not in recovered


def test_opencode_recover_text_from_db_none_for_missing_db(tmp_path: Path) -> None:
    assert _recover_text_from_db(tmp_path, "ses_1") is None


def test_opencode_recover_text_from_db_none_for_unknown_session(
    tmp_path: Path,
) -> None:
    _write_opencode_db(tmp_path, "ses_1", assistant_texts=["answer"])
    assert _recover_text_from_db(tmp_path, "ses_missing") is None


def test_opencode_recover_text_from_db_none_for_no_session_id(tmp_path: Path) -> None:
    _write_opencode_db(tmp_path, "ses_1", assistant_texts=["answer"])
    assert _recover_text_from_db(tmp_path, None) is None


def test_opencode_recover_text_from_db_none_on_schema_drift(tmp_path: Path) -> None:
    db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE unrelated (id TEXT)")
    conn.commit()
    conn.close()
    assert _recover_text_from_db(tmp_path, "ses_1") is None


def test_opencode_recover_text_from_db_none_when_text_blank(tmp_path: Path) -> None:
    _write_opencode_db(
        tmp_path, "ses_1", assistant_texts=["   ", ""], include_noise=False
    )
    assert _recover_text_from_db(tmp_path, "ses_1") is None


@pytest.mark.asyncio
async def test_opencode_run_recovers_dropped_text_from_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # opencode #31435: stdout carries only step_start (text/step_finish dropped),
    # but the full answer is in opencode.db. The adapter must recover it and
    # report success instead of a spurious rc125 empty-output failure.
    _write_opencode_db(
        tmp_path, "ses_drop", assistant_texts=["the full ", "recovered answer"]
    )

    class _Proc:
        def __init__(self) -> None:
            self.pid = 5555
            self.returncode = 0

    async def _spawn(*_a: object, **_kw: object) -> _Proc:
        return _Proc()

    async def _only_step_start(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        return (
            b'{"type":"step_start","sessionID":"ses_drop",'
            b'"part":{"type":"step-start"}}\n',
            b"",
            False,
        )

    monkeypatch.setattr(opencode_mod, "_resolve_openrouter_key", lambda: "fake-key")
    monkeypatch.setattr(opencode_mod, "_ensure_sandbox_home", lambda *a, **k: tmp_path)
    monkeypatch.setattr(opencode_mod, "_build_subprocess_env", lambda *a, **k: {})
    monkeypatch.setattr(opencode_mod, "_DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setattr(
        opencode_mod, "communicate_lines_or_kill", _only_step_start, raising=False
    )

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert result.output == "the full recovered answer"
    assert result.returncode == 0, "recovered text must clear the empty-output failure"


def test_opencode_session_id_from_stdout_skips_non_dict_json() -> None:
    # A line that is valid JSON but a scalar/array (not an object) must be
    # skipped, not crash with AttributeError on .get().
    stdout = b'null\n[]\n42\n{"type":"step_start","sessionID":"ses_q","part":{}}\n'
    assert _session_id_from_stdout(stdout) == "ses_q"


def test_opencode_extract_text_skips_non_dict_json() -> None:
    # Same hardening for the twin parser: a non-object JSON line must not abort
    # the run before the real text event is read.
    stdout = b'[]\n{"type":"text","part":{"type":"text","text":"answer"}}\n'
    text, error = _extract_text_and_error(stdout)
    assert text == "answer"
    assert error == ""


def test_opencode_recover_text_from_db_handles_special_chars_in_path(
    tmp_path: Path,
) -> None:
    # The prod sandbox path is fixed and clean, but a URI-significant char in the
    # home path (space, '#') must not silently break recovery -- passing the Path
    # directly to sqlite3.connect (no file: URI parsing) reads it fine.
    weird = tmp_path / "a #b"
    weird.mkdir()
    _write_opencode_db(weird, "ses_1", assistant_texts=["recovered"])
    assert _recover_text_from_db(weird, "ses_1") == "recovered"


@pytest.mark.asyncio
async def test_opencode_run_prefers_longer_db_text_over_short_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # opencode #31435 truncated variant: stdout carries a short stub answer but
    # the DB holds the full answer. The adapter must prefer the longer DB copy
    # (len(recovered) > len(text)) and report success.
    _write_opencode_db(
        tmp_path,
        "ses_stub",
        assistant_texts=["the full ", "recovered answer that is much longer"],
    )

    class _Proc:
        def __init__(self) -> None:
            self.pid = 5556
            self.returncode = 0

    async def _spawn(*_a: object, **_kw: object) -> _Proc:
        return _Proc()

    async def _short_stub(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        return (
            b'{"type":"step_start","sessionID":"ses_stub",'
            b'"part":{"type":"step-start"}}\n'
            b'{"type":"text","sessionID":"ses_stub",'
            b'"part":{"type":"text","text":"stub"}}\n',
            b"",
            False,
        )

    monkeypatch.setattr(opencode_mod, "_resolve_openrouter_key", lambda: "fake-key")
    monkeypatch.setattr(opencode_mod, "_ensure_sandbox_home", lambda *a, **k: tmp_path)
    monkeypatch.setattr(opencode_mod, "_build_subprocess_env", lambda *a, **k: {})
    monkeypatch.setattr(opencode_mod, "_DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(opencode_mod.asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setattr(
        opencode_mod, "communicate_lines_or_kill", _short_stub, raising=False
    )

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert result.output == "the full recovered answer that is much longer"
    assert result.returncode == 0


def test_opencode_extract_text_skips_non_dict_part() -> None:
    # A valid event object with a non-dict `part` (list/string/scalar) must not
    # crash part.get(...) -- the isinstance(event, dict) guard alone does not
    # cover this, since event IS a dict and only `part` is malformed.
    stdout = (
        b'{"type":"text","part":["bad"]}\n'
        b'{"type":"text","part":{"type":"text","text":"ok"}}\n'
    )
    text, error = _extract_text_and_error(stdout)
    assert text == "ok"
    assert error == ""


def test_opencode_extract_text_handles_invalid_utf8() -> None:
    # json.loads(bytes) raises UnicodeDecodeError (NOT JSONDecodeError) on
    # invalid UTF-8, so a single bad byte-line must be skipped, not crash.
    # (b'\xff' alone -> UnicodeDecodeError; b'\xff\xfe' would be the UTF-16 BOM
    # and decode to empty -> JSONDecodeError, which the old code already caught.)
    stdout = b'\xff\n{"type":"text","part":{"type":"text","text":"ok"}}\n'
    text, error = _extract_text_and_error(stdout)
    assert text == "ok"
    assert error == ""


def test_opencode_session_id_from_stdout_handles_invalid_utf8() -> None:
    stdout = b'\xff\n{"type":"step_start","sessionID":"ses_u","part":{}}\n'
    assert _session_id_from_stdout(stdout) == "ses_u"


def test_opencode_recover_text_from_db_ties_use_insertion_order(
    tmp_path: Path,
) -> None:
    # On equal time_created, parts must concatenate in insertion (rowid) order,
    # NOT id-alphabetical order -- opencode part ids are ULID-ish TEXT keys whose
    # alpha order is not their write order. Insert id "prt_zzz" first (rowid 1)
    # then "prt_aaa" (rowid 2), both at the same timestamp: correct output is
    # "FIRSTsecond" (insertion), the regression would be "secondFIRST" (alpha).
    db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, "
            "time_created INTEGER, time_updated INTEGER, data TEXT);"
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, "
            "session_id TEXT, time_created INTEGER, time_updated INTEGER, data TEXT);"
        )
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?,?)",
            ("msg_a", "ses_t", 2, 2, json.dumps({"role": "assistant"})),
        )
        # same time_created (10); id alpha order is the REVERSE of insertion order
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?,?)",
            (
                "prt_zzz",
                "msg_a",
                "ses_t",
                10,
                10,
                json.dumps({"type": "text", "text": "FIRST"}),
            ),
        )
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?,?)",
            (
                "prt_aaa",
                "msg_a",
                "ses_t",
                10,
                10,
                json.dumps({"type": "text", "text": "second"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    assert _recover_text_from_db(tmp_path, "ses_t") == "FIRSTsecond"


def test_opencode_recover_text_from_db_reads_wal_mode(tmp_path: Path) -> None:
    # The real opencode.db is WAL-mode; recovery must read it. Build the fixture,
    # switch it to WAL, and confirm recovery still returns the answer.
    _write_opencode_db(tmp_path, "ses_w", assistant_texts=["wal ", "answer"])
    db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=wal")
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?,?)",
            (
                "prt_extra",
                "msg_asst",
                "ses_w",
                20,
                20,
                json.dumps({"type": "text", "text": "!"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    assert _recover_text_from_db(tmp_path, "ses_w") == "wal answer!"


# --- make_codex_agent: model-selection wiring --------------------------------
# resolve_model/recorded_choice read quorum.model_config.CONFIG_PATH lazily
# (at call time, not import time), so monkeypatching mc.CONFIG_PATH is enough
# to point the factory at a tmp config without a path= parameter.


def _cfg(tmp_path: Path) -> Path:
    return tmp_path / "models.toml"


def test_make_codex_agent_uses_recorded_model_and_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("codex", {"model": "gpt-6-preview", "effort": "high"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    seat = make_codex_agent()
    assert seat.model == "gpt-6-preview"
    assert seat.effort == "high"


def test_make_codex_agent_env_model_beats_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("codex", {"model": "recorded-model"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setenv("CODE_QUORUM_CODEX_MODEL", "env-model")
    seat = make_codex_agent()
    assert seat.model == "env-model"


def test_make_codex_agent_env_effort_beats_recorded_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("codex", {"model": "recorded-model", "effort": "low"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setenv("CODE_QUORUM_CODEX_EFFORT", "xhigh")
    seat = make_codex_agent()
    assert seat.effort == "xhigh"


def test_make_codex_agent_recorded_without_effort_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("codex", {"model": "recorded-model"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    seat = make_codex_agent()
    assert seat.effort == codex_mod.DEFAULT_EFFORT


def test_make_codex_agent_absent_config_matches_shipped_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    assert not path.exists()
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    seat = make_codex_agent()
    assert seat.model == codex_mod.DEFAULT_MODEL
    assert seat.effort == codex_mod.DEFAULT_EFFORT


def test_make_codex_agent_absent_record_spawns_no_version_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Headless invariant: no recorded choice means no version probe at all.
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    calls: list[list[str]] = []
    monkeypatch.setattr(mc.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    make_codex_agent()
    assert calls == []


def test_make_codex_agent_version_drift_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice(
        "codex", {"model": "gpt-6-preview", "cli_version": "0.100.0"}, path
    )
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setattr(codex_mod, "_codex_version", lambda binary="codex": "0.200.0")
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        make_codex_agent()
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "quorum.model_config"
    ]
    assert len(warnings) == 1
    assert "codex" in warnings[0].message
    assert "0.200.0" in warnings[0].message
    assert "0.100.0" in warnings[0].message


def test_make_codex_agent_version_matches_recorded_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice(
        "codex", {"model": "gpt-6-preview", "cli_version": "0.100.0"}, path
    )
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setattr(codex_mod, "_codex_version", lambda binary="codex": "0.100.0")
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        make_codex_agent()
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "quorum.model_config"
    ]
    assert warnings == []


def test_codex_version_caches_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class _Proc:
        stdout = "codex-cli 0.150.0\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Proc()

    # The probe now runs in model_config.probe_cli_version -- patch the
    # subprocess it calls, not codex.py's (which no longer imports one).
    monkeypatch.setattr(mc.subprocess, "run", _fake_run)
    assert codex_mod._codex_version() == "0.150.0"
    assert codex_mod._codex_version() == "0.150.0"
    assert len(calls) == 1


def test_make_codex_agent_disabled_seat_invalid_effort_raises_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # load_config skips effort validation on disabled tables so a stale value
    # cannot abort every command. select_agents honors an explicit request
    # for a disabled seat, so resolve_seat_choice must validate the
    # recorded effort itself or the CLI gets fed garbage late.
    path = _cfg(tmp_path)
    path.write_text(
        '[codex]\ndisabled = "true"\nmodel = "gpt-6-preview"\neffort = "warp9"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    with pytest.raises(ValueError, match="warp9"):
        make_codex_agent()


def test_make_codex_agent_malformed_recorded_effort_raises_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    path.write_text(
        '[codex]\nmodel = "gpt-6-preview"\neffort = "turbo"\n', encoding="utf-8"
    )
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    with pytest.raises(mc.ModelConfigError):
        make_codex_agent()


# --- make_opencode_agent: model-selection wiring -----------------------------


def test_make_opencode_agent_uses_recorded_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("opencode", {"model": "openrouter/acme/custom-model"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    seat = make_opencode_agent()
    assert seat.model == "openrouter/acme/custom-model"


def test_make_opencode_agent_env_model_beats_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("opencode", {"model": "recorded-model"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setenv("CODE_QUORUM_OPENCODE_MODEL", "env-model")
    seat = make_opencode_agent()
    assert seat.model == "env-model"


def test_make_opencode_agent_recorded_model_reaches_sandbox_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The resolved model must flow into _build_opencode_config exactly like
    # an env override does today (existing tests call it with a hardcoded
    # model; this confirms a recorded choice reaches the same place).
    path = _cfg(tmp_path)
    mc.record_choice("opencode", {"model": "openrouter/deepseek/deepseek-v4-pro"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    seat = make_opencode_agent()
    cfg = _build_opencode_config(seat.model)
    assert cfg is not None
    assert "deepseek/deepseek-v4-pro" in cfg["provider"]["openrouter"]["models"]


def test_make_opencode_agent_absent_config_matches_shipped_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    seat = make_opencode_agent()
    assert seat.model == opencode_mod.DEFAULT_MODEL


def test_make_opencode_agent_absent_record_spawns_no_version_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    calls: list[list[str]] = []
    monkeypatch.setattr(mc.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    make_opencode_agent()
    assert calls == []


def test_make_opencode_agent_version_drift_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice(
        "opencode",
        {"model": "openrouter/deepseek/deepseek-v4-pro", "cli_version": "0.9.0"},
        path,
    )
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setattr(
        opencode_mod, "_opencode_version", lambda binary="opencode": "0.10.0"
    )
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        make_opencode_agent()
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "quorum.model_config"
    ]
    assert len(warnings) == 1
    assert "opencode" in warnings[0].message
    assert "0.10.0" in warnings[0].message
    assert "0.9.0" in warnings[0].message


def test_opencode_version_caches_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class _Proc:
        stdout = "0.42.0\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Proc()

    # Patch model_config's subprocess -- the probe's real home (see the
    # codex twin of this test).
    monkeypatch.setattr(mc.subprocess, "run", _fake_run)
    assert opencode_mod._opencode_version() == "0.42.0"
    assert opencode_mod._opencode_version() == "0.42.0"
    assert len(calls) == 1


# --- served-backend relay (records which OpenRouter backend served a turn) ---
# The provider pin (_PROVIDER_ORDER) is only auditable if the seat records
# who actually served. opencode drops OpenRouter's `provider` field, so run()
# points opencode at a loopback relay (quorum.agents.openrouter_relay) via
# the generated config's baseURL. The relay is best-effort: if it cannot
# bind, the seat runs direct to OpenRouter exactly as before.


def test_subprocess_env_points_openrouter_at_relay_per_process(tmp_path: Path) -> None:
    # The relay URL rides in OPENCODE_CONFIG_CONTENT (per process), never in
    # the shared sandbox opencode.json: two overlapping seat runs must each
    # keep their own relay. opencode deep-merges it over the file config
    # (verified on the wire against 1.18.25).
    env = _build_subprocess_env(
        tmp_path, "k", base={}, relay_base_url="http://127.0.0.1:43111/api/v1"
    )
    content = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert content == {
        "provider": {
            "openrouter": {"options": {"baseURL": "http://127.0.0.1:43111/api/v1"}}
        }
    }
    cfg = _build_opencode_config("openrouter/deepseek/deepseek-v4.1-flash")
    assert cfg is not None
    assert "baseURL" not in cfg["provider"]["openrouter"]["options"]


def test_subprocess_env_without_relay_sends_no_override(tmp_path: Path) -> None:
    # No relay -> no OPENCODE_CONFIG_CONTENT at all, so opencode uses its
    # built-in OpenRouter endpoint (the pre-relay behaviour, byte for byte).
    env = _build_subprocess_env(tmp_path, "k", base={})
    assert "OPENCODE_CONFIG_CONTENT" not in env


class _FakeRelay:
    """Stand-in for OpenRouterRelay: records lifecycle calls, never binds."""

    instances: list["_FakeRelay"] = []
    start_error: BaseException | None = None

    def __init__(self, **_kw: object) -> None:
        self.base_url: str | None = None
        self.started = 0
        self.stopped = 0
        self.turns: list[object] = []
        _FakeRelay.instances.append(self)

    async def start(self) -> str:
        if _FakeRelay.start_error is not None:
            raise _FakeRelay.start_error
        self.started += 1
        self.base_url = "http://127.0.0.1:43111/api/v1"
        return self.base_url

    async def stop(self) -> None:
        self.stopped += 1

    def summary(self) -> dict[str, int]:
        return {}


def _patch_relay_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    """Patch the run() seams like _patch_opencode_run_seams, but capture the
    kwargs _build_subprocess_env receives so a test can see the relay URL."""
    env_calls: list[dict[str, object]] = []

    def _record_env(*_a: object, **kw: object) -> dict[str, str]:
        env_calls.append(dict(kw))
        return {}

    async def _ok(
        proc: object, *, pgid: int, idle_timeout: float
    ) -> tuple[bytes, bytes, bool]:
        return (b'{"type":"text","part":{"text":"ok"}}\n', b"", False)

    _patch_opencode_run_seams(monkeypatch, tmp_path, _ok)
    monkeypatch.setattr(opencode_mod, "_build_subprocess_env", _record_env)
    _FakeRelay.instances = []
    _FakeRelay.start_error = None
    monkeypatch.setattr(opencode_mod, "OpenRouterRelay", _FakeRelay)
    monkeypatch.delenv("CODE_QUORUM_OPENCODE_RELAY", raising=False)
    return env_calls


@pytest.mark.asyncio
async def test_opencode_run_routes_through_relay_and_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_calls = _patch_relay_seams(monkeypatch, tmp_path)

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert result.returncode == 0
    assert len(_FakeRelay.instances) == 1, "one relay per seat run"
    relay = _FakeRelay.instances[0]
    assert relay.started == 1
    assert relay.stopped == 1, "the relay must be stopped after the run"
    # The subprocess env pointed opencode at the relay, not at OpenRouter.
    assert env_calls[0]["relay_base_url"] == "http://127.0.0.1:43111/api/v1"


@pytest.mark.asyncio
async def test_opencode_run_falls_back_to_direct_when_relay_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # S's constraint: the relay must never cost a call. A bind failure is
    # logged and the seat runs direct (no baseURL override).
    env_calls = _patch_relay_seams(monkeypatch, tmp_path)
    _FakeRelay.start_error = OSError(98, "address in use")

    with caplog.at_level(logging.WARNING, logger="quorum.agents.opencode"):
        result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert result.returncode == 0
    assert env_calls[0]["relay_base_url"] is None
    assert any("relay" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_opencode_run_relay_disabled_by_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_calls = _patch_relay_seams(monkeypatch, tmp_path)
    monkeypatch.setenv("CODE_QUORUM_OPENCODE_RELAY", "0")

    result = await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert result.returncode == 0
    assert _FakeRelay.instances == [], "CODE_QUORUM_OPENCODE_RELAY=0 skips the relay"
    assert env_calls[0]["relay_base_url"] is None


@pytest.mark.asyncio
async def test_opencode_run_stops_relay_even_when_attempt_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A crash inside the attempt must not leak the loopback listener.
    _patch_relay_seams(monkeypatch, tmp_path)

    async def _boom(*_a: object, **_kw: object) -> AgentResult:
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(OpenCodeAgent, "_attempt", _boom)

    with pytest.raises(RuntimeError):
        await OpenCodeAgent().run(prompt="x", cwd="/tmp")

    assert _FakeRelay.instances[0].stopped == 1


def test_unpinned_backends_flags_servers_outside_the_pin() -> None:
    # Ledger names are OpenRouter display names ("Novita"); the pin holds
    # slugs ("novita"). Comparison is case-insensitive; "(none)" is not a
    # backend. A model with no pin has nothing to flag.
    unpinned = opencode_mod._unpinned_backends
    model = "deepseek/deepseek-v4.1-flash"
    assert unpinned(model, {"Novita": 3, "Parasail": 1, "(none)": 1}) == []
    assert unpinned(model, {"Novita": 3, "DeepInfra": 2}) == ["DeepInfra"]
    assert unpinned("deepseek/deepseek-v4-pro", {"DeepInfra": 2}) == []
