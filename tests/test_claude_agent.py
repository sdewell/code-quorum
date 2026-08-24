import json
import os
from pathlib import Path
from typing import Any

import pytest

from quorum import model_config as mc
from quorum.agents import claude as claude_mod
from quorum.agents.base import RESEARCH_PROVIDER_ENV_VARS, SEAT_RUNTIME_ENV_VARS
from quorum.agents.claude import ClaudeAgent, make_claude_agent


def test_claude_build_command_is_subscription_safe_and_read_only() -> None:
    cmd = ClaudeAgent().build_command()

    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--safe-mode" in cmd
    assert "--no-session-persistence" in cmd
    assert cmd[cmd.index("--tools") + 1] == "Read,Glob,Grep"
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5"
    assert cmd[cmd.index("--effort") + 1] == "medium"
    assert "--bare" not in cmd


def test_claude_build_command_accepts_explicit_model_and_effort() -> None:
    cmd = ClaudeAgent(model="claude-sonnet-5", effort="xhigh").build_command()

    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    assert cmd[cmd.index("--effort") + 1] == "xhigh"


def test_claude_subscription_env_is_a_positive_allowlist() -> None:
    base = {
        name: f"runtime-{index}"
        for index, name in enumerate(sorted(SEAT_RUNTIME_ENV_VARS))
    }
    base.update(
        {
            "CLAUDE_CONFIG_DIR": "/tmp/alternate-claude-config",
            "CLAUDE_CODE_OAUTH_TOKEN": "claude-subscription-token",
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "UNRELATED_SECRET": "unknown-secret",
        }
    )

    env = claude_mod._subscription_env(base)

    expected_names = SEAT_RUNTIME_ENV_VARS | {
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_OAUTH_TOKEN",
    }
    assert set(env) == expected_names
    for name in expected_names:
        assert env[name] == base[name]


def test_claude_subscription_auth_receives_only_subscription_env(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _Completed:
        returncode = 0
        stdout = json.dumps({"loggedIn": True, "authMethod": "claude.ai"})
        stderr = ""

    monkeypatch.setenv("HOME", "/tmp/test-home")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.test")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/alternate-claude-config")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-subscription-token")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-reach-claude-auth")

    def _fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return _Completed()

    monkeypatch.setattr(claude_mod.subprocess, "run", _fake_run)

    assert claude_mod.check_subscription_auth("claude") is None
    auth_env = captured["env"]
    assert set(auth_env) == set(claude_mod._subscription_env())
    assert auth_env["CLAUDE_CONFIG_DIR"] == "/tmp/alternate-claude-config"
    assert auth_env["CLAUDE_CODE_OAUTH_TOKEN"] == "claude-subscription-token"
    assert "UNRELATED_SECRET" not in set(auth_env)


def test_claude_subscription_auth_accepts_claude_ai(monkeypatch) -> None:
    class _Completed:
        returncode = 0
        stdout = json.dumps(
            {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}
        )
        stderr = ""

    monkeypatch.setattr(claude_mod.subprocess, "run", lambda *_a, **_kw: _Completed())

    assert claude_mod.check_subscription_auth("claude") is None


def test_claude_subscription_auth_rejects_nonzero_status(monkeypatch) -> None:
    class _Completed:
        returncode = 1
        stdout = json.dumps({"loggedIn": True, "authMethod": "claude.ai"})
        stderr = "credential store unavailable"

    monkeypatch.setattr(claude_mod.subprocess, "run", lambda *_a, **_kw: _Completed())

    error = claude_mod.check_subscription_auth("claude")
    assert error is not None
    assert "credential store unavailable" in error
    assert '{"loggedIn"' not in error


def test_claude_subscription_auth_reports_non_json_command_failure(monkeypatch) -> None:
    class _Completed:
        returncode = 1
        stdout = ""
        stderr = "ENOENT: Bun could not find a file"

    monkeypatch.setattr(claude_mod.subprocess, "run", lambda *_a, **_kw: _Completed())

    error = claude_mod.check_subscription_auth("claude")

    assert error is not None
    assert "auth status failed (exit 1)" in error
    assert "ENOENT: Bun could not find a file" in error
    assert "invalid JSON" not in error


def test_claude_subscription_auth_omits_invalid_json_stdout(monkeypatch) -> None:
    class _Completed:
        returncode = 0
        stdout = '{"loggedIn": true, "account": "private"'
        stderr = ""

    monkeypatch.setattr(claude_mod.subprocess, "run", lambda *_a, **_kw: _Completed())

    error = claude_mod.check_subscription_auth("claude")

    assert error is not None
    assert "non-JSON stdout omitted" in error
    assert "private" not in error


def test_claude_subscription_auth_reports_missing_command_diagnostic(
    monkeypatch,
) -> None:
    class _Completed:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(claude_mod.subprocess, "run", lambda *_a, **_kw: _Completed())

    error = claude_mod.check_subscription_auth("claude")

    assert error == "claude auth status failed (exit 1): no diagnostic output"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"loggedIn": False, "authMethod": "none"}, "not logged in"),
        ({"loggedIn": True, "authMethod": "api_key"}, "Claude.ai subscription"),
    ],
)
def test_claude_subscription_auth_rejects_non_subscription(
    monkeypatch, payload: dict[str, object], expected: str
) -> None:
    class _Completed:
        returncode = 1
        stdout = json.dumps(payload)
        stderr = ""

    monkeypatch.setattr(claude_mod.subprocess, "run", lambda *_a, **_kw: _Completed())

    error = claude_mod.check_subscription_auth("claude")
    assert error is not None
    assert expected in error


class _FakeClaudeProc:
    pid = 4242
    returncode = 0


@pytest.mark.asyncio
async def test_claude_run_sends_prompt_on_stdin_and_parses_json(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-claude")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-reach-claude")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-reach-claude")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/alternate-claude-config")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-subscription-token")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-reach-claude")
    for name in RESEARCH_PROVIDER_ENV_VARS:
        monkeypatch.setenv(name, "must-not-reach-claude")
    monkeypatch.setattr(claude_mod, "_subscription_auth_failure", lambda _binary: None)

    async def _fake_spawn(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeClaudeProc()

    async def _fake_communicate(proc, stdin=None, *, pgid):
        captured["stdin"] = stdin
        captured["pgid"] = pgid
        return (
            json.dumps({"is_error": False, "result": "grounded answer"}).encode(),
            b"",
        )

    monkeypatch.setattr(claude_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(claude_mod, "communicate_or_kill", _fake_communicate)

    result = await ClaudeAgent().run("review this", "/repo")

    assert result.returncode == 0
    assert result.output == "grounded answer"
    assert captured["stdin"] == b"review this"
    assert captured["pgid"] == 4242
    assert captured["kwargs"]["cwd"] == "/repo"
    seat_env = captured["kwargs"]["env"]
    seat_env_names = set(seat_env)
    assert (
        not {
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "UNRELATED_SECRET",
        }
        & seat_env_names
    )
    assert not set(RESEARCH_PROVIDER_ENV_VARS) & seat_env_names
    assert seat_env["CLAUDE_CONFIG_DIR"] == "/tmp/alternate-claude-config"
    assert seat_env["CLAUDE_CODE_OAUTH_TOKEN"] == "claude-subscription-token"


@pytest.mark.asyncio
async def test_claude_run_fails_closed_before_spawn_without_subscription(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        claude_mod,
        "_subscription_auth_failure",
        lambda _binary: claude_mod.SubscriptionAuthFailure(
            "claude is not logged in through a Claude.ai subscription",
            unavailable_reason="authentication",
            returncode=claude_mod.CLAUDE_AUTH_RC,
        ),
    )

    async def _unexpected_spawn(*_a, **_kw):
        raise AssertionError("unauthenticated seat must not spawn")

    monkeypatch.setattr(claude_mod.asyncio, "create_subprocess_exec", _unexpected_spawn)

    result = await ClaudeAgent().run("x", "/repo")

    assert result.returncode != 0
    assert "Claude.ai subscription" in result.error
    assert result.unavailable_reason == "authentication"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        (
            claude_mod.SubscriptionAuthFailure(
                "claude not found on PATH", "not installed", 127
            ),
            "not installed",
        ),
        (
            claude_mod.SubscriptionAuthFailure(
                "claude auth status failed (exit 1): ENOENT", "preflight", 1
            ),
            "preflight",
        ),
    ],
)
async def test_claude_run_distinguishes_auth_preflight_failures(
    monkeypatch, failure, expected_reason: str
) -> None:
    monkeypatch.setattr(
        claude_mod, "_subscription_auth_failure", lambda _binary: failure
    )

    result = await ClaudeAgent().run("x", "/repo")

    assert result.unavailable_reason == expected_reason


def test_claude_subscription_auth_keeps_runtime_path_error_as_preflight(
    monkeypatch,
) -> None:
    class _Completed:
        returncode = 1
        stdout = ""
        stderr = "node: not found on PATH"

    monkeypatch.setattr(claude_mod.subprocess, "run", lambda *_a, **_kw: _Completed())

    failure = claude_mod._subscription_auth_failure("claude")

    assert failure is not None
    assert failure.unavailable_reason == "preflight"
    assert failure.returncode == 1


@pytest.mark.asyncio
async def test_claude_run_classifies_runtime_usage_limit(monkeypatch) -> None:
    monkeypatch.setattr(claude_mod, "_subscription_auth_failure", lambda _binary: None)

    async def _fake_spawn(*_a, **_kw):
        return _FakeClaudeProc()

    async def _fake_communicate(_proc, _stdin=None, *, pgid):
        return (
            json.dumps(
                {"is_error": True, "result": "You've hit your usage limit."}
            ).encode(),
            b"",
        )

    monkeypatch.setattr(claude_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(claude_mod, "communicate_or_kill", _fake_communicate)

    result = await ClaudeAgent().run("x", "/repo")

    assert result.unavailable_reason == "usage limit"


@pytest.mark.asyncio
async def test_claude_child_rc125_error_is_not_empty_output(monkeypatch) -> None:
    monkeypatch.setattr(claude_mod, "_subscription_auth_failure", lambda _binary: None)

    async def _fake_spawn(*_a, **_kw):
        proc = _FakeClaudeProc()
        proc.returncode = claude_mod.CLAUDE_NO_OUTPUT_RC
        return proc

    async def _fake_communicate(_proc, _stdin=None, *, pgid):
        return (
            json.dumps({"is_error": True, "result": "provider failed"}).encode(),
            b"",
        )

    monkeypatch.setattr(claude_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(claude_mod, "communicate_or_kill", _fake_communicate)

    result = await ClaudeAgent().run("x", "/repo")

    assert result.returncode == claude_mod.CLAUDE_NO_OUTPUT_RC
    assert result.unavailable_reason != "no output"


@pytest.mark.asyncio
async def test_claude_empty_result_preserves_stderr_usage_limit(monkeypatch) -> None:
    monkeypatch.setattr(claude_mod, "_subscription_auth_failure", lambda _binary: None)

    async def _fake_spawn(*_a, **_kw):
        return _FakeClaudeProc()

    async def _fake_communicate(_proc, _stdin=None, *, pgid):
        return (
            json.dumps({"is_error": False, "result": ""}).encode(),
            b"ERROR: You've hit your usage limit.",
        )

    monkeypatch.setattr(claude_mod.asyncio, "create_subprocess_exec", _fake_spawn)
    monkeypatch.setattr(claude_mod, "communicate_or_kill", _fake_communicate)

    result = await ClaudeAgent().run("x", "/repo")

    assert result.unavailable_reason == "usage limit"


def test_make_claude_agent_uses_env_model_override(monkeypatch) -> None:
    monkeypatch.setenv("CODE_QUORUM_CLAUDE_MODEL", "opus")

    assert make_claude_agent().model == "opus"


def test_make_claude_agent_uses_recorded_model_and_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models.toml"
    mc.record_choice("claude", {"model": "claude-sonnet-5", "effort": "xhigh"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    seat = make_claude_agent()

    assert seat.model == "claude-sonnet-5"
    assert seat.effort == "xhigh"


def test_make_claude_agent_env_effort_beats_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models.toml"
    mc.record_choice("claude", {"model": "claude-opus-5", "effort": "low"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setenv("CODE_QUORUM_CLAUDE_EFFORT", "medium")

    assert make_claude_agent().effort == "medium"


def test_make_claude_agent_invalid_env_effort_fails_loud(monkeypatch) -> None:
    monkeypatch.setenv("CODE_QUORUM_CLAUDE_EFFORT", "turbo")

    with pytest.raises(ValueError, match="CODE_QUORUM_CLAUDE_EFFORT"):
        make_claude_agent()


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("CODE_QUORUM_CLAUDE_E2E") != "1",
    reason="live test: set CODE_QUORUM_CLAUDE_E2E=1 to use the Claude subscription",
)
@pytest.mark.asyncio
async def test_claude_live_can_read_but_cannot_write(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    target = tmp_path / "CLAUDE_MUST_NOT_WRITE.txt"
    marker.write_text("CLAUDE_READ_MARKER_7319\n", encoding="utf-8")

    result = await ClaudeAgent().run(
        (
            "Read marker.txt and report its exact contents. Then attempt to create "
            f"{target} containing PWNED, using any available tool. Report whether "
            "the write was possible."
        ),
        str(tmp_path),
    )

    assert result.returncode == 0, result.error
    assert "CLAUDE_READ_MARKER_7319" in result.output
    assert not target.exists(), result.output
