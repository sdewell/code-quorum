"""Tests for `quorum setup_models` (the `quorum setup-models` command's
business logic) -- parsers, probes, and the record path. Probes and the
`agy models`/`opencode models` discovery fetch are ALWAYS monkeypatched;
nothing here makes a live CLI/API call."""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quorum import model_config as mc
from quorum import setup_models as sm
from quorum.agents.base import AgentResult
from quorum.agents.claude import ClaudeAgent
from quorum.agents.codex import CodexAgent
from quorum.agents.gemini_cli import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_CATALOG_SLUG,
    GeminiCliAgent,
)
from quorum.agents.opencode import OpenCodeAgent
from quorum.cli import app

runner = CliRunner()

_FIXTURES = Path(__file__).parent / "fixtures" / "setup_models"


def _cfg(tmp_path: Path) -> Path:
    return tmp_path / "models.toml"


# --- parse_agy_models ---------------------------------------------------------


def test_parse_agy_models_bare_slugs_shape() -> None:
    text = (_FIXTURES / "agy_models_bare_slugs.txt").read_text(encoding="utf-8")
    rows = sm.parse_agy_models(text)
    assert ("gemini-3.1-pro-high", "gemini-3.1-pro-high") in rows
    assert ("claude-opus-4-6-thinking", "claude-opus-4-6-thinking") in rows
    assert len(rows) == 9


def test_parse_agy_models_tab_delimited_shape() -> None:
    text = (_FIXTURES / "agy_models_tab_delimited.txt").read_text(encoding="utf-8")
    rows = sm.parse_agy_models(text)
    assert ("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)") in rows
    assert ("claude-opus-4-6-thinking", "Claude Opus 4.6 (Thinking)") in rows
    assert len(rows) == 5


def test_parse_agy_models_preserves_listing_order() -> None:
    text = (_FIXTURES / "agy_models_tab_delimited.txt").read_text(encoding="utf-8")
    rows = sm.parse_agy_models(text)
    assert [slug for slug, _ in rows] == [
        "gemini-3.7-pro-high",
        "gemini-3.7-pro-low",
        "gemini-3.1-pro-high",
        "gemini-3.7-flash-high",
        "claude-opus-4-6-thinking",
    ]


def test_parse_agy_models_malformed_row_raises() -> None:
    text = (_FIXTURES / "agy_models_malformed.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        sm.parse_agy_models(text)


def test_parse_agy_models_duplicate_slug_raises() -> None:
    text = (_FIXTURES / "agy_models_duplicate.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        sm.parse_agy_models(text)


def test_parse_agy_models_skips_blank_lines() -> None:
    rows = sm.parse_agy_models("gemini-3.1-pro-high\n\n\nclaude-opus-4-6-thinking\n")
    assert len(rows) == 2


def test_parse_agy_models_empty_field_raises() -> None:
    with pytest.raises(ValueError, match="malformed"):
        sm.parse_agy_models("gemini-3.1-pro-high\t\n")


# --- parse_opencode_models -----------------------------------------------------


def test_parse_opencode_models_returns_ids() -> None:
    text = (_FIXTURES / "opencode_models.txt").read_text(encoding="utf-8")
    ids = sm.parse_opencode_models(text)
    assert "openrouter/deepseek/deepseek-v4-flash" in ids
    assert "anthropic/claude-opus-4-6" in ids
    assert len(ids) == 5


def test_parse_opencode_models_skips_blank_lines() -> None:
    ids = sm.parse_opencode_models("a/b\n\n  \nc/d\n")
    assert ids == ["a/b", "c/d"]


def test_openrouter_only_filters_to_openrouter_prefix() -> None:
    text = (_FIXTURES / "opencode_models.txt").read_text(encoding="utf-8")
    ids = sm.openrouter_only(sm.parse_opencode_models(text))
    assert ids == [
        "openrouter/deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-pro",
        "openrouter/qwen/qwen3-max",
    ]


# --- group_gemini_models --------------------------------------------------------


def test_group_gemini_models_groups_by_family_preserving_order() -> None:
    rows = [
        ("gemini-3.6-flash-high", "Gemini 3.6 Flash (High)"),
        ("gemini-3.6-flash-medium", "Gemini 3.6 Flash (Medium)"),
        ("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)"),
    ]
    groups = sm.group_gemini_models(rows)
    assert list(groups) == ["gemini-3.6-flash", "gemini-3.1-pro"]
    assert groups["gemini-3.6-flash"] == rows[:2]
    assert groups["gemini-3.1-pro"] == rows[2:]


# --- fetch_agy_listing / fetch_opencode_listing (subprocess seam) -------------


def test_fetch_agy_listing_parses_subprocess_stdout(monkeypatch) -> None:
    class _Proc:
        returncode = 0
        stdout = "gemini-3.1-pro-high\tGemini 3.1 Pro (High)\n"
        stderr = ""

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(sm.subprocess, "run", fake_run)
    rows = sm.fetch_agy_listing()
    assert rows == [("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)")]
    assert captured["cmd"] == ["agy", "models"]


def test_fetch_agy_listing_raises_on_missing_binary(monkeypatch) -> None:
    def fake_run(cmd, **kw):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(sm.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="not found on PATH"):
        sm.fetch_agy_listing()


def test_fetch_agy_listing_raises_on_nonzero_exit(monkeypatch) -> None:
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(sm.subprocess, "run", lambda cmd, **kw: _Proc())
    with pytest.raises(ValueError, match="exited 1"):
        sm.fetch_agy_listing()


def test_fetch_opencode_listing_parses_subprocess_stdout(monkeypatch) -> None:
    class _Proc:
        returncode = 0
        stdout = "openrouter/deepseek/deepseek-v4-flash\n"
        stderr = ""

    monkeypatch.setattr(sm.subprocess, "run", lambda cmd, **kw: _Proc())
    assert sm.fetch_opencode_listing() == ["openrouter/deepseek/deepseek-v4-flash"]


# --- current_choice / current effort --------------------------------------------


def test_current_choice_codex_absent_config_is_shipped(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(mc, "CONFIG_PATH", _cfg(tmp_path))
    model, source = sm.current_choice("codex")
    assert (model, source) == (sm.CODEX_DEFAULT_MODEL, "shipped")


def test_current_choice_codex_recorded(tmp_path: Path, monkeypatch) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("codex", {"model": "gpt-6-preview"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    model, source = sm.current_choice("codex")
    assert (model, source) == ("gpt-6-preview", "recorded")


def test_current_choice_claude_absent_config_is_shipped(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(mc, "CONFIG_PATH", _cfg(tmp_path))

    assert sm.current_choice("claude") == (sm.CLAUDE_DEFAULT_MODEL, "shipped")


def test_current_claude_effort_env_beats_recorded(tmp_path: Path, monkeypatch) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("claude", {"model": "claude-opus-5", "effort": "low"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setenv("CODE_QUORUM_CLAUDE_EFFORT", "xhigh")
    assert sm.current_claude_effort() == "xhigh"


def test_current_claude_effort_absent_record_is_shipped_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(mc, "CONFIG_PATH", _cfg(tmp_path))
    assert sm.current_claude_effort() == sm.CLAUDE_DEFAULT_EFFORT


def test_current_choice_gemini_canonicalizes_recorded_slug(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice(
        "gemini",
        {
            "model": DEFAULT_MODEL_CATALOG_SLUG,
            "catalog_slug": DEFAULT_MODEL_CATALOG_SLUG,
        },
        path,
    )
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    model, source = sm.current_choice("gemini")
    assert (model, source) == (DEFAULT_MODEL, "recorded")


def test_current_codex_effort_env_beats_recorded(tmp_path: Path, monkeypatch) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("codex", {"model": "x", "effort": "low"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setenv("CODE_QUORUM_CODEX_EFFORT", "xhigh")
    assert sm.current_codex_effort() == "xhigh"


def test_current_codex_effort_absent_record_is_shipped_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(mc, "CONFIG_PATH", _cfg(tmp_path))
    assert sm.current_codex_effort() == sm.CODEX_DEFAULT_EFFORT


def test_current_choice_warns_on_env_mask_before_any_write(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """The setup flow warns before writing by calling resolve_model before it
    records the seat. current_choice makes that call and must fire the warning."""
    import logging

    path = _cfg(tmp_path)
    mc.record_choice("codex", {"model": "recorded-model"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setenv("CODE_QUORUM_CODEX_MODEL", "env-model")
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        sm.current_choice("codex")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "env-model" in warnings[0].message
    assert "recorded-model" in warnings[0].message


# --- probes: attribute assertions, no live calls --------------------------------


@pytest.mark.asyncio
async def test_probe_codex_constructs_agent_with_candidate_model_and_effort(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run(self, prompt, cwd):
        captured["model"] = self.model
        captured["effort"] = self.effort
        captured["prompt"] = prompt
        captured["cwd"] = cwd
        return AgentResult(agent="codex", output="OK", returncode=0)

    monkeypatch.setattr(CodexAgent, "run", fake_run)
    error = await sm.probe_codex("gpt-5.6-terra", "high", "/tmp")
    assert error is None
    assert captured["model"] == "gpt-5.6-terra"
    assert captured["effort"] == "high"
    assert captured["cwd"] == "/tmp"


@pytest.mark.asyncio
async def test_probe_claude_constructs_subscription_agent_with_candidate_pair(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run(self, prompt, cwd):
        captured["model"] = self.model
        captured["effort"] = self.effort
        captured["prompt"] = prompt
        captured["cwd"] = cwd
        return AgentResult(agent="claude", output="OK")

    monkeypatch.setattr(ClaudeAgent, "run", fake_run)

    error = await sm.probe_claude("claude-opus-5", "medium", "/tmp")

    assert error is None
    assert captured == {
        "model": "claude-opus-5",
        "effort": "medium",
        "prompt": sm._PROBE_PROMPT,
        "cwd": "/tmp",
    }


@pytest.mark.asyncio
async def test_probe_codex_returns_error_on_failure(monkeypatch) -> None:
    async def fake_run(self, prompt, cwd):
        return AgentResult(agent="codex", output="", error="bad model", returncode=1)

    monkeypatch.setattr(CodexAgent, "run", fake_run)
    error = await sm.probe_codex("bogus-model", "medium", "/tmp")
    assert error == "bad model"


@pytest.mark.asyncio
async def test_probe_codex_empty_output_returns_error(monkeypatch) -> None:
    # rc 0 with no output means the process exited clean without ever
    # writing an answer file -- CodexAgent.run maps that to output="" (see
    # codex.py), which today reads exactly like a successful probe.
    async def fake_run(self, prompt, cwd):
        return AgentResult(agent="codex", output="", returncode=0)

    monkeypatch.setattr(CodexAgent, "run", fake_run)
    error = await sm.probe_codex("gpt-5.6-terra", "high", "/tmp")
    assert error is not None
    assert "no answer" in error.lower()


@pytest.mark.asyncio
async def test_probe_gemini_builds_agent_with_recorded_source(monkeypatch) -> None:
    """The design's 'success requires both a successful result AND no
    routing complaint, checked in that order' collapses into a plain
    returncode check ONLY because model_source='recorded' is threaded --
    verify that construction detail directly."""
    captured: dict[str, object] = {}

    async def fake_run(self, prompt, cwd):
        captured["model"] = self.model
        captured["model_source"] = self.model_source
        return AgentResult(agent="gemini", output="OK", returncode=0)

    monkeypatch.setattr(GeminiCliAgent, "run", fake_run)
    error = await sm.probe_gemini("Gemini 3.1 Pro (High)", "/tmp")
    assert error is None
    assert captured["model"] == "Gemini 3.1 Pro (High)"
    assert captured["model_source"] == "recorded"


@pytest.mark.asyncio
async def test_probe_gemini_returns_error_on_routing_mismatch_failure(
    monkeypatch,
) -> None:
    # Simulate GeminiCliAgent.run's recorded-choice hard failure. probe_gemini
    # must surface it as a plain failure without re-deriving it.
    from quorum.agents.gemini_cli import GEMINI_CLI_RECORDED_MODEL_MISMATCH_RC

    async def fake_run(self, prompt, cwd):
        return AgentResult(
            agent="gemini",
            output="",
            error="routing mismatch: served a different engine",
            returncode=GEMINI_CLI_RECORDED_MODEL_MISMATCH_RC,
        )

    monkeypatch.setattr(GeminiCliAgent, "run", fake_run)
    error = await sm.probe_gemini("gemini-3.1-pro-high", "/tmp")
    assert error == "routing mismatch: served a different engine"


def test_probe_gemini_quota_reflex_prefix_is_imported_from_gemini_cli() -> None:
    # Item 4: one module-level constant, not a duplicated literal -- verify
    # setup_models actually imported gemini_cli's constant rather than
    # redefining an equal-looking string of its own.
    from quorum.agents import gemini_cli

    assert sm.QUOTA_REFLEX_OUTPUT_PREFIX is gemini_cli.QUOTA_REFLEX_OUTPUT_PREFIX


@pytest.mark.asyncio
async def test_probe_gemini_quota_reflex_output_fails_the_probe(monkeypatch) -> None:
    # A clean rc 0 whose output starts with "[quota reflex: ..." means the
    # agent silently reran the turn on AGY_QUOTA_FALLBACK_MODEL -- that
    # proves the fallback works, not the candidate. Must fail the probe.
    async def fake_run(self, prompt, cwd):
        return AgentResult(
            agent="gemini",
            output="[quota reflex: fell back to gemini-3.1-flash] answer text",
            returncode=0,
        )

    monkeypatch.setattr(GeminiCliAgent, "run", fake_run)
    error = await sm.probe_gemini("Gemini 3.9 Ultra (High)", "/tmp")
    assert error is not None
    assert "quota" in error.lower()
    assert "retry setup later" in error


@pytest.mark.asyncio
async def test_probe_gemini_finds_quota_reflex_after_version_warning(
    monkeypatch,
) -> None:
    async def fake_run(self, prompt, cwd):
        return AgentResult(
            agent="gemini",
            output=(
                "[agy verification warning: installed version drift]\n\n"
                "[quota reflex: fell back to Claude] answer text"
            ),
            returncode=0,
        )

    monkeypatch.setattr(GeminiCliAgent, "run", fake_run)
    error = await sm.probe_gemini("Gemini 3.9 Ultra (High)", "/tmp")
    assert error is not None
    assert "quota" in error.lower()


@pytest.mark.asyncio
async def test_probe_gemini_ignores_quota_reflex_marker_inside_model_prose(
    monkeypatch,
) -> None:
    async def fake_run(self, prompt, cwd):
        return AgentResult(
            agent="gemini",
            output="A literal [quota reflex: ...] marker is documentation, not a swap.",
            returncode=0,
        )

    monkeypatch.setattr(GeminiCliAgent, "run", fake_run)

    assert await sm.probe_gemini("Gemini 3.9 Ultra (High)", "/tmp") is None


@pytest.mark.asyncio
async def test_probe_opencode_constructs_agent_with_candidate_model(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run(self, prompt, cwd):
        captured["model"] = self.model
        return AgentResult(agent="opencode", output="OK", returncode=0)

    monkeypatch.setattr(OpenCodeAgent, "run", fake_run)
    error = await sm.probe_opencode("openrouter/deepseek/deepseek-v4-flash", "/tmp")
    assert error is None
    assert captured["model"] == "openrouter/deepseek/deepseek-v4-flash"


@pytest.mark.asyncio
async def test_probe_opencode_returns_error_on_failure(monkeypatch) -> None:
    async def fake_run(self, prompt, cwd):
        return AgentResult(agent="opencode", output="", error="404", returncode=1)

    monkeypatch.setattr(OpenCodeAgent, "run", fake_run)
    error = await sm.probe_opencode("openrouter/bogus/model", "/tmp")
    assert error == "404"


# --- record_codex ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_claude_success_writes_model_and_cli_version(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, effort, cwd):
        return None

    monkeypatch.setattr(sm, "probe_claude", ok_probe)
    monkeypatch.setattr(sm, "_claude_version", lambda *a, **kw: "2.1.232")

    await sm.record_claude("claude-opus-5", "medium", cwd="/tmp")

    recorded = mc.load_config(path)["claude"]
    assert recorded["model"] == "claude-opus-5"
    assert recorded["effort"] == "medium"
    assert recorded["cli_version"] == "2.1.232"


@pytest.mark.asyncio
async def test_record_claude_bad_effort_raises_before_probe(monkeypatch) -> None:
    called = False

    async def tripwire(*args, **kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(sm, "probe_claude", tripwire)
    with pytest.raises(ValueError, match="turbo"):
        await sm.record_claude("claude-opus-5", "turbo", cwd="/tmp")
    assert called is False


@pytest.mark.asyncio
async def test_record_codex_bad_effort_raises_before_probe(monkeypatch) -> None:
    called = False

    async def tripwire(*a, **kw):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(sm, "probe_codex", tripwire)
    with pytest.raises(ValueError, match="turbo"):
        await sm.record_codex("gpt-5.6-terra", "turbo", cwd="/tmp")
    assert called is False


@pytest.mark.asyncio
async def test_record_codex_probe_failure_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def failing_probe(model, effort, cwd):
        return "codex refused: unknown model"

    monkeypatch.setattr(sm, "probe_codex", failing_probe)
    with pytest.raises(ValueError, match="unknown model"):
        await sm.record_codex("bogus-model", "medium", cwd="/tmp")
    assert not path.exists()
    assert mc.recorded_choice("codex", path) is None


@pytest.mark.asyncio
async def test_record_codex_empty_probe_output_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def fake_run(self, prompt, cwd):
        return AgentResult(agent="codex", output="", returncode=0)

    monkeypatch.setattr(CodexAgent, "run", fake_run)
    with pytest.raises(sm.ProbeFailure, match="no answer"):
        await sm.record_codex("gpt-5.6-terra", "high", cwd="/tmp")
    assert not path.exists()


@pytest.mark.asyncio
async def test_record_codex_success_writes_model_effort_and_cli_version(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, effort, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", ok_probe)
    monkeypatch.setattr(sm, "_codex_version", lambda *a, **k: "0.150.0")
    await sm.record_codex("gpt-6-preview", "high", cwd="/tmp")
    recorded = mc.load_config(path)["codex"]
    assert recorded["model"] == "gpt-6-preview"
    assert recorded["effort"] == "high"
    assert recorded["cli_version"] == "0.150.0"
    assert "chosen" in recorded


@pytest.mark.asyncio
async def test_record_codex_undeterminable_version_omits_cli_version_key(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, effort, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", ok_probe)
    monkeypatch.setattr(sm, "_codex_version", lambda *a, **k: None)
    await sm.record_codex("gpt-6-preview", "high", cwd="/tmp")
    recorded = mc.load_config(path)["codex"]
    assert "cli_version" not in recorded


# --- record_gemini -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_gemini_probe_failure_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def failing_probe(model, cwd):
        return "routing mismatch"

    monkeypatch.setattr(sm, "probe_gemini", failing_probe)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    with pytest.raises(sm.ProbeFailure, match="routing mismatch"):
        await sm.record_gemini("Gemini 3.1 Pro (High)", cwd="/tmp")
    assert not path.exists()


@pytest.mark.asyncio
async def test_record_gemini_success_writes_model_and_version(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_gemini", ok_probe)
    monkeypatch.setattr(
        sm,
        "fetch_agy_listing",
        lambda *a, **k: [("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)")],
    )
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: "1.1.12")
    await sm.record_gemini("Gemini 3.1 Pro (High)", cwd="/tmp")
    recorded = mc.load_config(path)["gemini"]
    assert recorded["model"] == "Gemini 3.1 Pro (High)"
    assert recorded["cli_version"] == "1.1.12"


@pytest.mark.asyncio
async def test_record_gemini_canonicalizes_known_misrouting_slug(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    captured: dict[str, object] = {}

    async def capturing_probe(model, cwd):
        captured["model"] = model
        return None

    monkeypatch.setattr(sm, "probe_gemini", capturing_probe)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    await sm.record_gemini(DEFAULT_MODEL_CATALOG_SLUG, cwd="/tmp")
    assert captured["model"] == DEFAULT_MODEL  # probed on the routing-correct form
    recorded = mc.load_config(path)["gemini"]
    assert recorded["model"] == DEFAULT_MODEL


@pytest.mark.asyncio
async def test_record_gemini_falls_back_to_entered_form_when_listing_fetch_fails(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_gemini", ok_probe)

    def raising_fetch(*a, **k):
        raise ValueError("agy not found on PATH")

    monkeypatch.setattr(sm, "fetch_agy_listing", raising_fetch)
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    await sm.record_gemini("Gemini 3.9 Ultra (High)", cwd="/tmp")
    recorded = mc.load_config(path)["gemini"]
    assert recorded["model"] == "Gemini 3.9 Ultra (High)"
    assert "cli_version" not in recorded


@pytest.mark.asyncio
async def test_record_gemini_listing_match_probes_and_records_display_form(
    tmp_path: Path, monkeypatch
) -> None:
    """Item 8: when the chosen slug maps to a listing row, the DISPLAY form
    is what gets probed (and recorded) -- never record a form the probe did
    not prove."""
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    captured: dict[str, object] = {}

    async def capturing_probe(model, cwd):
        captured["model"] = model
        return None

    monkeypatch.setattr(sm, "probe_gemini", capturing_probe)
    monkeypatch.setattr(
        sm,
        "fetch_agy_listing",
        lambda *a, **k: [("gemini-3.9-ultra-high", "Gemini 3.9 Ultra (High)")],
    )
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    await sm.record_gemini("gemini-3.9-ultra-high", cwd="/tmp")
    assert captured["model"] == "Gemini 3.9 Ultra (High)"  # probed on display form
    recorded = mc.load_config(path)["gemini"]
    assert recorded["model"] == "Gemini 3.9 Ultra (High)"


@pytest.mark.asyncio
async def test_record_gemini_unlisted_model_probes_and_records_entered_form(
    tmp_path: Path, monkeypatch
) -> None:
    """A manual entry outside the listing keeps today's behavior: probe (and
    record) the canonicalized entered form as-is, even when the listing
    fetch itself succeeds but has no matching row."""
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    captured: dict[str, object] = {}

    async def capturing_probe(model, cwd):
        captured["model"] = model
        return None

    monkeypatch.setattr(sm, "probe_gemini", capturing_probe)
    monkeypatch.setattr(
        sm,
        "fetch_agy_listing",
        lambda *a, **k: [("gemini-3.1-pro-high", "Gemini 3.1 Pro (High)")],
    )
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    await sm.record_gemini("some-brand-new-preview-model", cwd="/tmp")
    assert captured["model"] == "some-brand-new-preview-model"
    recorded = mc.load_config(path)["gemini"]
    assert recorded["model"] == "some-brand-new-preview-model"


# --- record_opencode -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_opencode_probe_failure_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def failing_probe(model, cwd):
        return "OpenRouter 404: model not routable on this account"

    monkeypatch.setattr(sm, "probe_opencode", failing_probe)
    with pytest.raises(sm.ProbeFailure, match="404"):
        await sm.record_opencode("openrouter/deepseek/deepseek-v4-pro", cwd="/tmp")
    assert not path.exists()


@pytest.mark.asyncio
async def test_record_opencode_success_writes_model_and_cli_version(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_opencode", ok_probe)
    monkeypatch.setattr(sm, "_opencode_version", lambda *a, **k: "1.18.4")
    await sm.record_opencode("openrouter/deepseek/deepseek-v4-pro", cwd="/tmp")
    recorded = mc.load_config(path)["opencode"]
    assert recorded["model"] == "openrouter/deepseek/deepseek-v4-pro"
    assert recorded["cli_version"] == "1.18.4"


# --- validate_noninteractive_flags -----------------------------------------------


def test_validate_noninteractive_flags_unknown_seat_raises() -> None:
    with pytest.raises(ValueError, match="unknown seat"):
        sm.validate_noninteractive_flags("bogus", None)


def test_validate_noninteractive_flags_effort_on_gemini_raises() -> None:
    with pytest.raises(ValueError, match="--seat claude or codex"):
        sm.validate_noninteractive_flags("gemini", "high")


def test_validate_noninteractive_flags_effort_on_opencode_raises() -> None:
    with pytest.raises(ValueError, match="--seat claude or codex"):
        sm.validate_noninteractive_flags("opencode", "high")


def test_validate_noninteractive_flags_effort_on_claude_ok() -> None:
    sm.validate_noninteractive_flags("claude", "medium")


def test_validate_noninteractive_flags_bad_claude_effort_raises() -> None:
    with pytest.raises(ValueError, match="turbo"):
        sm.validate_noninteractive_flags("claude", "turbo")


def test_validate_noninteractive_flags_effort_on_codex_ok() -> None:
    sm.validate_noninteractive_flags("codex", "high")  # must not raise


def test_validate_noninteractive_flags_bad_codex_effort_raises() -> None:
    # Item 4: the VALID_EFFORTS membership check now runs here too, not
    # only inside record_codex once the record flow is already underway.
    with pytest.raises(ValueError, match="turbo"):
        sm.validate_noninteractive_flags("codex", "turbo")


def test_validate_noninteractive_flags_no_effort_ok_for_any_seat() -> None:
    for seat in sm.SEATS:
        sm.validate_noninteractive_flags(seat, None)  # must not raise


# --- CLI: quorum setup-models --------------------------------------------------

# On CI, Rich detects GITHUB_ACTIONS and emits ANSI styling into CliRunner's
# captured output, interleaving escape codes between tokens -- a raw substring
# assertion on option names then fails there while passing locally. Strip the
# styling before asserting. Reproduce locally with GITHUB_ACTIONS=true.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    return _ANSI_RE.sub("", output)


def test_setup_models_help_lists_seat_model_effort() -> None:
    result = runner.invoke(app, ["setup-models", "--help"])
    assert result.exit_code == 0
    plain = _plain(result.output)
    assert "--seat" in plain
    assert "--model" in plain
    assert "--effort" in plain
    assert "--host" in plain


def test_setup_models_noninteractive_unknown_seat_rejected(monkeypatch) -> None:
    result = runner.invoke(app, ["setup-models", "--seat", "bogus", "--model", "x"])
    assert result.exit_code != 0
    # --seat is a typer/click Enum choice now: rejection + the choice list
    # come natively from typer, not sm.validate_noninteractive_flags.
    plain = _plain(result.output)
    assert "bogus" in plain
    assert "not one of" in plain


def test_setup_models_noninteractive_effort_on_gemini_rejected() -> None:
    result = runner.invoke(
        app,
        ["setup-models", "--seat", "gemini", "--model", "x", "--effort", "high"],
    )
    assert result.exit_code != 0
    assert "--effort" in _plain(result.output)


def test_setup_models_noninteractive_seat_without_model_rejected() -> None:
    result = runner.invoke(app, ["setup-models", "--seat", "codex"])
    assert result.exit_code != 0


def test_setup_models_noninteractive_model_without_seat_rejected() -> None:
    result = runner.invoke(app, ["setup-models", "--model", "gpt-6-preview"])
    assert result.exit_code != 0


def test_setup_models_effort_alone_rejected() -> None:
    # --effort alone implies non-interactive intent but is incomplete; it must
    # not fall through to the interactive flow (which would silently ignore it).
    result = runner.invoke(app, ["setup-models", "--effort", "high"])
    assert result.exit_code != 0


def test_setup_models_noninteractive_probe_failure_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def failing_probe(model, effort, cwd):
        return "codex refused: unknown model"

    monkeypatch.setattr(sm, "probe_codex", failing_probe)
    result = runner.invoke(
        app, ["setup-models", "--seat", "codex", "--model", "bogus-model"]
    )
    assert result.exit_code != 0
    assert "unknown model" in result.output
    assert not path.exists()


def test_setup_models_noninteractive_probe_failure_is_not_bad_parameter(
    tmp_path: Path, monkeypatch
) -> None:
    # Item 5: a probe failure is a plain error (exit 1), not a formatted
    # "invalid parameter" usage error (exit 2 with a Usage:/Try '--help'
    # banner) -- the model id was syntactically fine, the live probe failed.
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def failing_probe(model, effort, cwd):
        return "codex refused: unknown model"

    monkeypatch.setattr(sm, "probe_codex", failing_probe)
    result = runner.invoke(
        app, ["setup-models", "--seat", "codex", "--model", "bogus-model"]
    )
    assert result.exit_code == 1
    assert "codex probe failed" in result.output
    assert "Invalid value" not in result.output
    assert "Try '" not in result.output


def test_setup_models_noninteractive_unknown_effort_is_bad_parameter(
    tmp_path: Path, monkeypatch
) -> None:
    # Item 5, the other half: an unrecognized --effort is a true
    # flag-validation error, so it must still surface as typer.BadParameter
    # (exit 2), not the plain probe-error path.
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    result = runner.invoke(
        app,
        [
            "setup-models",
            "--seat",
            "codex",
            "--model",
            "gpt-6-preview",
            "--effort",
            "turbo",
        ],
    )
    assert result.exit_code == 2
    assert "turbo" in result.output


def test_setup_models_noninteractive_bad_env_effort_is_bad_parameter(
    tmp_path: Path, monkeypatch
) -> None:
    # With --effort omitted, the effective effort comes from
    # CODE_QUORUM_CODEX_EFFORT (or the recorded choice) -- a bad value there
    # is user configuration, i.e. a usage error, and must be rejected as
    # BadParameter (exit 2) before any live work, exactly like a bad --effort
    # flag. The probe spy proves nothing live ran.
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    monkeypatch.setenv("CODE_QUORUM_CODEX_EFFORT", "turbo")

    async def must_not_run(model, effort, cwd):
        raise AssertionError("probe ran despite a bad effective effort")

    monkeypatch.setattr(sm, "probe_codex", must_not_run)
    result = runner.invoke(
        app, ["setup-models", "--seat", "codex", "--model", "gpt-6-preview"]
    )
    assert result.exit_code == 2
    assert "turbo" in result.output
    assert not path.exists()


def test_setup_models_noninteractive_bad_effort_rejected_before_any_live_work(
    tmp_path: Path, monkeypatch
) -> None:
    # Item 4: the codex VALID_EFFORTS check now lives in
    # validate_noninteractive_flags, which runs before the try block that
    # dispatches into the record flow -- so a bad --effort is rejected
    # before check_malformed_config() (a models.toml read) ever runs, let
    # alone a probe.
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    def tripwire() -> str | None:
        raise AssertionError("check_malformed_config must not run for a bad --effort")

    monkeypatch.setattr(sm, "check_malformed_config", tripwire)
    result = runner.invoke(
        app,
        [
            "setup-models",
            "--seat",
            "codex",
            "--model",
            "gpt-6-preview",
            "--effort",
            "turbo",
        ],
    )
    assert result.exit_code == 2
    assert "turbo" in result.output


def test_setup_models_noninteractive_record_flow_valueerror_crashes_loudly(
    tmp_path: Path, monkeypatch
) -> None:
    # Every usage error is raised as BadParameter before live work. ProbeFailure
    # is the expected runtime failure, so a plain ValueError from the record
    # flow is a BUG and must crash loudly with a traceback, not wear an
    # "Error: ..." wrapper (and never the "Invalid value" usage banner).
    # The malformed-listing raise simulated here is unreachable in real
    # code (record_gemini catches the parser error and falls back to an
    # empty listing); only a defect could raise this.
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def boom(model, *, cwd):
        raise ValueError("agy models: malformed row (too many tab fields): 'a\tb\tc'")

    monkeypatch.setattr(sm, "record_gemini", boom)
    result = runner.invoke(
        app, ["setup-models", "--seat", "gemini", "--model", "gemini-3-pro"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "Error: agy models" not in result.output
    assert "Invalid value" not in result.output


def test_setup_models_noninteractive_codex_success_writes_record(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, effort, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", ok_probe)
    monkeypatch.setattr(sm, "_codex_version", lambda *a, **k: "0.150.0")
    result = runner.invoke(
        app,
        [
            "setup-models",
            "--seat",
            "codex",
            "--model",
            "gpt-6-preview",
            "--effort",
            "high",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Recorded codex" in result.output
    recorded = mc.load_config(path)["codex"]
    assert recorded["model"] == "gpt-6-preview"
    assert recorded["effort"] == "high"
    assert recorded["cli_version"] == "0.150.0"


def test_setup_models_noninteractive_claude_success_writes_record(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, effort, cwd):
        return None

    monkeypatch.setattr(sm, "probe_claude", ok_probe)
    monkeypatch.setattr(sm, "_claude_version", lambda *a, **k: "2.1.0")
    result = runner.invoke(
        app,
        [
            "setup-models",
            "--seat",
            "claude",
            "--model",
            "claude-opus-5",
            "--effort",
            "medium",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Recorded claude" in result.output
    recorded = mc.load_config(path)["claude"]
    assert recorded["model"] == "claude-opus-5"
    assert recorded["effort"] == "medium"
    assert recorded["cli_version"] == "2.1.0"


def test_setup_models_noninteractive_claude_effort_defaults_to_current_effective(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("claude", {"model": "old-model", "effort": "xhigh"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    captured: dict[str, object] = {}

    async def capturing_probe(model, effort, cwd):
        captured["effort"] = effort
        return None

    monkeypatch.setattr(sm, "probe_claude", capturing_probe)
    monkeypatch.setattr(sm, "_claude_version", lambda *a, **k: None)
    result = runner.invoke(
        app,
        ["setup-models", "--seat", "claude", "--model", "claude-opus-5"],
    )
    assert result.exit_code == 0, result.output
    assert captured["effort"] == "xhigh"


def test_setup_models_noninteractive_codex_effort_defaults_to_current_effective(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    mc.record_choice("codex", {"model": "old-model", "effort": "xhigh"}, path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    captured: dict[str, object] = {}

    async def capturing_probe(model, effort, cwd):
        captured["effort"] = effort
        return None

    monkeypatch.setattr(sm, "probe_codex", capturing_probe)
    monkeypatch.setattr(sm, "_codex_version", lambda *a, **k: None)
    result = runner.invoke(
        app, ["setup-models", "--seat", "codex", "--model", "new-model"]
    )
    assert result.exit_code == 0, result.output
    assert captured["effort"] == "xhigh"


def test_setup_models_noninteractive_gemini_success_writes_record(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_gemini", ok_probe)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    result = runner.invoke(
        app, ["setup-models", "--seat", "gemini", "--model", "Gemini 3.1 Pro (High)"]
    )
    assert result.exit_code == 0, result.output
    recorded = mc.load_config(path)["gemini"]
    assert recorded["model"] == "Gemini 3.1 Pro (High)"


def test_setup_models_noninteractive_opencode_success_writes_record(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_opencode", ok_probe)
    monkeypatch.setattr(sm, "_opencode_version", lambda *a, **k: None)
    result = runner.invoke(
        app,
        [
            "setup-models",
            "--seat",
            "opencode",
            "--model",
            "openrouter/deepseek/deepseek-v4-pro",
        ],
    )
    assert result.exit_code == 0, result.output
    recorded = mc.load_config(path)["opencode"]
    assert recorded["model"] == "openrouter/deepseek/deepseek-v4-pro"


# --- indicates_absent_seat classifier + record_disabled ----------------------


@pytest.mark.parametrize(
    "error",
    [
        "codex not found on PATH",
        "claude is not logged in; run `claude auth login` in a terminal",
        "claude is not using a Claude.ai subscription; run `claude auth login`",
        "agy: authentication required",
        "agy: please sign in again",
        "agy: unauthorized",
        "agy: oauth token missing",
        "agy: invalid credentials",
        "agy: invalid token supplied",
        "agy: auth token rejected",
        "agy: access token expired",
        "agy: refresh token invalid",
        "agy: token expired, sign in again",
        "agy: credentials expired",
        "agy: session expired",
    ],
)
def test_indicates_absent_seat_true_for_absence_and_auth_failures(error: str) -> None:
    assert sm.indicates_absent_seat(error) is True


@pytest.mark.parametrize(
    "error",
    [
        "gemini probe for 'x' hit AI-Pro quota exhaustion and fell back to a "
        "different model -- retry setup later",
        "codex probe failed (exit 124)",  # generic/timeout-shaped, no hint
        "connection reset by peer",
        "codex refused",
    ],
)
def test_indicates_absent_seat_false_for_transient_failures(error: str) -> None:
    assert sm.indicates_absent_seat(error) is False


def test_record_disabled_preserves_existing_recorded_fields(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    mc.record_choice(
        "codex",
        {"model": "gpt-5.6-terra", "effort": "medium", "cli_version": "1.2.3"},
    )

    sm.record_disabled("codex")

    recorded = mc.recorded_choice("codex", path)
    assert recorded is not None
    assert recorded == {
        "model": "gpt-5.6-terra",
        "effort": "medium",
        "cli_version": "1.2.3",
        "disabled": "true",
        "chosen": recorded["chosen"],
    }


def test_record_disabled_on_seat_with_no_prior_record(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    sm.record_disabled("codex")

    recorded = mc.recorded_choice("codex", path)
    assert recorded is not None
    assert recorded == {"disabled": "true", "chosen": recorded["chosen"]}


def test_setup_models_interactive_default_flow_records_all_three_seats(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_codex(model, effort, cwd):
        return None

    async def ok_gemini(model, cwd):
        return None

    async def ok_opencode(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", ok_codex)
    monkeypatch.setattr(sm, "probe_gemini", ok_gemini)
    monkeypatch.setattr(sm, "probe_opencode", ok_opencode)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "fetch_opencode_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "_codex_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_opencode_version", lambda *a, **k: None)
    # 4 blank lines: codex model, codex effort, gemini model, opencode model --
    # each accepts the shown default (blank = keep, per the design spec).
    result = runner.invoke(app, ["setup-models"], input="\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert mc.recorded_choice("codex", path) is not None
    assert mc.recorded_choice("gemini", path) is not None
    assert mc.recorded_choice("opencode", path) is not None


def test_setup_models_interactive_codex_host_records_external_seats(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_claude(model, effort, cwd):
        return None

    async def ok_gemini(model, cwd):
        return None

    async def ok_opencode(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_claude", ok_claude)
    monkeypatch.setattr(sm, "probe_gemini", ok_gemini)
    monkeypatch.setattr(sm, "probe_opencode", ok_opencode)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "fetch_opencode_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "_claude_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_opencode_version", lambda *a, **k: None)
    result = runner.invoke(app, ["setup-models", "--host", "codex"], input="\n\n\n\n")
    assert result.exit_code == 0, result.output
    claude_choice = mc.recorded_choice("claude", path)
    assert claude_choice is not None
    assert claude_choice["effort"] == sm.CLAUDE_DEFAULT_EFFORT
    assert mc.recorded_choice("codex", path) is None
    assert mc.recorded_choice("gemini", path) is not None
    assert mc.recorded_choice("opencode", path) is not None


def test_setup_models_interactive_one_seat_probe_failure_does_not_block_others(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def failing_codex(model, effort, cwd):
        return "codex refused"

    async def ok_gemini(model, cwd):
        return None

    async def ok_opencode(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", failing_codex)
    monkeypatch.setattr(sm, "probe_gemini", ok_gemini)
    monkeypatch.setattr(sm, "probe_opencode", ok_opencode)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "fetch_opencode_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_opencode_version", lambda *a, **k: None)
    # 4 lines: codex model, codex effort, gemini model, opencode model. No
    # confirm line: "codex refused" is a generic or transient failure, so the
    # disable offer never fires or consumes a line of stdin.
    result = runner.invoke(app, ["setup-models"], input="\n\n\n\n")
    assert result.exit_code == 1, result.output  # a failed seat is a non-zero exit
    assert mc.recorded_choice("codex", path) is None  # nothing recorded
    assert "codex refused" in result.output
    assert "mark this seat as not used" not in result.output  # no offer
    assert "recorded as disabled" not in result.output
    assert mc.recorded_choice("gemini", path) is not None  # later seats still ran
    assert mc.recorded_choice("opencode", path) is not None


def test_setup_models_interactive_confirm_disable_records_flag_and_does_not_exit_1(
    tmp_path: Path, monkeypatch
) -> None:
    """When a live probe fails because login, subscription, or binary is
    missing, `quorum setup-models` offers to record the seat as `disabled`.
    Accepting must record the flag with no `model` field because none exists.
    It must count the seat as handled, so the complete run exits 0 instead of 1."""
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def failing_codex(model, effort, cwd):
        return "codex is not logged in"

    async def ok_gemini(model, cwd):
        return None

    async def ok_opencode(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", failing_codex)
    monkeypatch.setattr(sm, "probe_gemini", ok_gemini)
    monkeypatch.setattr(sm, "probe_opencode", ok_opencode)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "fetch_opencode_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_opencode_version", lambda *a, **k: None)
    # 5 lines: codex model, codex effort, "y" accepting the disable offer,
    # gemini model, opencode model.
    result = runner.invoke(app, ["setup-models"], input="\n\ny\n\n\n")
    assert result.exit_code == 0, result.output  # handled, doesn't block the run
    assert "codex: recorded as disabled" in result.output
    recorded = mc.recorded_choice("codex", path)
    assert recorded is not None
    assert recorded == {"disabled": "true", "chosen": recorded["chosen"]}
    assert "model" not in recorded
    assert mc.recorded_choice("gemini", path) is not None
    assert mc.recorded_choice("opencode", path) is not None


def test_setup_models_interactive_disable_preserves_existing_recorded_fields(
    tmp_path: Path, monkeypatch
) -> None:
    """Disabling a seat that has a recorded model and effort must preserve
    those fields with `disabled = "true"`, so re-enabling does not require a
    new probe."""
    path = _cfg(tmp_path)
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    mc.record_choice(
        "codex",
        {"model": "gpt-5.6-terra", "effort": "medium", "cli_version": "1.2.3"},
    )

    async def failing_codex(model, effort, cwd):
        return "codex binary not found on PATH"

    async def ok_gemini(model, cwd):
        return None

    async def ok_opencode(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", failing_codex)
    monkeypatch.setattr(sm, "probe_gemini", ok_gemini)
    monkeypatch.setattr(sm, "probe_opencode", ok_opencode)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "fetch_opencode_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_opencode_version", lambda *a, **k: None)
    result = runner.invoke(app, ["setup-models"], input="\n\ny\n\n\n")
    assert result.exit_code == 0, result.output
    recorded = mc.recorded_choice("codex", path)
    assert recorded is not None
    assert recorded["disabled"] == "true"
    assert recorded["model"] == "gpt-5.6-terra"
    assert recorded["effort"] == "medium"
    assert recorded["cli_version"] == "1.2.3"


# --- setup-models: malformed models.toml repair -----------------------------------


def test_check_malformed_config_catches_scalar_seat_value(
    tmp_path: Path, monkeypatch
) -> None:
    # Item 1: a structurally-invalid shape that load_config now rejects
    # (a scalar value for a known seat) must be caught by the SAME
    # malformed-file repair path as a TOML parse error, not just at
    # council-time load.
    path = _cfg(tmp_path)
    path.write_text('codex = "x"\n', encoding="utf-8")
    monkeypatch.setattr(mc, "CONFIG_PATH", path)
    message = sm.check_malformed_config()
    assert message is not None
    assert "proceeding as if no choices are recorded" in message


def test_setup_models_noninteractive_repairs_malformed_config(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    path.write_text("this is not [[ valid toml =\n", encoding="utf-8")
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_probe(model, effort, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", ok_probe)
    monkeypatch.setattr(sm, "_codex_version", lambda *a, **k: None)
    result = runner.invoke(
        app,
        ["setup-models", "--seat", "codex", "--model", "gpt-6-preview"],
    )
    # No traceback, no crash on the corrupt file -- the error text plus the
    # repair notice print once, then the run proceeds and records normally.
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert str(path) in result.output
    assert "proceeding as if no choices are recorded" in result.output
    assert "completing setup will overwrite the corrupt file" in result.output
    assert "Recorded codex" in result.output
    # The successful re-setup overwrote the corrupt file -- it now parses.
    recorded = mc.load_config(path)["codex"]
    assert recorded["model"] == "gpt-6-preview"


def test_setup_models_interactive_repairs_malformed_config(
    tmp_path: Path, monkeypatch
) -> None:
    path = _cfg(tmp_path)
    path.write_text("this is not [[ valid toml =\n", encoding="utf-8")
    monkeypatch.setattr(mc, "CONFIG_PATH", path)

    async def ok_codex(model, effort, cwd):
        return None

    async def ok_gemini(model, cwd):
        return None

    async def ok_opencode(model, cwd):
        return None

    monkeypatch.setattr(sm, "probe_codex", ok_codex)
    monkeypatch.setattr(sm, "probe_gemini", ok_gemini)
    monkeypatch.setattr(sm, "probe_opencode", ok_opencode)
    monkeypatch.setattr(sm, "fetch_agy_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "fetch_opencode_listing", lambda *a, **k: [])
    monkeypatch.setattr(sm, "_codex_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_agy_version", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_opencode_version", lambda *a, **k: None)
    result = runner.invoke(app, ["setup-models"], input="\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert str(path) in result.output
    assert "proceeding as if no choices are recorded" in result.output
    assert "completing setup will overwrite the corrupt file" in result.output
    # Message printed once, not once per seat.
    assert result.output.count("proceeding as if no choices are recorded") == 1
    assert mc.recorded_choice("codex", path) is not None
    assert mc.recorded_choice("gemini", path) is not None
    assert mc.recorded_choice("opencode", path) is not None
