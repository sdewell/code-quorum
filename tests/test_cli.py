import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quorum import cli as cli_mod
from quorum.agents.base import AgentResult
from quorum.cli import app
from quorum.orchestration import read_plan_file

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    # Rich colorizes --help and splits option names across ANSI segments
    # (e.g. --role -> "\x1b[36m-\x1b[0m\x1b[36m-role\x1b[0m"); CI forces
    # color on, so substring checks must run against stripped output.
    return _ANSI.sub("", text)


def test_root_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "q-plan" in out
    assert "q-brainstorm" in out
    assert "q-validate" in out
    assert "q-review" in out
    assert "research" in out
    assert "seat-helper" in out
    assert "install-seat-helper-launchagent" in out
    assert "update-codex" in out
    assert "auth-check" in out
    assert "doctor" in out


def test_seat_helper_status_reports_package_version(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli_mod, "launchagent_installed", lambda: True)
    monkeypatch.setattr(cli_mod, "launchagent_status", lambda: "loaded")
    monkeypatch.setattr(cli_mod, "helper_pid", lambda _spool: 123)
    monkeypatch.setattr(
        cli_mod,
        "helper_state",
        lambda _spool: {
            "protocol_version": 1,
            "code_quorum_version": "0.0.56",
            "allowed_roots": [],
        },
    )

    result = runner.invoke(app, ["seat-helper-status", "--spool-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Code Quorum version: 0.0.56" in result.output


def test_update_codex_reports_completed_version(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def _update(root: Path, *, progress):
        seen["root"] = root
        progress("Refreshing stable checkout")
        return "0.0.63"

    monkeypatch.setattr(cli_mod, "perform_codex_update", _update)

    result = runner.invoke(app, ["update-codex", "--project-root", str(tmp_path)])

    assert result.exit_code == 0
    assert seen["root"] == tmp_path.resolve()
    assert "Refreshing stable checkout" in result.output
    assert "Code Quorum 0.0.63 is installed for Codex" in result.output


def test_auth_check_claude_host_uses_direct_zero_quota_probe(
    monkeypatch, tmp_path
) -> None:
    seen = {}

    class _Gemini:
        async def check_auth(self, cwd: str):
            seen["cwd"] = cwd
            return AgentResult(agent="gemini", output="agy authentication is ready")

    monkeypatch.setattr(cli_mod, "GeminiCliAgent", _Gemini)

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["auth-check", "--seat", "gemini", "--host", "claude"])

    assert result.exit_code == 0
    assert result.output.strip() == "agy authentication is ready"
    assert seen["cwd"]


def test_auth_check_codex_host_uses_out_of_sandbox_helper(
    monkeypatch, tmp_path
) -> None:
    seen = {}

    async def _check(**kwargs):
        seen.update(kwargs)
        return AgentResult(agent="gemini", output="agy authentication is ready")

    monkeypatch.setattr(cli_mod, "run_gemini_auth_check_via_helper", _check)

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["auth-check", "--seat", "gemini", "--host", "codex"])

    assert result.exit_code == 0
    assert result.output.strip() == "agy authentication is ready"
    assert seen["cwd"]


def test_auth_check_honors_configured_runtime_host_when_flag_is_omitted(
    monkeypatch, tmp_path
) -> None:
    seen = {}

    async def _check(**kwargs):
        seen.update(kwargs)
        return AgentResult(agent="gemini", output="agy authentication is ready")

    monkeypatch.setenv("CODE_QUORUM_HOST", "codex")
    monkeypatch.setattr(cli_mod, "run_gemini_auth_check_via_helper", _check)

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["auth-check", "--seat", "gemini"])

    assert result.exit_code == 0
    assert seen["cwd"]


def test_q_validate_rejects_missing_plan_path() -> None:
    result = runner.invoke(app, ["q-validate", "/nonexistent/plan.md"])
    assert result.exit_code != 0


def test_q_validate_rejects_directory_argument(tmp_path: Path) -> None:
    result = runner.invoke(app, ["q-validate", str(tmp_path)])
    assert result.exit_code != 0


def test_q_validate_resolves_relative_plan_against_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    import quorum.cli as cli

    repo = tmp_path / "repo"
    repo.mkdir()
    plan = repo / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    seen: dict[str, Path] = {}

    class _ReadPlan(Exception):
        pass

    def capture(path: Path) -> str:
        seen["path"] = path
        raise _ReadPlan

    monkeypatch.setattr(cli, "_require_live_seats", lambda *_a: None)
    monkeypatch.setattr(cli, "read_plan_file", capture)

    result = runner.invoke(app, ["q-validate", "plan.md", "-C", str(repo)])

    assert isinstance(result.exception, _ReadPlan)
    assert seen["path"] == plan.resolve()


def test_q_validate_rejects_plan_outside_cwd(monkeypatch, tmp_path: Path) -> None:
    import quorum.cli as cli

    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "secret.md"
    secret.write_text("private", encoding="utf-8")
    monkeypatch.setattr(cli, "_require_live_seats", lambda *_a: None)

    async def unexpected_run(**_kwargs):
        raise AssertionError("outside plan must be rejected before council startup")

    monkeypatch.setattr(cli, "run_mode", unexpected_run)

    result = runner.invoke(app, ["q-validate", str(secret), "-C", str(repo)])

    assert result.exit_code != 0
    assert "outside working directory" in _plain(result.output)
    assert "Traceback" not in result.output


def test_read_plan_file_returns_text(tmp_path: Path) -> None:
    f = tmp_path / "plan.md"
    f.write_text("# A plan\n\nDo X.\n", encoding="utf-8")
    assert read_plan_file(f).startswith("# A plan")


def test_read_plan_file_rejects_too_large(tmp_path: Path) -> None:
    import pytest

    f = tmp_path / "huge.md"
    f.write_bytes(b"x" * 200_001)
    with pytest.raises(ValueError):
        read_plan_file(f)


def test_unknown_agent_rejected() -> None:
    result = runner.invoke(
        app, ["q-plan", "--no-context", "--agent", "bogus", "anything"]
    )
    assert result.exit_code != 0


def test_q_validate_help_has_extended_not_rounds() -> None:
    result = runner.invoke(app, ["q-validate", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--extended" in out
    assert "--rounds" not in out


def test_q_plan_help_has_role() -> None:
    result = runner.invoke(app, ["q-plan", "--help"])
    assert result.exit_code == 0
    assert "--role" in _plain(result.output)


def test_q_brainstorm_help_has_role() -> None:
    result = runner.invoke(app, ["q-brainstorm", "--help"])
    assert result.exit_code == 0
    assert "--role" in _plain(result.output)


def test_role_help_lists_every_stance() -> None:
    """The --role help must enumerate every stance, sourced from ROLES so it
    cannot drift (regression: visionary/pioneer were added everywhere but the
    CLI help string, and no test caught it)."""
    from quorum.cli import _ROLE_HELP
    from quorum.roles import ROLES

    for stance in ROLES:
        assert stance in _ROLE_HELP, f"{stance} missing from --role help"
    assert "visionary" in _ROLE_HELP
    assert "pioneer" in _ROLE_HELP


def test_role_bad_format_rejected() -> None:
    result = runner.invoke(
        app, ["q-plan", "--no-context", "--role", "skeptic", "anything"]
    )
    assert result.exit_code != 0


def test_role_unselected_agent_rejected() -> None:
    result = runner.invoke(
        app,
        [
            "q-plan",
            "--no-context",
            "--agent",
            "codex",
            "--role",
            "skeptic:gemini",
            "anything",
        ],
    )
    assert result.exit_code != 0


def test_q_plan_rejects_extended_flag() -> None:
    result = runner.invoke(app, ["q-plan", "--no-context", "--extended", "anything"])
    assert result.exit_code != 0


def test_research_subcommand_outputs_digest(monkeypatch):
    import quorum.cli as cli
    from quorum.research import Paper, ResearchDigest

    async def fake_research_topic(topic, **kw):
        return ResearchDigest(
            topic=topic,
            papers=(Paper("Found Paper", ("Ada",), 2024, "arxiv", "1", "u", "abs"),),
            libraries=(),
            errors=(),
        )

    monkeypatch.setattr(cli, "research_topic", fake_research_topic)
    result = runner.invoke(app, ["research", "diffusion models"])
    assert result.exit_code == 0
    assert "## Prior art for: diffusion models" in result.stdout
    assert "Found Paper" in result.stdout


def test_research_help_lists_source_and_limit():
    result = runner.invoke(app, ["research", "--help"])
    assert result.exit_code == 0
    # Drop Rich's option-box borders and collapse its line-wrapping so the
    # wrapped source list reads as one line.
    plain = " ".join(_plain(result.stdout).replace("│", " ").split())
    assert "--source" in plain
    assert "--limit" in plain
    # Pin the full source contract so a dropped source or a stale count (the
    # "all five" that omitted Europe PMC) fails here instead of drifting.
    assert "arxiv, openalex, europepmc, context7, github, huggingface" in plain
    assert "Default: all six." in plain


def test_q_research_is_reserved_for_host_skills() -> None:
    result = runner.invoke(app, ["q-research", "--help"])
    assert result.exit_code == 2
    assert "No such command 'q-research'" in _plain(result.output)
    root_help = runner.invoke(app, ["--help"])
    assert root_help.exit_code == 0
    assert "q-research" not in _plain(root_help.output)


def test_q_brainstorm_researches_by_default_and_seeds_the_prompt(monkeypatch):
    """A plain q-brainstorm runs research before the council, prints the
    digest, and seeds the digest into the council prompt as evidence."""
    import quorum.cli as cli
    from quorum.agents.base import AgentResult
    from quorum.research import Paper, ResearchDigest

    seen_prompts: list[str] = []

    async def fake_run_mode(**kw):
        seen_prompts.append(kw["prompt"])
        return [[AgentResult(agent="codex", output="agent idea", role="skeptic")]]

    async def fake_research_topic(topic, **kw):
        # Populate counts so the per-source footer and status-OK path run.
        return ResearchDigest(
            topic=topic,
            papers=(Paper("Prior Work", ("Ada",), 2024, "arxiv", "1", "u", "abs"),),
            libraries=(),
            errors=(),
            counts=(("arxiv", 1),),
        )

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    monkeypatch.setattr(cli, "research_topic", fake_research_topic)
    result = runner.invoke(app, ["q-brainstorm", "a topic"])  # no flag needed
    assert result.exit_code == 0
    assert "agent idea" in result.stdout  # council output
    assert "## Prior art for: a topic" in result.stdout  # digest printed
    # The digest is printed BEFORE the council output (research ran first).
    assert result.stdout.index("Prior Work") < result.stdout.index("agent idea")
    # And seeded into the council prompt as evidence, outside any do-NOT-repeat wrap.
    assert seen_prompts and "Prior Work" in seen_prompts[0]
    assert "do NOT repeat" not in seen_prompts[0]


def test_q_brainstorm_no_research_skips_research(monkeypatch):
    import quorum.cli as cli
    from quorum.agents.base import AgentResult

    called = False

    async def fake_run_mode(**kw):
        return [[AgentResult(agent="codex", output="agent idea", role="skeptic")]]

    async def fake_research_topic(topic, **kw):
        nonlocal called
        called = True
        raise AssertionError("research must not run under --no-research")

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    monkeypatch.setattr(cli, "research_topic", fake_research_topic)
    result = runner.invoke(app, ["q-brainstorm", "a topic", "--no-research"])
    assert result.exit_code == 0
    assert "agent idea" in result.stdout
    assert not called
    assert "Prior art" not in result.stdout


def test_q_brainstorm_does_not_seed_a_retry_recommended_digest(monkeypatch):
    """The CLI cannot act on RETRY-RECOMMENDED. It must show a known-bad
    digest to the user without seeding it into the quorum prompt as evidence."""
    import quorum.cli as cli
    from quorum.agents.base import AgentResult
    from quorum.research import ResearchDigest

    seen_prompts: list[str] = []

    async def fake_run_mode(**kw):
        seen_prompts.append(kw["prompt"])
        return [[AgentResult(agent="codex", output="agent idea", role="skeptic")]]

    async def fake_research_topic(topic, **kw):
        # Every paper source empty on a valid query -> RETRY-RECOMMENDED.
        return ResearchDigest(
            topic=topic, papers=(), counts=(("arxiv", 0), ("openalex", 0))
        )

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    monkeypatch.setattr(cli, "research_topic", fake_research_topic)
    result = runner.invoke(app, ["q-brainstorm", "diffusion flow matching"])
    assert result.exit_code == 0
    assert "RETRY-RECOMMENDED" in result.stdout  # user sees why
    assert seen_prompts and "RETRY-RECOMMENDED" not in seen_prompts[0]  # unseeded
    assert "Prior art" not in seen_prompts[0]


def test_q_brainstorm_research_crash_degrades_to_unseeded_run(monkeypatch):
    """A non-ValueError research crash (network stack or schema drift) must
    degrade to an unseeded run, not kill the command before the council starts."""
    import quorum.cli as cli
    from quorum.agents.base import AgentResult

    async def fake_run_mode(**kw):
        return [[AgentResult(agent="codex", output="agent idea", role="skeptic")]]

    async def fake_research_topic(topic, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    monkeypatch.setattr(cli, "research_topic", fake_research_topic)
    result = runner.invoke(app, ["q-brainstorm", "a topic"])
    assert result.exit_code == 0
    assert "agent idea" in result.stdout  # council still ran
    assert "[research unavailable: boom]" in result.stdout


def test_q_brainstorm_refused_topic_degrades_to_unseeded_run(monkeypatch):
    """A pre-flight-refused topic (all stopwords) must not kill the council:
    note the skip and run unseeded."""
    import quorum.cli as cli
    from quorum.agents.base import AgentResult

    seen_prompts: list[str] = []

    async def fake_run_mode(**kw):
        seen_prompts.append(kw["prompt"])
        return [[AgentResult(agent="codex", output="agent idea", role="skeptic")]]

    async def fake_research_topic(topic, **kw):
        raise ValueError("too generic")

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    monkeypatch.setattr(cli, "research_topic", fake_research_topic)
    result = runner.invoke(app, ["q-brainstorm", "a topic"])
    assert result.exit_code == 0
    assert "[research skipped: too generic]" in result.stdout
    assert "agent idea" in result.stdout
    assert seen_prompts and "Prior art" not in seen_prompts[0]  # unseeded


def test_research_rejects_unknown_source() -> None:
    # Bad --source is rejected up front (BadParameter), not run as an empty digest.
    result = runner.invoke(app, ["research", "--source", "arixv", "diffusion"])
    assert result.exit_code != 0
    assert "arixv" in _plain(result.output)


def test_research_rejects_non_positive_limit() -> None:
    result = runner.invoke(app, ["research", "--limit", "0", "diffusion"])
    assert result.exit_code != 0


def test_q_review_help_has_scope() -> None:
    result = runner.invoke(app, ["q-review", "--help"])
    assert result.exit_code == 0
    assert "--scope" in _plain(result.output)


@pytest.mark.parametrize(
    "command", ["q-plan", "q-brainstorm", "q-validate", "q-review"]
)
def test_council_command_help_has_host(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0
    assert "--host" in _plain(result.output)


def test_q_review_help_has_extended_not_rounds() -> None:
    result = runner.invoke(app, ["q-review", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--extended" in out
    assert "--rounds" not in out


def test_q_review_rejects_bad_target(monkeypatch) -> None:
    import quorum.cli as cli

    async def fake_resolve_review_target(target, cwd):
        raise ValueError(f"unrecognized review target: {target!r}")

    monkeypatch.setattr(cli, "resolve_review_target", fake_resolve_review_target)
    result = runner.invoke(app, ["q-review", "--no-context", "bogus-target"])
    assert result.exit_code != 0


def test_q_review_reports_bad_scope_without_traceback(monkeypatch, tmp_path) -> None:
    import quorum.cli as cli

    async def fake_resolve_review_target(target, cwd):
        return "diff", "branch review"

    monkeypatch.setattr(cli, "resolve_review_target", fake_resolve_review_target)
    result = runner.invoke(
        app,
        [
            "q-review",
            "--no-context",
            "--cwd",
            str(tmp_path),
            "--scope",
            str(tmp_path / "missing.md"),
        ],
    )

    assert result.exit_code != 0
    assert "scope file not found" in _plain(result.output)
    assert "Traceback" not in result.output


def test_q_review_empty_diff_short_circuits(monkeypatch) -> None:
    import quorum.cli as cli

    async def fake_resolve_review_target(target, cwd):
        return "", "no changes vs main"

    def fail_run_mode(**kw):
        raise AssertionError("council must not run on an empty diff")

    monkeypatch.setattr(cli, "resolve_review_target", fake_resolve_review_target)
    monkeypatch.setattr(cli, "run_mode", fail_run_mode)
    result = runner.invoke(app, ["q-review", "--no-context"])
    assert result.exit_code == 0
    assert "no changes" in result.stdout


def test_q_review_padded_all_target_runs_council(monkeypatch) -> None:
    # The resolver normalizes padding and returns the same explicit inspection
    # instruction as a bare `all` target.
    import quorum.cli as cli

    class _Ran(Exception):
        pass

    async def fake_resolve_review_target(target, cwd):
        return (
            "[WHOLE CODEBASE REVIEW]\nInspect the repository.",
            "whole-codebase review",
        )

    def sentinel_run_mode(**kw):
        raise _Ran

    monkeypatch.setattr(cli, "resolve_review_target", fake_resolve_review_target)
    monkeypatch.setattr(cli, "run_mode", sentinel_run_mode)
    result = runner.invoke(app, ["q-review", "  all  ", "--no-context"])
    # Padded "all" must NOT short-circuit — the council runs (run_mode reached).
    assert isinstance(result.exception, _Ran)


# --- setup-agy command -------------------------------------------------------
from quorum.agents import gemini_cli as gc  # noqa: E402


def test_setup_agy_writes_read_only_config(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "antigravity-cli" / "settings.json"
    real_default = gc.AGY_SETTINGS_PATH
    monkeypatch.setattr(gc, "AGY_SETTINGS_PATH", target)
    monkeypatch.setattr(gc, "_agy_version", lambda *a, **k: None)  # hermetic
    result = runner.invoke(app, ["setup-agy"])
    assert result.exit_code == 0, result.output
    assert target.exists()
    assert gc.verify_read_only_config(target) is None
    # The redirect is the guarantee the real home config is never touched.
    assert target != real_default


def test_setup_agy_refuses_malformed_existing(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text("{ broken", encoding="utf-8")
    monkeypatch.setattr(gc, "AGY_SETTINGS_PATH", target)
    monkeypatch.setattr(gc, "_agy_version", lambda *a, **k: None)  # hermetic
    result = runner.invoke(app, ["setup-agy"])
    assert result.exit_code != 0
    assert target.read_text(encoding="utf-8") == "{ broken"  # not clobbered


def test_setup_agy_warns_on_seat_version_drift(monkeypatch, tmp_path: Path) -> None:
    # After setup, surface that the installed agy differs from the version the
    # seat was live-verified against -- the place a user lands post-upgrade.
    target = tmp_path / "settings.json"
    monkeypatch.setattr(gc, "AGY_SETTINGS_PATH", target)
    monkeypatch.setattr(gc, "_agy_version", lambda *a, **k: "99.0.0")
    result = runner.invoke(app, ["setup-agy"])
    assert result.exit_code == 0, result.output
    assert "99.0.0" in result.output
    assert "SEAT_VERIFIED_AGY_VERSION" in result.output


@pytest.mark.parametrize(
    "command", ["q-plan", "q-brainstorm", "q-validate", "q-review"]
)
def test_council_command_help_has_verbose(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "--verbose" in _plain(result.output)


def test_q_plan_verbose_flag_reaches_run_mode(monkeypatch) -> None:
    import quorum.cli as cli
    from quorum.agents.base import AgentResult

    captured: dict[str, object] = {}

    async def fake_run_mode(**kw):
        captured.update(kw)
        return [[AgentResult(agent="codex", output="x", returncode=0)]]

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    result = runner.invoke(app, ["q-plan", "--no-context", "--verbose", "task"])
    assert result.exit_code == 0
    assert captured["verbose"] is True


def test_q_plan_host_flag_reaches_agent_selection(monkeypatch) -> None:
    import quorum.cli as cli
    from quorum.agents.base import AgentResult
    from quorum.agents.seat_helper import SeatHelperAgent

    captured: dict[str, object] = {}

    async def fake_run_mode(**kw):
        captured.update(kw)
        return [[AgentResult(agent="claude", output="x", returncode=0)]]

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    result = runner.invoke(
        app,
        [
            "q-plan",
            "--host",
            "codex",
            "--agent",
            "claude",
            "--no-context",
            "task",
        ],
    )

    assert result.exit_code == 0, result.output
    agents = captured["agents"]
    assert isinstance(agents, list)
    [agent] = agents
    assert isinstance(agent, SeatHelperAgent)


def test_q_brainstorm_verbose_flag_reaches_run_mode(monkeypatch) -> None:
    import quorum.cli as cli
    from quorum.agents.base import AgentResult

    captured: dict[str, object] = {}

    async def fake_run_mode(**kw):
        captured.update(kw)
        return [[AgentResult(agent="codex", output="x", returncode=0)]]

    async def tripwire_research(topic, **kw):
        raise AssertionError("unit test must never do live research")  # default-on

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    monkeypatch.setattr(cli, "research_topic", tripwire_research)
    result = runner.invoke(
        app, ["q-brainstorm", "--no-context", "--no-research", "--verbose", "topic"]
    )
    assert result.exit_code == 0
    assert captured["verbose"] is True


def test_q_validate_verbose_flag_reaches_run_mode(monkeypatch, tmp_path: Path) -> None:
    import quorum.cli as cli
    from quorum.agents.base import AgentResult

    captured: dict[str, object] = {}

    async def fake_run_mode(**kw):
        captured.update(kw)
        return [[AgentResult(agent="codex", output="x", returncode=0)]]

    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\nDo the thing.\n", encoding="utf-8")
    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    result = runner.invoke(
        app,
        ["q-validate", "--no-context", "--verbose", "-C", str(tmp_path), str(plan)],
    )
    assert result.exit_code == 0
    assert captured["verbose"] is True


def test_q_review_defaults_to_terse(monkeypatch) -> None:
    import quorum.cli as cli
    from quorum.agents.base import AgentResult

    captured: dict[str, object] = {}

    async def fake_resolve_review_target(target, cwd):
        return "diff", "branch review"

    async def fake_run_mode(**kw):
        captured.update(kw)
        return [[AgentResult(agent="codex", output="x", returncode=0)]]

    monkeypatch.setattr(cli, "resolve_review_target", fake_resolve_review_target)
    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    result = runner.invoke(app, ["q-review", "--no-context"])
    assert result.exit_code == 0
    assert captured["verbose"] is False


def test_q_review_verbose_flag_reaches_run_mode(monkeypatch) -> None:
    import quorum.cli as cli
    from quorum.agents.base import AgentResult

    captured: dict[str, object] = {}

    async def fake_resolve_review_target(target, cwd):
        return "diff", "branch review"

    async def fake_run_mode(**kw):
        captured.update(kw)
        return [[AgentResult(agent="codex", output="x", returncode=0)]]

    monkeypatch.setattr(cli, "resolve_review_target", fake_resolve_review_target)
    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    result = runner.invoke(app, ["q-review", "--no-context", "--verbose"])
    assert result.exit_code == 0
    assert captured["verbose"] is True


def test_q_plan_gemini_model_flag_reaches_seat(monkeypatch) -> None:
    # The per-run agy-model ask must match the MCP tools' gemini_model parameter
    # without mutating ambient environment state.
    import quorum.cli as cli
    from quorum.agents.base import AgentResult
    from quorum.agents.gemini_cli import GeminiCliAgent

    captured: dict[str, list] = {}

    async def fake_run_mode(**kw):
        captured["agents"] = kw["agents"]
        return [[AgentResult(agent="gemini", output="x", returncode=0)]]

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    result = runner.invoke(
        app,
        [
            "q-plan",
            "some task",
            "--agent",
            "gemini",
            "--gemini-model",
            "claude-opus-4-6-thinking",
            "--no-context",
        ],
    )
    assert result.exit_code == 0, result.output
    [seat] = captured["agents"]
    assert isinstance(seat, GeminiCliAgent)
    assert seat.model == "claude-opus-4-6-thinking"


def test_q_plan_all_disabled_default_roster_exits_nonzero(monkeypatch) -> None:
    """If every default-roster seat is disabled, the run must fail with a
    nonzero exit and name the disabled seats. It must not print blank output
    from `format_rounds` and exit successfully."""
    import quorum.cli as cli
    from quorum import model_config as mc

    async def tripwire_run_mode(**kw):
        raise AssertionError("run_mode must never be called with zero live seats")

    monkeypatch.setattr(cli, "run_mode", tripwire_run_mode)
    for seat in ("codex", "gemini", "opencode"):
        mc.record_choice(seat, {"disabled": "true"})

    result = runner.invoke(app, ["q-plan", "--no-context", "task"])

    assert result.exit_code != 0
    assert "No seats to run" in result.output
    assert "codex" in result.output
    assert "gemini" in result.output
    assert "opencode" in result.output
    assert "quorum setup-models" in result.output


def test_q_plan_partial_disable_reports_skipped_seat_on_cli(monkeypatch) -> None:
    """The CLI must report a skipped seat instead of omitting the disabled
    seat from format_rounds without an explanation."""
    import quorum.cli as cli
    from quorum import model_config as mc
    from quorum.agents.base import AgentResult

    async def fake_run_mode(**kw):
        return [[AgentResult(agent="gemini", output="x", returncode=0)]]

    monkeypatch.setattr(cli, "run_mode", fake_run_mode)
    mc.record_choice("codex", {"disabled": "true"})

    result = runner.invoke(app, ["q-plan", "--no-context", "task"])

    assert result.exit_code == 0, result.output
    assert "codex (disabled, skipped)" in result.output
