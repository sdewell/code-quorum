import json
import re
import tomllib
from pathlib import Path

import pytest

import quorum_mcp.server as mcp_server
from quorum.agents.opencode import DEFAULT_MODEL as OPENCODE_DEFAULT_MODEL
from quorum.agents.opencode import OPENROUTER_CHUNK_TIMEOUT_MS

_ROOT = Path(__file__).resolve().parent.parent
_SKILLS = _ROOT / "skills"
_SKILL_NAMES = tuple(sorted(path.name for path in _SKILLS.iterdir() if path.is_dir()))
_HOST_SLASH_RE = re.compile(
    rf"(?<![\w./~-])/(?:{'|'.join(map(re.escape, _SKILL_NAMES))})\b"
)
_PREFIX = "mcp__quorum_codex__"
_CLAUDE_PREFIX = "mcp__plugin_code-quorum_quorum__"
_COUNCIL_SKILLS = (
    "q-plan",
    "q-brainstorm",
    "q-skystorm",
    "q-validate",
    "q-review",
)


def _skill(name: str) -> str:
    return (_SKILLS / name / "SKILL.md").read_text(encoding="utf-8")


def _security() -> str:
    return (_ROOT / "SECURITY.md").read_text(encoding="utf-8")


def test_codex_manifest_uses_host_adapter_surfaces() -> None:
    manifest = json.loads(
        (_ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )

    assert manifest["name"] == "code-quorum"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"


def test_public_typecheck_excludes_generated_codex_plugin_copy() -> None:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "plugins/" in pyproject["tool"]["ty"]["src"]["exclude"]
    assert "plugins/" in pyproject["tool"]["ruff"]["exclude"]


def test_claude_source_mcp_identity_and_launcher_stay_unchanged() -> None:
    config = json.loads((_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    assert set(config["mcpServers"]) == {"quorum"}
    server = config["mcpServers"]["quorum"]

    assert server["command"] == "uv"
    assert "${CLAUDE_PLUGIN_ROOT}" in server["args"]


def test_codex_hook_trust_docs_match_per_definition_contract() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "review and trust each code-quorum command hook" in readme
    assert "each current definition hash is trusted" in readme
    assert "re-review and trust for that changed definition" in readme
    assert "each hook definition's current hash" in architecture


@pytest.mark.skipif(
    not (_ROOT / "docs" / "RELEASING.md").is_file(),
    reason="release runbook is workshop-only",
)
def test_codex_namespace_docs_match_generated_adapter() -> None:
    architecture = (_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    releasing = (_ROOT / "docs" / "RELEASING.md").read_text(encoding="utf-8")

    for text in (architecture, releasing):
        assert "`quorum_codex`" in text
        assert "`mcp__quorum_codex__*`" in text
    assert "cannot shadow" in releasing


@pytest.mark.parametrize("skill", (*_COUNCIL_SKILLS, "q-research"))
def test_codex_skills_use_codex_mcp_prefix(skill: str) -> None:
    text = _skill(skill)

    assert _PREFIX in text
    assert _CLAUDE_PREFIX in text
    assert "mcp__quorum__" not in text


@pytest.mark.parametrize(
    "skill,tool_names",
    (
        ("q-plan", ("q_plan_start", "q_await")),
        ("q-brainstorm", ("q_research", "q_brainstorm_start", "q_await")),
        ("q-validate", ("q_validate_start", "q_await")),
        ("q-review", ("q_review_start", "q_await")),
        ("q-research", ("q_research",)),
    ),
)
def test_codex_skills_name_each_exact_codex_tool(
    skill: str, tool_names: tuple[str, ...]
) -> None:
    text = _skill(skill)

    for tool_name in tool_names:
        assert f"{_PREFIX}{tool_name}" in text


def test_codex_skystorm_reference_names_each_exact_codex_tool() -> None:
    text = (_SKILLS / "q-skystorm" / "references" / "codex-host.md").read_text(
        encoding="utf-8"
    )

    for tool_name in ("q_research", "q_brainstorm_start", "q_await"):
        assert f"{_PREFIX}{tool_name}" in text
    assert "mcp__quorum__" not in text


@pytest.mark.parametrize("skill", ("q-plan", "q-brainstorm", "q-validate", "q-review"))
def test_shared_council_skills_document_host_routing(skill: str) -> None:
    text = _skill(skill)

    assert '`host: "claude"`' in text
    assert '`host: "codex"`' in text


def test_codex_skystorm_keeps_codex_native_contract() -> None:
    text = (_SKILLS / "q-skystorm" / "references" / "codex-host.md").read_text(
        encoding="utf-8"
    )

    assert 'agents`: `["claude", "gemini", "opencode"]`' in text
    assert (
        'roles`: `["visionary:claude", "pioneer:gemini", "visionary:opencode"]`' in text
    )
    assert "fresh Codex-native" in text
    assert '`fork_turns`: `"none"`' in text
    assert "supportive_grounder_<first-12-job-id>_retry" in text
    assert "orchestrator fallback" in text
    assert "mcp__quorum_codex__q_brainstorm_start" in text

    step5 = text.split("## Step 5")[1].split("## Step 6")[0]
    assert step5.split("\n", 1)[1].lstrip().startswith("Assemble a concise dream pool")
    assert "spawn_agent" in step5
    assert "mcp__quorum_codex__q_brainstorm_start" not in step5
    for label in (
        "Vision preserved",
        "Smallest falsifiable experiment",
        "Baseline and controls",
        "Success and stop thresholds",
        "Missing evidence or method gap",
        "Cheapest next implementation step",
    ):
        assert label in step5


def test_codex_skystorm_names_host_ideation_and_degraded_council_behavior() -> None:
    text = (_SKILLS / "q-skystorm" / "references" / "codex-host.md").read_text(
        encoding="utf-8"
    )

    assert "## Step 1: Write independent host ideas" in text
    assert "pioneer pass" not in text
    assert "Council degraded:" in text
    assert "each unavailable seat and its status" in text
    assert "seat helper as a likely cause" in text
    assert "uv run --directory /path/to/code-quorum quorum seat-helper-status" in text
    assert "uv run quorum seat-helper-status" not in text
    assert "available seats" in text
    assert "never describe the spread as complete or full" in text.lower()

    step5 = text.split("## Step 5")[1].split("## Step 6")[0]
    step6 = text.split("## Step 6")[1]
    assert "pioneer framing" not in step5
    assert "full source-attributed" not in step6


def test_codex_skystorm_ports_research_and_citation_contract() -> None:
    text = (_SKILLS / "q-skystorm" / "references" / "codex-host.md").read_text(
        encoding="utf-8"
    )

    assert "anchor → harvest → pivot → map" in text
    assert 'Anchor (`mode="grounded"`)' in text
    assert 'Pivot (`mode="exploratory"`)' in text
    assert "Absence is inconclusive" in text
    assert "Research status:" in text
    assert "backend/infrastructure failure" in text
    assert "re-anchor" in text
    assert "pivot digest" in text
    assert "load-bearing" in text
    assert "artifact itself" in text
    assert "abstract is not a full read" in text
    assert "future-dated" in text
    assert "tracking parameters" in text
    assert "results table (including N and" in text
    assert 'well-formed "no such record"' in text


def test_codex_quorum_cheat_sheet_matches_host_profile() -> None:
    text = _skill("q-help")

    assert "| Claude Code | Codex + Gemini + OpenCode |" in text
    assert "| Codex | Claude subscription + Gemini + OpenCode |" in text
    assert (
        "Codex uses external Claude, Gemini, and OpenCode dreamers plus a fresh "
        "native Codex supportive grounder."
    ) in text
    assert "other platform's seat is skeptic" in text


def test_public_invocation_docs_are_host_aware() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    help_skill = _skill("q-help")

    assert "`/q-plan <task>`" in readme
    assert "`$code-quorum:q-plan <task>`" in readme
    assert "`uv run quorum q-plan <task>`" in readme
    assert "`uv run quorum research <topic>`" in readme
    assert "host-only skill" in readme
    assert "Every command also runs from the shell" not in readme
    assert "codex plugin marketplace add sdewell/code-quorum --ref main" in readme
    codex_install = readme.split("**Codex:**", 1)[1].split("## Design", 1)[0]
    assert "build_codex_marketplace.py" not in codex_install

    assert "which slash command fits" not in help_skill
    assert "| Slash | When |" not in help_skill
    assert "`$code-quorum:q-plan <task>`" in help_skill
    assert "`uv run quorum research <topic>`" in help_skill


def test_public_readme_is_consolidated_and_links_public_guides() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")

    assert len(readme.split()) < 2_500
    assert "[Security and data boundaries](SECURITY.md)" in readme
    assert "[ARCHITECTURE.md](ARCHITECTURE.md)" in readme
    for target in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]+)?\)", readme):
        assert (_ROOT / target).is_file(), f"README link target is missing: {target}"


@pytest.mark.skipif(
    not (_ROOT / ".github" / "dependabot.yml").is_file(),
    reason="Dependabot configuration is workshop-only",
)
def test_workshop_dependency_monitoring_covers_uv_and_ci_audit() -> None:
    dependabot = (_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "package-ecosystem: uv" in dependabot
    assert "package-ecosystem: github-actions" in dependabot
    assert "uv run pip-audit --skip-editable" in ci
    assert '"pip-audit>=' in pyproject


def test_public_docs_explain_codex_openrouter_environment_and_restart() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    codex_install = readme.split("**Codex:**", 1)[1].split("## Design", 1)[0]

    assert "Codex CLI" in codex_install
    assert "ChatGPT desktop app" in codex_install
    assert "source ~/.zshrc.local" in codex_install
    assert 'launchctl setenv OPENROUTER_API_KEY "$OPENROUTER_API_KEY"' in codex_install
    assert "launchctl unsetenv OPENROUTER_API_KEY" in codex_install
    assert "subsequently launched application" in codex_install
    assert "current macOS login session" in codex_install
    assert "desktop app with a minimal `PATH`" in codex_install
    assert "`~/.local/bin`" in codex_install
    assert "`/opt/homebrew/bin`" in codex_install
    assert "fully quit and reopen" in codex_install
    assert "brand-new Codex thread" in codex_install
    assert "Do not resume" in codex_install


def test_public_codex_install_prepares_stable_mcp_runtime() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    codex_install = readme.split("**Codex:**", 1)[1].split("## Design", 1)[0]
    flat = " ".join(codex_install.split())

    assert "uv run quorum install-seat-helper-launchagent" in codex_install
    assert "prepared `.venv` directly" in codex_install
    assert "writable `uv` cache" in flat
    assert "uv run quorum update-codex" in codex_install


def test_public_install_logs_into_agy_before_code_quorum_setup() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    install = readme.split("## Install", 1)[1].split("## Read-only boundaries", 1)[0]

    assert "agy  # complete Google OAuth login" in install
    assert install.index("agy  # complete Google OAuth login") < install.index(
        "uv run quorum setup-agy"
    )


def test_public_codex_auth_check_follows_helper_installation() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    install = readme.split("## Install", 1)[1].split("## Read-only boundaries", 1)[0]

    helper = "uv run quorum install-seat-helper-launchagent"
    auth_check = "uv run quorum auth-check --seat gemini --host codex"
    assert helper in install
    assert auth_check in install
    assert install.index(helper) < install.index(auth_check)


def test_public_codex_upgrade_refreshes_checkout_and_helper_protocol() -> None:
    update = _security()

    assert "uv run quorum update-codex" in update
    assert "snapshots your chosen approval settings" in update
    assert "It never runs `codex plugin remove`" in update
    assert "helper protocol" in update.lower()


def test_public_docs_explain_claude_openrouter_environment_and_restart() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    claude_install = readme.split("**Claude Code:**", 1)[1].split("**Codex:**", 1)[0]
    flat = " ".join(claude_install.split())

    assert "Claude Code CLI" in claude_install
    assert "Claude desktop app" in claude_install
    assert "source ~/.zshrc.local" in claude_install
    assert 'launchctl setenv OPENROUTER_API_KEY "$OPENROUTER_API_KEY"' in claude_install
    assert "launchctl unsetenv OPENROUTER_API_KEY" in claude_install
    assert "quit Claude Code completely" in flat
    assert "start a new Claude Code session" in flat


def test_public_docs_explain_managed_opencode_configuration() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### OpenCode configuration", 1)[1].split(
        "### Disabling a seat", 1
    )[0]
    flat = " ".join(section.split())

    assert "does not load your personal OpenCode configuration" in flat
    assert "`OPENROUTER_API_KEY`" in section
    assert f"`{OPENCODE_DEFAULT_MODEL}`" in section
    assert f"`{OPENROUTER_CHUNK_TIMEOUT_MS}` milliseconds" in section
    assert "`OPENCODE_DISABLE_PROJECT_CONFIG=1`" in section
    assert "`OPENCODE_PURE=1`" in section
    assert "`Read`, `glob`, and `list`" in section
    assert "`.env.example`" in section
    assert "rebuilds this generated configuration" in flat
    assert "`~/.cache/code-quorum/opencode-debug`" in section
    assert "`0700`" in section and "`0600`" in section
    assert "20" in section
    assert "raw stdout/stderr" in flat


def test_public_docs_explain_research_sources_credentials_and_hf_check() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Research sources and credentials", 1)[1].split(
        "## Requirements", 1
    )[0]
    flat = " ".join(section.split())

    for source in (
        "arXiv",
        "OpenAlex",
        "Europe PMC",
        "Context7",
        "GitHub",
        "Hugging Face",
    ):
        assert source in section
    for variable in (
        "OPENALEX_API_KEY",
        "QUORUM_OPENALEX_API_KEY",
        "QUORUM_OPENALEX_EMAIL",
        "CONTEXT7_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "QUORUM_HF_TOKEN",
    ):
        assert variable in section
    assert "required for normal OpenAlex use" in flat
    assert "generated Codex adapter forwards" in flat
    assert "does not store their values" in flat
    assert "model IDs" in section
    assert "`[broadened]`" in section
    assert "`europepmc-published`" in section
    assert "`europepmc-preprints`" in section
    assert "all-time" in section and "recent" in section
    assert "Source/lane status" in section
    assert (
        "uv run pytest tests/test_research_live.py -m live -k huggingface -q" in section
    )


@pytest.mark.asyncio
async def test_public_docs_explain_codex_data_egress_and_opt_in_approval() -> None:
    security = _security()
    flat = " ".join(security.split())

    assert "## Codex approvals and authorization" in security
    assert "read-only does not mean data-local" in flat
    assert "read-only access to the selected `cwd`" in flat
    assert "bare `q-review`" in security
    assert "uncommitted tracked changes" in security
    assert "untracked filenames" in flat
    assert "`main...HEAD`" in security
    assert "does not restrict the external seats' read-only access" in flat
    assert "does not enforce universal `cwd` read confinement" in flat
    assert "Claude, Codex, and OpenCode" in flat
    assert "outside `$HOME`" in security
    assert "agy credential and runtime state" in flat
    assert "`CODE_QUORUM_HELPER_ALLOWED_ROOTS`" in security
    assert "every MCP workflow" in flat
    assert "Plan and scope documents" in flat
    assert "separate clean clone" in flat
    assert "not a security boundary" in flat
    assert "dedicated macOS VM or machine" in flat
    assert "read-only access to this repository directory" in flat
    assert 'approvals_reviewer = "auto_review"' in security
    approval_tools = re.findall(
        r"mcp_servers\.quorum_codex\.tools\.([a-z_]+)\]",
        security,
    )
    assert sorted(approval_tools) == [
        "q_await",
        "q_brainstorm_start",
        "q_plan_start",
        "q_research",
        "q_review_start",
        "q_validate_start",
    ]
    assert security.count('approval_mode = "approve"') == 6
    registered = {tool.name for tool in await mcp_server.mcp.list_tools()}
    assert set(approval_tools) <= registered
    assert "`q_await` is shared by every council mode" in flat
    assert "standing authorization" in flat
    assert "project `AGENTS.md`" in security
    assert "off-machine transmission" in flat
    assert "specific workflow" in flat
    assert "credential paths" in flat
    assert "supersedes per-run confirmation" in flat
    assert "not enforced by Code Quorum" in flat
    assert "`codex plugin add code-quorum@code-quorum`" in security
    assert "`codex plugin remove` deletes" in flat
    assert "one-time namespace migration" in flat
    assert "explicitly approve the six tools again" in flat
    assert (
        "old `quorum` approval blocks do not authorize `quorum_codex`" in flat.lower()
    )
    assert "`codex --strict-config --version`" in security
    assert "`codex plugin list`" in security
    assert "all six" in flat
    assert "verified with Codex CLI 0.148.0" in flat


def test_shipped_skill_cross_references_are_host_neutral() -> None:
    assert _SKILL_NAMES
    assert all(name.startswith("q-") for name in _SKILL_NAMES)
    exempted_claude_rows = 0
    for path in sorted(_SKILLS.rglob("*.md")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if path == _SKILLS / "q-help" / "SKILL.md" and re.fullmatch(
                r"\|\s*Claude Code\s*\|\s*`/q-[^`]+`\s*\|", line.strip()
            ):
                exempted_claude_rows += 1
                continue
            assert _HOST_SLASH_RE.search(line) is None, (
                f"{path.relative_to(_ROOT)}:{line_number} uses a Claude-only "
                "cross-reference"
            )
    assert exempted_claude_rows == 1, "expected one Claude invocation-table row"


def test_host_slash_pattern_distinguishes_commands_from_paths() -> None:
    assert _HOST_SLASH_RE.search('run "/q-review" now')
    assert _HOST_SLASH_RE.search("use **/q-plan**")
    assert _HOST_SLASH_RE.search("skills/q-review/SKILL.md") is None
    assert _HOST_SLASH_RE.search("~/q-plan") is None


def test_q_help_distinguishes_host_and_shell_research_flags() -> None:
    text = _skill("q-help")

    assert "`--exploratory` is host-only" in text
    assert "shell `research` command is the grounded digest" in text


@pytest.mark.parametrize("relative", ("quorum_mcp/server.py", "quorum_mcp/jobs.py"))
def test_mcp_module_docs_use_q_tool_names(relative: str) -> None:
    text = (_ROOT / relative).read_text(encoding="utf-8")

    assert "co_<mode>_start" not in text
    assert "quorum_co_<mode>_start" not in text
    assert _HOST_SLASH_RE.search(text) is None
