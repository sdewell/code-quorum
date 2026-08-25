from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_codex_marketplace as marketplace_builder

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "verify_codex_approvals.py"
pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason="approval verifier is workshop-only",
)

if _SCRIPT.exists():
    _SPEC = importlib.util.spec_from_file_location("verify_codex_approvals", _SCRIPT)
    assert _SPEC is not None and _SPEC.loader is not None
    verifier = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(verifier)


def _config(*, missing: str | None = None) -> str:
    blocks = []
    for tool in verifier.APPROVED_TOOLS:
        if tool == missing:
            continue
        blocks.append(
            "\n".join(
                (
                    '[plugins."code-quorum@code-quorum".mcp_servers.'
                    f"{verifier.MCP_SERVER_NAME}."
                    f"tools.{tool}]",
                    'approval_mode = "approve"',
                )
            )
        )
    return "\n\n".join(blocks) + "\n"


def test_verify_accepts_all_six_approved_tools(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_config(), encoding="utf-8")

    assert verifier.approval_errors(path) == []


def test_verify_targets_the_distinct_codex_server() -> None:
    assert verifier.MCP_SERVER_NAME == "quorum_codex"
    assert verifier.MCP_SERVER_NAME == marketplace_builder.CODEX_MCP_SERVER_NAME


def test_verify_explains_legacy_namespace_migration(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _config().replace("mcp_servers.quorum_codex", "mcp_servers.quorum"),
        encoding="utf-8",
    )

    assert verifier.approval_errors(path) == [
        "approval namespace migrated from quorum to quorum_codex; "
        "review and approve the six new tool entries"
    ]


def test_verify_accepts_redacted_codex_0148_schema_fixture() -> None:
    path = _ROOT / "tests" / "fixtures" / "codex" / "code_quorum_approvals.toml"

    assert verifier.approval_errors(path) == []


def test_verify_names_missing_tool(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_config(missing="q_await"), encoding="utf-8")

    assert verifier.approval_errors(path) == ["q_await: missing"]


def test_verify_reports_non_approved_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _config().replace('approval_mode = "approve"', 'approval_mode = "prompt"', 1),
        encoding="utf-8",
    )

    assert verifier.approval_errors(path) == [
        "q_plan_start: approval_mode='prompt', expected 'approve'"
    ]


def test_verify_reports_malformed_table(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[plugins."code-quorum@code-quorum"]\nmcp_servers = "bad"\n',
        encoding="utf-8",
    )

    assert verifier.approval_errors(path) == ["mcp_servers: expected table"]


def test_verify_rejects_alternate_code_quorum_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _config()
        + '\n[plugins."code-quorum@unexpected-marketplace"]\nenabled = true\n',
        encoding="utf-8",
    )

    assert verifier.approval_errors(path) == [
        "unexpected plugin identity: code-quorum@unexpected-marketplace"
    ]


def test_verify_reports_unreadable_config_path(tmp_path: Path) -> None:
    assert verifier.approval_errors(tmp_path) == [f"config unreadable: {tmp_path}"]


def test_verify_expands_tilde_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text(_config(), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert verifier.approval_errors(Path("~/.codex/config.toml")) == []


def test_run_outside_code_quorum_environment_reports_clean_error() -> None:
    # -S skips site-packages, so `quorum` is not importable -- reproduces
    # running the script with a bare/non-venv interpreter instead of `uv run`.
    # Must fail with a clean instruction on stderr and exit 2, not an opaque
    # ModuleNotFoundError traceback.
    result = subprocess.run(
        [sys.executable, "-S", str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert (
        "verify_codex_approvals.py must run inside the code-quorum environment: "
        f"uv run --directory {_ROOT} python scripts/verify_codex_approvals.py"
        in result.stderr
    )
    assert "ModuleNotFoundError" not in result.stderr


def test_workshop_release_runbook_owns_github_install_verification() -> None:
    releasing = (_ROOT / "docs" / "RELEASING.md").read_text(encoding="utf-8")
    flat = " ".join(releasing.split())

    assert "--project-root /path/to/stable/code-quorum" in releasing
    assert "plugins/cache/code-quorum" not in releasing
    assert "codex plugin marketplace add sdewell/code-quorum --ref main" in releasing
    assert "config.toml.pre-X.Y.Z.$(date +%s)" in releasing
    assert "git -C /path/to/stable/code-quorum pull --ff-only" in releasing
    assert "copy only the six Code Quorum" in flat
    assert "approval blocks from" in releasing
    assert "Code Quorum version" in releasing
    assert "required GitHub-install verification" in flat
    assert "GitHub-install verification" in releasing
    assert "scripts/verify_codex_approvals.py" in releasing
    assert "workshop-only" in releasing
    assert "~/.local/share/code-quorum/marketplace-local-X.Y.Z" in releasing
    assert "source-swap fallback" in releasing
    assert "true first install" in releasing
    assert "one-time namespace migration" in flat
    assert "approve the six `quorum_codex` tools" in flat
    assert "one workflow cannot prompt for all six approvals" in flat
    assert "configure every entry shown in `SECURITY.md`" in flat
    assert "source checkout can now exercise" in flat
