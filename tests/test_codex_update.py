from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from quorum import codex_update as codex_update_mod
from quorum.agents.seat_helper import legacy_default_allowed_roots
from quorum.codex_update import (
    APPROVED_TOOLS,
    CodexUpdateError,
    perform_codex_update,
)


def _write_project(root: Path, version: str = "0.0.63") -> None:
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "code-quorum"\nversion = "{version}"\n',
        encoding="utf-8",
    )


def _write_approvals(path: Path, *, missing: str | None = None) -> None:
    blocks = []
    for tool in APPROVED_TOOLS:
        if tool == missing:
            continue
        blocks.append(
            "\n".join(
                (
                    '[plugins."code-quorum@code-quorum".mcp_servers.'
                    f"quorum_codex.tools.{tool}]",
                    'approval_mode = "approve"',
                )
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _runner(
    calls: list[list[str]],
    *,
    plugin_version: str = "0.0.63",
    helper_version: str = "0.0.63",
    helper_ready_after: int = 1,
):
    status_calls = 0

    def run(argv, **_kwargs):
        nonlocal status_calls
        command = list(argv)
        calls.append(command)
        if command[-3:] == ["status", "--porcelain", "--untracked-files=normal"]:
            output = ""
        elif command[-2:] == ["branch", "--show-current"]:
            output = "main\n"
        elif command[-3:] == ["remote", "get-url", "origin"]:
            output = "https://github.com/sdewell/code-quorum.git\n"
        elif command[-3:] in (
            ["rev-parse", "HEAD", "origin/main"],
            ["rev-parse", "origin/main", "HEAD"],
        ):
            output = "abc123\nabc123\n"
        elif command[:3] == ["codex", "plugin", "add"]:
            output = json.dumps(
                {
                    "pluginId": "code-quorum@code-quorum",
                    "version": plugin_version,
                }
            )
        elif command[-1] == "seat-helper-status":
            status_calls += 1
            output = (
                f"Code Quorum version: {helper_version}\n"
                if status_calls >= helper_ready_after
                else "LaunchAgent state: loaded\nHelper running: no\n"
            )
        else:
            output = "ok\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    return run


def test_update_codex_runs_one_approval_preserving_sequence(tmp_path: Path) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)
    calls: list[list[str]] = []
    allowed = (tmp_path / "Code", tmp_path / "src")

    version = perform_codex_update(
        root, config_path=config, allowed_roots=allowed, run=_runner(calls)
    )

    assert version == "0.0.63"
    assert calls == [
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        ["git", "-C", str(root), "branch", "--show-current"],
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        ["git", "-C", str(root), "pull", "--ff-only"],
        ["git", "-C", str(root), "rev-parse", "HEAD", "origin/main"],
        ["uv", "sync", "--directory", str(root)],
        ["codex", "plugin", "marketplace", "upgrade", "code-quorum", "--json"],
        ["codex", "plugin", "add", "code-quorum@code-quorum", "--json"],
        ["codex", "--strict-config", "--version"],
        [
            "uv",
            "run",
            "--directory",
            str(root),
            "quorum",
            "install-seat-helper-launchagent",
            "--project-root",
            str(root),
            "--allowed-root",
            str(allowed[0]),
            "--allowed-root",
            str(allowed[1]),
        ],
        [
            "uv",
            "run",
            "--directory",
            str(root),
            "quorum",
            "seat-helper-status",
        ],
    ]
    assert all("plugin remove" not in " ".join(command) for command in calls)


def test_update_codex_omits_allowed_root_flags_when_live_roots_are_legacy_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A helper still running under the pre-upgrade two-root default must not
    # have that legacy pair re-forwarded as explicit --allowed-root flags --
    # doing so would freeze it there and the new default root would never
    # apply after the reinstall.
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)
    calls: list[list[str]] = []
    legacy = legacy_default_allowed_roots()
    monkeypatch.setattr(codex_update_mod, "active_allowed_roots", lambda: legacy)

    perform_codex_update(
        root, config_path=config, allowed_roots=None, run=_runner(calls)
    )

    install_call = next(
        call for call in calls if "install-seat-helper-launchagent" in call
    )
    assert "--allowed-root" not in install_call


def test_update_codex_forwards_custom_live_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)
    calls: list[list[str]] = []
    custom = (tmp_path / "work",)
    monkeypatch.setattr(codex_update_mod, "active_allowed_roots", lambda: custom)

    perform_codex_update(
        root, config_path=config, allowed_roots=None, run=_runner(calls)
    )

    install_call = next(
        call for call in calls if "install-seat-helper-launchagent" in call
    )
    assert install_call.count("--allowed-root") == 1
    assert str((tmp_path / "work").resolve()) in install_call


def test_update_codex_preserves_the_users_approval_choices(tmp_path: Path) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config, missing="q_await")
    before = config.read_text(encoding="utf-8")
    calls: list[list[str]] = []

    version = perform_codex_update(
        root, config_path=config, allowed_roots=(), run=_runner(calls)
    )

    assert version == "0.0.63"
    assert config.read_text(encoding="utf-8") == before


def test_update_codex_rejects_changed_approval_state(tmp_path: Path) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)
    base_run = _runner([])

    def change_approvals(argv, **kwargs):
        command = list(argv)
        result = base_run(command, **kwargs)
        if command[:3] == ["codex", "plugin", "add"]:
            _write_approvals(config, missing="q_await")
        return result

    with pytest.raises(CodexUpdateError, match="approval settings changed"):
        perform_codex_update(
            root, config_path=config, allowed_roots=(), run=change_approvals
        )


def test_update_codex_reports_the_failed_step(tmp_path: Path) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)
    base_run = _runner([])

    def fail_sync(argv, **kwargs):
        command = list(argv)
        if command[:2] == ["uv", "sync"]:
            return subprocess.CompletedProcess(
                command, 2, stdout="", stderr="lockfile is stale"
            )
        return base_run(command, **kwargs)

    with pytest.raises(
        CodexUpdateError, match="Preparing the stable runtime.*lockfile is stale"
    ):
        perform_codex_update(root, config_path=config, allowed_roots=(), run=fail_sync)


@pytest.mark.parametrize(
    ("plugin_version", "helper_version", "message"),
    [
        ("0.0.62", "0.0.63", "installed plugin version 0.0.62"),
        ("0.0.63", "0.0.62", "seat helper did not report version 0.0.63"),
    ],
)
def test_update_codex_rejects_version_mismatch(
    tmp_path: Path,
    plugin_version: str,
    helper_version: str,
    message: str,
) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)

    with pytest.raises(CodexUpdateError, match=message):
        perform_codex_update(
            root,
            config_path=config,
            allowed_roots=(),
            run=_runner(
                [], plugin_version=plugin_version, helper_version=helper_version
            ),
        )


def test_update_codex_requires_a_git_checkout(tmp_path: Path) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "code-quorum"\nversion = "0.0.63"\n',
        encoding="utf-8",
    )

    with pytest.raises(CodexUpdateError, match="stable Git checkout"):
        perform_codex_update(root, config_path=tmp_path / "missing.toml")


def test_update_codex_waits_for_the_restarted_helper(tmp_path: Path) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)
    calls: list[list[str]] = []
    sleeps: list[float] = []

    version = perform_codex_update(
        root,
        config_path=config,
        allowed_roots=(),
        run=_runner(calls, helper_ready_after=3),
        sleep=sleeps.append,
    )

    assert version == "0.0.63"
    assert sum(command[-1] == "seat-helper-status" for command in calls) == 3
    assert sleeps == [0.2, 0.2]


def test_update_codex_honors_codex_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_approvals(codex_home / "config.toml")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert perform_codex_update(root, allowed_roots=(), run=_runner([])) == "0.0.63"


@pytest.mark.parametrize(
    ("probe", "output", "message"),
    [
        ("status", " M quorum/cli.py\n", "uncommitted changes"),
        ("branch", "feature\n", "must be on main"),
        ("remote", "git@github.com:someone/fork.git\n", "official public repository"),
    ],
)
def test_update_codex_refuses_an_unstable_checkout(
    tmp_path: Path, probe: str, output: str, message: str
) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)
    base_run = _runner([])

    def unstable(argv, **kwargs):
        command = list(argv)
        if probe == "status" and "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        if probe == "branch" and "branch" in command:
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        if probe == "remote" and "get-url" in command:
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        return base_run(command, **kwargs)

    with pytest.raises(CodexUpdateError, match=message):
        perform_codex_update(root, config_path=config, allowed_roots=(), run=unstable)


def test_update_codex_reports_malformed_project_table(tmp_path: Path) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text('project = "bad"\n', encoding="utf-8")
    config = tmp_path / "config.toml"
    _write_approvals(config)

    with pytest.raises(CodexUpdateError, match="project table"):
        perform_codex_update(
            root, config_path=config, allowed_roots=(), run=_runner([])
        )


def test_update_codex_requires_pulled_main_to_match_origin(tmp_path: Path) -> None:
    root = tmp_path / "code-quorum"
    root.mkdir()
    _write_project(root)
    config = tmp_path / "config.toml"
    _write_approvals(config)
    base_run = _runner([])

    def stale_main(argv, **kwargs):
        command = list(argv)
        if command[-3:] == ["rev-parse", "HEAD", "origin/main"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="local\nremote\n", stderr=""
            )
        return base_run(command, **kwargs)

    with pytest.raises(CodexUpdateError, match="does not match origin/main"):
        perform_codex_update(root, config_path=config, allowed_roots=(), run=stale_main)
