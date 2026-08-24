from __future__ import annotations

import json
import os
import subprocess
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from .agents.seat_helper import active_allowed_roots

PLUGIN_ID = "code-quorum@code-quorum"
MARKETPLACE_NAME = "code-quorum"
MCP_SERVER_NAME = "quorum_codex"
LEGACY_MCP_SERVER_NAME = "quorum"
APPROVED_TOOLS = (
    "q_plan_start",
    "q_research",
    "q_brainstorm_start",
    "q_validate_start",
    "q_review_start",
    "q_await",
)

RunCommand = Callable[..., subprocess.CompletedProcess[str]]
Progress = Callable[[str], None]
Sleep = Callable[[float], None]
_HELPER_STATUS_ATTEMPTS = 25
_HELPER_STATUS_DELAY_S = 0.2
_OFFICIAL_REMOTES = frozenset(
    {
        "git@github.com:sdewell/code-quorum",
        "https://github.com/sdewell/code-quorum",
        "ssh://git@github.com/sdewell/code-quorum",
    }
)


class CodexUpdateError(RuntimeError):
    """A supported Codex update step failed without completing the upgrade."""


def approval_errors(path: Path) -> list[str]:
    path = path.expanduser().resolve()
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return [f"config unreadable: {path}"]
    except tomllib.TOMLDecodeError as exc:
        return [f"invalid TOML: {exc}"]

    errors: list[str] = []

    def child_table(parent: object, key: str, label: str) -> dict[str, object]:
        if not isinstance(parent, dict):
            errors.append(f"{label}: expected table")
            return {}
        table = cast(dict[str, object], parent)
        value = table.get(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            errors.append(f"{label}: expected table")
            return {}
        return cast(dict[str, object], value)

    plugins = child_table(config, "plugins", "plugins")
    unexpected = sorted(
        name
        for name in plugins
        if name.startswith("code-quorum@") and name != PLUGIN_ID
    )
    errors.extend(f"unexpected plugin identity: {name}" for name in unexpected)
    plugin = child_table(plugins, PLUGIN_ID, PLUGIN_ID)
    servers = child_table(plugin, "mcp_servers", "mcp_servers")
    if MCP_SERVER_NAME not in servers and isinstance(
        servers.get(LEGACY_MCP_SERVER_NAME), dict
    ):
        errors.append(
            "approval namespace migrated from quorum to quorum_codex; "
            "review and approve the six new tool entries"
        )
        return errors
    server_label = f"mcp_servers.{MCP_SERVER_NAME}"
    quorum = child_table(servers, MCP_SERVER_NAME, server_label)
    tools = child_table(quorum, "tools", f"{server_label}.tools")
    if errors:
        return errors

    for name in APPROVED_TOOLS:
        tool = tools.get(name)
        if not isinstance(tool, dict):
            errors.append(f"{name}: missing")
            continue
        observed = cast(dict[str, object], tool).get("approval_mode")
        if observed != "approve":
            errors.append(f"{name}: approval_mode={observed!r}, expected 'approve'")
    return errors


def approval_snapshot(path: Path) -> object:
    """Return the user's plugin-scoped approval state without interpreting it."""
    path = path.expanduser().resolve()
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CodexUpdateError(f"Codex config is unreadable: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CodexUpdateError(f"Codex config is invalid TOML: {exc}") from exc

    plugins = config.get("plugins", {})
    if not isinstance(plugins, dict):
        raise CodexUpdateError("Codex config plugins entry is not a table")
    plugin = plugins.get(PLUGIN_ID, {})
    if not isinstance(plugin, dict):
        raise CodexUpdateError(f"Codex config {PLUGIN_ID} entry is not a table")
    servers = plugin.get("mcp_servers", {})
    if not isinstance(servers, dict):
        raise CodexUpdateError("Codex config mcp_servers entry is not a table")
    return servers


def _project_version(root: Path) -> str:
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CodexUpdateError(f"could not read the checkout version: {exc}") from exc
    project = data.get("project")
    if not isinstance(project, dict):
        raise CodexUpdateError("pyproject.toml project table is missing or malformed")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise CodexUpdateError("pyproject.toml has no project version")
    return version


def _run_step(
    label: str,
    argv: list[str],
    *,
    root: Path,
    run: RunCommand,
    progress: Progress,
) -> str:
    progress(label)
    try:
        result = run(argv, cwd=root, capture_output=True, text=True)
    except OSError as exc:
        raise CodexUpdateError(f"{label} failed: {exc}") from exc
    if result.returncode == 0:
        return result.stdout
    detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    raise CodexUpdateError(f"{label} failed: {detail}")


def _installed_plugin_version(payload: str) -> str:
    try:
        parsed: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CodexUpdateError(
            "Codex returned an unreadable plugin install result"
        ) from exc
    if not isinstance(parsed, dict) or parsed.get("pluginId") != PLUGIN_ID:
        raise CodexUpdateError("Codex installed an unexpected plugin identity")
    version = parsed.get("version")
    if not isinstance(version, str) or not version:
        raise CodexUpdateError("Codex did not report the installed plugin version")
    return version


def _quiet(_message: str) -> None:
    pass


def _check_stable_checkout(root: Path, *, run: RunCommand, progress: Progress) -> None:
    progress("Checking stable checkout")
    status = _run_step(
        "Read checkout status",
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        root=root,
        run=run,
        progress=_quiet,
    )
    if status.strip():
        raise CodexUpdateError("stable checkout has uncommitted changes")
    branch = _run_step(
        "Read checkout branch",
        ["git", "-C", str(root), "branch", "--show-current"],
        root=root,
        run=run,
        progress=_quiet,
    ).strip()
    if branch != "main":
        raise CodexUpdateError(f"stable checkout must be on main, not {branch!r}")
    remote = _run_step(
        "Read checkout origin",
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        root=root,
        run=run,
        progress=_quiet,
    ).strip()
    if remote.rstrip("/").removesuffix(".git") not in _OFFICIAL_REMOTES:
        raise CodexUpdateError(
            "stable checkout origin is not the official public repository"
        )


def _verify_pulled_main(root: Path, *, run: RunCommand) -> None:
    revisions = _run_step(
        "Verify refreshed checkout",
        ["git", "-C", str(root), "rev-parse", "HEAD", "origin/main"],
        root=root,
        run=run,
        progress=_quiet,
    ).splitlines()
    if len(revisions) != 2 or revisions[0] != revisions[1]:
        raise CodexUpdateError("stable checkout does not match origin/main after pull")


def perform_codex_update(
    project_root: Path,
    *,
    config_path: Path | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
    run: RunCommand = subprocess.run,
    progress: Progress = lambda _message: None,
    sleep: Sleep = time.sleep,
) -> str:
    """Update Code Quorum's Codex plugin, checkout, and helper in one command."""
    root = project_root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        raise CodexUpdateError(
            f"update-codex requires a stable Git checkout of code-quorum: {root}"
        )

    _check_stable_checkout(root, run=run, progress=progress)
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    approvals = config_path or (
        Path(codex_home).expanduser() / "config.toml"
        if codex_home
        else Path.home() / ".codex" / "config.toml"
    )
    approvals_before = approval_snapshot(approvals)
    roots = tuple(
        path.expanduser().resolve()
        for path in (
            allowed_roots if allowed_roots is not None else active_allowed_roots()
        )
    )

    _run_step(
        "Refreshing stable checkout",
        ["git", "-C", str(root), "pull", "--ff-only"],
        root=root,
        run=run,
        progress=progress,
    )
    _verify_pulled_main(root, run=run)
    version = _project_version(root)
    _run_step(
        "Preparing the stable runtime",
        ["uv", "sync", "--directory", str(root)],
        root=root,
        run=run,
        progress=progress,
    )
    _run_step(
        "Refreshing the Code Quorum marketplace",
        [
            "codex",
            "plugin",
            "marketplace",
            "upgrade",
            MARKETPLACE_NAME,
            "--json",
        ],
        root=root,
        run=run,
        progress=progress,
    )
    plugin_result = _run_step(
        "Installing the refreshed Code Quorum plugin",
        ["codex", "plugin", "add", PLUGIN_ID, "--json"],
        root=root,
        run=run,
        progress=progress,
    )
    installed = _installed_plugin_version(plugin_result)
    if installed != version:
        raise CodexUpdateError(
            f"installed plugin version {installed} does not match checkout {version}"
        )
    _run_step(
        "Validating Codex configuration",
        ["codex", "--strict-config", "--version"],
        root=root,
        run=run,
        progress=progress,
    )
    helper_install = [
        "uv",
        "run",
        "--directory",
        str(root),
        "quorum",
        "install-seat-helper-launchagent",
        "--project-root",
        str(root),
    ]
    for allowed_root in roots:
        helper_install.extend(["--allowed-root", str(allowed_root)])
    _run_step(
        "Restarting the shared seat helper",
        helper_install,
        root=root,
        run=run,
        progress=progress,
    )
    status_command = [
        "uv",
        "run",
        "--directory",
        str(root),
        "quorum",
        "seat-helper-status",
    ]
    expected_status = f"Code Quorum version: {version}"
    for attempt in range(_HELPER_STATUS_ATTEMPTS):
        status = _run_step(
            "Verifying the shared seat helper",
            status_command,
            root=root,
            run=run,
            progress=progress if attempt == 0 else _quiet,
        )
        if expected_status in status.splitlines():
            break
        if attempt + 1 < _HELPER_STATUS_ATTEMPTS:
            sleep(_HELPER_STATUS_DELAY_S)
    else:
        raise CodexUpdateError(f"seat helper did not report version {version}")

    if approval_snapshot(approvals) != approvals_before:
        raise CodexUpdateError("Codex approval settings changed during the update")
    return version
