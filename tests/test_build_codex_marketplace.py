from __future__ import annotations

import importlib.util
import json
import plistlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

from quorum_mcp.jobs import JOB_TTL_SECONDS

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "build_codex_marketplace.py"
_SPEC = importlib.util.spec_from_file_location("build_codex_marketplace", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
builder = importlib.util.module_from_spec(_SPEC)
sys.modules["build_codex_marketplace"] = builder
_SPEC.loader.exec_module(builder)


def _use_plistbuddy_test_double(plugin: Path, tmp_path: Path) -> None:
    """Replace macOS PlistBuddy in the staged launcher for a hermetic test."""
    fake = tmp_path / "PlistBuddy"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import plistlib\n"
        "import sys\n"
        "with open(sys.argv[-1], 'rb') as fh:\n"
        "    value = plistlib.load(fh).get('WorkingDirectory')\n"
        "if not isinstance(value, str) or not value:\n"
        "    raise SystemExit(1)\n"
        "print(value)\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    launcher = plugin / "scripts" / "run_codex_mcp_from_helper.sh"
    source = launcher.read_text(encoding="utf-8")
    platform_reader = "/usr/libexec/PlistBuddy"
    assert source.count(platform_reader) == 1
    launcher.write_text(source.replace(platform_reader, str(fake)), encoding="utf-8")


def test_build_codex_marketplace_stages_installable_plugin(tmp_path: Path) -> None:
    output = tmp_path / "marketplace"

    plugin = builder.build_marketplace(output, source_root=_ROOT)

    manifest = json.loads(
        (plugin / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    marketplace_path = output / ".agents" / "plugins" / "marketplace.json"
    marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
    mcp = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))
    hooks = json.loads((plugin / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    entry = marketplace["plugins"][0]
    assert manifest["name"] == "code-quorum"
    assert ".mcp.json" not in builder._FILES
    assert (plugin / ".mcp.json").is_file()
    assert (plugin / "ARCHITECTURE.md").is_file()
    assert (plugin / "SECURITY.md").is_file()
    assert (plugin / "skills" / "q-plan" / "SKILL.md").is_file()
    readme = (plugin / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]+)?\)", readme):
        assert (plugin / target).is_file(), f"plugin README link is missing: {target}"
    assert (plugin / "quorum" / "orchestration.py").is_file()
    session_start = hooks["hooks"]["SessionStart"]
    assert isinstance(session_start, list)
    commands = [
        hook["command"]
        for entry in session_start
        for hook in entry["hooks"]
        if hook["type"] == "command"
    ]
    script_paths = []
    for command in commands:
        paths = re.findall(r"\$\{CLAUDE_PLUGIN_ROOT\}/([^\"\s]+)", command)
        assert paths == [f"scripts/{builder.CODEX_PATH_LAUNCHER}"]
        assert (plugin / paths[0]).is_file()
        referenced = [
            name for name in builder._HOOK_SCRIPTS if f"scripts/{name}" in command
        ]
        assert len(referenced) == 1
        script_paths.extend(f"scripts/{name}" for name in referenced)
    assert set(script_paths) == {f"scripts/{name}" for name in builder._HOOK_SCRIPTS}
    assert all((plugin / path).is_file() for path in script_paths)
    assert all(
        isinstance(entry["matcher"], str)
        and entry["matcher"]
        and all(
            hook["type"] == "command"
            and isinstance(hook["command"], str)
            and hook["command"]
            and isinstance(hook["timeout"], int)
            and hook["timeout"] > 0
            for hook in entry["hooks"]
        )
        for entry in session_start
    )
    launcher = plugin / "scripts" / "run_in_plugin_root_with_user_path.sh"
    mcp_launcher = plugin / "scripts" / "run_codex_mcp_from_helper.sh"
    assert launcher.is_file()
    assert mcp_launcher.is_file()
    assert set(mcp["mcpServers"]) == {"quorum_codex"}
    assert mcp["mcpServers"]["quorum_codex"]["command"] == "/bin/sh"
    assert mcp["mcpServers"]["quorum_codex"]["args"] == [
        "./scripts/run_codex_mcp_from_helper.sh",
        manifest["version"],
    ]
    assert mcp["mcpServers"]["quorum_codex"]["cwd"] == "."
    assert mcp["mcpServers"]["quorum_codex"]["env_vars"] == list(
        builder.CODEX_MCP_ENV_VARS
    )
    assert "OPENROUTER_API_KEY" in builder.CODEX_MCP_ENV_VARS
    assert mcp["mcpServers"]["quorum_codex"]["tool_timeout_sec"] > JOB_TTL_SECONDS
    assert entry["name"] == "code-quorum"
    assert entry["source"] == {
        "source": "local",
        "path": "./plugins/code-quorum",
    }
    assert entry["policy"] == {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL",
    }
    assert not (output / "marketplace.json").exists()


def test_codex_mcp_launcher_recovers_user_bins_from_desktop_path(
    tmp_path: Path,
) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)
    fake_home = tmp_path / "home"
    fake_bin = fake_home / ".local" / "bin"
    fake_bin.mkdir(parents=True)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'printf "PATH=%s\\n" "$PATH"\n'
        'printf "PWD=%s\\n" "$PWD"\n'
        'printf "KEY=%s\\n" "$OPENROUTER_API_KEY"\n'
        'printf "%s\\n" "$@"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    result = subprocess.run(
        [
            "/bin/sh",
            str(plugin / "scripts" / "run_in_plugin_root_with_user_path.sh"),
            "uv",
            "run",
            "--directory",
            ".",
            "quorum-mcp",
            "--host",
            "codex",
        ],
        cwd=tmp_path,
        env={
            "HOME": str(fake_home),
            "OPENROUTER_API_KEY": "test-key",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].removeprefix("PATH=").split(":", 1)[0] == str(fake_bin)
    assert lines[1] == f"PWD={plugin}"
    assert lines[2] == "KEY=test-key"
    assert lines[3:] == ["run", "--directory", ".", "quorum-mcp", "--host", "codex"]


def test_codex_mcp_uses_stable_helper_runtime_without_uv(tmp_path: Path) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)
    _use_plistbuddy_test_double(plugin, tmp_path)
    fake_home = tmp_path / "home"
    launchagents = fake_home / "Library" / "LaunchAgents"
    launchagents.mkdir(parents=True)
    runtime = tmp_path / "stable-checkout"
    runtime_bin = runtime / ".venv" / "bin"
    runtime_bin.mkdir(parents=True)
    version = json.loads(
        (plugin / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )["version"]
    python = runtime_bin / "python"
    python.write_text(f"#!/bin/sh\necho {version}\n", encoding="utf-8")
    python.chmod(0o755)
    quorum_mcp = runtime_bin / "quorum-mcp"
    quorum_mcp.write_text(
        '#!/bin/sh\nprintf "PATH=%s\\n" "$PATH"\n'
        'printf "RUNTIME=%s\\n" "$0"\nprintf "%s\\n" "$@"\n',
        encoding="utf-8",
    )
    quorum_mcp.chmod(0o755)
    (launchagents / "com.code-quorum.seat-helper.plist").write_bytes(
        plistlib.dumps({"WorkingDirectory": str(runtime)})
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    user_bin = fake_home / ".local" / "bin"
    user_bin.mkdir(parents=True)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/bin/sh\necho "uv must not run during Codex MCP startup" >&2\nexit 99\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    server = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))[
        "mcpServers"
    ]["quorum_codex"]

    result = subprocess.run(
        [server["command"], *server["args"]],
        cwd=plugin,
        env={
            "HOME": str(fake_home),
            "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].removeprefix("PATH=").split(":", 1)[0] == str(user_bin)
    assert lines[1:] == [
        f"RUNTIME={quorum_mcp}",
        "--host",
        "codex",
    ]


def test_codex_mcp_rejects_stale_stable_runtime(tmp_path: Path) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)
    _use_plistbuddy_test_double(plugin, tmp_path)
    fake_home = tmp_path / "home"
    launchagents = fake_home / "Library" / "LaunchAgents"
    launchagents.mkdir(parents=True)
    runtime = tmp_path / "stale-checkout"
    runtime_bin = runtime / ".venv" / "bin"
    runtime_bin.mkdir(parents=True)
    version = json.loads(
        (plugin / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )["version"]
    python = runtime_bin / "python"
    python.write_text(
        "#!/bin/sh\n"
        'runtime_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"\n'
        'if [ "$PWD" = "$runtime_root" ]; then\n'
        "    echo 0.0.1\n"
        "else\n"
        f"    echo {version}\n"
        "fi\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    quorum_mcp = runtime_bin / "quorum-mcp"
    quorum_mcp.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    quorum_mcp.chmod(0o755)
    (launchagents / "com.code-quorum.seat-helper.plist").write_bytes(
        plistlib.dumps({"WorkingDirectory": str(runtime)})
    )
    server = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))[
        "mcpServers"
    ]["quorum_codex"]

    result = subprocess.run(
        [server["command"], *server["args"]],
        cwd=plugin,
        env={
            "HOME": str(fake_home),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 78
    assert "stable runtime version 0.0.1 does not match plugin" in result.stderr
    assert "reinstall the seat helper" in result.stderr


def test_codex_mcp_reports_invalid_helper_plist(tmp_path: Path) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)
    _use_plistbuddy_test_double(plugin, tmp_path)
    fake_home = tmp_path / "home"
    launchagents = fake_home / "Library" / "LaunchAgents"
    launchagents.mkdir(parents=True)
    (launchagents / "com.code-quorum.seat-helper.plist").write_bytes(
        plistlib.dumps({"Label": "com.code-quorum.seat-helper"})
    )
    server = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))[
        "mcpServers"
    ]["quorum_codex"]

    result = subprocess.run(
        [server["command"], *server["args"]],
        cwd=plugin,
        env={"HOME": str(fake_home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 78
    assert "helper plist has no valid WorkingDirectory" in result.stderr


def test_codex_mcp_forwards_research_provider_credentials(tmp_path: Path) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)
    mcp = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))
    forwarded = set(mcp["mcpServers"]["quorum_codex"]["env_vars"])

    assert {
        "OPENALEX_API_KEY",
        "QUORUM_OPENALEX_API_KEY",
        "QUORUM_OPENALEX_EMAIL",
        "CONTEXT7_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "QUORUM_HF_TOKEN",
    } <= forwarded


def test_codex_path_runner_fails_clearly_for_missing_command(tmp_path: Path) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    missing = "code-quorum-command-that-does-not-exist"

    result = subprocess.run(
        [
            "/bin/sh",
            str(plugin / "scripts" / "run_in_plugin_root_with_user_path.sh"),
            missing,
        ],
        cwd=tmp_path,
        env={"HOME": str(fake_home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 127
    assert f"code-quorum: {missing} not found" in result.stderr


def test_codex_hooks_recover_user_bins_from_desktop_path(tmp_path: Path) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)
    fake_home = tmp_path / "home"
    fake_bin = fake_home / ".local" / "bin"
    fake_bin.mkdir(parents=True)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    hooks = json.loads((plugin / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    commands = [
        hook["command"]
        for entry in hooks["hooks"]["SessionStart"]
        for hook in entry["hooks"]
    ]
    env = {
        "CLAUDE_PLUGIN_ROOT": str(plugin),
        "HOME": str(fake_home),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }

    for command in commands:
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            shell=True,
            executable="/bin/sh",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{command}: {result.stderr}"


def test_build_codex_marketplace_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "marketplace"
    output.mkdir()

    try:
        builder.build_marketplace(output, source_root=_ROOT)
    except FileExistsError as exc:
        assert str(output) in str(exc)
    else:
        raise AssertionError("existing output must not be overwritten")


def test_build_codex_marketplace_excludes_python_bytecode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in builder._FILES:
        shutil.copy2(_ROOT / name, source / name)
    for name in (*builder._DIRECTORIES, "scripts"):
        shutil.copytree(_ROOT / name, source / name)
    bytecode = source / "quorum" / "__pycache__"
    bytecode.mkdir(exist_ok=True)
    (bytecode / "stale.cpython-313.pyc").write_bytes(b"not Python bytecode")

    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=source)

    assert not list(plugin.rglob("__pycache__"))
    assert not list(plugin.rglob("*.pyc"))
