"""Stage a self-contained local Codex marketplace from this source tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from quorum.agents.base import RESEARCH_PROVIDER_ENV_VARS
from quorum_mcp.jobs import JOB_TTL_SECONDS

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_NAME = "code-quorum"
CODEX_MCP_SERVER_NAME = "quorum_codex"
MARKETPLACE_TOP_LEVELS = (".agents", "plugins")
CODEX_MCP_ENV_VARS = (
    "OPENROUTER_API_KEY",
    *RESEARCH_PROVIDER_ENV_VARS,
)
CODEX_PATH_LAUNCHER = "run_in_plugin_root_with_user_path.sh"
CODEX_MCP_LAUNCHER = "run_codex_mcp_from_helper.sh"
_FILES = (
    "ARCHITECTURE.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
)
_DIRECTORIES = (".codex-plugin", "hooks", "quorum", "quorum_mcp", "skills")
_HOOK_SCRIPTS = (
    "agy-drift-nag.sh",
    "prune_opencode_db.py",
    "quorum_servers.py",
)
_CODEX_SCRIPTS = (*_HOOK_SCRIPTS, CODEX_PATH_LAUNCHER, CODEX_MCP_LAUNCHER)


def _marketplace_payload() -> dict[str, object]:
    return {
        "name": PLUGIN_NAME,
        "interface": {"displayName": "Code Quorum"},
        "plugins": [
            {
                "name": PLUGIN_NAME,
                "source": {
                    "source": "local",
                    "path": f"./plugins/{PLUGIN_NAME}",
                },
                "policy": {
                    "installation": "AVAILABLE",
                    "authentication": "ON_INSTALL",
                },
                "category": "Developer Tools",
            }
        ],
    }


def _codex_mcp_payload(version: str) -> dict[str, object]:
    return {
        "mcpServers": {
            CODEX_MCP_SERVER_NAME: {
                "command": "/bin/sh",
                "args": [
                    f"./scripts/{CODEX_MCP_LAUNCHER}",
                    version,
                ],
                "cwd": ".",
                "env_vars": list(CODEX_MCP_ENV_VARS),
                "tool_timeout_sec": int(JOB_TTL_SECONDS + 300),
            }
        }
    }


def build_marketplace(output: Path, *, source_root: Path = REPO_ROOT) -> Path:
    output = output.expanduser().resolve()
    source = source_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        plugin = stage / "plugins" / PLUGIN_NAME
        plugin.mkdir(parents=True)
        for name in _FILES:
            shutil.copy2(source / name, plugin / name)
        for name in _DIRECTORIES:
            shutil.copytree(
                source / name,
                plugin / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        manifest = json.loads(
            (plugin / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
        )
        version = manifest.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("Codex plugin manifest is missing a version")
        scripts = plugin / "scripts"
        scripts.mkdir()
        for name in _CODEX_SCRIPTS:
            shutil.copy2(source / "scripts" / name, scripts / name)
        (plugin / ".mcp.json").write_text(
            json.dumps(_codex_mcp_payload(version), indent=2) + "\n",
            encoding="utf-8",
        )
        marketplace = stage / ".agents" / "plugins" / "marketplace.json"
        marketplace.parent.mkdir(parents=True)
        marketplace.write_text(
            json.dumps(_marketplace_payload(), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output / "plugins" / PLUGIN_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dist" / "codex-marketplace",
        help="New marketplace directory to create (must not already exist).",
    )
    args = parser.parse_args()
    try:
        plugin = build_marketplace(args.output)
    except (FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(f"Codex marketplace: {plugin.parent.parent}")
    print(f"Plugin root: {plugin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
