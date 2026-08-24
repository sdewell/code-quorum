"""Sync derived version strings from pyproject.toml.

`pyproject.toml` `[project].version` is the single source of truth. This
script updates the derived locations:

- quorum/__init__.py                : __version__ = "..."
- .claude-plugin/plugin.json        : top-level version
- .codex-plugin/plugin.json         : top-level version
- skills/q-help/SKILL.md            : `code-quorum vX.Y.Z` cheat-sheet badge

Each host reads its own plugin manifest. Claude's manifest version silently
overrides any marketplace-entry version, so `marketplace.json` deliberately
carries no version field.

Plugin JSON is parsed for validation, then only the top-level `version` string
token is replaced. Formatting and every nested `version` field are left intact,
so the sync remains idempotent after Prettier formats the manifests.

Usage:
    uv run python scripts/sync_version.py

Exits non-zero if any file cannot be updated. Each write's own content is
computed and validated before the write happens, so a raise here means
nothing was mutated.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PYPROJECT = REPO_ROOT / "pyproject.toml"
INIT_PY = REPO_ROOT / "quorum" / "__init__.py"
CLAUDE_PLUGIN_JSON = REPO_ROOT / ".claude-plugin" / "plugin.json"
CODEX_PLUGIN_JSON = REPO_ROOT / ".codex-plugin" / "plugin.json"
Q_HELP_SKILL = REPO_ROOT / "skills" / "q-help" / "SKILL.md"


def read_pyproject_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    version = data.get("project", {}).get("version")
    if not isinstance(version, str):
        raise SystemExit(f"{path}: [project].version is missing or not a string")
    return version


def update_init_py(path: Path, version: str) -> str:
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'(?m)^__version__\s*=\s*"[^"]*"',
        f'__version__ = "{version}"',
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"{path}: could not find __version__ assignment to update")
    return new_text


def update_plugin_json_text(text: str, version: str) -> str:
    """Rewrite the top-level `"version"` string in a prettier-formatted
    plugin.json (2-space indent), leaving formatting and any nested
    `version` field untouched. Line-anchored on the 2-space top-level
    indent so a schema/dependency `version` nested deeper is never
    touched."""
    data = json.loads(text)
    if "version" not in data:
        raise SystemExit("plugin.json: missing top-level version")
    new_text, count = re.subn(
        r'(?m)^(  "version": )"[^"]*"',
        lambda m: f"{m.group(1)}{json.dumps(version)}",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("plugin.json: top-level version must be a string")
    json.loads(new_text)  # the substitution must still be well-formed JSON
    return new_text


def update_q_help_skill_text(text: str, version: str) -> str:
    """Rewrite the `code-quorum vX.Y.Z` badge line in the q-help cheat
    sheet. Line-anchored so unrelated backtick tokens (command names, other
    version-looking text) are never touched."""
    new_text, count = re.subn(
        r"(?m)^`code-quorum v[^`]+`",
        f"`code-quorum v{version}`",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(
            "q-help SKILL.md: could not find the `code-quorum vX.Y.Z` badge"
        )
    return new_text


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    version = read_pyproject_version(PYPROJECT)

    # Compute all new file contents before touching the filesystem. If any
    # helper raises (missing key, malformed JSON), no file has been mutated.
    planned: list[tuple[Path, str]] = [
        (INIT_PY, update_init_py(INIT_PY, version)),
        (
            CLAUDE_PLUGIN_JSON,
            update_plugin_json_text(
                CLAUDE_PLUGIN_JSON.read_text(encoding="utf-8"), version
            ),
        ),
        (
            CODEX_PLUGIN_JSON,
            update_plugin_json_text(
                CODEX_PLUGIN_JSON.read_text(encoding="utf-8"), version
            ),
        ),
        (
            Q_HELP_SKILL,
            update_q_help_skill_text(Q_HELP_SKILL.read_text(encoding="utf-8"), version),
        ),
    ]
    drift = [
        path
        for path, new_text in planned
        if path.read_text(encoding="utf-8") != new_text
    ]

    if check_only:
        if drift:
            print(
                "sync_version --check: derived versions out of sync; "
                f"run scripts/sync_version.py to sync them to {version!r}:",
                file=sys.stderr,
            )
            for p in drift:
                print(f"  {p.relative_to(REPO_ROOT)}", file=sys.stderr)
            return 1
        print(f"sync_version --check: all {len(planned)} locations synced at {version}")
        return 0

    drift_set = set(drift)
    for path, new_text in planned:
        if path in drift_set:
            path.write_text(new_text, encoding="utf-8")

    if drift:
        print(f"sync_version: updated to {version} ->")
        for p in drift:
            print(f"  {p.relative_to(REPO_ROOT)}")
    else:
        print(
            f"sync_version: all {len(planned)} locations already at {version} "
            "(no changes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
