"""Tests for scripts/sync_version.py — specifically, that the JSON
helpers update ONLY the targeted version fields and never overwrite
unrelated `version` keys that may appear elsewhere in the same file
(schema versions, dependency versions, future API versions, etc.).

Also guards the manifest shape: `plugin.json` carries the single
authoritative plugin `version` (the only one Claude Code reads for update
detection), and `marketplace.json` carries NO version field — the per-plugin
and metadata versions were removed because `plugin.json` silently overrides
them, so duplicating the string was redundant drift risk."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPT = _REPO_ROOT / "scripts" / "sync_version.py"
_spec = importlib.util.spec_from_file_location("sync_version", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sync_version = importlib.util.module_from_spec(_spec)
sys.modules["sync_version"] = sync_version
_spec.loader.exec_module(sync_version)

_PLUGIN_JSON = _REPO_ROOT / ".claude-plugin" / "plugin.json"
_CODEX_PLUGIN_JSON = _REPO_ROOT / ".codex-plugin" / "plugin.json"
_MARKETPLACE_JSON = _REPO_ROOT / ".claude-plugin" / "marketplace.json"


def test_plugin_json_is_sole_version_authority_with_display_name() -> None:
    # plugin.json holds the version Claude Code actually reads for `claude
    # plugin update`, plus the human-readable displayName for the UI.
    data = json.loads(_PLUGIN_JSON.read_text(encoding="utf-8"))
    assert isinstance(data.get("version"), str) and data["version"]
    assert data.get("displayName") == "Code Quorum"


def test_both_host_manifests_are_version_sync_targets() -> None:
    assert sync_version.CLAUDE_PLUGIN_JSON == _PLUGIN_JSON
    assert sync_version.CODEX_PLUGIN_JSON == _CODEX_PLUGIN_JSON
    assert sync_version.Q_HELP_SKILL == (_REPO_ROOT / "skills" / "q-help" / "SKILL.md")


def test_marketplace_json_has_no_redundant_version_and_has_display_name() -> None:
    # The per-plugin and metadata versions were removed: plugin.json wins
    # silently, so a marketplace copy is dead weight + drift risk. displayName
    # stays so the marketplace listing shows the friendly name.
    data = json.loads(_MARKETPLACE_JSON.read_text(encoding="utf-8"))
    assert "version" not in data.get("metadata", {})
    for plugin in data.get("plugins", []):
        assert "version" not in plugin, f"{plugin.get('name')} still pins a version"
        if plugin.get("name") == "code-quorum":
            assert plugin.get("displayName") == "Code Quorum"


def test_sync_version_no_longer_touches_marketplace() -> None:
    # The marketplace update helpers are gone; sync_version must not reference
    # marketplace.json as a version target anymore.
    assert not hasattr(sync_version, "update_marketplace_json_text")
    assert not hasattr(sync_version, "update_marketplace_json")


def test_plugin_update_touches_only_top_level_version() -> None:
    # Prettier-formatted (2-space indent), matching the real plugin.json
    # files update_plugin_json_text is line-anchored against.
    fixture = (
        "{\n"
        '  "name": "code-quorum",\n'
        '  "version": "0.0.0",\n'
        '  "skill_config": {\n'
        '    "version": "schema-v2"\n'
        "  },\n"
        '  "dependencies": [\n'
        "    {\n"
        '      "version": "7.7.7"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    result = sync_version.update_plugin_json_text(fixture, "1.0.0")
    out = json.loads(result)
    assert out["version"] == "1.0.0"
    assert out["skill_config"]["version"] == "schema-v2"
    assert out["dependencies"][0]["version"] == "7.7.7"


def test_plugin_update_preserves_prettier_formatting_and_nested_version() -> None:
    fixture = (
        '{\n  "nested": {\n    "version": "schema-v2"\n  },\n  "version": "0.0.0"\n}\n'
    )

    result = sync_version.update_plugin_json_text(fixture, "1.0.0")

    expected = (
        '{\n  "nested": {\n    "version": "schema-v2"\n  },\n  "version": "1.0.0"\n}\n'
    )
    assert result == expected


def test_plugin_update_missing_top_level_version_raises() -> None:
    fixture = {"name": "code-quorum"}
    with pytest.raises(SystemExit, match="top-level version"):
        sync_version.update_plugin_json_text(json.dumps(fixture), "1.0.0")


def test_q_help_skill_update_replaces_only_the_badge() -> None:
    fixture = (
        "## code-quorum — quick reference\n\n"
        "`code-quorum v0.0.0`\n\n"
        "### Modes\n"
        "Use `/q-plan <task>` — see `q-plan v1` notes elsewhere.\n"
    )
    out = sync_version.update_q_help_skill_text(fixture, "1.2.3")
    assert "`code-quorum v1.2.3`" in out
    assert "`code-quorum v0.0.0`" not in out
    # Unrelated backtick tokens (command names, other version-looking text)
    # must survive — only the line-anchored badge is rewritten.
    assert "`/q-plan <task>`" in out
    assert "`q-plan v1`" in out


def test_q_help_skill_update_missing_badge_raises() -> None:
    with pytest.raises(SystemExit, match="badge"):
        sync_version.update_q_help_skill_text("no badge in here\n", "1.0.0")
