"""Tests for scripts/quorum_servers.py — the MCP server doctor.

Covers the pure parse/classify functions (parsing `ps` output, pairing
launcher+worker into instances, and the orphan/stale classification that
decides what --reap kills). The actual os.kill path is glue and not unit
tested. ps sample lines are assembled from short path constants so no source
line trips the E501 commit gate (which counts physical length inside strings).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPT = _REPO_ROOT / "scripts" / "quorum_servers.py"
_spec = importlib.util.spec_from_file_location("quorum_servers", _SCRIPT)
assert _spec is not None and _spec.loader is not None
quorum_servers = importlib.util.module_from_spec(_spec)
sys.modules["quorum_servers"] = quorum_servers
_spec.loader.exec_module(quorum_servers)

qs = quorum_servers

# Short path stems so assembled ps lines stay well under 88 source chars.
_CACHE = "/h/.claude/plugins/cache/code-quorum/code-quorum"
_REPO = "/h/Code/tools/code-quorum"


def _ps_sample() -> str:
    return "\n".join(
        [
            # plugin-scope 0.0.11, parent 30203 (alive) -> live
            f"30212 30203 24:49 uv run --directory {_CACHE}/0.0.11 quorum-mcp",
            f"30216 30212 24:49 {_CACHE}/0.0.11/.venv/bin/quorum-mcp",
            # project-scope (live repo, dev), parent 25361 (alive) -> live
            f"25483 25361 38:48 uv run --directory {_REPO} quorum-mcp",
            f"25484 25483 38:48 {_REPO}/.venv/bin/quorum-mcp",
            # plugin-scope 0.0.9, parent 1 -> ORPHAN (and stale)
            f"9001 1 01:00 uv run --directory {_CACHE}/0.0.9 quorum-mcp",
            f"9002 9001 01:00 {_CACHE}/0.0.9/.venv/bin/quorum-mcp",
            # an unrelated process must be ignored (no marker)
            "7777 6666 02:00 /usr/bin/some-other-daemon --serve",
        ]
    )


# pids whose parents are "alive": 30203 and 25361 are sessions; the launchers
# 30212/25483 keep their workers' parents alive. 9001's parent (1) is not here.
_ALIVE = {30203, 30212, 25361, 25483, 9001, 9002, 7777, 6666}


def _fake_os_kill(alive: set[int]):
    """Stand in for os.kill(pid, 0): raise ProcessLookupError for any pid
    not in `alive`, otherwise do nothing (signal "delivered")."""

    def _kill(pid: int, _sig: int) -> None:
        if pid not in alive:
            raise ProcessLookupError

    return _kill


def test_parse_ps_filters_to_quorum_and_extracts_fields() -> None:
    procs = qs.parse_ps(_ps_sample())
    pids = {p.pid for p in procs}
    assert 7777 not in pids, "non-quorum processes must be filtered out"
    assert pids == {30212, 30216, 25483, 25484, 9001, 9002}
    launcher = next(p for p in procs if p.pid == 30212)
    assert launcher.ppid == 30203
    assert launcher.etime == "24:49"
    assert launcher.is_launcher is True


def test_parse_ps_skips_the_doctor_itself() -> None:
    line = "5555 1 00:01 uv run python scripts/quorum_servers.py --reap quorum-mcp"
    assert qs.parse_ps(line) == []


def test_parse_ps_keeps_server_whose_path_contains_grep() -> None:
    # A real server launched from a path containing "grep" must not be dropped;
    # the doctor never spawns grep itself, so there is no grep line to filter.
    line = "4242 4000 03:00 uv run --directory /h/grep/cq quorum-mcp"
    procs = qs.parse_ps(line)
    assert len(procs) == 1 and procs[0].pid == 4242


def test_scope_and_version_detection() -> None:
    procs = {p.pid: p for p in qs.parse_ps(_ps_sample())}
    assert procs[30212].scope == "plugin"
    assert procs[30212].version == "0.0.11"
    assert procs[25483].scope == "project"
    assert procs[25483].version is None  # live repo carries no cache version
    assert procs[9001].version == "0.0.9"


def test_group_instances_pairs_launcher_with_worker() -> None:
    instances = qs.group_instances(qs.parse_ps(_ps_sample()))
    by_launcher = {
        inst.launcher.pid: inst for inst in instances if inst.launcher is not None
    }
    plugin = by_launcher[30212]
    assert plugin.worker is not None and plugin.worker.pid == 30216
    assert plugin.pids == [30212, 30216]
    assert plugin.scope == "plugin" and plugin.version == "0.0.11"


def test_group_instances_surfaces_launcherless_worker() -> None:
    # A worker whose launcher already exited becomes its own instance.
    line = f"9002 9001 01:00 {_CACHE}/0.0.9/.venv/bin/quorum-mcp"
    instances = qs.group_instances(qs.parse_ps(line))
    assert len(instances) == 1
    assert instances[0].launcher is None
    assert instances[0].worker is not None and instances[0].worker.pid == 9002


def test_is_pid_alive_process_lookup_error_means_dead(monkeypatch) -> None:
    monkeypatch.setattr(qs.os, "kill", _fake_os_kill(set()))
    assert qs.is_pid_alive(12345) is False


def test_is_pid_alive_permission_error_still_means_alive(monkeypatch) -> None:
    def _kill(pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(qs.os, "kill", _kill)
    assert qs.is_pid_alive(12345) is True


def test_is_orphan_ppid_one_or_dead(monkeypatch) -> None:
    instances = {
        inst.pids[0]: inst for inst in qs.group_instances(qs.parse_ps(_ps_sample()))
    }
    monkeypatch.setattr(qs.os, "kill", _fake_os_kill(_ALIVE))
    orphan = instances[9001]  # parent pid 1
    assert qs.is_orphan(orphan) is True
    live = instances[30212]  # parent 30203 in alive set
    assert qs.is_orphan(live) is False
    # parent alive in the snapshot but absent from the alive set -> orphan
    monkeypatch.setattr(qs.os, "kill", _fake_os_kill(_ALIVE - {30203}))
    assert qs.is_orphan(live) is True


def test_is_stale_compares_versions() -> None:
    assert qs.is_stale("0.0.9", "0.0.11") is True
    assert qs.is_stale("0.0.11", "0.0.11") is False
    assert qs.is_stale("0.1.0", "0.0.11") is False  # newer is not stale
    assert qs.is_stale(None, "0.0.11") is False  # dev/live repo never stale
    assert qs.is_stale("0.0.9", None) is False  # unknown current -> can't judge


def test_status_orphan_takes_precedence_over_stale(monkeypatch) -> None:
    instances = {
        inst.pids[0]: inst for inst in qs.group_instances(qs.parse_ps(_ps_sample()))
    }
    monkeypatch.setattr(qs.os, "kill", _fake_os_kill(_ALIVE))
    # 9001 is BOTH stale (0.0.9 < 0.0.11) and orphan (ppid 1): orphan wins.
    assert qs.status_of(instances[9001], "0.0.11") == "ORPHAN"
    assert qs.status_of(instances[30212], "0.0.11") == "live"


def test_status_stale_when_parent_alive(monkeypatch) -> None:
    # A stale-version instance whose parent IS alive is STALE, not ORPHAN.
    line = "\n".join(
        [
            f"8001 8000 05:00 uv run --directory {_CACHE}/0.0.9 quorum-mcp",
            f"8002 8001 05:00 {_CACHE}/0.0.9/.venv/bin/quorum-mcp",
        ]
    )
    inst = qs.group_instances(qs.parse_ps(line))[0]
    monkeypatch.setattr(qs.os, "kill", _fake_os_kill({8000, 8001}))
    assert qs.status_of(inst, "0.0.11") == "STALE"


def test_version_tuple_orders_correctly() -> None:
    assert qs.version_tuple("0.0.9") < qs.version_tuple("0.0.11")
    assert qs.version_tuple("0.1.0") > qs.version_tuple("0.0.99")


def test_current_version_prefers_plugin_json(tmp_path: Path) -> None:
    pj = tmp_path / "plugin.json"
    pj.write_text(json.dumps({"version": "0.0.11"}), encoding="utf-8")
    assert qs.current_version(plugin_json=pj, cache_dir=tmp_path / "nope") == "0.0.11"


def test_current_version_falls_back_to_highest_cache_dir(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for v in ("0.0.9", "0.0.10", "0.0.11"):
        (cache / v).mkdir(parents=True)
    missing = tmp_path / "absent.json"
    assert qs.current_version(plugin_json=missing, cache_dir=cache) == "0.0.11"


def test_run_ps_pins_to_absolute_system_binary(monkeypatch) -> None:
    # --reap signals whatever PIDs ps reports, so ps must never be resolved via
    # PATH (a shim could run arbitrary code / steer the kill). Pin it absolute.
    captured: dict[str, list[str]] = {}

    class _Result:
        stdout = ""

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(qs.subprocess, "run", fake_run)
    qs._run_ps("pid=")
    assert captured["cmd"][0] == qs.PS_BIN
    assert qs.PS_BIN.startswith("/"), "ps must be an absolute path, not PATH-resolved"


def _patch_collect(monkeypatch) -> list[int]:
    """Point main()'s collect() at the ps sample and record SIGTERM targets.
    The sample's 9001/9002 instance has ppid 1 -> ORPHAN at any current ver."""
    instances = qs.group_instances(qs.parse_ps(_ps_sample()))
    monkeypatch.setattr(qs, "collect", lambda: (instances, "0.0.12"))
    monkeypatch.setattr(qs.os, "kill", _fake_os_kill(_ALIVE))
    killed: list[int] = []
    monkeypatch.setattr(qs, "kill_pid", lambda pid: killed.append(pid) or True)
    return killed


def test_main_reap_quiet_is_silent_but_reaps_orphans(monkeypatch, capsys) -> None:
    # The SessionStart hook runs `--reap --quiet`: it must emit NO stdout (so it
    # injects no session context) yet still SIGTERM the orphaned instance.
    killed = _patch_collect(monkeypatch)
    rc = qs.main(["--reap", "--quiet"])
    assert rc == 0
    assert capsys.readouterr().out == ""
    assert killed == [9001, 9002]  # the ppid-1 orphan's launcher + worker


def test_main_quiet_without_reap_kills_nothing(monkeypatch, capsys) -> None:
    # --quiet alone is a silent dry run: no output, and nothing is signalled.
    killed = _patch_collect(monkeypatch)
    rc = qs.main(["--quiet"])
    assert rc == 0
    assert capsys.readouterr().out == ""
    assert killed == []


def test_main_reap_without_quiet_still_prints(monkeypatch, capsys) -> None:
    # The human CLI path is unchanged: --reap without --quiet prints the table.
    _patch_collect(monkeypatch)
    qs.main(["--reap"])
    assert "ORPHAN" in capsys.readouterr().out
