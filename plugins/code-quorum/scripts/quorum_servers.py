"""Inspect and reap orphaned/stale code-quorum MCP server processes.

`uv run quorum-mcp` runs one server per Claude Code registration as TWO
processes (the `uv` launcher + the venv python it execs). The repo's
`.mcp.json` registers at plugin scope (`${CLAUDE_PLUGIN_ROOT}` -> the installed
cache dir); its project-scope copy is disabled by default via the committed
`.claude/settings.json` (`disabledMcpjsonServers`), but the live-repo escape
hatch (`CLAUDE_PLUGIN_ROOT=$PWD claude`) re-enables it, so a dev session can
still register two servers.
`/reload-plugins` restarts servers without reaping the old ones, so stale and
orphaned instances accumulate across reloads and versions. A stale-version
server can produce Gemini `exit 127` failures or route OpenCode through an old
model configuration, which can change billing.

This doctor lists every running quorum-mcp instance with its version, scope,
age, and parent state, and (with --reap) kills the ones safe to remove:

- ORPHAN: the launcher's parent session is gone (reparented to launchd / ppid
  not alive). Nobody is connected to it -> always safe to reap.
- STALE: running an older-than-current plugin version. Reaped only with
  --reap-stale, since it may still be attached to a live session that should
  instead reload/restart to pick up the current version.

Same-version instances under a live parent are left alone: they may be
legitimate separate sessions, which ps cannot tell apart from reload leftovers.

Usage:
    uv run python scripts/quorum_servers.py               # list only (dry-run)
    uv run python scripts/quorum_servers.py --reap        # + kill dead-parent orphans
    uv run python scripts/quorum_servers.py --reap-stale  # + kill stale-version too
    uv run python scripts/quorum_servers.py --reap --quiet  # silent, for the hook

The plugin's SessionStart hook (hooks/hooks.json) runs `--reap --quiet` so each
new session sweeps the orphans left by sessions that have since exited — no
manual run needed for the common dead-session case. STALE servers (older version
but parent still alive) are left for their session to retire on restart.

Lists nothing and exits 0 when no quorum-mcp processes are running.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HOME = Path.home()
MARKETPLACE_PLUGIN_JSON = (
    HOME
    / ".claude"
    / "plugins"
    / "marketplaces"
    / "code-quorum"
    / ".claude-plugin"
    / "plugin.json"
)
CACHE_DIR = HOME / ".claude" / "plugins" / "cache" / "code-quorum" / "code-quorum"

# Substring identifying a quorum MCP process (both launcher and worker carry it).
MARKER = "quorum-mcp"
# Pin ps to the system binary; never resolve via PATH. Its output drives which
# PIDs --reap signals, so a PATH-injected shim could run code or steer the kill.
PS_BIN = "/bin/ps"
# The cache path embeds the installed version: .../code-quorum/code-quorum/<v>/
_CACHE_VERSION_RE = re.compile(r"/cache/code-quorum/code-quorum/(\d+\.\d+\.\d+)")
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


@dataclass(frozen=True)
class Proc:
    """One process line from `ps`, already filtered to quorum-mcp."""

    pid: int
    ppid: int
    etime: str
    command: str

    @property
    def is_launcher(self) -> bool:
        # The `uv run ... quorum-mcp` launcher; the worker is the venv python.
        return self.command.startswith("uv ") or " uv run " in f" {self.command}"

    @property
    def scope(self) -> str:
        # parse_ps already filters out every line without MARKER, so a
        # non-plugin-cache command here is always project-scope.
        return "plugin" if "/cache/code-quorum/" in self.command else "project"

    @property
    def version(self) -> str | None:
        m = _CACHE_VERSION_RE.search(self.command)
        return m.group(1) if m else None


@dataclass(frozen=True)
class Instance:
    """A launcher + its worker child = one MCP server registration."""

    launcher: Proc | None
    worker: Proc | None
    scope: str
    version: str | None

    @property
    def pids(self) -> list[int]:
        return [p.pid for p in (self.launcher, self.worker) if p is not None]

    @property
    def session_ppid(self) -> int | None:
        # The session is the launcher's parent (the worker's parent is the
        # launcher). For a launcher-less worker, fall back to its own parent.
        rep = self.launcher or self.worker
        return rep.ppid if rep is not None else None

    @property
    def etime(self) -> str:
        rep = self.launcher or self.worker
        assert rep is not None  # group_instances always sets at least one side
        return rep.etime


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def parse_ps(output: str) -> list[Proc]:
    """Parse `ps -axo pid=,ppid=,etime=,command=` output into quorum Procs."""
    procs: list[Proc] = []
    for line in output.splitlines():
        if MARKER not in line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid_s, ppid_s, etime, command = parts
        if "quorum_servers.py" in command:
            continue  # never count this doctor itself as a server
        try:
            procs.append(Proc(int(pid_s), int(ppid_s), etime, command))
        except ValueError:
            continue
    return procs


def group_instances(procs: list[Proc]) -> list[Instance]:
    """Pair each launcher with its worker child; surface orphaned workers
    (whose launcher already exited) as launcher-less instances."""
    launchers = [p for p in procs if p.is_launcher]
    workers = [p for p in procs if not p.is_launcher]
    used: set[int] = set()
    instances: list[Instance] = []
    for launcher in launchers:
        worker = next(
            (w for w in workers if w.ppid == launcher.pid and w.pid not in used),
            None,
        )
        if worker is not None:
            used.add(worker.pid)
        version = launcher.version or (worker.version if worker else None)
        instances.append(Instance(launcher, worker, launcher.scope, version))
    for worker in workers:
        if worker.pid not in used:
            instances.append(Instance(None, worker, worker.scope, worker.version))
    return instances


def is_pid_alive(pid: int) -> bool:
    """True if `pid` currently exists. A PermissionError still means it
    exists (e.g. owned by another user); only ProcessLookupError means
    gone."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def is_orphan(instance: Instance) -> bool:
    """The owning session is gone: reparented to launchd (ppid 1) or the
    parent pid is no longer alive."""
    ppid = instance.session_ppid
    if ppid is None:
        return False
    return ppid == 1 or not is_pid_alive(ppid)


def is_stale(version: str | None, current: str | None) -> bool:
    """Running an older plugin version than the installed current one."""
    if version is None or current is None:
        return False
    return version_tuple(version) < version_tuple(current)


def status_of(instance: Instance, current: str | None) -> str:
    if is_orphan(instance):
        return "ORPHAN"
    if is_stale(instance.version, current):
        return "STALE"
    return "live"


def current_version(
    plugin_json: Path = MARKETPLACE_PLUGIN_JSON, cache_dir: Path = CACHE_DIR
) -> str | None:
    """The installed plugin version. Prefer the marketplace clone's plugin.json
    (what `claude plugin update` installs); fall back to the highest cached
    version directory."""
    try:
        data = json.loads(plugin_json.read_text(encoding="utf-8"))
        version = data.get("version")
        if isinstance(version, str) and _VERSION_RE.fullmatch(version):
            return version
    except (OSError, json.JSONDecodeError):
        pass
    try:
        cached = [
            d.name
            for d in cache_dir.iterdir()
            if d.is_dir() and _VERSION_RE.fullmatch(d.name)
        ]
    except OSError:
        cached = []
    return max(cached, key=version_tuple) if cached else None


def _run_ps(fmt: str) -> str:
    return subprocess.run(
        [PS_BIN, "-axo", fmt], capture_output=True, text=True, check=True
    ).stdout


def collect() -> tuple[list[Instance], str | None]:
    instances = group_instances(parse_ps(_run_ps("pid=,ppid=,etime=,command=")))
    return instances, current_version()


def kill_pid(pid: int) -> bool:
    """SIGTERM a pid. Returns True if the signal was delivered."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _format_row(instance: Instance, status: str) -> str:
    # instance.pids is never empty: group_instances always sets at least
    # one of launcher/worker.
    pids = "+".join(str(p) for p in instance.pids)
    version = instance.version or "dev"
    return (
        f"  {pids:<13} {version:<8} {instance.etime:>10}  {instance.scope:<8} {status}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and reap orphaned/stale code-quorum MCP servers."
    )
    parser.add_argument(
        "--reap", action="store_true", help="kill dead-parent ORPHAN instances"
    )
    parser.add_argument(
        "--reap-stale",
        action="store_true",
        help="also kill STALE (older-than-current version) instances",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress all stdout (for the SessionStart hook); still reaps",
    )
    args = parser.parse_args(argv)
    # --quiet silences the human listing so the hook injects no session context;
    # the reap below still runs. Route every print through emit.
    emit = (lambda *_: None) if args.quiet else print

    instances, current = collect()
    rows = [(inst, status_of(inst, current)) for inst in instances]

    if not rows:
        emit("No quorum-mcp servers running.")
        return 0

    emit(f"current installed version: {current or 'unknown'}")
    emit(f"  {'PIDS':<13} {'VER':<8} {'AGE':>10}  {'SCOPE':<8} STATUS")
    for inst, status in rows:
        emit(_format_row(inst, status))

    reapable = [
        inst
        for inst, status in rows
        if status == "ORPHAN" or (args.reap_stale and status == "STALE")
    ]
    if not (args.reap or args.reap_stale):
        n_orphan = sum(1 for _, s in rows if s == "ORPHAN")
        n_stale = sum(1 for _, s in rows if s == "STALE")
        if n_orphan or n_stale:
            emit(
                f"\n{n_orphan} orphan, {n_stale} stale. "
                "Re-run with --reap (orphans) or --reap-stale (also stale)."
            )
        return 0

    killed: list[int] = []
    for inst in reapable:
        for pid in inst.pids:
            if kill_pid(pid):
                killed.append(pid)
    emit(
        f"\nReaped {len(killed)} process(es): {killed}"
        if killed
        else "\nNothing to reap."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
