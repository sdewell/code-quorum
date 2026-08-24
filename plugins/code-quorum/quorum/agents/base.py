import asyncio
import atexit
import contextlib
import json
import logging
import os
import signal
import stat
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
# asyncio.TimeoutError is an alias for builtin TimeoutError on Python
# 3.11+; we only support 3.13+, so the builtin catches both.
_CLEANUP_WAIT_ERRORS = (TimeoutError, asyncio.CancelledError)

UNAVAILABLE_REASONS = frozenset(
    {
        "authentication",
        "network",
        "no output",
        "not installed",
        "preflight",
        "read-only configuration",
        "sandbox",
        "seat helper",
        "timeout",
        "usage limit",
    }
)

RESEARCH_PROVIDER_ENV_VARS = (
    "OPENALEX_API_KEY",
    "QUORUM_OPENALEX_API_KEY",
    "QUORUM_OPENALEX_EMAIL",
    "CONTEXT7_API_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "HF_TOKEN",
    "QUORUM_HF_TOKEN",
)

SEAT_RUNTIME_ENV_VARS = frozenset(
    {
        "ALL_PROXY",
        "AWS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NODE_EXTRA_CA_CERTS",
        "NO_PROXY",
        "NO_COLOR",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "USER",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


def allowlisted_seat_subprocess_env(
    *, extra: Iterable[str] = (), base: dict[str, str] | None = None
) -> dict[str, str]:
    """Build a minimal seat environment plus explicitly required names."""
    source = os.environ if base is None else base
    allowed = (SEAT_RUNTIME_ENV_VARS | frozenset(extra)) - frozenset(
        RESEARCH_PROVIDER_ENV_VARS
    )
    return {name: value for name, value in source.items() if name in allowed}


@dataclass
class AgentResult:
    agent: str
    output: str
    error: str = ""
    returncode: int = 0
    duration_s: float = 0.0
    role: str = "neutral"
    # Resolved model the seat ran (e.g. "gpt-5.6-terra"). Stamped centrally by
    # run_council after each round; "" when unknown. Reports lead with stance
    # and keep the seat and model for diagnosis.
    model: str = ""
    # User-facing reason when the seat did not run or could not return an
    # answer. This is explicit because child-process exit codes overlap across
    # providers and cannot safely identify authentication, timeout, helper, or
    # empty-output failures on their own.
    unavailable_reason: str = ""


class Agent(ABC):
    name: str
    default_role: str = "neutral"
    role: str = "neutral"
    # Resolved model this seat will run; every real seat sets it in __init__.
    # Declared here so run_council can stamp it onto results uniformly.
    model: str = ""

    @abstractmethod
    async def run(self, prompt: str, cwd: str) -> AgentResult: ...


def has_usage_limit_diagnostic(error: str) -> bool:
    """Recognize provider quota phrasing only at the diagnostic tail.

    Some CLIs echo the prompt before their real error. Limiting this to eight
    final non-empty lines and specific provider phrases keeps reviewed source
    text from becoming a false classification signal.
    """
    normalized = (
        error.casefold()
        .replace("\u2019", "'")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
    )
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    for line in lines[-8:]:
        if line.startswith(
            (
                "error: you've hit your usage limit",
                "error: you've hit your limit",
                "error: usage limit reached",
                "error: rate limit exceeded",
                "you've hit your usage limit",
                "you've hit your limit",
                "usage limit reached",
                "rate limit exceeded",
                "quota exhaustion (",
            )
        ):
            return True
        if (
            "resource_exhausted" in line
            and "429" in line
            and line.startswith(("error:", "429", "resource_exhausted"))
        ):
            return True
    return False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via temp-file + os.replace so a killed process can never
    leave a partially-written file. Shared by seat_helper.py (spool files)
    and gemini_cli.py (agy's settings.json) -- gemini_cli can't import this
    from seat_helper directly (seat_helper already imports gemini_cli, and a
    reverse import would cycle), so it lives here instead."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _ensure_private_dir(path: Path, *, parents: bool) -> None:
    """Create or verify an owner-only directory without following symlinks."""
    if path.is_symlink():
        raise OSError(f"private directory must not be a symlink: {path}")
    path.mkdir(mode=0o700, parents=parents, exist_ok=True)
    if path.is_symlink():
        raise OSError(f"private directory became a symlink: {path}")
    path.chmod(0o700)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise OSError(f"private path is not a directory: {path}")
    if info.st_uid != os.getuid():
        raise OSError(f"private directory is not owned by uid {os.getuid()}: {path}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise OSError(f"private directory is not mode 0700: {path}")


# How often the monitor refreshes the descendant snapshot while proc is
# live. Tight enough to catch a short-lived shell leader that forks a
# setsid descendant and exits — once the leader is reaped, the descendant
# reparents to init and lineage is unrecoverable. The trade-off is ps
# overhead: at every tick we spawn ``ps -ax``. Personal-macOS agent CLIs
# (codex, gemini) take seconds to minutes per turn and do not
# fork setsid descendants in practice; 500 ms keeps total ps work at
# ~one call/second of agent runtime instead of 20/second. A residual
# race remains for descendants that fork+setsid+exit faster than this
# interval; closing that race requires an OS containment primitive
# (cgroups on Linux, jobobjects on Windows, kqueue PROC_TRACK on
# macOS) — all out of scope for this codebase.
_MONITOR_INTERVAL_S = 0.5

# Bound on proc.wait() after we've signalled everyone we know about.
_WAIT_BOUND_S = 2.0

# Per-call timeout on the ps subprocess so a stuck ps can't stall the
# monitor.
_PS_TIMEOUT_S = 2.0

# Width of the BSD `lstart` field on macOS (e.g. "Mon May 25 14:30:25
# 2026"). The timestamp contains spaces and ps right-pads each line,
# so we rstrip and slice from the end rather than splitting by
# whitespace.
_LSTART_WIDTH = 24

# Absolute path to ps. Using bare "ps" would let a project-supplied
# binary on PATH (venv, direnv, hostile checkout) forge the snapshot
# we use to decide which PIDs get signalled. macOS ships ps at /bin/ps.
_PS_BINARY = "/bin/ps"


# --- Parent shutdown safety -------------------------------------------------
#
# Agents are spawned with ``start_new_session=True`` so they have their own
# process group and survive Ctrl-Z, pipe death, etc. The flip side is that
# a parent shutdown via SIGTERM/SIGHUP (terminal close, kill, MCP server
# shutdown) no longer reaches the agents through the foreground-pgrp signal
# path. We need an explicit registry plus shutdown handlers, otherwise a
# killed parent leaves codex/gemini processes burning API time and pipes.
#
# The registry is module-global and threading-safe; ``communicate_or_kill``
# registers each agent's pgid at entry and removes it on exit. atexit covers
# normal shutdown (sys.exit, end of main); SIGTERM and SIGHUP handlers cover
# violent termination. SIGINT is intentionally left alone -- asyncio installs
# its own SIGINT handling that turns Ctrl-C into task cancellation, which
# then runs the normal cleanup path.

# Reentrant on purpose: signal handlers run on the main thread between
# bytecodes, so a SIGTERM/SIGHUP delivered while the main thread is
# already inside _register_pgid / _unregister_pgid / _kill_all_live_pgids
# would deadlock on a non-reentrant Lock when the handler re-enters the
# registry. RLock allows the same-thread re-acquire; cross-thread mutual
# exclusion is still enforced.
_LIVE_PGIDS: set[int] = set()
_pgid_lock = threading.RLock()
_shutdown_handlers_installed = False


def _kill_all_live_pgids() -> None:
    """SIGKILL every registered agent process group. Called from atexit
    and from the shutdown signal handler; idempotent and safe to call
    on an empty registry."""
    with _pgid_lock:
        pgids = list(_LIVE_PGIDS)
        _LIVE_PGIDS.clear()
    for pgid in pgids:
        if pgid == os.getpgid(0):
            continue  # never signal our own group on shutdown
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


def _shutdown_signal_handler(signum: int, _frame: object) -> None:
    """Reap agent pgrps then re-raise the signal with default
    disposition so the parent terminates as the caller intended."""
    _kill_all_live_pgids()
    with contextlib.suppress(Exception):
        signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _ensure_shutdown_handlers() -> None:
    """Install atexit + SIGTERM/SIGHUP handlers on first use. Idempotent.
    Silent if called from a non-main thread, where signal.signal() is
    not permitted."""
    global _shutdown_handlers_installed
    with _pgid_lock:
        if _shutdown_handlers_installed:
            return
        _shutdown_handlers_installed = True
    atexit.register(_kill_all_live_pgids)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            existing = signal.getsignal(sig)
            # Only install over the default. SIG_IGN means a caller
            # (nohup, disown, supervisor wrapper) intentionally chose
            # to ignore this signal -- overriding with our handler
            # would turn an explicitly-ignored signal into process
            # death and lose an in-flight quorum run. Any custom
            # handler is also caller-owned and presumably does its
            # own cleanup.
            if existing in (signal.SIG_DFL, None):
                signal.signal(sig, _shutdown_signal_handler)
        except (ValueError, OSError):
            # ValueError: not in main thread. OSError: signal not
            # available on this platform. Both are non-fatal -- we
            # still have atexit, which covers normal exits.
            pass


def _register_pgid(pgid: int) -> None:
    _ensure_shutdown_handlers()
    with _pgid_lock:
        _LIVE_PGIDS.add(pgid)


def _unregister_pgid(pgid: int) -> None:
    with _pgid_lock:
        _LIVE_PGIDS.discard(pgid)


_snapshot_unavailable_warned = False


def _warn_snapshot_unavailable(reason: str) -> None:
    """Log a one-time warning when process snapshots become unavailable.
    Cleanup degrades gracefully (regular descendants still die via
    killpg) but setsid escapees won't be caught — the user should know."""
    global _snapshot_unavailable_warned
    if not _snapshot_unavailable_warned:
        _snapshot_unavailable_warned = True
        logger.warning(
            "code-quorum: process snapshots unavailable (%s) -- agent "
            "cleanup will not catch setsid-escaped descendants. "
            "Process-group kill still fires for non-escaped children.",
            reason,
        )


async def _run_ps(
    args: list[str], *, on_unavailable: Callable[[str], None] | None = None
) -> tuple[bytes, bytes, int | None] | None:
    """Spawn `_PS_BINARY` with `args`, bounded by `_PS_TIMEOUT_S`. Returns
    `(stdout, stderr, returncode)` on completion, or None if ps could not be
    spawned or timed out -- in which case `on_unavailable(reason)` is called
    first, if given. A timeout kills and reaps the stuck ps before returning;
    an external CancelledError does the same before re-raising, so a
    cancelled caller never leaves a ps process behind."""
    try:
        proc = await asyncio.create_subprocess_exec(
            _PS_BINARY,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, FileNotFoundError) as exc:
        if on_unavailable is not None:
            on_unavailable(f"ps spawn failed: {exc}")
        return None
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_PS_TIMEOUT_S
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(BaseException):
            await proc.wait()
        if on_unavailable is not None:
            on_unavailable("ps timed out")
        return None
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(BaseException):
            await proc.wait()
        raise
    return stdout, stderr, proc.returncode


def _parse_ps_lstart_rows(
    stdout: bytes, n_head_fields: int
) -> list[tuple[list[int], str]]:
    """Parse `ps -o ...,lstart=` output lines shaped as `<n_head_fields
    integer columns><lstart>`, where lstart is the fixed-width, right-padded
    BSD timestamp field (`_LSTART_WIDTH`) -- so we rstrip and slice from the
    end rather than splitting by whitespace. Skips lines that are too short,
    have the wrong head field count, or have a non-integer head; returns
    `(head_ints, lstart)` per well-formed line."""
    rows: list[tuple[list[int], str]] = []
    for raw in stdout.decode(errors="replace").splitlines():
        line = raw.rstrip()
        if len(line) < _LSTART_WIDTH + 2:
            continue
        lstart = line[-_LSTART_WIDTH:]
        head = line[:-_LSTART_WIDTH].split()
        if len(head) != n_head_fields:
            continue
        try:
            head_ints = [int(field) for field in head]
        except ValueError:
            continue
        rows.append((head_ints, lstart))
    return rows


async def _snapshot() -> dict[int, tuple[int, str]]:
    """Single ps snapshot of `pid -> (ppid, lstart)`. Empty on failure;
    a one-time warning is logged the first time snapshots are
    unavailable so the degraded cleanup path doesn't fail open
    silently."""
    result = await _run_ps(
        ["-o", "pid=,ppid=,lstart=", "-ax"], on_unavailable=_warn_snapshot_unavailable
    )
    if result is None:
        return {}
    stdout, stderr, returncode = result
    if returncode != 0:
        _warn_snapshot_unavailable(
            f"ps exited {returncode}: {stderr.decode(errors='replace').strip()[:120]}"
        )
        return {}
    return {
        pid: (ppid, lstart) for (pid, ppid), lstart in _parse_ps_lstart_rows(stdout, 2)
    }


def _descendants_with_starts(
    root_pid: int,
    snap: dict[int, tuple[int, str]],
) -> dict[int, str]:
    """Transitive descendants of `root_pid` in `snap`, mapped to each
    pid's lstart. Empty if no descendants have intact lineage back to
    `root_pid` in the snapshot."""
    children: dict[int, list[int]] = {}
    for pid, (ppid, _lstart) in snap.items():
        children.setdefault(ppid, []).append(pid)
    seen: dict[int, str] = {}
    stack = [root_pid]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, ()):
            if child in seen:
                continue
            seen[child] = snap[child][1]
            stack.append(child)
    return seen


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def _verify_lstart_batch(pids: list[int]) -> dict[int, str]:
    """Targeted batch ``ps -p PID1,PID2,...`` returning ``pid -> lstart``
    for live pids. Empty dict if ps cannot answer. One ps call per
    cleanup, bounded by ``_PS_TIMEOUT_S`` regardless of how many pids
    were accrued -- per-pid verification would otherwise turn a
    bounded agent timeout into unbounded cleanup work when the
    monitor accumulated stale entries over a long agent lifetime."""
    if not pids:
        return {}
    result = await _run_ps(["-p", ",".join(str(p) for p in pids), "-o", "pid=,lstart="])
    if result is None:
        return {}
    stdout, _stderr, _returncode = result
    # ps -p with a list returns non-zero if any pid is gone, but still
    # emits live pids on stdout -- parse what we got regardless of code.
    return {pid: lstart for (pid,), lstart in _parse_ps_lstart_rows(stdout, 1)}


def _kill_process_group(proc: asyncio.subprocess.Process, pgid: int) -> None:
    """SIGKILL the spawn-time process group.

    Scope decision (personal macOS CLI, single user, no cross-site
    deployment): we do not gate killpg on a snapshot-time pgid identity
    check. POSIX PGID reuse requires the leader's PID to be both reaped
    AND recycled by the kernel; on macOS the kernel cycles PIDs
    sequentially through ~100k values, so a desktop with low fork churn
    needs hours of fork activity to recycle. Our cleanup window after
    cancellation is bounded at a few seconds, so the probability of
    killpg hitting a reused PGID is operationally zero. The defenses
    that would close this race (snapshot-and-verify-pgid-owner, or a
    fail-closed posture when the monitor never accrued an in-group
    descendant) would either break the leader-exited-before-monitor
    safety net or require an OS containment primitive (cgroups /
    jobobjects / kqueue PROC_TRACK) -- both out of scope for this
    codebase. See the xfail boundary test for the related fast-fork
    daemonization gap.
    """
    if pgid == os.getpgid(0):
        # Same group as us -- refuse killpg or we'd self-terminate.
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


async def _terminate_tree(
    proc: asyncio.subprocess.Process,
    pgid: int,
    escapees: dict[int, str],
    current_snap: dict[int, tuple[int, str]],
) -> None:
    """Cleanup the process group and monitor-captured escapees.

    Process-group kill is the primary containment for non-setsid
    descendants -- pgid stays valid as long as the group has any
    surviving member, even after the leader exits, and pgid reuse is
    not a concern because we hold ``proc`` (the group leader) and its
    pid cannot be recycled until we reap it.

    Monitor-captured escapees (setsid descendants in their own group)
    are killed only with a fresh per-pid lstart match. We never kill
    an accrued pid blindly: ``known`` accumulates over the agent's
    full lifetime, so a long-lived agent that spawned and reaped
    short-lived helpers could have stale pids whose value has since
    been reused. The bulk cleanup snapshot is preferred (already paid
    for); for the rest, one targeted batch ``ps -p`` covers them all
    in a single bounded call. If both passes fail (ps degraded or pid
    gone), we skip rather than risk SIGKILL against an unrelated
    process. Callers MUST cancel the monitor before invoking this so
    ``escapees`` is no longer mutated during the awaits below.
    """
    _kill_process_group(proc, pgid)

    accrued = list(escapees.items())
    needs_verify = [pid for pid, _ in accrued if pid not in current_snap]
    fallback = await _verify_lstart_batch(needs_verify)

    for pid, captured_lstart in accrued:
        live = current_snap.get(pid)
        if live is not None:
            if live[1] == captured_lstart:
                _kill(pid)
            continue
        live_lstart = fallback.get(pid)
        if live_lstart is None or live_lstart != captured_lstart:
            continue
        _kill(pid)


async def _monitor_descendants(root_pid: int, known: dict[int, str]) -> None:
    """Background task: accrue (pid, lstart) into `known` for the
    lifetime of `proc`. The polling captures descendants before their
    parent dies -- once the leader is reaped, ps walks from `root_pid`
    return nothing, so we must capture (pid, lstart) while lineage is
    still intact."""
    try:
        while True:
            snap = await _snapshot()
            for pid, lstart in _descendants_with_starts(root_pid, snap).items():
                # First-write wins so we keep the lstart from the
                # snapshot where the pid was first observed; reuse is
                # detected at kill time, not at capture.
                known.setdefault(pid, lstart)
            await asyncio.sleep(_MONITOR_INTERVAL_S)
    except asyncio.CancelledError:
        return


async def _teardown_process_tree(
    proc: asyncio.subprocess.Process,
    pgid: int,
    known: dict[int, str],
    monitor: asyncio.Task[None] | None,
) -> None:
    """Stop the monitor, take a final lineage snapshot, kill the process
    group plus verified setsid escapees, then bounded-wait with a
    transport-close fallback. Shared by every teardown path (external
    cancellation and the idle-timeout kill) so they get identical
    containment.

    Cancels the monitor first so it can't mutate `known` while
    _terminate_tree iterates it; the monitor already wrote every observation
    it had, and the final snapshot recovers any in-group descendant a missed
    tick left out. Snapshots BEFORE killpg so a setsid descendant still
    reachable through the leader is captured while lineage is intact -- once
    the leader is reaped the escapee reparents to init and ps walks from
    proc.pid stop finding it. Callers MUST null their own monitor reference
    after this returns so a finally block doesn't double-cancel it.
    """
    if monitor is not None:
        monitor.cancel()
        with contextlib.suppress(BaseException):
            await monitor
    try:
        current = await asyncio.wait_for(_snapshot(), timeout=_PS_TIMEOUT_S)
    except (TimeoutError, OSError, asyncio.CancelledError):
        current = {}
    for pid, lstart in _descendants_with_starts(proc.pid, current).items():
        known.setdefault(pid, lstart)
    await _terminate_tree(proc, pgid, known, current)
    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=_WAIT_BOUND_S)
        except _CLEANUP_WAIT_ERRORS:
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.close()


class _Containment:
    """State + teardown handle yielded by `_contained`. `known` accrues
    (pid, lstart) pairs for the monitored process tree -- the caller's own
    read loop keeps observing it via _monitor_descendants, which is already
    writing into this same dict. Call `teardown()` from a caller's own
    non-cancellation timeout path (e.g. communicate_lines_or_kill's idle
    timeout or its bounded clean-EOF wait) to run the same three-layer
    cleanup `_contained` runs on cancellation, without waiting for an
    exception; afterward `_contained`'s own finally is a no-op for the
    monitor since `teardown()` already nulled it."""

    def __init__(self, proc: asyncio.subprocess.Process, pgid: int) -> None:
        self._proc = proc
        self._pgid = pgid
        self.known: dict[int, str] = {}
        self.monitor: asyncio.Task[None] | None = None

    async def teardown(self) -> None:
        await _teardown_process_tree(self._proc, self._pgid, self.known, self.monitor)
        self.monitor = None


@contextlib.asynccontextmanager
async def _contained(
    proc: asyncio.subprocess.Process, pgid: int
) -> AsyncIterator[_Containment]:
    """Shared setup/teardown for communicate_or_kill and
    communicate_lines_or_kill.

    Cleanup uses three independent containment layers (see
    _teardown_process_tree): killpg(pgid) against the spawn-time process
    group; best-effort SIGKILL of each monitor-captured setsid escapee whose
    start token still matches; and a bounded proc.wait() with a
    transport-close fallback. The hard guarantee is process-group cleanup --
    a process that fork+setsid+exits faster than the first snapshot can
    still escape lineage tracking, which requires OS containment beyond this
    personal macOS CLI's scope to close.

    Registers `pgid` for shutdown-signal cleanup, seeds the yielded
    `_Containment.known` with the initial descendant snapshot, starts the
    background monitor, and guarantees the same teardown on cancellation.
    Callers must spawn with start_new_session=True and pass `pgid=proc.pid`
    snapshotted right after spawn; that pgid stays valid for killpg even
    after the leader exits, as long as the group has surviving members.
    """
    state = _Containment(proc, pgid)
    _register_pgid(pgid)
    try:
        initial_snap = await _snapshot()
        for pid, lstart in _descendants_with_starts(proc.pid, initial_snap).items():
            state.known.setdefault(pid, lstart)
        state.monitor = asyncio.create_task(_monitor_descendants(proc.pid, state.known))
        yield state
    except BaseException:
        # Teardown on ANY escape, not just cancellation. A BrokenPipeError from
        # a child that dies before reading its prompt can otherwise skip the
        # kill while the finally block unregisters the pgid from shutdown
        # cleanup, which permanently orphans the process tree.
        try:
            await state.teardown()
        except asyncio.CancelledError:
            # A cancellation arriving DURING teardown must win -- swallowing
            # it would make the task ignore its cancel signal. The original
            # error rides along as __context__.
            raise
        except BaseException:
            # A teardown failure must not mask the exception that triggered
            # it. Log the teardown failure and re-raise the original exception.
            logger.warning(
                "containment teardown failed while handling an escaping "
                "exception; re-raising the original",
                exc_info=True,
            )
        raise
    finally:
        if state.monitor is not None:
            state.monitor.cancel()
            with contextlib.suppress(BaseException):
                await state.monitor
        _unregister_pgid(pgid)


async def communicate_or_kill(
    proc: asyncio.subprocess.Process,
    stdin: bytes | None = None,
    *,
    pgid: int,
) -> tuple[bytes, bytes]:
    """proc.communicate() with cancellation cleanup for agent children. See
    `_contained` for the three containment layers and the pgid contract."""
    async with _contained(proc, pgid):
        return await proc.communicate(input=stdin)


# Size of each incremental stdout read in communicate_lines_or_kill. We read
# fixed-size chunks rather than whole lines: opencode emits newline-delimited
# JSON whose individual events can exceed asyncio's default 64 KiB StreamReader
# line limit (a large file-read tool result on one line), which would make
# readline() raise. read(n) returns as soon as ANY bytes are buffered, so it
# both sidesteps the line limit and gives true "did output arrive" idle
# detection.
_READ_CHUNK_BYTES = 65536


async def communicate_lines_or_kill(
    proc: asyncio.subprocess.Process,
    *,
    pgid: int,
    idle_timeout: float,
) -> tuple[bytes, bytes, bool]:
    """Stream proc.stdout incrementally, killing the whole process group if
    no output arrives within `idle_timeout` seconds. Returns
    ``(stdout_bytes, stderr_bytes, idle_timed_out)``.

    Unlike communicate_or_kill -- which reads everything via proc.communicate
    and only reacts to external cancellation -- this observes the stream as it
    flows, so a provider that stops emitting is detected as a stall and torn
    down without waiting for the council's outer cap. opencode streams
    newline-delimited JSON; a healthy run emits output every few seconds, so a
    multi-minute silence is a genuine stall, not normal think time. stderr is
    drained concurrently to avoid a pipe-buffer deadlock (a child blocked
    writing stderr would also stall stdout).

    Uses the same three containment layers as communicate_or_kill on the
    idle-kill and cancellation paths; see `_contained`. Callers must spawn
    with start_new_session=True and pass pgid=proc.pid snapshotted at spawn.
    """
    if proc.stdout is None or proc.stderr is None:
        raise ValueError("communicate_lines_or_kill requires piped stdout and stderr")
    stdout_chunks: list[bytes] = []
    idle_timed_out = False
    stderr_task: asyncio.Task[bytes] | None = None
    try:
        async with _contained(proc, pgid) as state:
            # Drain stderr in the background so a chatty child can't fill its
            # stderr pipe and deadlock the stdout read we block on below.
            stderr_task = asyncio.create_task(proc.stderr.read())
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(_READ_CHUNK_BYTES), timeout=idle_timeout
                    )
                except TimeoutError:
                    idle_timed_out = True
                    await state.teardown()
                    break
                if not chunk:
                    break  # EOF: opencode closed stdout, the run is finishing
                stdout_chunks.append(chunk)
            if not idle_timed_out and proc.returncode is None:
                # Clean EOF: stdout closed as the process exits. Reap it (bounded)
                # so returncode is set and no zombie lingers. If the process closed
                # stdout but WON'T exit, this wait times out -- escalate to the full
                # teardown so we neither leak the group nor return a false success
                # (returncode would stay None and the adapter maps `None or 0` -> 0).
                # Catch TimeoutError ONLY: a CancelledError here is the council's
                # outer cap firing and MUST propagate to the teardown+re-raise block
                # below, not be swallowed.
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_WAIT_BOUND_S)
                except TimeoutError:
                    idle_timed_out = True
                    await state.teardown()
            # Collect the concurrently-drained stderr with the same bound, so a
            # missed escapee holding the inherited stderr pipe can't stall us. Again
            # catch TimeoutError only -- an outer cancellation must propagate.
            try:
                stderr_bytes = await asyncio.wait_for(
                    stderr_task, timeout=_WAIT_BOUND_S
                )
            except TimeoutError:
                stderr_bytes = b""
            stderr_task = None
            return b"".join(stdout_chunks), stderr_bytes, idle_timed_out
    finally:
        if stderr_task is not None:
            stderr_task.cancel()
            with contextlib.suppress(BaseException):
                await stderr_task
