"""Cancellation-cleanup guarantees for agent subprocesses.

The council's per-agent timeout cancels the agent task and propagates
CancelledError up through communicate_or_kill. After that point there must
be no surviving descendant processes and no orphan temp files.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

from quorum.agents import base as base_mod
from quorum.agents.base import communicate_or_kill
from quorum.agents.codex import CodexAgent


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_until_dead(pid: int, timeout_s: float) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if not _pid_alive(pid):
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_communicate_or_kill_terminates_descendant_processes() -> None:
    """A subprocess that forked a detached descendant must not leave the
    descendant alive after communicate_or_kill is cancelled. The shell child
    backgrounds a long sleeper (well past the test window so natural expiry
    can't mask the bug), prints its PID, then waits forever. After cancel
    the whole tree should be torn down — not just the direct child."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 300 & echo $! ; wait",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    assert proc.stdout is not None
    line = await proc.stdout.readline()
    grandchild_pid = int(line.strip())
    assert _pid_alive(grandchild_pid), "grandchild should be running before cancel"

    task = asyncio.create_task(communicate_or_kill(proc, pgid=pgid))
    try:
        await asyncio.sleep(0.05)
        task.cancel()
        assert await _wait_until_dead(grandchild_pid, timeout_s=2.0), (
            f"grandchild PID {grandchild_pid} survived 2s past cancel — "
            "communicate_or_kill did not terminate the full process group"
        )
    finally:
        for pid in (grandchild_pid, proc.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_non_cancel_exception_still_kills_group() -> None:
    """Any exception from communicate_or_kill must leave the process group
    dead. This includes BrokenPipeError from a child that dies before it reads
    the prompt."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 300 & echo $! ; wait",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    grandchild_pid: int | None = None
    # Enter the cleanup finally BEFORE parsing/asserting anything (q-review
    # round 2, codex): an assertion failure above the old try left the
    # 300-second tree running.
    try:
        assert proc.stdout is not None
        line = await proc.stdout.readline()
        grandchild_pid = int(line.strip())
        assert _pid_alive(grandchild_pid), (
            "grandchild should be running before the raise"
        )

        async def _broken_communicate(
            input: bytes | None = None,
        ) -> tuple[bytes, bytes]:
            raise BrokenPipeError("child closed its pipes")

        proc.communicate = _broken_communicate  # ty: ignore[invalid-assignment]
        with pytest.raises(BrokenPipeError):
            await communicate_or_kill(proc, pgid=pgid)
        assert await _wait_until_dead(grandchild_pid, timeout_s=2.0), (
            f"grandchild PID {grandchild_pid} survived a non-cancellation "
            "exception — _contained did not tear down the process group"
        )
    finally:
        for pid in (grandchild_pid, proc.pid):
            if pid is None:
                continue
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_teardown_failure_does_not_mask_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If teardown raises while handling a non-cancel exception, the original
    exception must still propagate. A cleanup OSError must not replace the
    BrokenPipeError that triggered it."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 30",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    async def _broken_communicate(input: bytes | None = None) -> tuple[bytes, bytes]:
        raise BrokenPipeError("child closed its pipes")

    async def _broken_teardown(proc_, pgid_, known_, monitor_) -> None:
        raise OSError("ps exploded during teardown")

    proc.communicate = _broken_communicate  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(base_mod, "_teardown_process_tree", _broken_teardown)
    try:
        with pytest.raises(BrokenPipeError):
            await communicate_or_kill(proc, pgid=proc.pid)
    finally:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_cancellation_during_teardown_still_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the one deliberate exception to the rule above: a CancelledError
    raised DURING teardown must propagate (the task is being cancelled;
    swallowing it would make the task ignore its cancel signal). The
    original error rides along as __context__."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 30",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    async def _broken_communicate(input: bytes | None = None) -> tuple[bytes, bytes]:
        raise BrokenPipeError("child closed its pipes")

    async def _cancelled_teardown(proc_, pgid_, known_, monitor_) -> None:
        raise asyncio.CancelledError()

    proc.communicate = _broken_communicate  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(base_mod, "_teardown_process_tree", _cancelled_teardown)
    try:
        with pytest.raises(asyncio.CancelledError):
            await communicate_or_kill(proc, pgid=proc.pid)
    finally:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_communicate_or_kill_terminates_descendants_when_leader_exits() -> None:
    """The shell forks a backgrounded sleep, prints its PID, and exits. The
    grandchild keeps the inherited stdout pipe open, so proc.communicate() is
    still pending when we cancel. At that point, proc.pid is a zombie or has
    already been reaped, and an os.getpgid
    lookup at cancel time can't recover the group — cleanup must rely on a
    pgid snapshotted right after spawn (when start_new_session=True
    guarantees pgid==pid)."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 300 & echo $! ; exit 0",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid  # captured at spawn — pgid == pid under start_new_session
    assert proc.stdout is not None
    line = await proc.stdout.readline()
    grandchild_pid = int(line.strip())

    # Give the shell time to exit so the bug condition (leader gone, grandchild
    # alive) actually holds before we cancel.
    await asyncio.sleep(0.2)
    assert _pid_alive(grandchild_pid), "grandchild should outlive shell"

    task = asyncio.create_task(communicate_or_kill(proc, pgid=pgid))
    try:
        await asyncio.sleep(0.05)
        task.cancel()
        assert await _wait_until_dead(grandchild_pid, timeout_s=2.0), (
            f"grandchild PID {grandchild_pid} survived cancel after the shell "
            "leader exited — cleanup must use a spawn-time pgid"
        )
    finally:
        for pid in (grandchild_pid, proc.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_cancellation_during_initial_snapshot_kills_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup must be armed before the first snapshot await. If the
    caller cancels while setup is still taking a process snapshot, the
    already-spawned child process group must still be killed."""
    snapshot_started = asyncio.Event()
    never = asyncio.Event()
    calls = 0

    async def fake_snapshot() -> dict[int, tuple[int, str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            snapshot_started.set()
            await never.wait()
        return {}

    monkeypatch.setattr(base_mod, "_snapshot", fake_snapshot)

    proc = await asyncio.create_subprocess_exec(
        "/bin/sleep",
        "300",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=proc.pid))

    try:
        await asyncio.wait_for(snapshot_started.wait(), timeout=1.0)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail("communicate_or_kill hung after setup-time cancellation")
        assert await _wait_until_dead(proc.pid, timeout_s=2.0), (
            f"child PID {proc.pid} survived cancellation during initial snapshot"
        )
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_kills_group_when_leader_exits_before_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-setsid descendant remains in the spawn-time process group
    after the shell leader exits. Even if the monitor missed the child,
    cleanup must still signal the process group; gating killpg on the
    leader's current ps/libproc presence leaves this child alive."""

    async def dormant_monitor(_root_pid: int, _known: dict[int, str]) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(base_mod, "_monitor_descendants", dormant_monitor)

    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 300 & echo $! ; sleep 0.3 ; exit 0",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    assert proc.stdout is not None
    line = await proc.stdout.readline()
    grandchild_pid = int(line.strip())

    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=pgid))
    await asyncio.sleep(0.5)
    assert _pid_alive(grandchild_pid), "grandchild should outlive shell leader"

    try:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail(
                "communicate_or_kill cleanup hung when the leader exited "
                "before the monitor accrued descendants"
            )
        assert await _wait_until_dead(grandchild_pid, timeout_s=2.0), (
            f"grandchild PID {grandchild_pid} survived cancel after leader exit "
            "with an empty monitor set — cleanup must still killpg"
        )
    finally:
        for pid in (grandchild_pid, proc.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_monitor_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the monitor is cancelled while it is awaiting a snapshot, that
    cancellation must propagate. Swallowing CancelledError makes
    communicate_or_kill's finally block wait forever on the monitor."""
    calls = 0
    monitor_snapshot_started = asyncio.Event()
    monitor_cancelled = asyncio.Event()
    never = asyncio.Event()

    async def fake_snapshot() -> dict[int, tuple[int, str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {}
        if calls == 2:
            monitor_snapshot_started.set()
            try:
                await never.wait()
            except asyncio.CancelledError:
                monitor_cancelled.set()
                raise
        return {}

    monkeypatch.setattr(base_mod, "_snapshot", fake_snapshot)

    proc = await asyncio.create_subprocess_exec(
        "/bin/sleep",
        "300",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=proc.pid))

    try:
        await asyncio.wait_for(monitor_snapshot_started.wait(), timeout=1.0)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail("communicate_or_kill hung waiting for cancelled monitor")
        assert monitor_cancelled.is_set(), "monitor snapshot did not see cancellation"
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.xfail(
    reason=(
        "fast fork+setsid+leader-exit before the first snapshot is outside "
        "the personal macOS CLI containment guarantee"
    ),
    strict=False,
)
@pytest.mark.asyncio
async def test_communicate_or_kill_fast_setsid_escapee_boundary() -> None:
    """Document the residual boundary: if a child daemonizes with setsid
    and its original leader exits before lineage is observable, process
    group cleanup cannot reach it and polling cannot rediscover it."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        f'{sys.executable} -c "'
        "import os, sys, time; os.setsid(); "
        "sys.stdout.write(str(os.getpid()) + chr(10)); "
        "sys.stdout.flush(); time.sleep(300)"
        '" & echo $! ; exit 0',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    assert proc.stdout is not None
    _shell_pid_line = await proc.stdout.readline()
    setsid_line = await proc.stdout.readline()
    escapee_pid = int(setsid_line.strip())

    task = asyncio.create_task(communicate_or_kill(proc, pgid=pgid))
    try:
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail("cleanup hung at the fast daemonization boundary")
        assert await _wait_until_dead(escapee_pid, timeout_s=2.0), (
            f"fast setsid escapee PID {escapee_pid} survived cancellation"
        )
    finally:
        for pid in (escapee_pid, proc.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_terminates_setsid_escapee() -> None:
    """The immediate child forks a Python descendant that calls os.setsid().
    This call escapes the original process group. The descendant keeps the
    inherited stdout pipe open and
    sleeps. killpg(pgid) cannot reach it; cleanup must walk the process
    tree and SIGKILL the escapee directly, and the whole cleanup must
    return within a bounded window (no unbounded await proc.wait())."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        f'{sys.executable} -c "'
        "import os, sys, time; os.setsid(); "
        "sys.stdout.write(str(os.getpid()) + chr(10)); "
        "sys.stdout.flush(); time.sleep(300)"
        '" & echo $! ; wait',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    assert proc.stdout is not None
    # First line: shell's $! (python's PID). Second line: python's own
    # getpid() after setsid — same PID, but confirms setsid ran.
    _shell_pid_line = await proc.stdout.readline()
    setsid_line = await proc.stdout.readline()
    escapee_pid = int(setsid_line.strip())
    assert _pid_alive(escapee_pid), "escapee should be running before cancel"

    task = asyncio.create_task(communicate_or_kill(proc, pgid=pgid))
    try:
        await asyncio.sleep(0.1)
        task.cancel()
        # Cleanup must complete promptly — if proc.wait() hangs on a
        # missed escapee's pipes, the bounded wait + transport.close()
        # fallback ensures we return. Cap at 5s to flag a regression.
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail(
                "communicate_or_kill cleanup hung past 5s — "
                "setsid escapee likely held stdio pipes and bounded "
                "wait/transport-close fallback failed"
            )
        assert await _wait_until_dead(escapee_pid, timeout_s=2.0), (
            f"setsid escapee PID {escapee_pid} survived cancel — "
            "process-tree walk did not catch the process-group escape"
        )
    finally:
        for pid in (escapee_pid, proc.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_terminates_setsid_escapee_after_leader_exits() -> (
    None
):
    """Combined adversarial case: shell forks a setsid python descendant
    and then EXITS itself. By the time we cancel, the leader is gone and
    the descendant has been reparented to init — a one-shot ps walk from
    proc.pid finds nothing. Cleanup must rely on the background monitor
    that captured the descendant's pid while the lineage was still
    intact, and SIGKILL it from the accrued set."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        # Shell sleeps briefly after forking so the monitor (100ms tick)
        # has time to observe the python descendant before the leader
        # exits. Without this window the test would race the monitor.
        f'{sys.executable} -c "'
        "import os, sys, time; os.setsid(); "
        "sys.stdout.write(str(os.getpid()) + chr(10)); "
        "sys.stdout.flush(); time.sleep(300)"
        '" & echo $! ; sleep 0.5 ; exit 0',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    assert proc.stdout is not None
    _shell_pid_line = await proc.stdout.readline()
    setsid_line = await proc.stdout.readline()
    escapee_pid = int(setsid_line.strip())

    # The cleanup runs inside communicate_or_kill — start it BEFORE the
    # shell exits so the monitor task is alive while the lineage is
    # still walkable from proc.pid.
    task = asyncio.create_task(communicate_or_kill(proc, pgid=pgid))

    # Wait long enough for the shell to exit (0.5s sleep + reap) so the
    # adversarial condition (leader-gone, escapee-alive) actually holds
    # before cancel.
    await asyncio.sleep(0.8)
    assert _pid_alive(escapee_pid), "escapee should outlive the shell leader"

    try:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail(
                "communicate_or_kill cleanup hung past 5s — combined "
                "leader-exit + setsid case is not contained"
            )
        assert await _wait_until_dead(escapee_pid, timeout_s=2.0), (
            f"setsid escapee PID {escapee_pid} survived cancel after the "
            "shell leader exited — descendant monitor did not accrue the pid"
        )
    finally:
        for pid in (escapee_pid, proc.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_cancel_during_initial_snapshot_kills_setsid_escapee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation lands during the first _snapshot await after the child
    forks a live setsid descendant. If cleanup kills the leader before the
    snapshot,
    the descendant reparents to init and the post-kill snapshot can no
    longer reach it through proc.pid. The initial snapshot must capture the
    descendant while lineage is still intact."""
    orig_snapshot = base_mod._snapshot
    snapshot_started = asyncio.Event()
    never = asyncio.Event()
    calls = 0

    async def fake_snapshot() -> dict[int, tuple[int, str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            snapshot_started.set()
            await never.wait()
        return await orig_snapshot()

    monkeypatch.setattr(base_mod, "_snapshot", fake_snapshot)

    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        f'{sys.executable} -c "'
        "import os, sys, time; os.setsid(); "
        "sys.stdout.write(str(os.getpid()) + chr(10)); "
        "sys.stdout.flush(); time.sleep(300)"
        '" & echo $! ; wait',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    assert proc.stdout is not None
    _shell_pid_line = await proc.stdout.readline()
    setsid_line = await proc.stdout.readline()
    escapee_pid = int(setsid_line.strip())
    assert _pid_alive(escapee_pid), "setsid escapee should be running before cancel"

    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=pgid))
    try:
        await asyncio.wait_for(snapshot_started.wait(), timeout=1.0)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail(
                "communicate_or_kill hung when cancelled mid-initial-snapshot "
                "with a live setsid descendant"
            )
        assert await _wait_until_dead(escapee_pid, timeout_s=2.0), (
            f"setsid escapee PID {escapee_pid} survived setup-time cancellation "
            "— cleanup must snapshot before killpg so lineage stays walkable"
        )
        assert await _wait_until_dead(proc.pid, timeout_s=2.0), (
            f"shell leader PID {proc.pid} survived setup-time cancellation"
        )
    finally:
        for pid in (escapee_pid, proc.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_kills_monitored_escapee_when_cleanup_ps_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The monitor accrued a setsid escapee while ps worked normally. The
    cleanup-time snapshot returns empty because ps timed out or was unavailable.
    The accrued pid must
    still be killed best-effort — gating the per-pid kill on a fresh
    snapshot makes cleanup silently leak orphans on ps degradation."""
    orig_snapshot = base_mod._snapshot
    ps_failed = False

    async def fake_snapshot() -> dict[int, tuple[int, str]]:
        if ps_failed:
            return {}
        return await orig_snapshot()

    monkeypatch.setattr(base_mod, "_snapshot", fake_snapshot)

    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        f'{sys.executable} -c "'
        "import os, sys, time; os.setsid(); "
        "sys.stdout.write(str(os.getpid()) + chr(10)); "
        "sys.stdout.flush(); time.sleep(300)"
        '" & echo $! ; wait',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    assert proc.stdout is not None
    _shell_pid_line = await proc.stdout.readline()
    setsid_line = await proc.stdout.readline()
    escapee_pid = int(setsid_line.strip())
    assert _pid_alive(escapee_pid), "setsid escapee should be running before cancel"

    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=pgid))
    try:
        # Give the monitor time to accrue the escapee via real ps.
        await asyncio.sleep(0.25)
        ps_failed = True
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail("communicate_or_kill hung when cleanup-time ps was unavailable")
        assert await _wait_until_dead(escapee_pid, timeout_s=2.0), (
            f"setsid escapee PID {escapee_pid} survived cancel with ps unavailable "
            "— accrued pids must be killed even when the cleanup snapshot is empty"
        )
    finally:
        for pid in (escapee_pid, proc.pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_does_not_kill_unverified_pid_when_ps_fully_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety property: when neither the bulk snapshot nor the per-pid
    ``ps -p`` fallback can verify a monitored escapee's identity, we
    MUST NOT issue ``_kill`` against the accrued pid. The monitor
    accumulates pids over the agent's full lifetime, so a long-lived
    agent with reaped helpers could have stale entries whose PIDs have
    since been reused. Killing an unverified accrued pid would SIGKILL
    a random user process. Skipping (leaking, logged) is the correct
    failure mode."""
    base_mod._snapshot_unavailable_warned = False  # reset warn-once

    async def empty_snapshot() -> dict[int, tuple[int, str]]:
        return {}

    async def no_verify_batch(_pids: list[int]) -> dict[int, str]:
        return {}

    kill_calls: list[int] = []

    def record_kill(pid: int) -> None:
        kill_calls.append(pid)

    monkeypatch.setattr(base_mod, "_snapshot", empty_snapshot)
    monkeypatch.setattr(base_mod, "_verify_lstart_batch", no_verify_batch)
    monkeypatch.setattr(base_mod, "_kill", record_kill)

    # Seed an obviously-bogus pid into `known` via a dormant monitor
    # that injects directly. The shell child itself doesn't matter -- we
    # care only that _terminate_tree refuses to call _kill(stale_pid).
    stale_pid = 999999

    async def seed_monitor(_root_pid: int, known: dict[int, str]) -> None:
        known[stale_pid] = "Mon May 25 14:30:25 2026"
        await asyncio.Event().wait()

    monkeypatch.setattr(base_mod, "_monitor_descendants", seed_monitor)

    proc = await asyncio.create_subprocess_exec(
        "/bin/sleep",
        "300",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=proc.pid))
    try:
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail("communicate_or_kill hung when ps was fully unavailable")
        assert stale_pid not in kill_calls, (
            f"cleanup called _kill({stale_pid}) without identity verification "
            "— stale accrued pids must be skipped when ps is unavailable"
        )
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_cleanup_is_bounded_with_many_stale_accrued_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-lived or hostile agent can leave thousands of stale pids in the
    monitor's accrued set. Cleanup must bound its total ps work to one bulk
    verification pass, not one
    ps -p per stale entry), so the agent timeout cannot turn into
    minutes of cleanup churn under degraded ps. We simulate 500 stale
    entries plus an empty cleanup snapshot and a slow batch verify; the
    whole cancellation must still return in well under five seconds."""
    batch_calls = 0

    async def empty_snapshot() -> dict[int, tuple[int, str]]:
        return {}

    async def slow_batch(_pids: list[int]) -> dict[int, str]:
        nonlocal batch_calls
        batch_calls += 1
        # Simulate a degraded ps that returns after a noticeable
        # delay but well under the per-call timeout. If cleanup is
        # serial-per-pid, this dominates total time.
        await asyncio.sleep(0.05)
        return {}

    async def seed_monitor(_root_pid: int, known: dict[int, str]) -> None:
        for pid in range(900000, 900500):
            known[pid] = "Mon May 25 14:30:25 2026"
        await asyncio.Event().wait()

    monkeypatch.setattr(base_mod, "_snapshot", empty_snapshot)
    monkeypatch.setattr(base_mod, "_verify_lstart_batch", slow_batch)
    monkeypatch.setattr(base_mod, "_monitor_descendants", seed_monitor)

    proc = await asyncio.create_subprocess_exec(
        "/bin/sleep",
        "300",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=proc.pid))
    try:
        await asyncio.sleep(0.1)
        loop = asyncio.get_event_loop()
        cancel_start = loop.time()
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail(
                "cleanup did not return within 5s with 500 stale accrued pids "
                "and a slow batch verifier — verification must be bounded"
            )
        elapsed = loop.time() - cancel_start
        # Even with the 50 ms slow batch, total cleanup work should be
        # ~one batch call plus a final snapshot, not 500 serial waits.
        assert elapsed < 4.0, f"cleanup took {elapsed:.2f}s — should be ~one batch call"
        assert batch_calls <= 1, (
            f"_verify_lstart_batch was called {batch_calls} times — "
            "cleanup must batch into a single bulk verification"
        )
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


@pytest.mark.asyncio
async def test_communicate_or_kill_monitor_race_does_not_break_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the monitor must be cancelled (and awaited) before
    _terminate_tree iterates the accrued set; otherwise a still-running
    monitor can add entries during the per-pid verify await and raise
    ``RuntimeError: dictionary changed size during iteration``. We use a
    monitor that keeps inserting while cleanup is running to catch this
    even if the monitor-cancel ordering regresses."""
    inserted_after_cancel: list[int] = []

    async def busy_monitor(_root_pid: int, known: dict[int, str]) -> None:
        # Seed a known entry so _terminate_tree iterates a non-empty
        # dict and reaches the await in _verify_lstart_batch.
        known[900001] = "Mon May 25 14:30:25 2026"
        next_pid = 900002
        try:
            while True:
                known[next_pid] = "Mon May 25 14:30:25 2026"
                inserted_after_cancel.append(next_pid)
                next_pid += 1
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

    async def slow_batch(_pids: list[int]) -> dict[int, str]:
        # Hold long enough that, without the monitor-cancel-first
        # ordering, the monitor could insert during iteration.
        await asyncio.sleep(0.05)
        return {}

    monkeypatch.setattr(base_mod, "_monitor_descendants", busy_monitor)
    monkeypatch.setattr(base_mod, "_verify_lstart_batch", slow_batch)

    proc = await asyncio.create_subprocess_exec(
        "/bin/sleep",
        "300",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=proc.pid))
    try:
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pytest.fail("cleanup hung; monitor race may have crashed _terminate_tree")
        # Done. No RuntimeError observed; iteration order was stable.
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass


def test_ps_binary_is_absolute_path() -> None:
    """Snapshot helpers must invoke ps via an absolute path. Resolving
    ``ps`` through PATH would let a project-supplied binary in a venv,
    direnv shim, or hostile checkout execute in the parent process
    (outside any agent sandbox) and forge the process snapshot used to
    decide which PIDs we signal."""
    assert os.path.isabs(base_mod._PS_BINARY), (
        f"_PS_BINARY must be absolute; got {base_mod._PS_BINARY!r}"
    )
    assert os.path.exists(base_mod._PS_BINARY), (
        f"_PS_BINARY {base_mod._PS_BINARY!r} does not exist"
    )


@pytest.mark.asyncio
async def test_register_pgid_tracks_live_agents_for_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent process can be killed via SIGTERM/SIGHUP without
    entering communicate_or_kill's CancelledError path -- agents are
    in their own sessions and don't share a foreground process group
    with the parent. The registry + signal-handler path must SIGKILL
    every registered pgid on parent shutdown, otherwise codex/gemini
    keep running after the CLI/MCP server is gone."""
    base_mod._LIVE_PGIDS.clear()

    pgid = 12345

    proc = await asyncio.create_subprocess_exec(
        "/bin/sleep",
        "300",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    task = asyncio.create_task(base_mod.communicate_or_kill(proc, pgid=proc.pid))
    try:
        await asyncio.sleep(0.1)
        assert proc.pid in base_mod._LIVE_PGIDS, (
            "communicate_or_kill did not register its pgid for shutdown cleanup"
        )
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass

    assert proc.pid not in base_mod._LIVE_PGIDS, (
        "communicate_or_kill did not unregister its pgid on completion"
    )

    # Use a fake pgid for the handler test so we don't risk signalling
    # a real process group. The handler must call os.killpg with
    # SIGKILL for every registered pgid then re-raise the signal with
    # default disposition.
    killpg_calls: list[tuple[int, int]] = []

    def fake_killpg(p: int, s: int) -> None:
        killpg_calls.append((p, s))

    def fake_kill(p: int, s: int) -> None:
        # Block the handler's final os.kill(getpid, sig) so we don't
        # actually take down the test process.
        pass

    monkeypatch.setattr(base_mod.os, "killpg", fake_killpg)
    monkeypatch.setattr(base_mod.os, "kill", fake_kill)
    monkeypatch.setattr(base_mod.signal, "signal", lambda *_a, **_kw: None)
    base_mod._LIVE_PGIDS.add(pgid)

    base_mod._shutdown_signal_handler(15, None)  # SIGTERM

    assert (pgid, base_mod.signal.SIGKILL) in killpg_calls, (
        "shutdown handler did not SIGKILL the registered pgid"
    )
    assert pgid not in base_mod._LIVE_PGIDS, (
        "shutdown handler did not clear the registry"
    )


def test_shutdown_handler_skips_own_pgid_to_avoid_self_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a registered pgid somehow matches our own process group, the
    shutdown handler must skip it -- killpg against our own group would
    SIGKILL the test process (and a user's parent shell) before we get
    a chance to re-raise the signal cleanly."""
    base_mod._LIVE_PGIDS.clear()
    our_pgid = os.getpgid(0)
    base_mod._LIVE_PGIDS.add(our_pgid)

    killpg_calls: list[tuple[int, int]] = []

    def fake_killpg(p: int, s: int) -> None:
        killpg_calls.append((p, s))

    monkeypatch.setattr(base_mod.os, "killpg", fake_killpg)

    base_mod._kill_all_live_pgids()

    assert killpg_calls == [], (
        f"_kill_all_live_pgids called killpg on our own pgid {our_pgid} "
        "-- would terminate the parent before clean shutdown"
    )


def test_shutdown_handler_does_not_deadlock_on_held_registry_lock() -> None:
    """Signal handlers run on the main thread between bytecodes. If a
    SIGTERM/SIGHUP is delivered while the main thread is inside
    _register_pgid / _unregister_pgid / _kill_all_live_pgids, the
    handler re-enters the registry lock on the SAME thread. With a
    non-reentrant ``threading.Lock`` that re-acquire deadlocks and the
    parent never terminates -- precisely the failure mode the shutdown
    feature is supposed to prevent. The lock must be reentrant."""
    import threading

    completed = threading.Event()

    def held_then_killall() -> None:
        # Hold the registry lock, then call the handler-equivalent
        # function on the SAME thread. Regular Lock blocks forever
        # here; RLock allows the re-entry.
        with base_mod._pgid_lock:
            base_mod._kill_all_live_pgids()
        completed.set()

    t = threading.Thread(target=held_then_killall, daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert completed.is_set(), (
        "_kill_all_live_pgids deadlocked when the registry lock was already "
        "held on the same thread -- shutdown handler would hang under signal "
        "delivery during registry mutation"
    )


def test_shutdown_handler_install_respects_sig_ign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run launched under nohup/disown/supervisor intentionally sets
    SIGHUP or SIGTERM to SIG_IGN. Installing our handler over SIG_IGN
    would convert an explicitly-ignored signal into process death and
    lose an in-flight multi-minute quorum run when a terminal closes.
    Only install over SIG_DFL."""
    import signal as signal_mod

    set_calls: list[tuple[int, object]] = []

    def fake_getsignal(sig: int) -> object:
        # Pretend the caller has set SIG_IGN on both shutdown signals.
        if sig in (signal_mod.SIGHUP, signal_mod.SIGTERM):
            return signal_mod.SIG_IGN
        return signal_mod.SIG_DFL

    def fake_signal(sig: int, handler: object) -> object:
        set_calls.append((sig, handler))
        return signal_mod.SIG_DFL

    monkeypatch.setattr(base_mod.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(base_mod.signal, "signal", fake_signal)
    # Force re-install path so the test exercises the install branch
    # regardless of whether earlier tests already installed handlers.
    monkeypatch.setattr(base_mod, "_shutdown_handlers_installed", False)
    monkeypatch.setattr(
        base_mod, "atexit", type("X", (), {"register": lambda *_: None})()
    )

    base_mod._ensure_shutdown_handlers()

    sighup_installs = [c for c in set_calls if c[0] == signal_mod.SIGHUP]
    sigterm_installs = [c for c in set_calls if c[0] == signal_mod.SIGTERM]
    assert sighup_installs == [], (
        "shutdown handler installed over SIG_IGN'd SIGHUP -- would break nohup runs"
    )
    assert sigterm_installs == [], (
        "shutdown handler installed over SIG_IGN'd SIGTERM -- would break "
        "supervisor-managed runs"
    )


@pytest.mark.asyncio
async def test_codex_agent_unlinks_temp_file_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CodexAgent.run allocates a temp file for codex's -o output flag.
    When the run task is cancelled mid-flight (e.g. by the council per-agent
    timeout), the temp file must be removed — otherwise /tmp accumulates
    quorum-codex-* files on every timeout."""
    captured: dict[str, str] = {}
    orig_named = tempfile.NamedTemporaryFile

    def capture(*args, **kwargs):  # type: ignore[no-untyped-def]
        f = orig_named(*args, **kwargs)
        captured["path"] = f.name
        return f

    monkeypatch.setattr("quorum.agents.codex.tempfile.NamedTemporaryFile", capture)

    agent = CodexAgent(binary="/bin/sleep")

    def _stub_cmd(**_kw: str) -> list[str]:
        return ["/bin/sleep", "300"]

    monkeypatch.setattr(agent, "build_command", _stub_cmd)

    task = asyncio.create_task(agent.run(prompt="x", cwd="/"))
    try:
        await asyncio.sleep(0.05)
        task.cancel()
        # Brief window for the task to settle and the unlink to happen.
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except (TimeoutError, asyncio.CancelledError):
            pass

        assert "path" in captured, "tempfile was never allocated"
        assert not Path(captured["path"]).exists(), (
            f"codex temp file {captured['path']} survived cancellation"
        )
    finally:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
        if "path" in captured:
            Path(captured["path"]).unlink(missing_ok=True)


# --- communicate_lines_or_kill: streaming read with idle timeout ----------
#
# opencode streams its run as newline-delimited JSON; a healthy run emits a
# line every few seconds. A stalled provider stream produces no line for
# minutes. communicate_lines_or_kill reads stdout incrementally and kills the
# whole process group if no line arrives within idle_timeout, reusing the same
# process-group + setsid-escapee teardown as communicate_or_kill.


@pytest.mark.asyncio
async def test_communicate_lines_or_kill_idle_timeout_kills_process_group() -> None:
    """A subprocess that emits one line then goes silent past the idle
    timeout must be reported as an idle timeout, and the whole process group
    (including a backgrounded descendant) must be torn down — not just the
    direct child. The line produced before the stall is preserved."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        # Background a long sleeper, print its PID, then go silent: the shell
        # itself blocks in `sleep 300` so no further stdout line ever arrives.
        "sleep 300 & echo $! ; sleep 300",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    grandchild_pid: int | None = None
    try:
        stdout_b, _stderr_b, idle = await base_mod.communicate_lines_or_kill(
            proc, pgid=pgid, idle_timeout=0.3
        )
        grandchild_pid = int(stdout_b.strip())
        assert idle is True, "a stalled stream must be reported as an idle timeout"
        assert await _wait_until_dead(grandchild_pid, timeout_s=2.0), (
            f"backgrounded sleeper PID {grandchild_pid} survived the idle "
            "timeout — the idle-kill path must tear down the whole process group"
        )
        assert await _wait_until_dead(proc.pid, timeout_s=2.0), (
            f"shell leader PID {proc.pid} survived the idle timeout"
        )
    finally:
        for pid in (grandchild_pid, proc.pid):
            if pid is None:
                continue
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_communicate_lines_or_kill_returns_all_output_on_clean_exit() -> None:
    """Normal completion: every stdout line is captured, stderr is captured,
    and idle_timed_out is False when the process exits on its own before the
    idle timeout. The streaming read must not corrupt or drop the happy
    path."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "echo one ; echo two ; echo oops 1>&2 ; exit 0",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    stdout_b, stderr_b, idle = await base_mod.communicate_lines_or_kill(
        proc, pgid=proc.pid, idle_timeout=5.0
    )
    assert idle is False, "a clean exit must not be reported as an idle timeout"
    assert stdout_b == b"one\ntwo\n", f"unexpected stdout: {stdout_b!r}"
    assert stderr_b == b"oops\n", f"unexpected stderr: {stderr_b!r}"
    assert proc.returncode == 0


@pytest.mark.asyncio
async def test_communicate_lines_or_kill_cancellation_terminates_group() -> None:
    """The council's 600s outer cap cancels the agent task; that cancellation
    propagates through communicate_lines_or_kill, which must tear down the
    process group exactly like communicate_or_kill. The descendant PID is
    captured via a temp file so the test doesn't consume the stdout the
    function is reading."""
    pidfile = tempfile.NamedTemporaryFile(mode="r", suffix=".pid", delete=False)
    pidfile.close()
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        f"sleep 300 & echo $! > {pidfile.name} ; wait",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    task = asyncio.create_task(
        base_mod.communicate_lines_or_kill(proc, pgid=pgid, idle_timeout=60.0)
    )
    grandchild_pid: int | None = None
    try:
        # Wait for the shell to background the sleeper and record its PID.
        for _ in range(100):
            await asyncio.sleep(0.02)
            text = Path(pidfile.name).read_text().strip()
            if text:
                grandchild_pid = int(text)
                break
        assert grandchild_pid is not None, "shell never recorded the sleeper PID"
        assert _pid_alive(grandchild_pid), "sleeper should be running before cancel"

        task.cancel()
        assert await _wait_until_dead(grandchild_pid, timeout_s=2.0), (
            f"sleeper PID {grandchild_pid} survived cancel — "
            "communicate_lines_or_kill did not tear down the process group"
        )
    finally:
        for pid in (grandchild_pid, proc.pid):
            if pid is None:
                continue
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
        Path(pidfile.name).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_communicate_lines_or_kill_eof_without_exit_is_flagged_not_success() -> (
    None
):
    """A process that closes stdout (EOF) but does NOT exit must not be
    reported as a clean success or leaked. The bounded clean-EOF reap times
    out; the function must escalate to a full teardown, flag the kill, and
    leave no surviving process group. Otherwise returncode stays None and the
    opencode adapter maps `None or 0` to a false success."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "exec 1>&- ; sleep 300",  # close stdout (EOF on parent), stay alive
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    try:
        _stdout_b, _stderr_b, idle = await base_mod.communicate_lines_or_kill(
            proc, pgid=pgid, idle_timeout=60.0
        )
        assert idle is True, (
            "stdout-EOF-but-process-alive must be flagged as killed, not "
            "returned as a clean (idle=False) success"
        )
        assert await _wait_until_dead(proc.pid, timeout_s=2.0), (
            f"process PID {proc.pid} survived — EOF-without-exit must still "
            "tear down the process group, not leak it"
        )
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_communicate_lines_or_kill_cancel_during_eof_wait_propagates() -> None:
    """If the council's outer cap cancels the task while it is inside the
    bounded clean-EOF proc.wait()/stderr-collect, the CancelledError MUST
    propagate (and tear down the group) -- it must not be swallowed by the
    cleanup-timeout except clause and turned into a normal return, which would
    silently defeat the outer timeout."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "exec 1>&- ; sleep 300",  # EOF on stdout, process stays alive
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    task = asyncio.create_task(
        base_mod.communicate_lines_or_kill(proc, pgid=pgid, idle_timeout=60.0)
    )
    try:
        # Let it hit stdout EOF and enter the bounded clean-EOF wait, then
        # cancel while it is parked there.
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await _wait_until_dead(proc.pid, timeout_s=2.0), (
            f"process PID {proc.pid} survived cancel — a swallowed "
            "CancelledError skipped the teardown"
        )
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
