"""In-process job registry for the start/await MCP tool surface.

Each `q_<mode>_start` call schedules an asyncio.Task and returns a job_id; the
paired `q_await(job_id)` waits for the task and
returns its result. One-shot retention — the entry is removed when await
completes (success or error). A TTL reaper removes abandoned jobs so a
caller that never returns doesn't leak agent subprocesses indefinitely.

Cancellation semantics:
- If the awaiting MCP request is cancelled (client disconnect), we cancel
  the underlying task too so agent subprocesses don't orphan. The agents'
  own cancellation handlers (start_new_session + SIGKILL on group) tear
  down their subprocesses.
- If the reaper cancels an expired task, a subsequent await_job raises a
  meaningful ValueError instead of a bare CancelledError.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

# Default TTL: a job older than this is reaped even if the caller never
# returns to retrieve it. Generous because q-validate extended can run
# 10+ minutes on large codebases.
JOB_TTL_SECONDS: float = 30 * 60.0

# How often the reaper wakes to scan for expired entries.
REAPER_INTERVAL_SECONDS: float = 60.0

# Bound on how long we wait for a cancelled task to finish its cleanup
# before giving up. The agent layer's own communicate_or_kill timeouts
# are bounded; this is a second-order ceiling.
DRAIN_TIMEOUT_SECONDS: float = 5.0


@dataclass
class _Job:
    task: asyncio.Task[Any]
    created_at: float
    claimed: bool = False  # set when an await_job is in progress


_jobs: dict[str, _Job] = {}
_reaper_task: asyncio.Task[None] | None = None
_reaper_lock: asyncio.Lock | None = None

# Outstanding drain tasks scheduled by _cancel_and_drain. We hold strong
# refs here so the asyncio loop doesn't garbage-collect them mid-cleanup
# (asyncio.create_task only holds a weak ref via the loop).
_drain_tasks: set[asyncio.Task[None]] = set()


def _get_lock() -> asyncio.Lock:
    """Lazy-construct the lock so it binds to the running event loop.
    Calling asyncio.Lock() at module import time would attach it to a
    loop that may not be the one FastMCP runs."""
    global _reaper_lock
    if _reaper_lock is None:
        _reaper_lock = asyncio.Lock()
    return _reaper_lock


def _retrieve_orphan_exception(task: asyncio.Task[Any]) -> None:
    """Done-callback that retrieves any post-drain exception so asyncio
    doesn't log 'Task exception was never retrieved'. The drainer may
    give up at its deadline while the underlying task is still running;
    if that task subsequently raises a non-cancellation exception, no
    one will await it. Calling task.exception() marks it as retrieved."""
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.InvalidStateError:
        pass


def _cancel_and_drain(task: asyncio.Task[Any]) -> None:
    """Cancel `task` and schedule a bounded background drain.

    `task.cancel()` alone returns immediately and the cleanup runs
    whenever the loop next services the task. That's usually fine —
    agent subprocess cleanup is already bounded by the agent layer's
    own timeouts — but a drain task gives us an observation point and
    a second-order ceiling. Failures are suppressed: the goal is
    letting the task finish its teardown, not surfacing errors nobody
    is watching for in this surface.

    The drainer loops on shield(task) under an absolute deadline so it
    keeps observing even under event-loop-shutdown cancellation
    pressure (asyncio.shield only blocks ONE cancellation per await).
    A done-callback on the underlying task suppresses asyncio's
    'unhandled exception' warning if the task raises after the drainer
    has given up."""
    if not task.done():
        task.cancel()
    task.add_done_callback(_retrieve_orphan_exception)

    async def _drain() -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + DRAIN_TIMEOUT_SECONDS
        while not task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except (TimeoutError, asyncio.CancelledError, Exception):
                # Loop and re-check; bounded by the absolute deadline.
                pass

    drainer = asyncio.create_task(_drain())
    _drain_tasks.add(drainer)
    drainer.add_done_callback(_drain_tasks.discard)


async def _reaper_loop() -> None:
    """Wake periodically; cancel jobs older than TTL and drop their entries.
    Runs for the lifetime of the MCP server."""
    while True:
        await asyncio.sleep(REAPER_INTERVAL_SECONDS)
        now = time.monotonic()
        expired = [
            job_id
            for job_id, job in list(_jobs.items())
            if now - job.created_at > JOB_TTL_SECONDS
        ]
        for job_id in expired:
            job = _jobs.pop(job_id, None)
            if job is not None and not job.task.done():
                _cancel_and_drain(job.task)


async def _ensure_reaper() -> None:
    """Start the reaper on first use. Idempotent across concurrent calls."""
    global _reaper_task
    async with _get_lock():
        if _reaper_task is None or _reaper_task.done():
            _reaper_task = asyncio.create_task(_reaper_loop())


async def start_job(coro: Coroutine[Any, Any, Any]) -> str:
    """Schedule `coro` as a background asyncio.Task and return a job_id.

    If `_ensure_reaper()` is cancelled before the task is created, close
    `coro` explicitly — otherwise the caller's coroutine is never
    scheduled or closed and Python emits a 'coroutine was never awaited'
    warning. After create_task, ownership passes to the asyncio loop."""
    try:
        await _ensure_reaper()
    except BaseException:
        coro.close()
        raise
    job_id = uuid.uuid4().hex
    task = asyncio.create_task(coro)
    _jobs[job_id] = _Job(task=task, created_at=time.monotonic())
    return job_id


async def await_job(job_id: str) -> Any:
    """Block until the job completes; return its result.

    Keeps the entry in `_jobs` until the await resolves (one-shot via
    the `claimed` flag) so the reaper can still cancel a task whose
    caller has hung past TTL — without that, an await_job blocked on a
    stuck agent would wait forever. On caller cancellation, cancels the
    underlying task and drains it in the background. If the task was
    cancelled externally (reaper), raises a descriptive ValueError
    instead of CancelledError.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise ValueError(
            f"job_id not found (expired, already retrieved, or invalid): {job_id}"
        )
    if job.claimed:
        raise ValueError(f"job_id {job_id} is already being awaited by another caller")
    job.claimed = True
    try:
        return await asyncio.shield(job.task)
    except asyncio.CancelledError:
        if job.task.cancelled():
            raise ValueError(
                f"job {job_id} was cancelled (TTL exceeded or server shutdown)"
            ) from None
        # Caller's await was cancelled (client disconnect). Cancel the
        # underlying task so agent subprocesses tear down cleanly, and
        # drain it in the background so the cleanup is bounded and
        # observable rather than fire-and-forget.
        _cancel_and_drain(job.task)
        raise
    finally:
        _jobs.pop(job_id, None)


def active_job_count() -> int:
    """Test helper: number of registered jobs (incl. still-running)."""
    return len(_jobs)


def active_drain_count() -> int:
    """Test helper: number of in-flight drain tasks."""
    return len(_drain_tasks)


def _reset_for_tests() -> None:
    """Test helper: wipe registry and stop the reaper. Tests that want
    a clean module state between cases call this in their fixtures."""
    global _reaper_task, _reaper_lock
    for job in list(_jobs.values()):
        if not job.task.done():
            job.task.cancel()
    _jobs.clear()
    for drainer in list(_drain_tasks):
        if not drainer.done():
            drainer.cancel()
    _drain_tasks.clear()
    if _reaper_task is not None and not _reaper_task.done():
        _reaper_task.cancel()
    _reaper_task = None
    _reaper_lock = None
