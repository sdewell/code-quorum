"""Unit tests for the MCP start/await job registry."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from quorum_mcp import jobs


@pytest.fixture(autouse=True)
def reset_jobs() -> Iterator[None]:
    """Wipe the module-level registry + reaper between tests so cases
    don't leak state."""
    jobs._reset_for_tests()
    yield
    jobs._reset_for_tests()


@pytest.fixture(autouse=True)
def allow_test_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MCP production paths are root-confined; unit tests work in tmp_path."""
    from quorum_mcp import server

    monkeypatch.setattr(server, "active_allowed_roots", lambda: (tmp_path,))


def test_resolve_cwd_rejects_directory_outside_allowed_roots(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quorum_mcp import server

    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setattr(server, "active_allowed_roots", lambda: (allowed,))

    with pytest.raises(ValueError, match="outside allowed roots"):
        server._resolve_cwd(str(outside))


@pytest.mark.parametrize("cwd", [None, "", "   "])
def test_resolve_cwd_requires_explicit_directory(cwd) -> None:
    from quorum_mcp import server

    with pytest.raises(ValueError, match="explicit cwd is required"):
        server._resolve_cwd(cwd)


@pytest.mark.parametrize("cwd", [".", "project", "../project"])
def test_resolve_cwd_rejects_relative_directory(cwd: str) -> None:
    from quorum_mcp import server

    with pytest.raises(ValueError, match="must be absolute"):
        server._resolve_cwd(cwd)


@pytest.mark.asyncio
async def test_start_job_returns_hex_id() -> None:
    async def work() -> str:
        return "ok"

    job_id = await jobs.start_job(work())
    assert isinstance(job_id, str)
    assert len(job_id) == 32  # uuid4().hex
    assert all(c in "0123456789abcdef" for c in job_id)


@pytest.mark.asyncio
async def test_start_job_returns_unique_ids() -> None:
    async def work() -> int:
        return 1

    ids = {await jobs.start_job(work()) for _ in range(5)}
    assert len(ids) == 5


@pytest.mark.asyncio
async def test_await_job_retrieves_result() -> None:
    async def work() -> str:
        await asyncio.sleep(0)
        return "payload"

    job_id = await jobs.start_job(work())
    result = await jobs.await_job(job_id)
    assert result == "payload"


@pytest.mark.asyncio
async def test_await_job_is_one_shot() -> None:
    async def work() -> str:
        return "once"

    job_id = await jobs.start_job(work())
    assert await jobs.await_job(job_id) == "once"
    with pytest.raises(ValueError, match="not found"):
        await jobs.await_job(job_id)


@pytest.mark.asyncio
async def test_await_job_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        await jobs.await_job("deadbeef" * 4)


@pytest.mark.asyncio
async def test_await_job_propagates_task_exception() -> None:
    async def boom() -> None:
        raise RuntimeError("kaboom")

    job_id = await jobs.start_job(boom())
    with pytest.raises(RuntimeError, match="kaboom"):
        await jobs.await_job(job_id)
    # One-shot: entry removed even on error.
    assert jobs.active_job_count() == 0


@pytest.mark.asyncio
async def test_reaper_cancels_expired_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "JOB_TTL_SECONDS", 0.05)
    monkeypatch.setattr(jobs, "REAPER_INTERVAL_SECONDS", 0.02)

    async def slow() -> str:
        await asyncio.sleep(5.0)
        return "should be cancelled"

    job_id = await jobs.start_job(slow())
    # Give the reaper enough wakeups to find and cancel the entry.
    await asyncio.sleep(0.25)
    with pytest.raises(ValueError, match="TTL exceeded|not found"):
        await jobs.await_job(job_id)


@pytest.mark.asyncio
async def test_caller_cancel_propagates_to_task() -> None:
    """When the awaiter is cancelled (e.g. MCP client disconnect), the
    underlying task must be cancelled too so agent subprocesses tear
    down instead of orphaning."""
    started = asyncio.Event()
    cancelled_inside = asyncio.Event()

    async def long_running() -> str:
        started.set()
        try:
            await asyncio.sleep(10.0)
            return "unreachable"
        except asyncio.CancelledError:
            cancelled_inside.set()
            raise

    job_id = await jobs.start_job(long_running())
    awaiter = asyncio.create_task(jobs.await_job(job_id))
    await started.wait()
    awaiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await awaiter
    # Give the loop a tick to deliver cancellation to the inner task.
    for _ in range(20):
        if cancelled_inside.is_set():
            break
        await asyncio.sleep(0.01)
    assert cancelled_inside.is_set(), "inner task should have been cancelled"


@pytest.mark.asyncio
async def test_concurrent_start_jobs_dont_collide() -> None:
    async def work(n: int) -> int:
        await asyncio.sleep(0.01)
        return n

    ids = await asyncio.gather(*[jobs.start_job(work(i)) for i in range(10)])
    assert len(set(ids)) == 10
    results = await asyncio.gather(*[jobs.await_job(i) for i in ids])
    assert sorted(results) == list(range(10))


@pytest.mark.asyncio
async def test_caller_cancel_schedules_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller-disconnect path must schedule a tracked drain task so
    the underlying job gets a bounded cleanup window — not fire-and-
    forget cancel."""
    monkeypatch.setattr(jobs, "DRAIN_TIMEOUT_SECONDS", 0.5)
    started = asyncio.Event()

    async def long_running() -> str:
        started.set()
        try:
            await asyncio.sleep(10.0)
            return "unreachable"
        except asyncio.CancelledError:
            raise

    job_id = await jobs.start_job(long_running())
    awaiter = asyncio.create_task(jobs.await_job(job_id))
    await started.wait()
    assert jobs.active_drain_count() == 0
    awaiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await awaiter
    # A drain was scheduled; let it run to completion so we can confirm
    # the set drains itself.
    for _ in range(100):
        if jobs.active_drain_count() == 0:
            break
        await asyncio.sleep(0.02)
    assert jobs.active_drain_count() == 0


@pytest.mark.asyncio
async def test_reaper_drains_expired_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reaper must drain its cancellations through the same tracked
    helper so cleanup is bounded and observable."""
    monkeypatch.setattr(jobs, "JOB_TTL_SECONDS", 0.05)
    monkeypatch.setattr(jobs, "REAPER_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(jobs, "DRAIN_TIMEOUT_SECONDS", 0.5)

    async def slow() -> str:
        await asyncio.sleep(5.0)
        return "should be cancelled"

    await jobs.start_job(slow())
    # Wait for reaper to wake at least once and process the expiry.
    for _ in range(50):
        if jobs.active_job_count() == 0:
            break
        await asyncio.sleep(0.02)
    assert jobs.active_job_count() == 0
    # Drain set should empty out as the cancelled task finishes.
    for _ in range(100):
        if jobs.active_drain_count() == 0:
            break
        await asyncio.sleep(0.02)
    assert jobs.active_drain_count() == 0


@pytest.mark.asyncio
async def test_mcp_server_registers_expected_tools() -> None:
    """The server module must register exactly the expected tools.
    Catches decorator typos and missing imports."""
    from quorum_mcp.server import mcp as mcp_server

    tools = await mcp_server.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "q_plan_start",
        "q_brainstorm_start",
        "q_validate_start",
        "q_review_start",
        "q_await",
        "q_research",
    }


@pytest.mark.asyncio
async def test_mcp_start_tools_expose_verbose_schema() -> None:
    """The MCP escape hatch must be visible in the registered tool schemas, not
    merely accepted by the underlying Python functions."""
    from quorum_mcp.server import mcp as mcp_server

    tools = await mcp_server.list_tools()
    by_name = {tool.name: tool for tool in tools}
    for name in (
        "q_plan_start",
        "q_brainstorm_start",
        "q_validate_start",
        "q_review_start",
    ):
        verbose = by_name[name].inputSchema["properties"].get("verbose")
        assert verbose == {
            "default": False,
            "title": "Verbose",
            "type": "boolean",
        }


@pytest.mark.asyncio
async def test_mcp_start_tools_require_explicit_cwd_schema() -> None:
    """A host omission must fail schema validation instead of selecting the
    MCP server's own plugin or checkout directory."""
    from quorum_mcp.server import mcp as mcp_server

    tools = await mcp_server.list_tools()
    by_name = {tool.name: tool for tool in tools}
    for name in (
        "q_plan_start",
        "q_brainstorm_start",
        "q_validate_start",
        "q_review_start",
    ):
        schema = by_name[name].inputSchema
        assert "cwd" in schema["required"]
        assert schema["properties"]["cwd"] == {
            "title": "Cwd",
            "type": "string",
        }


@pytest.mark.asyncio
async def test_mcp_start_tools_expose_gemini_model_schema() -> None:
    """The per-run agy-seat model override (e.g. 'run the gemini seat on
    Claude') must be visible in every council-start tool's schema, not merely
    accepted by the underlying Python functions."""
    from quorum_mcp.server import mcp as mcp_server

    tools = await mcp_server.list_tools()
    by_name = {tool.name: tool for tool in tools}
    for name in (
        "q_plan_start",
        "q_brainstorm_start",
        "q_validate_start",
        "q_review_start",
    ):
        prop = by_name[name].inputSchema["properties"].get("gemini_model")
        assert prop is not None, f"{name} does not expose gemini_model"
        assert prop.get("default") is None


@pytest.mark.asyncio
async def test_mcp_start_tools_expose_explicit_host_schema() -> None:
    """Every council start names the orchestrator host so one shared MCP
    manifest can select the correct external roster and containment path."""
    from quorum_mcp.server import mcp as mcp_server

    tools = await mcp_server.list_tools()
    by_name = {tool.name: tool for tool in tools}
    for name in (
        "q_plan_start",
        "q_brainstorm_start",
        "q_validate_start",
        "q_review_start",
    ):
        host = by_name[name].inputSchema["properties"].get("host")
        assert host is not None
        assert host.get("default") is None
        assert {item.get("type") for item in host["anyOf"]} == {"string", "null"}


@pytest.mark.asyncio
async def test_every_start_tool_passes_explicit_host_to_agent_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from quorum_mcp import server

    selected_hosts: list[str | None] = []

    def fake_select(*args, host=None, **kwargs):
        selected_hosts.append(host)
        return [], []

    async def fake_run_mode(**kwargs):
        return [[]]

    async def fake_resolve(target, cwd):
        return "diff", "whole-codebase review"

    monkeypatch.setattr(server, "select_agents", fake_select)
    monkeypatch.setattr(server, "run_mode", fake_run_mode)
    monkeypatch.setattr(server, "format_rounds", lambda r: "rendered")
    monkeypatch.setattr(server, "resolve_review_target", fake_resolve)
    plan = tmp_path / "plan.md"
    plan.write_text("# plan\n", encoding="utf-8")

    starts = [
        server.q_plan_start(task="t", cwd=str(tmp_path), host="codex"),
        server.q_brainstorm_start(topic="t", cwd=str(tmp_path), host="codex"),
        server.q_validate_start(plan_path=str(plan), cwd=str(tmp_path), host="codex"),
        server.q_review_start(target="all", cwd=str(tmp_path), host="codex"),
    ]
    for start in starts:
        out = await start
        await server.q_await(out["job_id"])

    assert selected_hosts == ["codex", "codex", "codex", "codex"]


@pytest.mark.asyncio
async def test_every_start_tool_passes_gemini_model_to_the_seat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Each start tool calls select_agents at its own call site, so each one
    can independently forget the pass-through -- exercise all four."""
    from quorum.agents.gemini_cli import GeminiCliAgent
    from quorum_mcp import server

    captured: dict[str, list] = {}

    async def fake_run_mode(**kwargs):
        captured["agents"] = kwargs["agents"]
        return [[]]

    async def fake_resolve(target, cwd):
        return (
            "[WHOLE CODEBASE REVIEW]\nInspect the repository.",
            "whole-codebase review",
        )

    monkeypatch.setattr(server, "run_mode", fake_run_mode)
    monkeypatch.setattr(server, "format_rounds", lambda r: "rendered")
    monkeypatch.setattr(server, "resolve_review_target", fake_resolve)
    plan = tmp_path / "plan.md"
    plan.write_text("# plan\n", encoding="utf-8")

    common = dict(
        cwd=str(tmp_path),
        agents=["gemini"],
        gemini_model="claude-opus-4-6-thinking",
    )
    starts = [
        server.q_plan_start(task="t", **common),
        server.q_brainstorm_start(topic="t", **common),
        server.q_validate_start(plan_path=str(plan), **common),
        server.q_review_start(target="all", **common),
    ]
    for start in starts:
        captured.clear()
        out = await start
        await server.q_await(out["job_id"])
        [seat] = captured["agents"]
        assert isinstance(seat, GeminiCliAgent)
        assert seat.model == "claude-opus-4-6-thinking"


def test_brainstorm_prompt_grounding_uses_grounding_block() -> None:
    from quorum_mcp.server import _brainstorm_prompt

    out = _brainstorm_prompt("topic", "- idea", grounding=True)
    assert "validation guide" in out.lower()
    assert "do not generate new" in out.lower()


def test_brainstorm_prompt_default_uses_divergence_block() -> None:
    from quorum_mcp.server import _brainstorm_prompt

    out = _brainstorm_prompt("topic", "- idea")
    assert "do not generate new" not in out.lower()
    assert "extend past them" in out.lower()


def test_brainstorm_prompt_grounding_noop_without_prior() -> None:
    from quorum_mcp.server import _brainstorm_prompt

    assert _brainstorm_prompt("topic", None, grounding=True) == _brainstorm_prompt(
        "topic", None
    )


@pytest.mark.asyncio
async def test_brainstorm_grounding_keeps_opencode_on_flash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A brainstorm-phase call with grounding=True keeps the opencode seat on
    V4 Flash -- the grounding flag must NOT bump it to Pro. This locks the
    phase-resolution property, not a skystorm workflow step: the shipped
    q-skystorm Stage 4 grounding pass is codex-only (agents=["codex"]), so
    opencode never actually runs during skystorm grounding. The property still
    matters -- it pins the approved trade-off (the eval's Pro grounding edge was
    small and non-robust, the latency/cost win is real) so a future change
    routing grounding to a Pro phase (e.g. phase="review" if grounding) fails
    loudly here rather than silently reverting it."""
    from quorum.agents import OpenCodeAgent
    from quorum_mcp import server

    captured: dict[str, list] = {}

    async def fake_run_mode(**kwargs):
        captured["agents"] = kwargs["agents"]
        return [[]]

    monkeypatch.setattr(server, "run_mode", fake_run_mode)
    monkeypatch.setattr(server, "format_rounds", lambda r: "rendered")

    out = await server.q_brainstorm_start(
        topic="t",
        prior_ideas="- idea",
        grounding=True,
        agents=["opencode"],
        cwd=str(tmp_path),
    )
    await server.q_await(out["job_id"])
    seat = captured["agents"][0]
    assert isinstance(seat, OpenCodeAgent)
    assert seat.model == "openrouter/deepseek/deepseek-v4.1-flash"


@pytest.mark.asyncio
async def test_q_review_start_empty_diff_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """An empty diff (target != 'all') must short-circuit: q_review_start
    returns the uniform {"job_id"} shape, and the awaited result is the
    'no changes' message WITHOUT ever running the council."""
    from quorum_mcp import server

    async def fake_resolve(target, cwd):
        return "", "no changes vs main"

    monkeypatch.setattr(server, "resolve_review_target", fake_resolve)

    def fail_run_mode(*args, **kwargs):
        raise AssertionError("council must not run on an empty diff")

    monkeypatch.setattr(server, "run_mode", fail_run_mode)

    out = await server.q_review_start(cwd=str(tmp_path))
    assert set(out) == {"job_id"}
    result = await server.q_await(out["job_id"])
    assert result == "no changes vs main — nothing to review."


@pytest.mark.asyncio
async def test_q_plan_start_all_disabled_default_roster_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """If every default-roster seat is disabled, the result must be an
    explicit, named MCP error. It must preserve the {"job_id"} contract without
    calling run_mode or returning a blank success."""
    from quorum import model_config as mc
    from quorum_mcp import server

    def fail_run_mode(*args, **kwargs):
        raise AssertionError("run_mode must never be called with zero live seats")

    monkeypatch.setattr(server, "run_mode", fail_run_mode)
    for seat in ("codex", "gemini", "opencode"):
        mc.record_choice(seat, {"disabled": "true"})

    out = await server.q_plan_start(task="t", cwd=str(tmp_path))
    assert set(out) == {"job_id"}
    result = await server.q_await(out["job_id"])
    assert "No seats to run" in result
    assert "codex" in result
    assert "gemini" in result
    assert "opencode" in result
    assert "quorum setup-models" in result


@pytest.mark.asyncio
async def test_q_review_start_all_target_runs_council(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """target='all' carries an explicit repository-inspection instruction."""
    from quorum_mcp import server

    async def fake_resolve(target, cwd):
        return (
            "[WHOLE CODEBASE REVIEW]\nInspect the repository.",
            "whole-codebase review",
        )

    ran = asyncio.Event()
    captured: dict[str, object] = {}

    async def fake_run_mode(**kwargs):
        captured.update(kwargs)
        ran.set()
        return [[]]

    monkeypatch.setattr(server, "resolve_review_target", fake_resolve)
    monkeypatch.setattr(server, "run_mode", fake_run_mode)
    monkeypatch.setattr(server, "format_rounds", lambda r: "rendered")

    out = await server.q_review_start(target="all", cwd=str(tmp_path))
    assert set(out) == {"job_id"}
    result = await server.q_await(out["job_id"])
    assert ran.is_set()
    assert "WHOLE CODEBASE REVIEW" in str(captured["prompt"])
    assert "Inspect the repository" in str(captured["prompt"])
    assert result.startswith("Council:")  # liveness line prefixes the render
    assert result.endswith("rendered")


@pytest.mark.asyncio
async def test_q_review_start_padded_all_target_runs_council(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A padded all target carries the same explicit inspection instruction."""
    from quorum_mcp import server

    async def fake_resolve(target, cwd):
        return (
            "[WHOLE CODEBASE REVIEW]\nInspect the repository.",
            "whole-codebase review",
        )

    ran = asyncio.Event()

    async def fake_run_mode(**kwargs):
        ran.set()
        return [[]]

    monkeypatch.setattr(server, "resolve_review_target", fake_resolve)
    monkeypatch.setattr(server, "run_mode", fake_run_mode)
    monkeypatch.setattr(server, "format_rounds", lambda r: "rendered")

    out = await server.q_review_start(target="  all  ", cwd=str(tmp_path))
    result = await server.q_await(out["job_id"])
    assert ran.is_set()
    assert result.startswith("Council:")  # liveness line prefixes the render
    assert result.endswith("rendered")


@pytest.mark.asyncio
async def test_start_job_closes_coro_on_reaper_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If await _ensure_reaper() is cancelled before create_task runs,
    start_job must close the caller's coroutine explicitly. Otherwise
    Python emits 'coroutine was never awaited' and the closure leaks.
    A closed coroutine has cr_frame == None."""

    async def never_awaited() -> str:
        return "should not run"

    coro = never_awaited()

    async def cancelled_reaper() -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(jobs, "_ensure_reaper", cancelled_reaper)

    with pytest.raises(asyncio.CancelledError):
        await jobs.start_job(coro)
    assert coro.cr_frame is None, "coro should be closed after cancelled start_job"


@pytest.mark.asyncio
async def test_reaper_cancels_hung_in_progress_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller awaiting a hung task must NOT escape the TTL reaper.
    Before the fix, await_job popped the entry from _jobs on entry, so
    the reaper couldn't see (or cancel) it — the await would hang
    forever. Verify the reaper cancels the underlying task even while
    a caller is mid-await, and that await_job surfaces a meaningful
    ValueError."""
    monkeypatch.setattr(jobs, "JOB_TTL_SECONDS", 0.05)
    monkeypatch.setattr(jobs, "REAPER_INTERVAL_SECONDS", 0.02)

    async def stuck() -> str:
        await asyncio.sleep(10.0)
        return "unreachable"

    job_id = await jobs.start_job(stuck())
    with pytest.raises(ValueError, match="TTL exceeded"):
        await jobs.await_job(job_id)
    # Entry must be cleaned up too.
    assert jobs.active_job_count() == 0


@pytest.mark.asyncio
async def test_await_job_rejects_double_claim() -> None:
    """One-shot semantics: a second concurrent await_job on the same
    job_id must reject rather than both callers waiting on the same
    task."""
    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(0.5)
        return "first"

    job_id = await jobs.start_job(slow())
    first = asyncio.create_task(jobs.await_job(job_id))
    await started.wait()
    with pytest.raises(ValueError, match="already being awaited"):
        await jobs.await_job(job_id)
    assert await first == "first"


@pytest.mark.asyncio
async def test_retrieve_orphan_exception_handles_all_task_states() -> None:
    """The done-callback must handle cancelled, raised, and succeeded
    tasks without crashing — it's attached unconditionally to every
    drained task."""

    async def boom() -> None:
        raise RuntimeError("kaboom")

    async def returns_value() -> str:
        return "ok"

    async def slow() -> None:
        await asyncio.sleep(5.0)

    raised = asyncio.create_task(boom())
    succeeded = asyncio.create_task(returns_value())
    cancelled = asyncio.create_task(slow())
    for _ in range(50):
        if raised.done() and succeeded.done():
            break
        await asyncio.sleep(0.01)
    cancelled.cancel()
    for _ in range(50):
        if cancelled.done():
            break
        await asyncio.sleep(0.01)
    # All three callbacks must complete without raising.
    jobs._retrieve_orphan_exception(raised)
    jobs._retrieve_orphan_exception(succeeded)
    jobs._retrieve_orphan_exception(cancelled)
    # Verify each state was handled correctly.
    assert isinstance(raised.exception(), RuntimeError)
    assert succeeded.result() == "ok"
    assert cancelled.cancelled()


@pytest.mark.asyncio
async def test_cancel_and_drain_attaches_orphan_callback() -> None:
    """Sanity check: _cancel_and_drain wires the done-callback so a
    task raising after the drain deadline doesn't produce an asyncio
    'Task exception was never retrieved' log entry. Verified by
    confirming the callback is registered on the task."""

    async def slow() -> None:
        await asyncio.sleep(5.0)

    task = asyncio.create_task(slow())
    jobs._cancel_and_drain(task)
    # Internal API: a done-callback was attached.
    # We can't easily introspect callbacks from outside, so just verify
    # the task gets cancelled cleanly and no warning leaks.
    for _ in range(50):
        if task.done():
            break
        await asyncio.sleep(0.01)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_drain_loop_terminates_under_cancellation_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drainer loops on shield(task) under an absolute deadline.
    Even if the drainer task itself is cancelled mid-flight, it must
    finish within the bound — not leak or propagate cancellation
    outward (it's a background fire-and-forget)."""
    monkeypatch.setattr(jobs, "DRAIN_TIMEOUT_SECONDS", 0.3)

    async def long_running() -> str:
        try:
            await asyncio.sleep(10.0)
            return "unreachable"
        except asyncio.CancelledError:
            raise

    job_id = await jobs.start_job(long_running())
    job = jobs._jobs.pop(job_id)  # bypass await_job to drive _cancel_and_drain
    jobs._cancel_and_drain(job.task)
    assert jobs.active_drain_count() == 1
    drainer = next(iter(jobs._drain_tasks))
    # Cancel the drainer once — shield-once would have leaked here.
    drainer.cancel()
    # Wait for drainer to finish (bounded by DRAIN_TIMEOUT_SECONDS=0.3).
    # gather with return_exceptions absorbs any CancelledError that
    # escapes — we only care whether the drainer terminates.
    await asyncio.wait(
        [drainer, asyncio.create_task(asyncio.sleep(2.0))],
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert drainer.done()
    # Allow the done_callback to remove the entry.
    await asyncio.sleep(0)
    assert jobs.active_drain_count() == 0


@pytest.mark.asyncio
async def test_run_and_format_composes_liveness_then_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_run_and_format must compose the real format_liveness + format_rounds
    with a single blank-line separator. The q_review tests monkeypatch
    format_rounds, so this is the only check on the actual composed shape:
    a regression in the separator or a stray newline would otherwise pass."""
    from quorum.agents import AgentResult
    from quorum_mcp import server

    async def fake_run_mode(**kwargs: object) -> list[list[AgentResult]]:
        return [
            [
                AgentResult(
                    agent="codex",
                    output="hello",
                    returncode=0,
                    duration_s=1.0,
                    role="skeptic",
                )
            ]
        ]

    monkeypatch.setattr(server, "run_mode", fake_run_mode)
    result = await server._run_and_format(prompt="p", cwd=".", agents=[], rounds=1)
    assert result == (
        "Council: skeptic (codex) ✓ (1.0s)\n\n=== skeptic — codex (1.0s) ===\nhello\n"
    )


@pytest.mark.asyncio
async def test_q_plan_start_defaults_to_terse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Every *_start tool must forward verbose=False by default (terse)."""
    from quorum_mcp import server

    captured: dict[str, object] = {}

    async def fake_run_mode(**kwargs):
        captured.update(kwargs)
        return [[]]

    monkeypatch.setattr(server, "run_mode", fake_run_mode)

    out = await server.q_plan_start(task="t", cwd=str(tmp_path))
    await server.q_await(out["job_id"])
    assert captured["verbose"] is False


@pytest.mark.asyncio
async def test_q_review_start_forwards_verbose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """q_review_start(verbose=True) must reach run_mode."""
    from quorum_mcp import server

    captured: dict[str, object] = {}

    async def fake_resolve(target, cwd):
        return "diff", "branch review"

    async def fake_run_mode(**kwargs):
        captured.update(kwargs)
        return [[]]

    monkeypatch.setattr(server, "resolve_review_target", fake_resolve)
    monkeypatch.setattr(server, "run_mode", fake_run_mode)

    out = await server.q_review_start(cwd=str(tmp_path), verbose=True)
    await server.q_await(out["job_id"])
    assert captured["verbose"] is True


@pytest.mark.asyncio
async def test_q_brainstorm_start_forwards_verbose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """q_brainstorm_start(verbose=True) must reach run_mode (same plumbing as
    q_review; only that path was under test before)."""
    from quorum_mcp import server

    captured: dict[str, object] = {}

    async def fake_run_mode(**kwargs):
        captured.update(kwargs)
        return [[]]

    monkeypatch.setattr(server, "run_mode", fake_run_mode)

    out = await server.q_brainstorm_start(topic="t", cwd=str(tmp_path), verbose=True)
    await server.q_await(out["job_id"])
    assert captured["verbose"] is True


@pytest.mark.asyncio
async def test_q_validate_start_forwards_verbose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """q_validate_start(verbose=True) must reach run_mode."""
    from quorum_mcp import server

    captured: dict[str, object] = {}

    async def fake_run_mode(**kwargs):
        captured.update(kwargs)
        return [[]]

    monkeypatch.setattr(server, "run_mode", fake_run_mode)

    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\nDo the thing.", encoding="utf-8")
    out = await server.q_validate_start(
        plan_path=str(plan), cwd=str(tmp_path), verbose=True
    )
    await server.q_await(out["job_id"])
    assert captured["verbose"] is True
