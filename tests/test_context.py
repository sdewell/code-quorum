import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from quorum.context import (
    ContextBlock,
    _run,
    _run_rc,
    build_description,
    extract_file_mentions,
    extract_keywords,
    scan_learnings,
)


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


def test_extract_keywords_drops_stopwords_and_short_words() -> None:
    kws = extract_keywords("Add a --version flag to the CLI for typer apps")
    assert "version" in kws
    assert "typer" in kws
    assert "apps" in kws
    assert "the" not in kws
    assert "for" not in kws


def test_extract_keywords_lowercases() -> None:
    kws = extract_keywords("Refactor CodexAgent and Typer Callback")
    assert "codexagent" in kws
    assert "typer" in kws
    assert "callback" in kws


def test_extract_file_mentions_finds_paths() -> None:
    text = "Update quorum/cli.py and tests/test_context.py — also pyproject.toml"
    mentions = extract_file_mentions(text)
    assert "quorum/cli.py" in mentions
    assert "tests/test_context.py" in mentions
    assert "pyproject.toml" in mentions


def test_extract_file_mentions_dedupes() -> None:
    mentions = extract_file_mentions("a.py and a.py again, plus a.py")
    assert mentions == ["a.py"]


def test_build_description_uses_h2_headers(tmp_path: Path) -> None:
    f = tmp_path / "LEARNINGS.md"
    f.write_text(
        "# Top\n\n## First section\n\nbody\n\n## Second section\n\nmore\n",
        encoding="utf-8",
    )
    desc = build_description(f.read_text(encoding="utf-8"))
    assert "First section" in desc
    assert "Second section" in desc


def test_build_description_falls_back_to_first_paragraph(tmp_path: Path) -> None:
    f = tmp_path / "CLAUDE.md"
    f.write_text(
        "# Title\n\nThis project does X and Y.\n\nMore content.\n",
        encoding="utf-8",
    )
    desc = build_description(f.read_text(encoding="utf-8"))
    assert desc.startswith("This project does X")


def test_scan_learnings_matches_keyword(tmp_path: Path) -> None:
    (tmp_path / "LEARNINGS.md").write_text(
        "# L\n\n## Typer auto-promotion\n\nbody\n", encoding="utf-8"
    )
    (tmp_path / "LEARNINGS-DB.md").write_text(
        "# L\n\n## Postgres connection pooling\n\nbody\n", encoding="utf-8"
    )
    matched = scan_learnings(tmp_path, {"typer"})
    names = [p.name for p, _ in matched]
    assert "LEARNINGS.md" in names
    assert "LEARNINGS-DB.md" not in names


def test_scan_learnings_always_includes_claude_md(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text(
        "Project conventions live here.", encoding="utf-8"
    )
    matched = scan_learnings(tmp_path, {"unrelated"})
    assert any(p.name == "CLAUDE.md" for p, _ in matched)


def test_scan_learnings_caps_at_limit(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"LEARNINGS-{i}.md").write_text(
            f"# T\n\n## section keyword{i}\n", encoding="utf-8"
        )
    matched = scan_learnings(
        tmp_path,
        {f"keyword{i}" for i in range(10)},
    )
    assert len(matched) == 5


def test_context_block_render_empty() -> None:
    assert ContextBlock().render() == ""


def test_context_block_render_full() -> None:
    block = ContextBlock(
        learnings=[(Path("LEARNINGS.md"), "Typer gotcha; ruff fix")],
        git_branch="main",
        git_uncommitted=True,
        git_recent_commits=["abc1234 Fix the thing"],
        gh_prs=["#5 Add --version flag"],
    )
    rendered = block.render()
    assert "<project_context>" in rendered
    assert "</project_context>" in rendered
    assert "LEARNINGS.md" in rendered
    assert "Typer gotcha" in rendered
    assert "main" in rendered
    assert "uncommitted" in rendered
    assert "abc1234" in rendered
    assert "#5" in rendered


@pytest.mark.asyncio
async def test_run_kills_subprocess_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When wait_for fires, communicate() releases its pipes but the
    underlying subprocess keeps running unless _run explicitly kills it.
    Spawn `sleep 60` with a 100ms timeout, then verify the pid dies."""
    captured: dict[str, int] = {}
    real_spawn = asyncio.create_subprocess_exec

    async def capture(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_spawn(*args, **kwargs)
        captured["pid"] = proc.pid
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    result = await _run(tmp_path, "sleep", "60", timeout=0.1)
    assert result == ""
    assert "pid" in captured
    assert await _wait_until_dead(captured["pid"], timeout_s=3.0)


@pytest.mark.asyncio
async def test_run_kills_subprocess_on_outer_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the enclosing task (e.g. a quorum_mcp job) is cancelled,
    _run must kill its subprocess before re-raising — otherwise the
    git/gh helper orphans on every TTL or caller-disconnect."""
    captured: dict[str, int] = {}
    real_spawn = asyncio.create_subprocess_exec

    async def capture(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_spawn(*args, **kwargs)
        captured["pid"] = proc.pid
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    runner = asyncio.create_task(_run(tmp_path, "sleep", "60", timeout=30.0))
    for _ in range(100):
        if "pid" in captured:
            break
        await asyncio.sleep(0.01)
    assert "pid" in captured, "subprocess never spawned"
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    assert await _wait_until_dead(captured["pid"], timeout_s=3.0)


@pytest.mark.asyncio
async def test_run_kills_subprocess_under_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stress the _kill_and_reap loop: cancel the outer task multiple
    times during cleanup (simulates event-loop-shutdown pressure).
    The subprocess must still die and the task must end with
    CancelledError without busy-looping forever."""
    captured: dict[str, int] = {}
    real_spawn = asyncio.create_subprocess_exec

    async def capture(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_spawn(*args, **kwargs)
        captured["pid"] = proc.pid
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    runner = asyncio.create_task(_run(tmp_path, "sleep", "60", timeout=30.0))
    for _ in range(100):
        if "pid" in captured:
            break
        await asyncio.sleep(0.01)
    assert "pid" in captured, "subprocess never spawned"
    # Hammer cancel during the cleanup phase.
    for _ in range(5):
        runner.cancel()
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(runner, timeout=4.0)
    assert await _wait_until_dead(captured["pid"], timeout_s=3.0)


# _run_rc is the return-code-aware core `_run` wraps, also used directly by
# the q-review diff resolver (orchestration.py). It must reap its subprocess
# on both timeout and outer cancellation exactly as _run does — a hung
# `git diff` on a stalled mount would otherwise orphan and (on timeout)
# block the review flow indefinitely.


@pytest.mark.asyncio
async def test_run_rc_kills_subprocess_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, int] = {}
    real_spawn = asyncio.create_subprocess_exec

    async def capture(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_spawn(*args, **kwargs)
        captured["pid"] = proc.pid
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    rc, out = await _run_rc(tmp_path, "sleep", "60", timeout=0.1)
    assert (rc, out) == (127, "")
    assert "pid" in captured
    assert await _wait_until_dead(captured["pid"], timeout_s=3.0)


@pytest.mark.asyncio
async def test_run_rc_kills_subprocess_on_outer_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, int] = {}
    real_spawn = asyncio.create_subprocess_exec

    async def capture(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_spawn(*args, **kwargs)
        captured["pid"] = proc.pid
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    runner = asyncio.create_task(_run_rc(tmp_path, "sleep", "60", timeout=30.0))
    for _ in range(100):
        if "pid" in captured:
            break
        await asyncio.sleep(0.01)
    assert "pid" in captured, "subprocess never spawned"
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    assert await _wait_until_dead(captured["pid"], timeout_s=3.0)
