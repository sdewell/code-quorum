"""Context resolver: scans LEARNINGS files + git/gh state, produces a small
prompt-injectable block to prepend to agent prompts."""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS: frozenset[str] = frozenset(
    """
    the this that these those they them their your yours ours
    have has had are was were been being
    should would could will shall must
    make made want need
    with from into about
    what when where while
    just like very really quite
    more less many much some most
    than then also
    task todo issue code codebase file files implement implementation feature
    """.split()
)

_KEYWORD_RE = re.compile(r"\b[a-zA-Z][\w-]{3,}\b")
_FILE_RE = re.compile(r"[\w./-]+\.[a-zA-Z]{1,5}\b")
_H2_RE = re.compile(r"^##\s+(.+)$", flags=re.MULTILINE)

LEARNINGS_LIMIT = 5
COMMIT_LIMIT = 10
PR_LIMIT = 5
DESCRIPTION_LEN = 120


@dataclass
class ContextBlock:
    learnings: list[tuple[Path, str]] = field(default_factory=list)
    git_branch: str | None = None
    git_uncommitted: bool = False
    git_recent_commits: list[str] = field(default_factory=list)
    gh_prs: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not (
            self.learnings or self.git_branch or self.git_recent_commits or self.gh_prs
        ):
            return ""
        parts: list[str] = ["<project_context>"]
        if self.learnings:
            parts.append("Project context files (read what's relevant):")
            for path, desc in self.learnings:
                line = f"- {path.name}"
                if desc:
                    line += f": {desc}"
                parts.append(line)
        if self.git_branch or self.git_recent_commits:
            parts.append("")
            parts.append("Repo state:")
            if self.git_branch:
                tail = " (uncommitted changes)" if self.git_uncommitted else ""
                parts.append(f"- Branch: {self.git_branch}{tail}")
            if self.git_recent_commits:
                parts.append("- Recent commits touching mentioned files:")
                for c in self.git_recent_commits:
                    parts.append(f"  {c}")
        if self.gh_prs:
            parts.append("")
            parts.append("Open PRs matching keywords:")
            for pr in self.gh_prs:
                parts.append(f"- {pr}")
        parts.append("</project_context>")
        return "\n".join(parts)


def extract_keywords(prompt: str) -> set[str]:
    return {
        w
        for w in (m.group(0).lower() for m in _KEYWORD_RE.finditer(prompt))
        if w not in STOPWORDS
    }


def extract_file_mentions(prompt: str) -> list[str]:
    return list(dict.fromkeys(m.group(0) for m in _FILE_RE.finditer(prompt)))


def build_description(text: str) -> str:
    headers = _H2_RE.findall(text)
    if headers:
        joined = "; ".join(h.strip() for h in headers[:3])
        return joined[:DESCRIPTION_LEN]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:DESCRIPTION_LEN]
    return ""


def scan_learnings(cwd: Path, keywords: set[str]) -> list[tuple[Path, str]]:
    candidates: list[Path] = []
    for pattern in ("LEARNINGS*.md", "CLAUDE.md"):
        candidates.extend(sorted(cwd.glob(pattern)))
    matched: list[tuple[int, Path, str]] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        haystack = (path.name + " " + text[:2000]).lower()
        hits = sum(1 for kw in keywords if kw in haystack)
        always_include = path.name == "CLAUDE.md"
        if hits > 0 or always_include:
            score = hits + (100 if always_include else 0)
            matched.append((score, path, build_description(text)))
    matched.sort(key=lambda t: -t[0])
    return [(p, d) for _, p, d in matched[:LEARNINGS_LIMIT]]


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Kill `proc` (if still running) and await its exit with a small
    bounded wait. Both wait_for's internal timeout and an outer task
    cancellation can release `proc.communicate()` without killing the
    child — the subprocess keeps running, holding its pipes.

    asyncio.shield only blocks ONE cancellation per await. If the caller
    is cancelled twice in rapid succession (e.g. during event-loop
    shutdown), a single shielded await would surface the second cancel
    and leave the reap task dangling. Loop until the reap completes
    (its own 2s wait_for ceiling guarantees termination), then re-raise
    any pending cancellation after reap is done."""
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    async def _reap_inner() -> None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except (TimeoutError, ProcessLookupError):
            pass

    reap_task = asyncio.create_task(_reap_inner())
    pending_cancel = False
    while not reap_task.done():
        try:
            await asyncio.shield(reap_task)
        except asyncio.CancelledError:
            pending_cancel = True
    if pending_cancel:
        raise asyncio.CancelledError()


async def _run_rc(cwd: Path, *args: str, timeout: float = 15.0) -> tuple[int, str]:
    """Return-code-aware subprocess runner. `_run` (below) collapses
    missing-tool / timeout / nonzero-exit / empty-output ALL to "" — fine for
    best-effort orientation, fatal for a review diff (a transient git failure
    would read as an empty diff). This variant surfaces the return code so the
    caller distinguishes a genuinely empty diff (rc 0, "") from a failure
    (rc != 0). A missing binary or timeout returns rc 127."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        return 127, ""
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        await _kill_and_reap(proc)
        return 127, ""
    except asyncio.CancelledError:
        await _kill_and_reap(proc)
        raise
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


async def _run(cwd: Path, *args: str, timeout: float = 5.0) -> str:
    rc, out = await _run_rc(cwd, *args, timeout=timeout)
    return out if rc == 0 else ""


async def gather_git(
    cwd: Path, file_mentions: list[str]
) -> tuple[str | None, bool, list[str]]:
    if shutil.which("git") is None:
        return None, False, []
    if not (await _run(cwd, "git", "rev-parse", "--git-dir")).strip():
        return None, False, []
    branch = (await _run(cwd, "git", "rev-parse", "--abbrev-ref", "HEAD")).strip()
    status = (await _run(cwd, "git", "status", "--short")).strip()
    uncommitted = bool(status)
    commits: list[str] = []
    for f in file_mentions[:3]:
        log = await _run(cwd, "git", "log", "--oneline", "-5", "--", f)
        for line in log.strip().splitlines():
            if line:
                commits.append(line)
        if len(commits) >= COMMIT_LIMIT:
            break
    return branch or None, uncommitted, commits[:COMMIT_LIMIT]


async def gather_gh_prs(cwd: Path, keywords: set[str]) -> list[str]:
    if shutil.which("gh") is None or not keywords:
        return []
    query = " ".join(sorted(keywords)[:4])
    if not query:
        return []
    out = await _run(
        cwd,
        "gh",
        "pr",
        "list",
        "--state",
        "open",
        "--search",
        query,
        "--limit",
        str(PR_LIMIT),
        "--json",
        "number,title",
        "--jq",
        '.[] | "#\\(.number) \\(.title)"',
        timeout=8.0,
    )
    return [line for line in out.strip().splitlines() if line][:PR_LIMIT]


async def resolve(
    prompt: str,
    cwd: Path,
    *,
    no_context: bool = False,
    skip_gh: bool = False,
) -> ContextBlock:
    if no_context:
        return ContextBlock()
    keywords = extract_keywords(prompt)
    file_mentions = extract_file_mentions(prompt)
    learnings = scan_learnings(cwd, keywords)
    git_task = gather_git(cwd, file_mentions)
    gh_task = (
        gather_gh_prs(cwd, keywords) if not skip_gh else asyncio.sleep(0, result=[])
    )
    (branch, uncommitted, commits), prs = await asyncio.gather(git_task, gh_task)
    return ContextBlock(
        learnings=learnings,
        git_branch=branch,
        git_uncommitted=uncommitted,
        git_recent_commits=commits,
        gh_prs=prs,
    )
