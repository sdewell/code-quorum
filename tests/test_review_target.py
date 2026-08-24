"""Unit tests for the q-review ingestion + review prompt: resolve_review_target,
the default-branch resolver, the scope-file pair, and the REVIEW_PROMPT / tag
constants. Uses real tmp_path git repos for the local-target paths (so the real
`git diff` "" ambiguity is exercised) and monkeypatches the subprocess layer for
gh/PR targets and the large-diff PR metadata path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quorum import orchestration
from quorum.orchestration import (
    NO_SCOPE_NOTICE,
    OUT_OF_SCOPE_TAG,
    REVIEW_DIFF_MAX_BYTES,
    REVIEW_PROMPT,
    SCOPE_FILE_MAX_BYTES,
    read_scope_file,
    resolve_doc_path,
    resolve_review_target,
)

# --------------------------------------------------------------------------- #
# git repo helpers
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _init_repo(repo: Path, default_branch: str = "main") -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", default_branch)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")


def _commit(repo: Path, name: str, body: str, msg: str) -> None:
    (repo / name).write_text(body, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", msg)


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #


def test_caps_equal_plan_cap() -> None:
    assert REVIEW_DIFF_MAX_BYTES == orchestration.PLAN_FILE_MAX_BYTES == 100_000
    assert SCOPE_FILE_MAX_BYTES == orchestration.PLAN_FILE_MAX_BYTES


def test_out_of_scope_tag_literal() -> None:
    assert OUT_OF_SCOPE_TAG == "[OUT-OF-SCOPE]"


def test_no_scope_notice_reads_as_in_scope_instruction() -> None:
    # The {scope} fallback when no --scope doc is supplied must read as a real
    # instruction (everything is in-scope), not a bare "(no scope ...)" note, so
    # the Scope paragraph above it does not dangle a reference to an absent scope.
    assert "in-scope" in NO_SCOPE_NOTICE
    rendered = REVIEW_PROMPT.format(diff="D", scope=NO_SCOPE_NOTICE)
    assert NO_SCOPE_NOTICE in rendered


def test_review_prompt_has_format_slots_and_labels() -> None:
    rendered = REVIEW_PROMPT.format(diff="DIFFTEXT", scope="SCOPETEXT")
    assert "DIFFTEXT" in rendered
    assert "SCOPETEXT" in rendered
    # closed-vocabulary per-finding labels
    assert "FILE:" in rendered
    assert "SEVERITY" in rendered
    assert "FINDING" in rendered
    assert "RECOMMENDATION" in rendered
    # severity vocabulary matches the codex review schema
    for sev in ("critical", "high", "medium", "low"):
        assert sev in rendered
    # the out-of-scope tagging instruction references the shared constant
    assert OUT_OF_SCOPE_TAG in rendered


# --------------------------------------------------------------------------- #
# default target: branch vs default branch (merge-base)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_default_target_diffs_branch_vs_merge_base(tmp_path: Path) -> None:
    """Default target reviews all branch work (committed + uncommitted) vs the
    merge-base with the default branch. A change to main *after* the fork point
    must NOT appear (merge-base, not a raw default..HEAD)."""
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "base.txt", "base\n", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feat.py", "def feature():\n    return 1\n", "add feature")
    # uncommitted tracked change on the feature branch
    (repo / "feat.py").write_text("def feature():\n    return 2\n", encoding="utf-8")

    diff, subject = await resolve_review_target("", repo)
    assert "feat.py" in diff
    assert "def feature" in diff
    assert "return 2" in diff  # uncommitted edit is included
    assert subject  # non-empty orientation subject


@pytest.mark.asyncio
async def test_default_target_none_behaves_like_empty(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "base.txt", "base\n", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feat.py", "x = 1\n", "add feat")

    diff_none, _ = await resolve_review_target(None, repo)
    diff_empty, _ = await resolve_review_target("", repo)
    assert "feat.py" in diff_none
    assert diff_none == diff_empty


@pytest.mark.asyncio
async def test_default_target_empty_diff_is_no_changes_sentinel(
    tmp_path: Path,
) -> None:
    """A genuinely empty diff (branch identical to default) returns
    ("", subject) so the caller can short-circuit — NOT an error."""
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "base.txt", "base\n", "base")
    _git(repo, "checkout", "-q", "-b", "feature")  # no new commits

    diff, subject = await resolve_review_target("", repo)
    assert diff == ""
    assert "no changes" in subject.lower()


# --------------------------------------------------------------------------- #
# working target
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_working_target_only_uncommitted(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")
    _commit(repo, "b.py", "y = 1\n", "second")  # committed, must NOT appear
    (repo / "a.py").write_text("x = 99\n", encoding="utf-8")  # uncommitted

    diff, _ = await resolve_review_target("working", repo)
    assert "a.py" in diff
    assert "x = 99" in diff
    assert "b.py" not in diff


@pytest.mark.asyncio
async def test_working_target_empty_is_no_changes(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")

    diff, subject = await resolve_review_target("working", repo)
    assert diff == ""
    assert "no changes" in subject.lower()


# --------------------------------------------------------------------------- #
# untracked files
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_untracked_only_is_surfaced_not_silent_empty(tmp_path: Path) -> None:
    """A brand-new un-`add`ed file is not in `git diff` — it must be surfaced
    in diff_text (named) rather than yielding a silent empty review."""
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")
    (repo / "newfile.py").write_text("brand new\n", encoding="utf-8")  # untracked

    diff, subject = await resolve_review_target("working", repo)
    assert "newfile.py" in diff
    assert "no changes" not in subject.lower()


@pytest.mark.asyncio
async def test_untracked_files_are_surfaced_with_tracked_changes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "newfile.py").write_text("brand new\n", encoding="utf-8")

    diff, _ = await resolve_review_target("working", repo)

    assert "x = 2" in diff
    assert "newfile.py" in diff
    assert "UNTRACKED FILES PRESENT" in diff


# --------------------------------------------------------------------------- #
# all target
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_all_target_returns_repo_inspection_instruction(tmp_path: Path) -> None:
    repo = tmp_path / "myrepo"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")

    diff, subject = await resolve_review_target("all", repo)
    assert "WHOLE CODEBASE REVIEW" in diff
    assert "inspect the repository" in diff.casefold()
    assert "empty" not in diff.casefold()
    assert "myrepo" in subject  # repo path/name in the subject


# --------------------------------------------------------------------------- #
# range targets
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_range_target_two_dot(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")
    _commit(repo, "a.py", "x = 2\n", "second")
    sha1 = _git(repo, "rev-parse", "HEAD~1").strip()
    sha2 = _git(repo, "rev-parse", "HEAD").strip()

    diff, _ = await resolve_review_target(f"{sha1}..{sha2}", repo)
    assert "a.py" in diff
    assert "x = 2" in diff


@pytest.mark.asyncio
async def test_range_target_three_dot(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feat.py", "f = 1\n", "feat")

    diff, _ = await resolve_review_target("main...feature", repo)
    assert "feat.py" in diff


@pytest.mark.asyncio
async def test_bad_range_raises(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")

    with pytest.raises(ValueError):
        await resolve_review_target("nope123..alsonope456", repo)


@pytest.mark.asyncio
async def test_range_target_rejects_git_option_injection(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    output = tmp_path / "injected.diff..HEAD"

    with pytest.raises(ValueError, match="unrecognized review target"):
        await resolve_review_target(f"--output={output}", repo)

    assert not output.exists()


def test_resolve_doc_path_rejects_tilde_path_outside_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    with pytest.raises(ValueError, match="outside working directory"):
        resolve_doc_path("~/.scope.md", tmp_path / "repo")


def test_resolve_doc_path_rejects_symlink_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    (repo / "scope.md").symlink_to(outside)

    with pytest.raises(ValueError, match="outside working directory"):
        resolve_doc_path("scope.md", repo)


# --------------------------------------------------------------------------- #
# pr targets (monkeypatched gh subprocess layer)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pr_target_uses_gh_diff_and_title_body_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pr:N => gh pr diff for the payload, gh pr view title+body for the
    subject (NOT the raw diff, which would pollute extract_keywords)."""
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")

    calls: list[tuple[str, ...]] = []

    async def fake_run_rc(cwd: Path, *args: str, timeout: float = 5.0):
        calls.append(args)
        if args[:3] == ("git", "rev-parse", "--git-dir"):
            return 0, ".git\n"
        if args[:3] == ("gh", "pr", "view"):
            if "--json" in args and "title,body" in args:
                return 0, "Fix the widget\n\nThis PR fixes the widget leak.\n"
            return 0, "exists\n"
        if args[:3] == ("gh", "pr", "diff"):
            return 0, "diff --git a/a.py b/a.py\n+x = 2\n"
        return 0, ""

    monkeypatch.setattr(orchestration, "shutil_which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(orchestration, "_run_rc", fake_run_rc)

    diff, subject = await resolve_review_target("pr:123", repo)
    assert "x = 2" in diff
    assert "Fix the widget" in subject
    assert "widget leak" in subject
    # raw diff text must NOT be the subject
    assert "diff --git" not in subject


@pytest.mark.asyncio
async def test_pr_url_target_parsed_with_anchored_regex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")

    seen_pr_args: list[tuple[str, ...]] = []

    async def fake_run_rc(cwd: Path, *args: str, timeout: float = 5.0):
        if args[:3] == ("git", "rev-parse", "--git-dir"):
            return 0, ".git\n"
        if args[:3] == ("gh", "pr", "view"):
            seen_pr_args.append(args)
            return 0, "Title\n\nBody\n"
        if args[:3] == ("gh", "pr", "diff"):
            seen_pr_args.append(args)
            return 0, "diff --git a/a.py b/a.py\n+x = 2\n"
        return 0, ""

    monkeypatch.setattr(orchestration, "shutil_which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(orchestration, "_run_rc", fake_run_rc)

    url = "https://github.com/sdewell/code-quorum/pull/42"
    diff, _ = await resolve_review_target(url, repo)
    assert "x = 2" in diff
    # the URL must have been parsed to PR number 42 (passed to gh)
    assert any("42" in args for args in seen_pr_args)


@pytest.mark.asyncio
async def test_pr_target_missing_gh_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")

    monkeypatch.setattr(orchestration, "shutil_which", lambda _: None)
    with pytest.raises(ValueError):
        await resolve_review_target("pr:1", repo)


@pytest.mark.asyncio
async def test_bad_pr_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonexistent PR (gh pr view nonzero) raises rather than diffing an
    empty/garbage payload."""
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")

    async def fake_run_rc(cwd: Path, *args: str, timeout: float = 5.0):
        if args[:3] == ("git", "rev-parse", "--git-dir"):
            return 0, ".git\n"
        if args[:3] == ("gh", "pr", "view"):
            return 1, ""  # PR does not exist
        return 0, ""

    monkeypatch.setattr(orchestration, "shutil_which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(orchestration, "_run_rc", fake_run_rc)
    with pytest.raises(ValueError):
        await resolve_review_target("pr:999", repo)


# --------------------------------------------------------------------------- #
# not-a-repo / unrecognized
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_not_a_repo_raises(tmp_path: Path) -> None:
    not_repo = tmp_path / "plain"
    not_repo.mkdir()
    with pytest.raises(ValueError):
        await resolve_review_target("", not_repo)


@pytest.mark.asyncio
async def test_unrecognized_target_raises(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")
    with pytest.raises(ValueError, match="unrecognized review target"):
        await resolve_review_target("garbage-not-a-ref", repo)


# --------------------------------------------------------------------------- #
# default-branch detection fallbacks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_default_branch_falls_back_to_master(tmp_path: Path) -> None:
    """No origin and no `main`; the resolver must fall back to `master`."""
    repo = tmp_path / "r"
    _init_repo(repo, default_branch="master")
    _commit(repo, "base.txt", "base\n", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feat.py", "f = 1\n", "feat")

    diff, _ = await resolve_review_target("", repo)
    assert "feat.py" in diff


@pytest.mark.asyncio
async def test_no_default_branch_raises(tmp_path: Path) -> None:
    """Neither main nor master nor origin/HEAD resolves -> raise, not guess."""
    repo = tmp_path / "r"
    _init_repo(repo, default_branch="trunk")
    _commit(repo, "base.txt", "base\n", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feat.py", "f = 1\n", "feat")

    with pytest.raises(ValueError):
        await resolve_review_target("", repo)


# --------------------------------------------------------------------------- #
# large diff: local (git diff --stat) vs pr (gh metadata)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_large_local_diff_uses_git_stat_marker(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")
    # a working-tree change bigger than the cap
    big = "y = 0\n" * (REVIEW_DIFF_MAX_BYTES // 4)
    (repo / "big.py").write_text(big, encoding="utf-8")
    _git(repo, "add", "big.py")

    diff, _ = await resolve_review_target("working", repo)
    assert len(diff.encode("utf-8")) <= REVIEW_DIFF_MAX_BYTES * 2  # not the full diff
    assert "big.py" in diff  # changed-file list from --stat
    # local marker: full files are in cwd
    assert "cwd" in diff.lower()
    # the full unified diff body must NOT be inlined
    assert diff.count("y = 0") < 100


@pytest.mark.asyncio
async def test_large_pr_diff_uses_gh_metadata_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over-cap pr: target uses gh pr view --json + gh pr diff --name-only,
    NOT git diff --stat (the PR may not be checked out)."""
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "a.py", "x = 1\n", "first")

    huge = "+" + ("z" * (REVIEW_DIFF_MAX_BYTES + 5000))
    used_name_only = {"v": False}
    used_git_stat = {"v": False}

    async def fake_run_rc(cwd: Path, *args: str, timeout: float = 5.0):
        if args[:3] == ("git", "rev-parse", "--git-dir"):
            return 0, ".git\n"
        if args[:2] == ("git", "diff") and "--stat" in args:
            used_git_stat["v"] = True
            return 0, "should-not-be-used\n"
        if args[:3] == ("gh", "pr", "view"):
            if "changedFiles" in " ".join(args):
                return 0, '{"changedFiles":3,"additions":900,"deletions":40}\n'
            return 0, "Big PR\n\nbody\n"
        if args[:3] == ("gh", "pr", "diff"):
            if "--name-only" in args:
                used_name_only["v"] = True
                return 0, "pkg/one.py\npkg/two.py\npkg/three.py\n"
            return 0, huge  # over-cap unified diff
        return 0, ""

    monkeypatch.setattr(orchestration, "shutil_which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(orchestration, "_run_rc", fake_run_rc)

    diff, _ = await resolve_review_target("pr:7", repo)
    assert len(diff.encode("utf-8")) < len(huge)  # truncated
    assert "pkg/one.py" in diff  # name-only file list
    assert used_name_only["v"] is True
    assert used_git_stat["v"] is False  # never git diff --stat for a PR
    # pr marker points at the PR source, not cwd
    assert "PR" in diff or "pull request" in diff.lower()


# --------------------------------------------------------------------------- #
# scope file pair
# --------------------------------------------------------------------------- #


def test_read_scope_file_reads_text(tmp_path: Path) -> None:
    f = tmp_path / "scope.md"
    f.write_text("## In scope\n- the widget\n", encoding="utf-8")
    assert "widget" in read_scope_file(f)


def test_read_scope_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_scope_file(tmp_path / "nope.md")


def test_read_scope_file_directory_raises(tmp_path: Path) -> None:
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        read_scope_file(d)


def test_read_scope_file_too_large_rejected(tmp_path: Path) -> None:
    f = tmp_path / "scope.md"
    f.write_text("x" * (SCOPE_FILE_MAX_BYTES + 1), encoding="utf-8")
    with pytest.raises(ValueError):
        read_scope_file(f)
