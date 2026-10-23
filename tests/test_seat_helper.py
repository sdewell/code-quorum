import asyncio
import os
import plistlib
import subprocess
import time
from pathlib import Path

import pytest

from quorum import __version__
from quorum.agents import seat_helper as sh
from quorum.agents.base import AgentResult


def test_default_spool_is_shared_seat_helper_namespace() -> None:
    assert "code-quorum-seat-helper" in str(sh.default_spool_dir())


def test_spool_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    spool = tmp_path / "spool"
    spool.symlink_to(target, target_is_directory=True)

    try:
        sh._ensure_spool(spool)
    except OSError as exc:
        assert "symlink" in str(exc)
    else:
        raise AssertionError("helper spool must not follow a symlink")


def test_helper_pid_rejects_stale_heartbeat(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    sh._ensure_spool(spool)
    sh._atomic_write_json(
        spool / "helper.json",
        {"pid": os.getpid(), "heartbeat_at": 0},
    )

    assert sh.helper_pid(spool) is None


def test_helper_client_rejects_unversioned_protocol_before_queueing(
    monkeypatch, tmp_path: Path
) -> None:
    spool = tmp_path / "spool"
    sh._ensure_spool(spool)
    sh._atomic_write_json(
        spool / "helper.json",
        {
            "pid": os.getpid(),
            "heartbeat_at": time.time(),
        },
    )

    async def unexpected_poll(_delay: float) -> None:
        raise AssertionError("incompatible helper must be rejected before polling")

    monkeypatch.setattr(sh.asyncio, "sleep", unexpected_poll)

    result = asyncio.run(
        sh.run_via_helper(
            seat="claude",
            prompt="p",
            cwd=str(tmp_path),
            model="claude-opus-5",
            effort="medium",
            timeout_s=0.01,
            spool_dir=spool,
            poll_interval=0.01,
        )
    )

    assert result.returncode == sh.HELPER_UNAVAILABLE_RC
    assert "protocol version missing is incompatible" in result.error
    assert "install-seat-helper-launchagent" in result.error
    assert result.unavailable_reason == "seat helper"
    assert list((spool / "requests").iterdir()) == []


def test_helper_current_protocol_is_compatible(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    sh._ensure_spool(spool)
    sh._atomic_write_json(
        spool / "helper.json",
        {
            "pid": os.getpid(),
            "heartbeat_at": time.time(),
            "protocol_version": sh.HELPER_PROTOCOL_VERSION,
        },
    )

    assert sh.HELPER_PROTOCOL_VERSION == 2
    assert sh.helper_compatibility_error(spool) is None


def test_active_allowed_roots_reads_live_helper_state(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    allowed = tmp_path / "projects"
    allowed.mkdir()
    sh._ensure_spool(spool)
    sh._atomic_write_json(
        spool / "helper.json",
        {
            "pid": os.getpid(),
            "heartbeat_at": time.time(),
            "protocol_version": sh.HELPER_PROTOCOL_VERSION,
            "allowed_roots": [str(allowed)],
        },
    )

    assert sh.active_allowed_roots(spool) == (allowed.resolve(),)


def test_default_allowed_roots_includes_worktrunk_worktree_root(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(sh._ALLOWED_ROOTS_ENV, raising=False)

    assert sh.default_allowed_roots() == (
        (tmp_path / "Code").resolve(),
        (tmp_path / "src").resolve(),
        (tmp_path / ".codex" / "agent-worktrees").resolve(),
    )


def test_cwd_allowed_error_permits_worktrunk_worktree(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(sh._ALLOWED_ROOTS_ENV, raising=False)
    worktree = tmp_path / ".codex" / "agent-worktrees" / "sdewell" / "some-repo"
    worktree.mkdir(parents=True)

    assert sh.cwd_allowed_error(str(worktree), sh.default_allowed_roots()) is None


def test_cwd_allowed_error_rejects_codex_siblings_of_agent_worktrees(
    monkeypatch, tmp_path: Path
) -> None:
    # ~/.codex/agent-worktrees is allowed, but that must not widen to its
    # siblings under ~/.codex, or to ~/.codex itself -- those hold plugin
    # cache and session state, not worktrees.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(sh._ALLOWED_ROOTS_ENV, raising=False)
    plugins_cache = tmp_path / ".codex" / "plugins" / "cache"
    sessions = tmp_path / ".codex" / "sessions"
    codex_dir = tmp_path / ".codex"
    plugins_cache.mkdir(parents=True)
    sessions.mkdir(parents=True)

    roots = sh.default_allowed_roots()
    for candidate in (plugins_cache, sessions, codex_dir):
        assert sh.cwd_allowed_error(str(candidate), roots) is not None


def test_default_allowed_roots_env_override_replaces_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    only_root = tmp_path / "only-allowed"
    monkeypatch.setenv(sh._ALLOWED_ROOTS_ENV, str(only_root))

    assert sh.default_allowed_roots() == (only_root.resolve(),)


def _git(*args: str, cwd: Path) -> None:
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    )


def _repo_with_linked_worktree(main: Path, worktree: Path) -> None:
    """A real repository at `main` with a linked worktree at `worktree`."""
    main.mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=main)
    _git("commit", "-q", "--allow-empty", "-m", "root", cwd=main)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "-q", "-b", "task", str(worktree), cwd=main)


def test_cwd_allowed_error_permits_linked_worktree_of_repo_under_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Worktrees land wherever the user's tooling puts them (Worktrunk's
    # ~/.agent-worktrees, a sibling directory, ...). A worktree whose main
    # checkout is under an allowed root IS that repository, so it is allowed
    # without the roots naming the worktree location.
    root = tmp_path / "Code"
    worktree = tmp_path / "elsewhere" / "repo" / "task"
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "ambient.git"))
    _repo_with_linked_worktree(root / "repo", worktree)

    assert sh.cwd_allowed_error(str(worktree), (root.resolve(),)) is None


def test_cwd_allowed_error_rejects_linked_worktree_of_repo_outside_roots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Code"
    root.mkdir()
    worktree = tmp_path / "elsewhere" / "repo" / "task"
    _repo_with_linked_worktree(tmp_path / "outside" / "repo", worktree)

    error = sh.cwd_allowed_error(str(worktree), (root.resolve(),))

    assert error is not None
    assert "outside allowed roots" in error


def test_cwd_allowed_error_rejects_main_checkout_outside_roots(
    tmp_path: Path,
) -> None:
    # The worktree rule widens nothing for a repository's own checkout: a main
    # checkout outside the roots is refused exactly as a plain directory is.
    root = tmp_path / "Code"
    root.mkdir()
    main = tmp_path / "outside" / "repo"
    _repo_with_linked_worktree(main, tmp_path / "elsewhere" / "task")

    error = sh.cwd_allowed_error(str(main), (root.resolve(),))

    assert error is not None
    assert "outside allowed roots" in error
    assert "a worktree of" not in error


def test_cwd_allowed_error_ignores_ambient_git_dir(monkeypatch, tmp_path: Path) -> None:
    # GIT_DIR overrides `git -C <path>` discovery. Exported by hooks and
    # tooling, it would make every directory look like a worktree of that
    # repository, and the gate would admit anything.
    root = tmp_path / "Code"
    outside = tmp_path / "outside"
    outside.mkdir()
    _repo_with_linked_worktree(root / "repo", tmp_path / "elsewhere" / "task")
    monkeypatch.setenv("GIT_DIR", str(root / "repo" / ".git" / "worktrees" / "task"))

    error = sh.cwd_allowed_error(str(outside), (root.resolve(),))

    assert error is not None
    assert "outside allowed roots" in error


def test_cwd_allowed_error_judges_a_separate_git_dir_by_the_git_dir(
    tmp_path: Path,
) -> None:
    # `--separate-git-dir` puts the git directory anywhere, and git records
    # no path back to the checkout. The gate follows the git directory, which
    # is the only anchor git can vouch for, and the refusal names it.
    root = tmp_path / "Code"
    checkout_inside = root / "repo"
    checkout_inside.mkdir(parents=True)
    store_outside = tmp_path / "outside" / "store.git"
    store_outside.parent.mkdir()
    _git(
        "init",
        "-q",
        "-b",
        "main",
        "--separate-git-dir",
        str(store_outside),
        cwd=checkout_inside,
    )
    _git("commit", "-q", "--allow-empty", "-m", "root", cwd=checkout_inside)
    worktree = tmp_path / "elsewhere" / "task"
    worktree.parent.mkdir(parents=True)
    _git("worktree", "add", "-q", "-b", "task", str(worktree), cwd=checkout_inside)

    error = sh.cwd_allowed_error(str(worktree), (root.resolve(),))

    assert error is not None
    assert "outside allowed roots" in error
    assert f"a worktree of {store_outside.resolve()}" in error


def test_helper_v1_is_rejected_after_auth_check_operation_was_added(
    tmp_path: Path,
) -> None:
    spool = tmp_path / "spool"
    sh._ensure_spool(spool)
    sh._atomic_write_json(
        spool / "helper.json",
        {
            "pid": os.getpid(),
            "heartbeat_at": time.time(),
            "protocol_version": 1,
        },
    )

    error = sh.helper_compatibility_error(spool)

    assert error is not None
    assert "protocol version 1 is incompatible; expected 2" in error
    assert "install-seat-helper-launchagent" in error


def test_helper_sweeps_stale_orphan_results(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    sh._ensure_spool(spool)
    stale = spool / "results" / f"{'a' * 32}.json"
    fresh = spool / "results" / f"{'b' * 32}.json"
    stale.write_text("{}", encoding="utf-8")
    fresh.write_text("{}", encoding="utf-8")
    now = 10_000.0
    os.utime(stale, (now - sh.RESULT_TTL_S - 1, now - sh.RESULT_TTL_S - 1))
    os.utime(fresh, (now, now))

    sh._sweep_stale_results(spool, now=now)

    assert not stale.exists()
    assert fresh.exists()


def test_build_launchagent_runs_shared_helper(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\n", encoding="utf-8")
    spool = tmp_path / "spool"
    logs = tmp_path / "logs"

    data = plistlib.loads(
        sh.build_launchagent_plist(
            project_root=project,
            uv_binary=uv,
            seat_binaries=(tmp_path / "agy", tmp_path / "claude"),
            spool_dir=spool,
            allowed_roots=(tmp_path / "allowed",),
            log_dir=logs,
        )
    )

    assert data["Label"] == "com.code-quorum.seat-helper"
    assert "seat-helper" in data["ProgramArguments"]
    assert data["ProgramArguments"][-2:] == ["--spool-dir", str(spool)]
    assert data["EnvironmentVariables"]["CODE_QUORUM_HELPER_ALLOWED_ROOTS"] == str(
        tmp_path / "allowed"
    )


def test_build_launchagent_preserves_empty_allowed_roots(tmp_path: Path) -> None:
    data = plistlib.loads(
        sh.build_launchagent_plist(
            project_root=tmp_path,
            uv_binary=tmp_path / "uv",
            seat_binaries=(),
            allowed_roots=(),
        )
    )

    assert data["EnvironmentVariables"]["CODE_QUORUM_HELPER_ALLOWED_ROOTS"] == ""


def test_install_launchagent_rejects_replaceable_codex_plugin_cache(
    monkeypatch, tmp_path: Path
) -> None:
    cache = tmp_path / ".codex" / "plugins" / "cache"
    project = cache / "code-quorum" / "code-quorum" / "0.0.60"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setattr(sh, "CODEX_PLUGIN_CACHE", cache)

    with pytest.raises(ValueError, match="replaceable Codex plugin cache"):
        sh.install_launchagent(
            project_root=project,
            uv_binary=tmp_path / "uv",
            seat_binaries=(tmp_path / "agy",),
            plist_path=tmp_path / "helper.plist",
            log_dir=tmp_path / "logs",
            load=False,
        )


def test_helper_client_reports_missing_helper(tmp_path: Path) -> None:
    result = asyncio.run(
        sh.run_via_helper(
            seat="claude",
            prompt="p",
            cwd=str(tmp_path),
            model="sonnet",
            timeout_s=1.0,
            spool_dir=tmp_path / "spool",
            poll_interval=0.01,
        )
    )

    assert result.agent == "claude"
    assert result.returncode == sh.HELPER_UNAVAILABLE_RC
    assert "not running" in result.error
    assert result.unavailable_reason == "seat helper"


def test_helper_client_reports_request_spool_write_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sh, "helper_compatibility_error", lambda _spool: None)

    def fail_write(path, payload):
        raise OSError("read-only spool")

    monkeypatch.setattr(sh, "_atomic_write_json", fail_write)
    result = asyncio.run(
        sh.run_via_helper(
            seat="claude",
            prompt="p",
            cwd=str(tmp_path),
            model="sonnet",
            timeout_s=1.0,
            spool_dir=tmp_path / "spool",
            poll_interval=0.01,
        )
    )

    assert result.returncode == sh.HELPER_UNAVAILABLE_RC
    assert "could not queue" in result.error


def test_helper_round_trips_claude_request(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    async def _fake_run_request(payload, **_kwargs):
        captured.update(payload)
        return AgentResult(
            agent=payload["seat"],
            output=f"read:{payload['prompt']}:{payload['cwd']}",
            # An engine-swap stamp (the agy quota reflex) must survive the
            # spool round-trip, or run_council re-stamps the configured model.
            model="fallback-model",
            unavailable_reason="usage limit",
        )

    monkeypatch.setattr(sh, "_run_request", _fake_run_request)

    async def _run() -> AgentResult:
        spool = tmp_path / "spool"
        server = asyncio.create_task(
            sh.serve_helper(
                spool_dir=spool,
                allowed_roots=(tmp_path.resolve(),),
                poll_interval=0.01,
                once=True,
            )
        )
        for _ in range(100):
            if (spool / "helper.json").exists():
                break
            await asyncio.sleep(0.01)
        state = sh.helper_state(spool)
        assert state is not None
        assert state["protocol_version"] == sh.HELPER_PROTOCOL_VERSION
        assert state["code_quorum_version"] == __version__
        result = await sh.run_via_helper(
            seat="claude",
            prompt="marker",
            cwd=str(tmp_path),
            model="claude-opus-5",
            effort="medium",
            timeout_s=1.0,
            spool_dir=spool,
            poll_interval=0.01,
        )
        await server
        return result

    result = asyncio.run(_run())

    assert result.returncode == 0
    assert result.output == f"read:marker:{tmp_path}"
    assert result.model == "fallback-model"
    assert result.unavailable_reason == "usage limit"
    assert captured["model"] == "claude-opus-5"
    assert captured["effort"] == "medium"


def test_helper_client_round_trips_gemini_auth_check(monkeypatch, tmp_path) -> None:
    captured = {}

    async def _fake_run_request(payload, **_kwargs):
        captured.update(payload)
        return AgentResult(agent="gemini", output="agy authentication is ready")

    monkeypatch.setattr(sh, "_run_request", _fake_run_request)

    async def _run() -> AgentResult:
        spool = tmp_path / "spool"
        server = asyncio.create_task(
            sh.serve_helper(
                spool_dir=spool,
                allowed_roots=(tmp_path.resolve(),),
                poll_interval=0.01,
                once=True,
            )
        )
        for _ in range(100):
            if (spool / "helper.json").exists():
                break
            await asyncio.sleep(0.01)
        result = await sh.run_gemini_auth_check_via_helper(
            cwd=str(tmp_path),
            timeout_s=1.0,
            spool_dir=spool,
            poll_interval=0.01,
        )
        await server
        return result

    result = asyncio.run(_run())

    assert result.returncode == 0
    assert result.output == "agy authentication is ready"
    assert captured["operation"] == "auth-check"
    assert captured["seat"] == "gemini"
    assert "prompt" not in captured


def test_result_payload_rejects_unknown_unavailable_reason() -> None:
    for value in (None, 0, "future typo"):
        result = sh._result_from_payload(
            {"returncode": 1, "unavailable_reason": value}, "claude"
        )
        assert result.unavailable_reason == ""


def test_codex_helper_preserves_gemini_network_failure(monkeypatch, tmp_path) -> None:
    captured = {}

    class _Gemini:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def run(self, prompt: str, cwd: str) -> AgentResult:
            captured.update(prompt=prompt, cwd=cwd)
            return AgentResult(
                agent="gemini",
                output="",
                error="agy OAuth refresh failed after one retry",
                returncode=2,
                unavailable_reason="network",
            )

    monkeypatch.setattr(sh, "GeminiCliAgent", _Gemini)

    result = asyncio.run(
        sh._run_request(
            {
                "seat": "gemini",
                "cwd": str(tmp_path),
                "prompt": "review",
                "model": "Gemini 3.7 Flash (High)",
                "model_source": "recorded",
                "recorded_cli_version": "1.1.13",
                "timeout_s": 30,
            },
            allowed_roots=(tmp_path.resolve(),),
        )
    )

    assert result.returncode == 2
    assert result.unavailable_reason == "network"
    assert captured["prompt"] == "review"
    assert captured["cwd"] == str(tmp_path)
    assert captured["model_source"] == "recorded"


def test_helper_gemini_auth_check_uses_zero_quota_path(monkeypatch, tmp_path) -> None:
    captured = {}

    class _Gemini:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def check_auth(self, cwd: str) -> AgentResult:
            captured["cwd"] = cwd
            return AgentResult(agent="gemini", output="agy authentication is ready")

        async def run(self, prompt: str, cwd: str) -> AgentResult:
            raise AssertionError("auth-check must not spend a model turn")

    monkeypatch.setattr(sh, "GeminiCliAgent", _Gemini)

    result = asyncio.run(
        sh._run_request(
            {
                "operation": "auth-check",
                "seat": "gemini",
                "cwd": str(tmp_path),
                "model": "Gemini 3.7 Flash (High)",
            },
            allowed_roots=(tmp_path.resolve(),),
        )
    )

    assert result.returncode == 0
    assert result.output == "agy authentication is ready"
    assert captured["cwd"] == str(tmp_path)


def test_helper_rejects_unknown_seat(tmp_path: Path) -> None:
    result = asyncio.run(
        sh._run_request(
            {"seat": "codex", "cwd": str(tmp_path), "prompt": "read"},
            allowed_roots=(tmp_path.resolve(),),
        )
    )

    assert result.returncode == 1
    assert "unknown helper seat" in result.error
    assert result.unavailable_reason == "seat helper"


def test_helper_rejects_non_string_prompt(tmp_path: Path) -> None:
    result = asyncio.run(
        sh._run_request(
            {"seat": "claude", "cwd": str(tmp_path), "prompt": None},
            allowed_roots=(tmp_path.resolve(),),
        )
    )

    assert result.returncode == 1
    assert "non-empty cwd and prompt" in result.error
    assert result.unavailable_reason == "seat helper"


def test_helper_rejects_cwd_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    result = asyncio.run(
        sh._run_request(
            {"seat": "claude", "cwd": str(outside), "prompt": "read"},
            allowed_roots=(allowed.resolve(),),
        )
    )

    assert result.returncode == 1
    assert "outside allowed roots" in result.error
    assert result.unavailable_reason == "seat helper"


def test_codex_host_routes_subscription_seats_through_helper() -> None:
    from quorum.orchestration import select_agents

    chosen = {agent.name: agent for agent in select_agents(None, host="codex").agents}

    assert isinstance(chosen["claude"], sh.SeatHelperAgent)
    assert isinstance(chosen["gemini"], sh.SeatHelperAgent)
    assert chosen["claude"].model == "claude-opus-5"
    assert chosen["claude"].effort == "medium"
    assert chosen["claude"].timeout_s == 900.0


def test_codex_helper_preserves_recorded_gemini_routing_metadata() -> None:
    from quorum import model_config
    from quorum.orchestration import select_agents

    model_config.record_choice(
        "gemini",
        {
            "model": "Gemini 3.1 Pro (High)",
            "catalog_slug": "gemini-3.1-pro-high",
            "cli_version": "1.1.12",
        },
    )

    [seat] = select_agents(["gemini"], host="codex").agents

    assert isinstance(seat, sh.SeatHelperAgent)
    assert seat.model_source == "recorded"
    assert seat.recorded_cli_version == "1.1.12"


# --- #125 hardening: timeout cap, effort check, inflight cap, cancel, grace ---


def test_default_timeout_is_the_seat_print_timeout_and_fits_the_cap() -> None:
    """One budget per review whichever host runs it, and the helper's client
    deadline (timeout + gemini grace + margin) stays under the council cap."""
    from quorum.agents import gemini_cli as gc
    from quorum.council import AGENT_TIMEOUT_S

    assert sh._HELPER_DEFAULT_TIMEOUT_S == gc.GEMINI_CLI_PRINT_TIMEOUT_S
    deadline = (
        sh._HELPER_DEFAULT_TIMEOUT_S
        + sh._seat_grace_s("gemini")
        + sh._HELPER_CLIENT_MARGIN_S
    )
    assert deadline < AGENT_TIMEOUT_S


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (10_000, 900.0),
        ("abc", sh._HELPER_DEFAULT_TIMEOUT_S),
        (-1, sh._HELPER_DEFAULT_TIMEOUT_S),
        (30, 30.0),
        (None, sh._HELPER_DEFAULT_TIMEOUT_S),
    ],
)
def test_effective_timeout_caps_and_defaults(raw: object, expected: float) -> None:
    assert sh._effective_timeout_s(raw) == expected


def test_helper_timeout_cap_mirrors_council_agent_timeout() -> None:
    # seat_helper cannot import quorum.council (cycle via quorum.agents), so
    # the cap is a mirrored constant; this pins the two together.
    from quorum.council import AGENT_TIMEOUT_S

    assert sh.HELPER_MAX_TIMEOUT_S == AGENT_TIMEOUT_S


def test_helper_rejects_invalid_claude_effort_without_building_agent(
    monkeypatch, tmp_path: Path
) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("ClaudeAgent must not be constructed")

    monkeypatch.setattr(sh, "ClaudeAgent", _boom)
    result = asyncio.run(
        sh._run_request(
            {
                "operation": "run",
                "seat": "claude",
                "cwd": str(tmp_path),
                "prompt": "p",
                "effort": "turbo",
            },
            allowed_roots=(tmp_path.resolve(),),
        )
    )

    assert result.returncode == 1
    assert result.unavailable_reason == "seat helper"
    assert result.error == (
        "seat helper request effort 'turbo' is not one of high, low, max, medium, xhigh"
    )


def _write_request(spool: Path, request_id: str) -> Path:
    path = spool / "requests" / f"{request_id}.json"
    sh._atomic_write_json(
        path,
        {
            "id": request_id,
            "operation": "run",
            "seat": "claude",
            "cwd": "/",
            "prompt": "p",
        },
    )
    return path


def test_process_one_request_limiter_bounds_concurrent_runs(
    monkeypatch, tmp_path: Path
) -> None:
    spool = tmp_path / "spool"
    sh._ensure_spool(spool)
    state = {"now": 0, "peak": 0}

    async def _fake_run_request(payload, **_kwargs):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.05)
        state["now"] -= 1
        return AgentResult(agent="claude", output="ok")

    monkeypatch.setattr(sh, "_run_request", _fake_run_request)

    async def _drive(capacity: int, ids: list[str]) -> int:
        state.update(now=0, peak=0)
        limiter = asyncio.Semaphore(capacity)
        paths = [_write_request(spool, rid) for rid in ids]
        await asyncio.gather(
            *(
                sh._process_one_request(
                    p, spool, allowed_roots=(tmp_path,), limiter=limiter
                )
                for p in paths
            )
        )
        return state["peak"]

    assert asyncio.run(_drive(1, [c * 32 for c in "abc"])) == 1
    assert asyncio.run(_drive(3, [c * 32 for c in "def"])) == 3


def test_serve_helper_inflight_cap_is_eight() -> None:
    assert sh.HELPER_MAX_INFLIGHT == 8


def test_ensure_spool_creates_cancel_directory(tmp_path: Path) -> None:
    sh._ensure_spool(tmp_path / "spool")
    assert (tmp_path / "spool" / "cancel").is_dir()


def test_client_timeout_after_claim_leaves_cancel_marker(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sh, "helper_compatibility_error", lambda _spool: None)
    monkeypatch.setattr(sh, "_HELPER_CLIENT_MARGIN_S", 0.0)
    spool = tmp_path / "spool"

    async def _run() -> AgentResult:
        async def _claim_and_ignore() -> None:
            requests = spool / "requests"
            while True:
                for req in requests.glob("*.json"):
                    os.replace(req, spool / "inflight" / req.name)
                    return
                await asyncio.sleep(0.005)

        claimer = asyncio.create_task(_claim_and_ignore())
        result = await sh.run_via_helper(
            seat="claude",
            prompt="p",
            cwd=str(tmp_path),
            model="sonnet",
            timeout_s=0.2,
            spool_dir=spool,
            poll_interval=0.01,
        )
        await claimer
        return result

    result = asyncio.run(_run())

    assert result.returncode == sh.HELPER_TIMEOUT_RC
    markers = list((spool / "cancel").iterdir())
    assert len(markers) == 1
    inflight = list((spool / "inflight").glob("*.json"))
    assert [m.name for m in markers] == [inflight[0].stem]


def test_client_timeout_before_claim_removes_request_and_leaves_no_marker(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sh, "helper_compatibility_error", lambda _spool: None)
    monkeypatch.setattr(sh, "_HELPER_CLIENT_MARGIN_S", 0.0)
    spool = tmp_path / "spool"
    result = asyncio.run(
        sh.run_via_helper(
            seat="claude",
            prompt="p",
            cwd=str(tmp_path),
            model="sonnet",
            timeout_s=0.1,
            spool_dir=spool,
            poll_interval=0.01,
        )
    )

    assert result.returncode == sh.HELPER_TIMEOUT_RC
    assert list((spool / "requests").iterdir()) == []
    assert list((spool / "cancel").iterdir()) == []


def test_helper_cancel_marker_stops_run_and_writes_no_result(
    monkeypatch, tmp_path: Path
) -> None:
    spool = tmp_path / "spool"
    sh._ensure_spool(spool)
    request_id = "c" * 32
    path = _write_request(spool, request_id)

    async def _slow_run_request(payload, **_kwargs):
        await asyncio.sleep(5)
        return AgentResult(agent="claude", output="late")

    monkeypatch.setattr(sh, "_run_request", _slow_run_request)

    async def _drive() -> tuple[bool, float]:
        async def _mark() -> None:
            await asyncio.sleep(0.05)
            (spool / "cancel" / request_id).touch()

        marker = asyncio.create_task(_mark())
        start = time.monotonic()
        handled = await sh._process_one_request(
            path, spool, allowed_roots=(tmp_path,), poll_interval=0.01
        )
        await marker
        return handled, time.monotonic() - start

    handled, elapsed = asyncio.run(_drive())

    assert handled is False
    assert elapsed < 1.0
    assert list((spool / "results").iterdir()) == []
    assert list((spool / "cancel").iterdir()) == []
    assert list((spool / "inflight").iterdir()) == []


def test_sweep_removes_old_cancel_markers_and_keeps_fresh(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    sh._ensure_spool(spool)
    stale = spool / "cancel" / ("a" * 32)
    fresh = spool / "cancel" / ("b" * 32)
    stale.touch()
    fresh.touch()
    now = time.time()
    os.utime(stale, (now - sh.RESULT_TTL_S - 10, now - sh.RESULT_TTL_S - 10))

    sh._sweep_stale_results(spool, now=now)

    assert not stale.exists()
    assert fresh.exists()


def test_seat_grace_only_for_gemini() -> None:
    from quorum.agents.gemini_cli import (
        AGY_STARTUP_CRASH_WINDOW_S,
        AGY_STARTUP_RETRY_DELAY_S,
        GEMINI_CLI_TIMEOUT_GRACE_S,
    )

    assert sh._seat_grace_s("gemini") == (
        GEMINI_CLI_TIMEOUT_GRACE_S
        + AGY_STARTUP_CRASH_WINDOW_S
        + AGY_STARTUP_RETRY_DELAY_S
        + 5.0
    )
    assert sh._seat_grace_s("claude") == 0.0


def test_helper_wrapper_waits_for_gemini_agent_own_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    # Patch the grace constant small so the test stays fast; the helper's own
    # wait_for must still allow timeout_s + grace, not just timeout_s.
    monkeypatch.setattr(sh, "GEMINI_CLI_TIMEOUT_GRACE_S", 0.0)

    async def _agent_run(self, prompt, cwd):
        await asyncio.sleep(0.3)
        return AgentResult(
            agent="gemini", output="", error="agent-own-timeout", returncode=124
        )

    monkeypatch.setattr(sh.GeminiCliAgent, "run", _agent_run)
    result = asyncio.run(
        sh._run_request(
            {
                "operation": "run",
                "seat": "gemini",
                "cwd": str(tmp_path),
                "prompt": "p",
                "timeout_s": 0.2,
            },
            allowed_roots=(tmp_path.resolve(),),
        )
    )

    assert result.returncode == 124
    assert result.error == "agent-own-timeout"
