from __future__ import annotations

import asyncio
import contextlib
import json
import os
import plistlib
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import __version__
from .base import (
    UNAVAILABLE_REASONS,
    Agent,
    AgentResult,
    _atomic_write_json,
    _ensure_private_dir,
)
from .claude import DEFAULT_EFFORT as CLAUDE_DEFAULT_EFFORT
from .claude import DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL
from .claude import ClaudeAgent
from .gemini_cli import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from .gemini_cli import GeminiCliAgent

HELPER_UNAVAILABLE_RC = 5
HELPER_TIMEOUT_RC = 124
HELPER_LABEL = "com.code-quorum.seat-helper"
HELPER_PROTOCOL_VERSION = 2
CODEX_PLUGIN_CACHE = Path("~/.codex/plugins/cache").expanduser()
RESULT_TTL_S = 300.0
_ROOT_NAME = f"code-quorum-seat-helper-{os.getuid()}"
_ALLOWED_ROOTS_ENV = "CODE_QUORUM_HELPER_ALLOWED_ROOTS"
_REQUEST_ID_CHARS = frozenset("0123456789abcdef")
_SUPPORTED_SEATS = frozenset({"claude", "gemini"})
_HEARTBEAT_INTERVAL_S = 1.0
_HEARTBEAT_MAX_AGE_S = 5.0


def default_spool_dir() -> Path:
    return Path("/tmp") / _ROOT_NAME


def default_launchagent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{HELPER_LABEL}.plist"


def default_log_dir() -> Path:
    return Path.home() / "Library" / "Logs"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_project_root(cwd: Path | None = None) -> Path:
    candidate = (cwd or Path.cwd()).resolve()
    if (candidate / "pyproject.toml").is_file() and (candidate / "quorum").is_dir():
        return candidate
    return _package_root()


def default_allowed_roots() -> tuple[Path, ...]:
    raw = os.environ.get(_ALLOWED_ROOTS_ENV, "").strip()
    if raw:
        return tuple(
            Path(part).expanduser().resolve()
            for part in raw.split(os.pathsep)
            if part.strip()
        )
    home = Path.home()
    return (
        (home / "Code").resolve(),
        (home / "src").resolve(),
        (home / ".codex" / "agent-worktrees").resolve(),
    )


def legacy_default_allowed_roots() -> tuple[Path, ...]:
    """The two-root default set before ~/.codex/agent-worktrees was added.

    A helper installed under those old defaults reports live roots of
    exactly this pair; ``update-codex`` uses this to detect "still on
    defaults" and avoid freezing it in place -- see
    ``codex_update.perform_codex_update``.
    """
    home = Path.home()
    return ((home / "Code").resolve(), (home / "src").resolve())


def _allowed_roots_env(roots: tuple[Path, ...]) -> str:
    return os.pathsep.join(str(root) for root in roots)


def _path_env(uv_binary: Path, seat_binaries: tuple[Path, ...]) -> str:
    candidates = [str(uv_binary.parent), *(str(path.parent) for path in seat_binaries)]
    candidates.extend(os.environ.get("PATH", "").split(os.pathsep))
    return os.pathsep.join(dict.fromkeys(part for part in candidates if part))


def cwd_allowed_error(cwd: str, roots: tuple[Path, ...]) -> str | None:
    try:
        path = Path(cwd).expanduser().resolve()
    except OSError as exc:
        return f"requested cwd cannot be resolved: {exc}"
    if not path.is_dir():
        return f"requested cwd is not a directory: {path}"
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return None
    allowed = ", ".join(str(root) for root in roots)
    return (
        f"requested cwd is outside allowed roots ({allowed}): {path}. Set "
        f"{_ALLOWED_ROOTS_ENV} before starting the host; on Codex, reinstall "
        "the seat helper with --allowed-root as well."
    )


def _ensure_spool(spool: Path) -> None:
    _ensure_private_dir(spool, parents=True)
    for name in ("requests", "inflight", "results"):
        _ensure_private_dir(spool / name, parents=False)


def _sweep_stale_results(spool: Path, *, now: float | None = None) -> None:
    cutoff = (time.time() if now is None else now) - RESULT_TTL_S
    for result in (spool / "results").glob("*.json"):
        try:
            if result.stat(follow_symlinks=False).st_mtime < cutoff:
                result.unlink()
        except OSError:
            continue


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _state_path(spool: Path) -> Path:
    return spool / "helper.json"


def helper_state(spool_dir: Path | None = None) -> dict[str, Any] | None:
    try:
        return _read_json(_state_path(spool_dir or default_spool_dir()))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None


def active_allowed_roots(spool_dir: Path | None = None) -> tuple[Path, ...]:
    """Use a live helper's installed roots, else the environment/defaults."""
    state = helper_state(spool_dir)
    if (
        _helper_pid_from_state(state) is not None
        and state is not None
        and state.get("protocol_version") == HELPER_PROTOCOL_VERSION
    ):
        raw = state.get("allowed_roots")
        if isinstance(raw, list):
            roots = tuple(
                Path(value).expanduser().resolve()
                for value in raw
                if isinstance(value, str) and value.strip()
            )
            if roots:
                return roots
    return default_allowed_roots()


def _helper_pid_from_state(state: dict[str, Any] | None) -> int | None:
    if state is None:
        return None
    try:
        pid = int(state["pid"])
        heartbeat_at = float(state["heartbeat_at"])
        heartbeat_age = time.time() - heartbeat_at
        if not -_HEARTBEAT_MAX_AGE_S <= heartbeat_age <= _HEARTBEAT_MAX_AGE_S:
            return None
        os.kill(pid, 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError):
        return None
    except PermissionError:
        pass
    return pid


def helper_pid(spool_dir: Path | None = None) -> int | None:
    return _helper_pid_from_state(helper_state(spool_dir))


def helper_compatibility_error(spool_dir: Path | None = None) -> str | None:
    spool = spool_dir or default_spool_dir()
    state = helper_state(spool)
    if _helper_pid_from_state(state) is None:
        return (
            f"seat helper is not running at {spool}. Start it outside the "
            "Codex sandbox with `uv run quorum seat-helper`."
        )
    version = state.get("protocol_version") if state is not None else None
    if version != HELPER_PROTOCOL_VERSION:
        observed = "missing" if version is None else str(version)
        return (
            f"seat helper protocol version {observed} is incompatible; expected "
            f"{HELPER_PROTOCOL_VERSION}. Reinstall and restart it outside the Codex "
            "sandbox with `uv run quorum install-seat-helper-launchagent`."
        )
    return None


def build_launchagent_plist(
    *,
    project_root: Path,
    uv_binary: Path,
    seat_binaries: tuple[Path, ...],
    spool_dir: Path | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
    log_dir: Path | None = None,
) -> bytes:
    spool = spool_dir or default_spool_dir()
    roots = default_allowed_roots() if allowed_roots is None else allowed_roots
    logs = log_dir or default_log_dir()
    payload = {
        "Label": HELPER_LABEL,
        "ProgramArguments": [
            str(uv_binary),
            "run",
            "--directory",
            str(project_root),
            "quorum",
            "seat-helper",
            "--spool-dir",
            str(spool),
        ],
        "WorkingDirectory": str(project_root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": {
            "UV_CACHE_DIR": str(project_root / ".uv-cache"),
            _ALLOWED_ROOTS_ENV: _allowed_roots_env(roots),
            "PATH": _path_env(uv_binary, seat_binaries),
        },
        "StandardOutPath": str(logs / "code-quorum-seat-helper.log"),
        "StandardErrorPath": str(logs / "code-quorum-seat-helper.err.log"),
    }
    return plistlib.dumps(payload, sort_keys=False)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _run_launchctl(argv: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, check=check, text=True, timeout=30)


def install_launchagent(
    *,
    project_root: Path | None = None,
    uv_binary: Path | None = None,
    seat_binaries: tuple[Path, ...] | None = None,
    spool_dir: Path | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
    plist_path: Path | None = None,
    log_dir: Path | None = None,
    load: bool = True,
) -> Path:
    root = (project_root or default_project_root()).resolve()
    try:
        root.relative_to(CODEX_PLUGIN_CACHE.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            "project root is inside the replaceable Codex plugin cache: "
            f"{root}. Use --project-root with a stable code-quorum checkout."
        )
    uv = uv_binary.resolve() if uv_binary else _required_binary("uv")
    binaries = seat_binaries or (_required_binary("agy"), _required_binary("claude"))
    if not (root / "pyproject.toml").is_file():
        raise ValueError(f"project root missing pyproject.toml: {root}")
    plist = plist_path or default_launchagent_path()
    logs = log_dir or default_log_dir()
    logs.mkdir(mode=0o755, parents=True, exist_ok=True)
    _atomic_write_bytes(
        plist,
        build_launchagent_plist(
            project_root=root,
            uv_binary=uv,
            seat_binaries=tuple(path.resolve() for path in binaries),
            spool_dir=spool_dir,
            allowed_roots=allowed_roots,
            log_dir=logs,
        ),
    )
    if load:
        domain = f"gui/{os.getuid()}"
        _run_launchctl(["launchctl", "bootout", domain, str(plist)], check=False)
        _run_launchctl(["launchctl", "bootstrap", domain, str(plist)], check=True)
        _run_launchctl(
            ["launchctl", "kickstart", "-k", f"{domain}/{HELPER_LABEL}"],
            check=True,
        )
    return plist


def _required_binary(name: str) -> Path:
    found = shutil.which(name)
    if found is None:
        raise ValueError(f"{name} not found on PATH; cannot install seat helper")
    return Path(found).resolve()


def launchagent_installed(plist_path: Path | None = None) -> bool:
    return (plist_path or default_launchagent_path()).is_file()


def launchagent_status() -> str:
    proc = _run_launchctl(
        ["launchctl", "print", f"gui/{os.getuid()}/{HELPER_LABEL}"], check=False
    )
    return "loaded" if proc.returncode == 0 else "not loaded"


def _result_from_payload(payload: dict[str, Any], seat: str) -> AgentResult:
    raw_reason = payload.get("unavailable_reason")
    unavailable_reason = (
        raw_reason
        if isinstance(raw_reason, str) and raw_reason in UNAVAILABLE_REASONS
        else ""
    )
    return AgentResult(
        agent=str(payload.get("agent", seat)),
        output=str(payload.get("output", "")),
        error=str(payload.get("error", "")),
        returncode=int(payload.get("returncode", 1)),
        duration_s=float(payload.get("duration_s", 0.0)),
        role=str(payload.get("role", "neutral")),
        # A seat that swapped engines mid-run (the agy quota reflex) stamps
        # the model that answered; dropping it here would let run_council
        # re-stamp the configured one across the IPC boundary.
        model=str(payload.get("model", "")),
        unavailable_reason=unavailable_reason,
    )


class SeatHelperAgent(Agent):
    def __init__(
        self,
        seat: str,
        model: str,
        effort: str | None = None,
        model_source: str = "shipped",
        recorded_cli_version: str | None = None,
        timeout_s: float = 600.0,
        spool_dir: Path | None = None,
        poll_interval: float = 0.2,
    ) -> None:
        if seat not in _SUPPORTED_SEATS:
            raise ValueError(f"unknown helper seat {seat!r}")
        self.name = seat
        self.model = model
        self.effort = effort
        self.model_source = model_source
        self.recorded_cli_version = recorded_cli_version
        self.timeout_s = timeout_s
        self.spool_dir = spool_dir
        self.poll_interval = poll_interval

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        return await self._submit(
            {
                "operation": "run",
                "seat": self.name,
                "cwd": cwd,
                "prompt": prompt,
                "model": self.model,
                "effort": self.effort,
                "model_source": self.model_source,
                "recorded_cli_version": self.recorded_cli_version,
                "timeout_s": self.timeout_s,
            }
        )

    async def check_auth(self, cwd: str) -> AgentResult:
        if self.name != "gemini":
            raise ValueError("helper auth-check supports only the gemini seat")
        return await self._submit(
            {
                "operation": "auth-check",
                "seat": self.name,
                "cwd": cwd,
                "model": self.model,
                "timeout_s": self.timeout_s,
            }
        )

    async def _submit(self, payload: dict[str, Any]) -> AgentResult:
        start = time.monotonic()
        spool = self.spool_dir or default_spool_dir()
        try:
            _ensure_spool(spool)
        except OSError as exc:
            return AgentResult(
                agent=self.name,
                output="",
                error=f"seat helper spool unavailable at {spool}: {exc}",
                returncode=HELPER_UNAVAILABLE_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="seat helper",
            )
        helper_error = helper_compatibility_error(spool)
        if helper_error is not None:
            return AgentResult(
                agent=self.name,
                output="",
                error=helper_error,
                returncode=HELPER_UNAVAILABLE_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="seat helper",
            )

        request_id = uuid.uuid4().hex
        request_path = spool / "requests" / f"{request_id}.json"
        result_path = spool / "results" / f"{request_id}.json"
        try:
            _atomic_write_json(
                request_path,
                {
                    "id": request_id,
                    "created_at": time.time(),
                    **payload,
                },
            )
        except OSError as exc:
            return AgentResult(
                agent=self.name,
                output="",
                error=f"seat helper could not queue request at {request_path}: {exc}",
                returncode=HELPER_UNAVAILABLE_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="seat helper",
            )
        deadline = start + self.timeout_s + 30.0
        try:
            while time.monotonic() < deadline:
                if result_path.exists():
                    try:
                        result = _result_from_payload(
                            _read_json(result_path), self.name
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        return AgentResult(
                            agent=self.name,
                            output="",
                            error=f"seat helper result was unreadable: {exc}",
                            returncode=1,
                            duration_s=time.monotonic() - start,
                        )
                    result.duration_s = time.monotonic() - start
                    return result
                await asyncio.sleep(self.poll_interval)
        finally:
            with contextlib.suppress(FileNotFoundError):
                result_path.unlink()
            with contextlib.suppress(FileNotFoundError):
                request_path.unlink()
        return AgentResult(
            agent=self.name,
            output="",
            error=f"seat helper timed out waiting for result {request_id}",
            returncode=HELPER_TIMEOUT_RC,
            duration_s=time.monotonic() - start,
            unavailable_reason="timeout",
        )


async def run_via_helper(
    *,
    seat: str,
    prompt: str,
    cwd: str,
    model: str,
    effort: str | None = None,
    model_source: str = "shipped",
    recorded_cli_version: str | None = None,
    timeout_s: float,
    spool_dir: Path | None = None,
    poll_interval: float = 0.2,
) -> AgentResult:
    """Thin delegate kept for direct callers (tests exercise the polling
    knobs -- poll_interval, spool_dir -- without going through
    SeatHelperAgent's Agent-ABC-shaped run(prompt, cwd) interface); the real
    implementation lives on SeatHelperAgent.run."""
    agent = SeatHelperAgent(
        seat=seat,
        model=model,
        effort=effort,
        model_source=model_source,
        recorded_cli_version=recorded_cli_version,
        timeout_s=timeout_s,
        spool_dir=spool_dir,
        poll_interval=poll_interval,
    )
    return await agent.run(prompt, cwd)


async def run_gemini_auth_check_via_helper(
    *,
    cwd: str,
    timeout_s: float = 30.0,
    spool_dir: Path | None = None,
    poll_interval: float = 0.2,
) -> AgentResult:
    agent = SeatHelperAgent(
        seat="gemini",
        model=GEMINI_DEFAULT_MODEL,
        timeout_s=timeout_s,
        spool_dir=spool_dir,
        poll_interval=poll_interval,
    )
    return await agent.check_auth(cwd)


async def _run_request(
    payload: dict[str, Any], *, allowed_roots: tuple[Path, ...]
) -> AgentResult:
    operation = str(payload.get("operation") or "run")
    raw_seat = payload.get("seat")
    raw_cwd = payload.get("cwd")
    raw_prompt = payload.get("prompt")
    seat = raw_seat if isinstance(raw_seat, str) else ""
    cwd = raw_cwd if isinstance(raw_cwd, str) else ""
    prompt = raw_prompt if isinstance(raw_prompt, str) else ""
    if seat not in _SUPPORTED_SEATS:
        return AgentResult(
            agent=seat or "unknown",
            output="",
            error=f"unknown helper seat {seat!r}",
            returncode=1,
            unavailable_reason="seat helper",
        )
    if operation not in {"run", "auth-check"}:
        return AgentResult(
            agent=seat,
            output="",
            error=f"unknown helper operation {operation!r}",
            returncode=1,
            unavailable_reason="seat helper",
        )
    if operation == "auth-check" and seat != "gemini":
        return AgentResult(
            agent=seat,
            output="",
            error="helper auth-check supports only the gemini seat",
            returncode=1,
            unavailable_reason="seat helper",
        )
    if not cwd or (operation == "run" and not prompt):
        return AgentResult(
            agent=seat,
            output="",
            error=(
                "seat helper request must include non-empty cwd"
                + (" and prompt" if operation == "run" else "")
            ),
            returncode=1,
            unavailable_reason="seat helper",
        )
    allowed_error = cwd_allowed_error(cwd, allowed_roots)
    if allowed_error is not None:
        return AgentResult(
            agent=seat,
            output="",
            error=allowed_error,
            returncode=1,
            unavailable_reason="seat helper",
        )
    default_model = CLAUDE_DEFAULT_MODEL if seat == "claude" else GEMINI_DEFAULT_MODEL
    model = str(payload.get("model") or default_model)
    model_source = str(payload.get("model_source") or "shipped")
    raw_recorded_version = payload.get("recorded_cli_version")
    recorded_cli_version = (
        str(raw_recorded_version) if raw_recorded_version is not None else None
    )
    try:
        timeout_s = float(payload.get("timeout_s", 600.0))
    except (TypeError, ValueError):
        timeout_s = 600.0
    if timeout_s <= 0:
        timeout_s = 600.0
    agent: Agent
    if seat == "claude":
        effort = str(payload.get("effort") or CLAUDE_DEFAULT_EFFORT)
        agent = ClaudeAgent(model=model, effort=effort, model_source=model_source)
    else:
        gemini_agent = GeminiCliAgent(
            model=model,
            print_timeout=timeout_s,
            model_source=model_source,
            recorded_cli_version=recorded_cli_version,
        )
        if operation == "auth-check":
            try:
                return await asyncio.wait_for(
                    gemini_agent.check_auth(cwd), timeout=timeout_s
                )
            except TimeoutError:
                return AgentResult(
                    agent=seat,
                    output="",
                    error=f"{seat} helper auth-check timed out after {timeout_s:g}s",
                    returncode=HELPER_TIMEOUT_RC,
                    unavailable_reason="timeout",
                )
        agent = gemini_agent
    try:
        return await asyncio.wait_for(agent.run(prompt, cwd), timeout=timeout_s)
    except TimeoutError:
        return AgentResult(
            agent=seat,
            output="",
            error=f"{seat} helper request timed out after {timeout_s:g}s",
            returncode=HELPER_TIMEOUT_RC,
            unavailable_reason="timeout",
        )


def _valid_request_id(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 32:
        return None
    return value if all(ch in _REQUEST_ID_CHARS for ch in value) else None


async def _process_one_request(
    path: Path, spool: Path, *, allowed_roots: tuple[Path, ...]
) -> bool:
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return False
    request_id = _valid_request_id(payload.get("id"))
    if request_id is None:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return False
    inflight = spool / "inflight" / path.name
    try:
        os.replace(path, inflight)
    except FileNotFoundError:
        return False
    try:
        result = await _run_request(payload, allowed_roots=allowed_roots)
        _atomic_write_json(spool / "results" / f"{request_id}.json", asdict(result))
    finally:
        with contextlib.suppress(FileNotFoundError):
            inflight.unlink()
    return True


async def serve_helper(
    *,
    spool_dir: Path | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
    poll_interval: float = 0.2,
    once: bool = False,
) -> None:
    spool = spool_dir or default_spool_dir()
    roots = default_allowed_roots() if allowed_roots is None else allowed_roots
    _ensure_spool(spool)
    _sweep_stale_results(spool)
    started_at = time.time()

    def _write_state() -> None:
        _atomic_write_json(
            _state_path(spool),
            {
                "pid": os.getpid(),
                "protocol_version": HELPER_PROTOCOL_VERSION,
                "code_quorum_version": __version__,
                "started_at": started_at,
                "heartbeat_at": time.time(),
                "spool_dir": str(spool),
                "allowed_roots": [str(root) for root in roots],
            },
        )

    _write_state()
    last_heartbeat = time.monotonic()
    tasks: set[asyncio.Task[bool]] = set()
    try:
        while True:
            now = time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                _sweep_stale_results(spool)
                _write_state()
                last_heartbeat = now
            for request in sorted((spool / "requests").glob("*.json")):
                if once and tasks:
                    break
                tasks.add(
                    asyncio.create_task(
                        _process_one_request(request, spool, allowed_roots=roots)
                    )
                )
            if once and tasks:
                await asyncio.gather(*tasks)
                return
            done = {task for task in tasks if task.done()}
            for task in done:
                with contextlib.suppress(Exception):
                    task.result()
            tasks.difference_update(done)
            await asyncio.sleep(poll_interval)
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(FileNotFoundError):
            _state_path(spool).unlink()
