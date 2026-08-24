from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass

from .agents.claude import check_subscription_auth
from .agents.gemini_cli import check_sandbox_available, verify_read_only_config
from .agents.seat_helper import HELPER_PROTOCOL_VERSION, helper_compatibility_error
from .model_config import ModelConfigError, load_config, seat_disabled

DOCTOR_HOSTS = ("claude", "codex", "both")

# The seat each checked binary belongs to -- "uv" has no seat (it's a
# runtime dependency, not a council member), so it's absent here and never
# skipped by a `disabled` record. "agy" is gemini's binary, not its seat
# name (see gemini_cli.py); every other binary name matches its seat name.
_SEAT_BY_BINARY = {
    "agy": "gemini",
    "codex": "codex",
    "claude": "claude",
    "opencode": "opencode",
}


@dataclass(frozen=True)
class Diagnostic:
    name: str
    ok: bool
    detail: str


def _result(name: str, error: str | None, success: str) -> Diagnostic:
    return Diagnostic(name=name, ok=error is None, detail=error or success)


def run_doctor(host: str) -> list[Diagnostic]:
    """Check the binaries and containment boundaries needed by a host profile."""
    normalized = host.strip().lower()
    if normalized not in DOCTOR_HOSTS:
        raise ValueError(
            f"unknown host {host!r}; choose one of: {', '.join(DOCTOR_HOSTS)}"
        )

    includes_claude_host = normalized in {"claude", "both"}
    includes_codex_host = normalized in {"codex", "both"}
    binaries = {"agy", "opencode", "uv"}
    if includes_claude_host:
        binaries.add("codex")
    if includes_codex_host:
        binaries.add("claude")

    results: list[Diagnostic] = []

    # Probe models.toml exactly ONCE, here, so a malformed file surfaces as
    # one clear diagnostic instead of aborting the whole report -- doctor is
    # an environment-triage command and must always finish, precisely when
    # the user most needs it to. On failure every seat reads as NOT
    # disabled (never silently skip a check because the file is corrupt);
    # `config_readable` gates every seat_disabled() call below so a
    # confirmed-malformed file is never probed again (and never raises a
    # second time).
    try:
        load_config()
        config_readable = True
    except ModelConfigError as exc:
        config_readable = False
        results.append(
            Diagnostic(name="models.toml", ok=False, detail=f"unreadable -- {exc}")
        )

    def _disabled(seat: str) -> bool:
        return config_readable and seat_disabled(seat)

    def _gated(
        name: str,
        seats: tuple[str, ...],
        check_fn: Callable[[], str | None],
        success_msg: str,
    ) -> Diagnostic:
        """`name`'s real check, unless every seat in `seats` is disabled --
        then a skipped Diagnostic naming them instead. The four boundary
        checks below (agy-config, seatbelt, claude-subscription, seat-
        helper) are each gated on the seat(s) they exist to protect."""
        if all(_disabled(seat) for seat in seats):
            label = " and ".join(seats)
            plural = "s" if len(seats) > 1 else ""
            return Diagnostic(
                name=name,
                ok=True,
                detail=f"skipped -- {label} seat{plural} disabled by user",
            )
        return _result(name, check_fn(), success_msg)

    for binary in sorted(binaries):
        seat = _SEAT_BY_BINARY.get(binary)
        if seat is not None and _disabled(seat):
            results.append(
                Diagnostic(name=f"seat:{seat}", ok=True, detail="disabled by user")
            )
            continue
        path = shutil.which(binary)
        results.append(
            Diagnostic(
                name=f"binary:{binary}",
                ok=path is not None,
                detail=path or f"{binary} not found on PATH",
            )
        )

    # agy-config and seatbelt both validate the gemini seat's read-only
    # sandbox (settings file, Seatbelt enforcement) -- both are gemini-owned
    # and skip together.
    results.append(
        _gated(
            "agy-config",
            ("gemini",),
            verify_read_only_config,
            "agy defence-in-depth settings are present",
        )
    )
    if includes_claude_host:
        results.append(
            _gated(
                "seatbelt",
                ("gemini",),
                check_sandbox_available,
                "macOS Seatbelt profile smoke test passed",
            )
        )
    if includes_codex_host:
        results.append(
            _gated(
                "claude-subscription",
                ("claude",),
                check_subscription_auth,
                "Claude CLI is logged in with a Claude.ai subscription",
            )
        )
        # The helper serves both claude and gemini -- it still matters if
        # either one is live, so it's skipped only when both are disabled.
        results.append(
            _gated(
                "seat-helper",
                ("claude", "gemini"),
                helper_compatibility_error,
                "shared Claude/agy seat helper is running with protocol "
                f"{HELPER_PROTOCOL_VERSION}",
            )
        )

    return results
