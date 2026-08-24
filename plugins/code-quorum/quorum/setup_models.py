"""Business logic behind `quorum setup-models`: parse each seat's model
listing, live-probe a candidate before it can be recorded, and write the
result via model_config.record_choice. quorum/cli.py owns all prompting/
echo (this module never imports typer, matching every other business-logic
module in the package) and dispatches into the functions here.

The probe is the safety gate the whole command exists for: nothing is
recorded unless the candidate model actually routes and answers *right
now* -- catalog presence (an id appearing in `agy models` / `opencode
models`) is necessary but never sufficient (see gemini_cli.py's routing
mis-route history, and the opencode account-routability note in the design
spec)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Awaitable, Callable

from .agents.base import Agent
from .agents.claude import (
    DEFAULT_EFFORT as CLAUDE_DEFAULT_EFFORT,
)
from .agents.claude import (
    DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL,
)
from .agents.claude import (
    EFFORT_ENV_VAR as CLAUDE_EFFORT_ENV_VAR,
)
from .agents.claude import (
    MODEL_ENV_VAR as CLAUDE_MODEL_ENV_VAR,
)
from .agents.claude import (
    VALID_EFFORTS as CLAUDE_VALID_EFFORTS,
)
from .agents.claude import ClaudeAgent, _claude_version
from .agents.codex import (
    DEFAULT_EFFORT as CODEX_DEFAULT_EFFORT,
)
from .agents.codex import (
    DEFAULT_MODEL as CODEX_DEFAULT_MODEL,
)
from .agents.codex import (
    EFFORT_ENV_VAR as CODEX_EFFORT_ENV_VAR,
)
from .agents.codex import (
    MODEL_ENV_VAR as CODEX_MODEL_ENV_VAR,
)
from .agents.codex import (
    VALID_EFFORTS as CODEX_VALID_EFFORTS,
)
from .agents.codex import (
    CodexAgent,
    _codex_version,
)
from .agents.gemini_cli import (
    DEFAULT_MODEL as GEMINI_DEFAULT_MODEL,
)
from .agents.gemini_cli import (
    QUOTA_REFLEX_OUTPUT_PREFIX,
    GeminiCliAgent,
    _agy_version,
    canonical_model,
    routing_key,
)
from .agents.opencode import (
    DEFAULT_MODEL as OPENCODE_DEFAULT_MODEL,
)
from .agents.opencode import (
    MODEL_ENV_VAR as OPENCODE_MODEL_ENV_VAR,
)
from .agents.opencode import (
    OpenCodeAgent,
    _opencode_version,
)
from .model_config import (
    ModelConfigError,
    load_config,
    record_choice,
    recorded_choice,
    resolve_model,
)

SEATS = ("claude", "codex", "gemini", "opencode")

# Not exported from gemini_cli.py (the seat inlines the literal at its one
# call site in orchestration.py) -- named here since this module reads it
# more than once (current choice display, env-mismatch warning, record).
GEMINI_MODEL_ENV_VAR = "CODE_QUORUM_GEMINI_MODEL"

# What every probe asks. Trivial on purpose -- the probe exists to prove
# the model routes and answers, not to exercise it.
_PROBE_PROMPT = "Reply with exactly the single word: OK"


# seat -> (effort env var, shipped default). current_claude_effort and
# current_codex_effort are otherwise identical bodies.
_EFFORT_ENV_DEFAULT = {
    "claude": (CLAUDE_EFFORT_ENV_VAR, CLAUDE_DEFAULT_EFFORT),
    "codex": (CODEX_EFFORT_ENV_VAR, CODEX_DEFAULT_EFFORT),
}

# seat -> (valid efforts, wording each seat's "not one of" ValueError uses).
_EFFORT_VALIDATION = {
    "claude": (CLAUDE_VALID_EFFORTS, "Claude effort"),
    "codex": (CODEX_VALID_EFFORTS, "effort"),
}


# --- Parsers -----------------------------------------------------------------


def parse_agy_models(text: str) -> list[tuple[str, str]]:
    """Parse `agy models` stdout into (slug, display) pairs, in listing
    order. agy lists newest-first, so a new generation precedes older families.

    Accepts both observed shapes. agy >=1.1.12 uses tab-delimited
    `slug<TAB>Display Name` rows. 'Fetching available models...' goes to stderr,
    so stdout is pure data. Earlier versions use bare single-field lines
    (display-name-only before 1.1.5, or bare-slug in 1.1.5-1.1.11). The parser
    cannot tell those
    apart structurally, so a bare line's one field is used as BOTH slug and
    display). A row with more than one tab, an empty field (e.g. a trailing
    tab with nothing after it), or a slug that repeats an earlier row raises
    ValueError -- setup refuses to record on an ambiguous listing rather
    than silently picking one of two candidate rows.

    Splits on tabs BEFORE stripping each field (not the reverse): stripping
    the whole line first would silently absorb a leading/trailing tab as
    outer whitespace, making a genuinely empty display/slug field
    indistinguishable from an ordinary bare-slug line."""
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split("\t")]
        if len(parts) == 1:
            slug = display = parts[0]
        elif len(parts) == 2:
            slug, display = parts
        else:
            raise ValueError(
                f"agy models: malformed row (too many tab fields): {raw!r}"
            )
        if not slug or not display:
            raise ValueError(f"agy models: malformed row (empty field): {raw!r}")
        if slug in seen:
            raise ValueError(f"agy models: duplicate slug {slug!r} in listing")
        seen.add(slug)
        rows.append((slug, display))
    return rows


def parse_opencode_models(text: str) -> list[str]:
    """Parse `opencode models` stdout into model ids, one per non-blank
    line. Filtering to `openrouter/*` (the only provider this seat ever
    runs) is the caller's job -- see `openrouter_only` -- so this stays a
    faithful transcript of everything opencode printed."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def openrouter_only(model_ids: list[str]) -> list[str]:
    return [m for m in model_ids if m.startswith("openrouter/")]


def _gemini_family(slug: str) -> str:
    """Grouping key for a gemini slug/display id: the reasoning-tier suffix
    stripped (agy's own convention -- the tier is part of the id, not a
    separate axis; see gemini_cli.py's DEFAULT_MODEL comment)."""
    low = slug.lower()
    for tier in ("-high", "-medium", "-low", " (high)", " (medium)", " (low)"):
        if low.endswith(tier):
            return slug[: -len(tier)]
    return slug


def group_gemini_models(
    rows: list[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Group (slug, display) rows by family, preserving `rows`' own order
    within and across groups -- since agy already lists newest-first, the
    resulting dict iterates newest-first too (Python dicts preserve
    insertion order)."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for slug, display in rows:
        groups.setdefault(_gemini_family(slug), []).append((slug, display))
    return groups


# --- Live listing fetch (subprocess module seams) -----------------------------


def _run_models_command(binary: str) -> str:
    """Run `<binary> models` and return stdout text. Raises ValueError on a
    missing binary, a spawn failure, or a non-zero exit -- setup cannot
    show (or match against) a discovery list it couldn't fetch."""
    try:
        proc = subprocess.run(
            [binary, "models"], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError as exc:
        raise ValueError(f"{binary} not found on PATH") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"`{binary} models` failed: {exc}") from exc
    if proc.returncode != 0:
        raise ValueError(
            f"`{binary} models` exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


def fetch_agy_listing(binary: str = "agy") -> list[tuple[str, str]]:
    return parse_agy_models(_run_models_command(binary))


def fetch_opencode_listing(binary: str = "opencode") -> list[str]:
    return parse_opencode_models(_run_models_command(binary))


def _lookup_catalog_row(
    model: str, listing: list[tuple[str, str]]
) -> tuple[str, str] | None:
    """The (slug, display) row from `agy models` matching `model` (which may
    itself be a slug or a display name), matched via routing_key -- the same
    case/punctuation-insensitive comparison the seat's own routing check
    uses (gemini_cli.routing_complaint). None when the listing has no match:
    a missing catalog match is absent evidence, not an error -- the live
    probe, not this lookup, is what actually proves the model works (mirrors
    routing_complaint's own 'no labels is NOT a complaint' rule)."""
    want = routing_key(model)
    for slug, display in listing:
        if routing_key(slug) == want or routing_key(display) == want:
            return slug, display
    return None


# --- Malformed-file repair: the setup flow tolerates what council-time load
# never does (model_config.load_config/resolve_model/recorded_choice stay
# loud everywhere else -- only these setup-flow lookups degrade). ------------


def check_malformed_config() -> str | None:
    """Detect a malformed models.toml once, at setup-flow entry. Returns the
    user-facing message (the ModelConfigError text plus the repair-mode
    notice) when malformed, else None. cli.py prints this once per command
    invocation; every lookup below (`_resolve`, `current_codex_effort`) then
    tolerates the SAME malformed file for the rest of that invocation
    without re-raising, so the flow can still show/record choices, and
    `record_choice` lets a successful record overwrite the corrupt file
    (it treats a malformed existing file as empty unconditionally).
    Council-time loading (model_config.load_config, used by the seat
    factories) is untouched and stays loud."""
    try:
        load_config()
    except ModelConfigError as exc:
        return (
            f"{exc}\nproceeding as if no choices are recorded; completing "
            "setup will overwrite the corrupt file."
        )
    return None


# --- Current effective choice (also satisfies the env-mismatch warning) -------


def _resolve(seat: str, env_var: str, shipped: str) -> tuple[str, str]:
    env_value = os.environ.get(env_var, "").strip() or None
    try:
        return resolve_model(seat, per_run=None, env_value=env_value, shipped=shipped)
    except ModelConfigError:
        # models.toml is malformed -- already reported once by
        # check_malformed_config() at setup-flow entry. Every later lookup
        # in this module treats the file as if nothing were recorded rather
        # than re-raising the same error on every seat's display/record call.
        return (env_value, "env") if env_value is not None else (shipped, "shipped")


def current_choice(seat: str) -> tuple[str, str]:
    """(model, source) for `seat`'s current effective choice -- the exact
    ladder each factory uses (env > recorded > shipped). This call is also
    what satisfies "warn before writing." It uses the same resolve_model call
    as each factory, so invoking it here before any record_choice write fires
    the env-masks-recorded warning (deduped once per seat per process) if the
    ambient environment disagrees with the recorded value.

    gemini always resolves the agy-scoped (cli-backend) record regardless
    of the ambient CODE_QUORUM_GEMINI_BACKEND: setup governs that record,
    not whichever seat the council would build right now if the sdk
    backend is active."""
    if seat == "claude":
        return _resolve("claude", CLAUDE_MODEL_ENV_VAR, CLAUDE_DEFAULT_MODEL)
    if seat == "codex":
        return _resolve("codex", CODEX_MODEL_ENV_VAR, CODEX_DEFAULT_MODEL)
    if seat == "opencode":
        return _resolve("opencode", OPENCODE_MODEL_ENV_VAR, OPENCODE_DEFAULT_MODEL)
    model, source = _resolve("gemini", GEMINI_MODEL_ENV_VAR, GEMINI_DEFAULT_MODEL)
    return canonical_model(model), source


def _current_effort(seat: str, env_var: str, shipped: str) -> str:
    """A seat's effective effort without factory version-check side effects."""
    env_effort = os.environ.get(env_var, "").strip()
    if env_effort:
        return env_effort
    try:
        recorded = recorded_choice(seat)
    except ModelConfigError:
        # See _resolve: tolerate the same malformed file, already reported
        # once by check_malformed_config().
        recorded = None
    return (recorded or {}).get("effort") or shipped


def current_claude_effort() -> str:
    env_var, default = _EFFORT_ENV_DEFAULT["claude"]
    return _current_effort("claude", env_var, default)


def current_codex_effort() -> str:
    env_var, default = _EFFORT_ENV_DEFAULT["codex"]
    return _current_effort("codex", env_var, default)


# --- Probes: None on success, an error string on failure ----------------------


async def _probe(
    agent: Agent, label: str, cwd: str, *, check_empty: bool = True
) -> str | None:
    """Run `agent` against the trivial probe prompt. Success = a non-error
    result (returncode 0) AND, when `check_empty`, a non-empty answer:
    CodexAgent/ClaudeAgent.run map a missing output file to "" (codex.py,
    claude.py), which is what a process that exits 0 without ever writing
    an answer looks like -- rc alone would record that as probed-valid.
    OpenCodeAgent has no such failure mode, so it skips the empty check."""
    result = await agent.run(_PROBE_PROMPT, cwd)
    if result.returncode != 0:
        return result.error or f"{label} probe failed (exit {result.returncode})"
    if check_empty and not result.output.strip():
        return f"{label} probe produced no answer"
    return None


async def probe_codex(model: str, effort: str, cwd: str) -> str | None:
    """CodexAgent with the candidate model/effort, one trivial prompt."""
    return await _probe(CodexAgent(model=model, effort=effort), "codex", cwd)


async def probe_claude(model: str, effort: str, cwd: str) -> str | None:
    """Claude subscription seat with the candidate pair and a trivial prompt."""
    return await _probe(ClaudeAgent(model=model, effort=effort), "claude", cwd)


async def probe_gemini(model: str, cwd: str) -> str | None:
    """GeminiCliAgent with the candidate model, built with
    model_source='recorded' -- this is what the design calls for
    ("success requires BOTH a successful result AND no routing complaint,
    checked in that order"): GeminiCliAgent.run already hard-fails a
    routing mismatch when model_source is 'recorded', so a plain returncode
    check here is that combined check, with no duplicated routing logic.

    A clean rc 0 is not enough on its own: on quota exhaustion the agent
    reruns the turn on AGY_QUOTA_FALLBACK_MODEL and labels the swap with a
    "[quota reflex: ...]" marker (gemini_cli.py) -- a success there
    proves the FALLBACK routes and answers, not the candidate. Treat that as
    a probe failure too, or setup would record a model whose own run was
    never actually validated."""
    agent = GeminiCliAgent(model=model, model_source="recorded")
    result = await agent.run(_PROBE_PROMPT, cwd)
    if result.returncode != 0:
        return result.error or f"gemini probe failed (exit {result.returncode})"
    if any(
        line.startswith(QUOTA_REFLEX_OUTPUT_PREFIX)
        for line in result.output.splitlines()
    ):
        return (
            f"gemini probe for {model!r} hit AI-Pro quota exhaustion and fell "
            "back to a different model -- retry setup later"
        )
    return None


async def probe_opencode(model: str, cwd: str) -> str | None:
    """OpenCodeAgent with the candidate model, one trivial prompt. This is
    the account-routability gate -- catalog presence in `opencode models`
    does not prove OpenRouter will actually serve this account/model pair
    (data-policy filters 404 models the list still shows)."""
    return await _probe(OpenCodeAgent(model=model), "opencode", cwd, check_empty=False)


# --- Record: validate, warn, probe, capture version, write --------------------


class ProbeFailure(ValueError):
    """A live probe against a candidate model failed. Distinct from a plain
    ValueError (a flag-validation error, e.g. an unrecognized effort) so
    cli.py's non-interactive path can format it as a plain probe error
    instead of typer.BadParameter -- the model id itself may be
    syntactically fine; it's the live probe that failed."""


# seat -> (model env var, shipped default). Not probe/version_fn -- those
# are passed in fresh at each record_claude/record_codex call site (never
# captured into a module-level table) so a test's `monkeypatch.setattr(sm,
# "probe_claude", ...)` still takes effect: a table built once at import
# time would freeze the pre-patch function object instead.
_RECORD_MODEL_DEFAULT = {
    "claude": (CLAUDE_MODEL_ENV_VAR, CLAUDE_DEFAULT_MODEL),
    "codex": (CODEX_MODEL_ENV_VAR, CODEX_DEFAULT_MODEL),
}


async def _record_effort_seat(
    seat: str,
    model: str,
    effort: str,
    *,
    cwd: str,
    probe: Callable[[str, str, str], Awaitable[str | None]],
    version_fn: Callable[[], str | None],
) -> None:
    _validate_effort_for_seat(seat, effort)
    env_var, default_model = _RECORD_MODEL_DEFAULT[seat]
    _resolve(seat, env_var, default_model)  # env-mismatch warn
    error = await probe(model, effort, cwd)
    if error is not None:
        raise ProbeFailure(
            f"{seat} probe failed for {model!r} (effort {effort!r}): {error}"
        )
    values = {"model": model, "effort": effort}
    version = version_fn()
    if version is not None:
        values["cli_version"] = version
    record_choice(seat, values)


async def record_claude(model: str, effort: str, *, cwd: str) -> None:
    await _record_effort_seat(
        "claude", model, effort, cwd=cwd, probe=probe_claude, version_fn=_claude_version
    )


async def record_codex(model: str, effort: str, *, cwd: str) -> None:
    await _record_effort_seat(
        "codex", model, effort, cwd=cwd, probe=probe_codex, version_fn=_codex_version
    )


async def record_gemini(model: str, *, cwd: str) -> None:
    _resolve("gemini", GEMINI_MODEL_ENV_VAR, GEMINI_DEFAULT_MODEL)  # env-mismatch warn
    canon = canonical_model(model)
    try:
        listing = fetch_agy_listing()
    except ValueError:
        listing = []  # best-effort: falls back to probing/recording canon as-is
    # Never record a form the probe did not prove: when the listing maps the
    # chosen slug to a display-name row, probe (and record) THAT DISPLAY
    # FORM, not the slug -- a manual entry outside the listing keeps today's
    # shape, probing and recording the entered (canonicalized) form as-is.
    row = _lookup_catalog_row(canon, listing)
    probe_model = row[1] if row else canon
    error = await probe_gemini(probe_model, cwd)
    if error is not None:
        raise ProbeFailure(f"gemini probe failed for {model!r}: {error}")
    values = {"model": probe_model}
    version = _agy_version()
    if version is not None:
        values["cli_version"] = version
    record_choice("gemini", values)


# Positive allowlist: substrings that indicate the seat itself is genuinely
# absent (no subscription, not logged in, binary missing) -- the only
# failure classes worth a durable "disabled" declaration. Deliberately
# excludes quota exhaustion, network blips, and timeouts: those are
# transient, and offering to disable on a bad day would durably drop a seat
# over a one-time hiccup. Anything NOT matching here (quota, network,
# timeout, an unclassified error) does not trigger the offer -- the safe
# default is to ask again next time, not to disable. Phrases are the exact
# wording each probe's wrapped error carries: claude.py's "not found on
# PATH" / "is not logged in" / "not using a Claude.ai subscription"
# (matched case-insensitively below), codex.py/opencode.py's identical
# "not found on PATH", and gemini_cli.py's full _AUTH_HINTS set (its
# quota-exhaustion message deliberately contains none of these).
_ABSENT_SEAT_HINTS = (
    "not found on path",
    "not logged in",
    "not using a claude.ai subscription",
    "authentication",
    "please sign in",
    "unauthorized",
    "oauth",
    "invalid credentials",
    "invalid token",
    "auth token",
    "access token",
    "refresh token",
    "token expired",
    "credentials expired",
    "session expired",
)


def indicates_absent_seat(error: str) -> bool:
    """True when a probe failure's text indicates the seat is genuinely
    absent (no subscription, not logged in, binary missing) rather than a
    transient failure (quota, network, timeout) -- the gate for offering
    `quorum setup-models`'s durable "mark this seat as not used?" prompt."""
    low = error.lower()
    return any(hint in low for hint in _ABSENT_SEAT_HINTS)


def record_disabled(seat: str) -> None:
    """Record `seat` as `disabled = "true"` -- the durable "I don't use this
    one" declaration `quorum setup-models` offers when a seat's live probe
    indicates it's genuinely absent (see `indicates_absent_seat`; public
    users rarely hold all four subscriptions). Preserves any already-
    recorded fields (model/effort/cli_version) so re-enabling the seat later
    doesn't require re-probing from scratch -- model_config.load_config
    tolerates a disabled table that also carries `model`. A malformed
    existing file is treated as empty (nothing to preserve) rather than
    raising: this write is itself a repair, like every other record_* here
    (record_choice treats a malformed existing file as empty
    unconditionally)."""
    try:
        existing = recorded_choice(seat) or {}
    except ModelConfigError:
        existing = {}
    record_choice(seat, {**existing, "disabled": "true"})


async def record_opencode(model: str, *, cwd: str) -> None:
    _resolve(
        "opencode", OPENCODE_MODEL_ENV_VAR, OPENCODE_DEFAULT_MODEL
    )  # env-mismatch warn
    error = await probe_opencode(model, cwd)
    if error is not None:
        raise ProbeFailure(f"opencode probe failed for {model!r}: {error}")
    values = {"model": model}
    version = _opencode_version()
    if version is not None:
        values["cli_version"] = version
    record_choice("opencode", values)


# --- Non-interactive flag validation -------------------------------------------


def validate_noninteractive_flags(seat: str, effort: str | None) -> None:
    """Raise ValueError on an unrecognized seat, a flag that seat doesn't
    take, or an effort value that Claude/Codex doesn't accept. Pure and
    synchronous, so a bad-flag rejection never needs a probe, a version
    subprocess, or a models.toml read."""
    if seat not in SEATS:
        raise ValueError(f"unknown seat {seat!r}; choose one of: {', '.join(SEATS)}")
    if effort is not None and seat not in {"claude", "codex"}:
        raise ValueError(
            f"--effort is only valid for --seat claude or codex (got --seat {seat!r})"
        )
    if effort is not None and seat == "claude":
        validate_claude_effort(effort)
    elif effort is not None:
        validate_codex_effort(effort)


def _validate_effort_for_seat(seat_name: str, effort: str) -> None:
    """The one place the VALID_EFFORTS membership check lives, for both
    seats: rejects a bad effort before any live work (probe, version
    subprocess, or models.toml read), whether it came from --effort,
    CODE_QUORUM_{SEAT}_EFFORT, or the interactive setup flow."""
    valid, label = _EFFORT_VALIDATION[seat_name]
    if effort not in valid:
        allowed = ", ".join(sorted(valid))
        raise ValueError(f"{label} {effort!r} is not one of: {allowed}")


def validate_claude_effort(effort: str) -> None:
    _validate_effort_for_seat("claude", effort)


def validate_codex_effort(effort: str) -> None:
    _validate_effort_for_seat("codex", effort)
