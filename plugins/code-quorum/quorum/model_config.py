"""Model-selection config: reads and writes ~/.config/code-quorum/models.toml,
resolving each council seat's model through a fixed priority chain (per-run
override > env var > recorded choice > shipped default) and recording new
choices atomically.

Must NOT import seat modules at module level: they import this module to resolve
their models, and a module-level import back here would create a cycle. Each
VALID_EFFORTS set is imported inside the effort-validation branch instead,
where it is actually needed."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
import tempfile
import tomllib
from collections.abc import Callable
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("~/.config/code-quorum/models.toml").expanduser()

_RERUN_HINT = "re-run `quorum setup-models` or delete the file"

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


@functools.cache
def probe_cli_version(binary: str) -> str | None:
    """Best-effort installed CLI version for `binary` (e.g. "1.0.13"), parsed
    from `<binary> --version`'s stdout+stderr. None if the binary is missing,
    the probe times out, or the output has no dotted-triple version. Cached
    per binary name (process-wide) so a council run spawns the probe at most
    once per binary regardless of how many seats ask.

    Shared by every seat's own `_x_version` wrapper (claude, codex, opencode,
    agy) -- each module keeps its thin wrapper as its own monkeypatch seam;
    this is only the underlying subprocess+regex+cache."""
    try:
        proc = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _VERSION_RE.search(f"{proc.stdout}\n{proc.stderr}")
    return match.group(1) if match else None


class ModelConfigError(ValueError):
    """models.toml is malformed or a seat table fails validation."""


# A models.toml file is valid only when every top-level value is a table and
# every value inside every table is a string. This is the one shape _serialize
# can round-trip losslessly and the shape setup-models writes (for example,
# chosen is an ISO string). The values are
# deliberately all-string). A non-dict top-level value or a non-string
# table value is malformed for ANY seat name: neither shape survives a
# rewrite intact, so tolerating one now just defers the corruption to
# whichever future write mangles it.
#
# `model` must ALSO be present in every seat's table, UNLESS it carries
# `disabled = "true"` (a user's durable "I don't use this seat" declaration --
# setup-models records that table with no `model` at all):
# nothing ever writes a seat outside claude/codex/gemini/opencode, so
# there is no forward-compat case to carve out.


def _config_path(path: Path | None) -> Path:
    return path if path is not None else CONFIG_PATH


def load_config(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Parse models.toml into {seat: {field: value}}. `{}` when the file is
    absent -- a single open()/FileNotFoundError round trip, no separate
    exists() probe. A models.toml is valid iff every top-level value is a
    table and every value inside every table is a string -- anything else
    raises ModelConfigError, for ANY seat name. `model` is additionally
    required unless the table is marked `disabled = "true"`. A [claude] or
    [codex] table's `effort` (if any) must also be one that seat actually
    accepts."""
    config_path = _config_path(path)
    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        raise ModelConfigError(
            f"{config_path}: invalid TOML ({exc}) -- {_RERUN_HINT}."
        ) from exc

    for seat, table in data.items():
        if not isinstance(table, dict):
            raise ModelConfigError(
                f"{config_path}: [{seat}] must be a table, got "
                f"{type(table).__name__} -- {_RERUN_HINT}."
            )
        for field, value in table.items():
            if not isinstance(value, str):
                # A hand-edited `disabled = true` (an unquoted TOML boolean)
                # is the one shape of this mistake worth naming explicitly --
                # it now gates doctor and every default-roster run, so a
                # generic "must be a string" leaves the fix non-obvious.
                hint = (
                    ' -- use disabled = "true" or disabled = "false" (a '
                    "quoted string, not a TOML boolean)"
                    if field == "disabled"
                    else ""
                )
                raise ModelConfigError(
                    f"{config_path}: [{seat}] {field} must be a string, got "
                    f"{type(value).__name__}{hint} -- {_RERUN_HINT}."
                )
            if field == "disabled" and value not in {"true", "false"}:
                # A string value that isn't exactly "true"/"false" (a typo'd
                # "True", "yes", etc.) passes the string check above but
                # today silently means "enabled" -- loud and specific beats
                # a seat someone thinks they disabled quietly staying live.
                raise ModelConfigError(
                    f"{config_path}: [{seat}] disabled={value!r} must be "
                    f'"true" or "false" -- {_RERUN_HINT}.'
                )
        if "model" not in table and table.get("disabled") != "true":
            raise ModelConfigError(
                f"{config_path}: [{seat}] is missing `model` -- {_RERUN_HINT}."
            )
        if (
            seat in {"claude", "codex"}
            and "effort" in table
            and table.get("disabled") != "true"
        ):
            # A disabled seat skips effort validation for the same reason it
            # skips the `model` requirement: nothing runs it, and a stale
            # effort must not abort the whole config load for every seat.
            if seat == "claude":
                from quorum.agents.claude import VALID_EFFORTS
            else:
                from quorum.agents.codex import VALID_EFFORTS

            if table["effort"] not in VALID_EFFORTS:
                allowed = ", ".join(sorted(VALID_EFFORTS))
                raise ModelConfigError(
                    f"{config_path}: [{seat}] effort {table['effort']!r} is not "
                    f"one of {{{allowed}}} -- {_RERUN_HINT}."
                )
    return data


def recorded_choice(seat: str, path: Path | None = None) -> dict[str, str] | None:
    return load_config(path).get(seat)


def seat_disabled(seat: str, path: Path | None = None) -> bool:
    """True when `seat`'s recorded models.toml choice carries
    `disabled = "true"` -- a user's durable "I don't use this one"
    declaration from `quorum setup-models`. The single predicate shared by
    orchestration.py (seat selection) and doctor.py (per-seat diagnostics)
    so the decoding of `disabled` lives in exactly one place.

    Raises ModelConfigError exactly like recorded_choice/load_config on a
    malformed file -- a caller that must never crash on a corrupt config
    (doctor.py is an environment-triage command) catches that once around
    its own disabled-state lookup rather than relying on this function to
    silently swallow it."""
    recorded = recorded_choice(seat, path)
    return recorded is not None and recorded.get("disabled") == "true"


_env_masks_recorded_warned: set[str] = set()


def _warn_env_masks_recorded(seat: str, env_value: str, recorded_model: str) -> None:
    """Warn once per seat per process when an env override silently outranks
    the recorded model choice -- otherwise the mismatch is invisible until
    the caller notices the wrong model ran. The env var name is derived, not
    passed in: every seat follows CODE_QUORUM_{SEAT}_MODEL (codex.py,
    opencode.py, orchestration.py's gemini lookup all follow it)."""
    if seat in _env_masks_recorded_warned:
        return
    _env_masks_recorded_warned.add(seat)
    env_var = f"CODE_QUORUM_{seat.upper()}_MODEL"
    logger.warning(
        "%s: env override selected %r but models.toml has %r recorded -- "
        "unset %s and restart Claude Code (/reload-plugins does not change "
        "env).",
        seat,
        env_value,
        recorded_model,
        env_var,
    )


_version_drift_warned: set[str] = set()


def warn_version_drift(seat: str, installed: str, recorded_version: str) -> None:
    """Warn once per process per seat that the installed CLI version no longer
    matches the version recorded at the last `quorum setup-models` run. Soft
    by design -- the recorded model still runs; this only nudges a re-verify,
    mirroring the agy SEAT_VERIFIED_AGY_VERSION warning pattern. Callers
    decide WHETHER the two differ (both known and unequal); this function
    only handles the per-seat dedup and message, the same split as
    _warn_env_masks_recorded."""
    if seat in _version_drift_warned:
        return
    _version_drift_warned.add(seat)
    logger.warning(
        "%s: installed CLI is v%s but models.toml recorded v%s (as of the "
        "last `quorum setup-models` run) -- the recorded model still runs; "
        "re-run `quorum setup-models` to re-verify and refresh the record.",
        seat,
        installed,
        recorded_version,
    )


def append_recorded_source_hint(error: str, model: str, source: str) -> str:
    """When a failed seat run's model came from a recorded `quorum
    setup-models` choice (`source == "recorded"`), append one conditional
    line naming the model and pointing at the remedy -- conditional phrasing
    on purpose: this never claims the model IS the cause (the run could have
    failed for any reason), and it never parses stderr. Returns `error`
    unchanged for any other source.

    Callers gate this on the failure's return code themselves BEFORE calling
    -- failure classes that are definitely not model problems (a missing
    binary, a missing API key, an idle stall) must never see this hint."""
    if source != "recorded":
        return error
    hint = (
        f"model {model!r} came from a recorded choice; if it is no longer "
        "available, re-run `quorum setup-models`."
    )
    return f"{error}\n{hint}" if error else hint


def resolve_model(
    seat: str,
    *,
    per_run: str | None,
    env_value: str | None,
    shipped: str,
    path: Path | None = None,
) -> tuple[str, str]:
    """Resolve `seat`'s model: first non-None of per_run, env_value, the
    recorded choice, shipped. `env_value` is the caller's own env var read,
    already resolved -- passed in rather than looked up here so this module
    stays decoupled from each seat's os.environ.get() call site, even though
    the CODE_QUORUM_{SEAT}_MODEL name itself is derivable (see
    _warn_env_masks_recorded)."""
    recorded = recorded_choice(seat, path)
    recorded_model = recorded.get("model") if recorded else None

    if (
        per_run is None
        and env_value is not None
        and recorded_model is not None
        and env_value != recorded_model
    ):
        _warn_env_masks_recorded(seat, env_value, recorded_model)

    if per_run is not None:
        return per_run, "per-run"
    if env_value is not None:
        return env_value, "env"
    if recorded_model is not None:
        return recorded_model, "recorded"
    return shipped, "shipped"


def resolve_seat_choice(
    seat: str,
    *,
    model_env_var: str,
    effort_env_var: str,
    shipped_model: str,
    shipped_effort: str,
    valid_efforts: frozenset[str],
    effort_label: str,
    version_fn: Callable[[], str | None],
) -> tuple[str, str, str]:
    """Resolve `seat`'s (model, effort, source) -- shared by make_claude_agent
    and make_codex_agent, which otherwise duplicated this whole chain modulo a
    few strings.

    Model resolves via resolve_model: per_run(none here) > env `model_env_var`
    > a recorded `quorum setup-models` choice > `shipped_model`. Effort
    resolves env `effort_env_var` > a recorded choice's `effort` >
    `shipped_effort`; the env and recorded values are both validated against
    `valid_efforts` (raising ValueError with `effort_label` naming what was
    expected -- recorded values need it here because load_config skips effort
    validation on disabled tables). When a recorded choice carries a
    `cli_version`, calls `version_fn()` and warns once (never raises) on
    drift against the installed CLI -- the recorded model still runs.

    `version_fn` is the caller's OWN `_x_version` wrapper, passed in by the
    caller's bare reference at its own call site -- so a test that
    monkeypatches e.g. `codex_mod._codex_version` before calling
    `make_codex_agent()` is still honored; this function never imports or
    captures a seat module's version function itself."""
    model, source = resolve_model(
        seat,
        per_run=None,
        env_value=os.environ.get(model_env_var, "").strip() or None,
        shipped=shipped_model,
    )
    recorded = recorded_choice(seat)
    env_effort = os.environ.get(effort_env_var, "").strip()
    if env_effort:
        if env_effort not in valid_efforts:
            allowed = ", ".join(sorted(valid_efforts))
            raise ValueError(
                f"{effort_env_var}={env_effort!r} is not a recognized "
                f"{effort_label}; use one of: {allowed}."
            )
        effort = env_effort
    else:
        recorded_effort = (recorded or {}).get("effort")
        if recorded_effort and recorded_effort not in valid_efforts:
            # Normally unreachable for an enabled seat (load_config already
            # validated it), but load skips effort validation on disabled
            # tables -- and select_agents deliberately honors an explicit
            # request for a disabled seat. Validate here, where the value is
            # consumed.
            allowed = ", ".join(sorted(valid_efforts))
            raise ValueError(
                f"models.toml records effort {recorded_effort!r} for [{seat}], "
                f"which is not a recognized {effort_label}; use one of: "
                f"{allowed}, or re-run `quorum setup-models`."
            )
        effort = recorded_effort or shipped_effort
    recorded_version = recorded.get("cli_version") if recorded else None
    if recorded_version is not None:
        installed = version_fn()
        if installed is not None and installed != recorded_version:
            warn_version_drift(seat, installed, recorded_version)
    return model, effort, source


_BARE_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")


def _serialize(config: dict[str, dict[str, str]]) -> str:
    """Hand-rolled TOML writer for flat string-valued tables only -- the one
    shape this file ever contains. Seats and fields both sorted so re-runs
    produce a stable diff.

    Escaping reuses json.dumps rather than hand-rolling a TOML basic-string
    escape table: with ensure_ascii=False, JSON escapes exactly backslash,
    quote, and C0 controls (\\b \\f \\n \\r \\t, \\uXXXX for the rest) --
    all valid TOML escapes -- and passes other characters through literally.
    ensure_ascii=True would encode non-BMP characters as \\uXXXX surrogate
    pairs, which TOML rejects on the next load. DEL is the one character JSON
    leaves literal but TOML forbids, so it is escaped by hand. Seats and keys
    are written bare, so anything
    outside TOML's bare-key alphabet fails loudly here instead of producing
    a file the next load rejects."""
    lines: list[str] = []
    for seat in sorted(config):
        table = config[seat]
        for name in (seat, *table):
            if not _BARE_KEY_RE.fullmatch(name):
                raise ModelConfigError(
                    f"cannot write models.toml: {name!r} is not a bare TOML "
                    "key (letters, digits, `_`, `-` only)"
                )
        lines.append(f"[{seat}]")
        for key in sorted(table):
            escaped = json.dumps(str(table[key]), ensure_ascii=False)[1:-1]
            lines.append(f'{key} = "{escaped.replace("\x7f", "\\u007F")}"')
        lines.append("")
    return "\n".join(lines)


def record_choice(seat: str, values: dict[str, str], path: Path | None = None) -> None:
    """Merge `values` into `seat`'s table (replacing any prior table for that
    seat -- a fresh setup run always records a complete choice, not a patch),
    stamp `chosen`, and write the whole config back atomically so a killed
    process can never leave a partial models.toml (temp file in the same
    dir + os.replace).

    A malformed existing file is treated as empty rather than raising, so a
    successful record repairs the corrupt file instead of failing at the
    same load that made it corrupt in the first place -- this is the whole
    reason the `quorum setup-models` repair path (setup_models.py) works."""
    config_path = _config_path(path)
    try:
        config = load_config(config_path)
    except ModelConfigError:
        config = {}
    config[seat] = {**values, "chosen": date.today().isoformat()}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialize(config)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=config_path.parent,
        delete=False,
        prefix=".models-",
        suffix=".toml.tmp",
    ) as tmp:
        tmp.write(serialized)
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, config_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
