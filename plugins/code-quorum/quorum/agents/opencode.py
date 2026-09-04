import asyncio
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

from ..model_config import (
    append_recorded_source_hint,
    probe_cli_version,
    recorded_choice,
    resolve_model,
    warn_version_drift,
)
from .base import (
    Agent,
    AgentResult,
    _ensure_private_dir,
    allowlisted_seat_subprocess_env,
    communicate_lines_or_kill,
    has_usage_limit_diagnostic,
)

logger = logging.getLogger(__name__)

# Env vars that disable opencode's project-config discovery. These must
# survive the _XDG/OPENCODE_* strip so the sandbox stays sealed when the
# agent runs in a project with its own opencode.json(c) or .opencode/
# (which could otherwise shadow our council.md). See codex review 2.
_PROJECT_DISABLE_ENV = {
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
    "OPENCODE_PURE": "1",
}

# opencode auto-loads $HOME/.config/opencode/AGENTS.md and the top-level
# "instructions" array in opencode.jsonc into every agent's context. There
# is no documented or undocumented per-agent override (tested 2026-05-26:
# `instructions: []` in agent frontmatter ignored; `--pure` only disables
# external plugins). For council use the user's interactive opencode rules
# would contaminate the neutral voice -- the model's reasoning trace
# explicitly cited "smart caveman style from the rules" in a smoke test.
# We sandbox HOME to a pristine dir containing only our own council agent
# definition; opencode falls back to defaults and authenticates via
# OPENROUTER_API_KEY since the sandbox has no DB-stored OAuth account.
#
# Tool gating uses the documented `permission:` frontmatter (each tool name
# mapped to allow/deny). opencode is a READ-ONLY council peer: read/glob/list
# are allowed so it can ground its reasoning in the caller's repo (--dir
# points there), while edit/write/bash and the rest stay denied -- a
# read-only posture like codex (-s read-only) and gemini (plan mode). `read`
# uses the object form to KEEP opencode's shipped default of denying *.env /
# *.env.* -- a flat `read: allow` scalar would replace that default and let
# repo secrets reach OpenRouter (codex review P1, 2026-06-02). grep is DENIED
# outright: its permission matches the search PATTERN (not a file path) and
# scans the whole tree, so .env contents cannot be path-excluded for grep the
# way they are for read -- a planted .env canary leaked via grep until we
# denied it. The leading `"*": deny` is a catch-all for tools we didn't
# enumerate; the read-only allows sit below it and win (specific overrides the
# wildcard, verified against opencode 1.15.13). The undocumented `tools:` key
# is treated as non-enforcing options, so enforcement rides on `permission:`.
#
# `doom_loop: allow` is NOT a tool grant -- it disables opencode's loop-breaker.
# That guard (opencode processor.ts, DOOM_LOOP_THRESHOLD=3) fires when the last
# 3 tool calls are byte-identical -- same tool, same input. Kimi K2.6
# intermittently gets stuck re-issuing an identical call: it receives a
# completed tool result and re-requests the same read anyway, and at 3 in a row
# the guard trips (verified against opencode source, 2026-06-08). It is
# INTERMITTENT -- most runs make all-distinct calls and never trip it. The
# guard's permission defaults to `ask`, and under headless `opencode run`
# (stdin=DEVNULL) an "ask" is unanswerable -- opencode then raises a fatal
# UnknownError and the run exits 1 with empty output. `deny` aborts the same way
# (tested); only `allow` lets the run finish (the model breaks out of the repeat
# on its own). The council's 900s AGENT_TIMEOUT_S (council.py) is a wall-clock
# cap on every production call routed through run_council (CLI and MCP); the
# adapter's OPENCODE_IDLE_TIMEOUT_S idle cap (below) additionally kills a
# stalled stream -- including a doom-loop that somehow spun without emitting --
# and also bounds a direct OpenCodeAgent.run() that bypasses run_council. Do
# NOT "harden" doom_loop back to deny.
COUNCIL_AGENT_MD = """\
---
description: Council member that reasons and may read repo files read-only.
mode: primary
permission:
  "*": deny
  read:
    "*": allow
    "*.env": deny
    "*.env.*": deny
    "*.env.example": allow
  glob: allow
  list: allow
  bash: deny
  grep: deny
  edit: deny
  write: deny
  webfetch: deny
  task: deny
  todowrite: deny
  websearch: deny
  lsp: deny
  skill: deny
  doom_loop: allow
---

You are a council member responding to a single prompt. Ground your answer in
the repository: use the Read tool to open any file the prompt refers to (and
glob/list to locate files by name) before you answer.

The ONLY tools available to you are Read, glob, and list. You do not have
grep, bash, web, edit, or write. Do not attempt them, and do not treat their
absence as an error -- reach the same goal by reading files directly.

Never invent or guess file contents, paths, headings, or code. If a file you
need cannot be read (e.g. a denied .env), say so plainly and reason only from
what you actually read.

You are running non-interactively: no one can answer follow-up questions. If
the prompt is ambiguous or refers to something you cannot find in the repo,
state your assumption explicitly and give your best substantive answer -- do
not ask clarifying questions or wait for input.

Respond with plain text or markdown only. Beyond Read/glob/list, emit no tool
calls, function calls, or execution markup. Keep your response focused and
substantive.

Always finish by writing your full answer as plain text. Reading is only to
inform that answer -- do NOT end your turn on a tool call, and never reply with
only whitespace. Once you have read what you need, stop reading and write the
response.
"""

# V4 Flash -- the throughput-tier DeepSeek model (same family/provider as the
# heavier V4 Pro), markedly cheaper and ~1.75x faster. A blinded 3-judge eval
# found it statistically indistinguishable from V4 Pro on long-form ideation,
# and its grounding edge did not survive multiplicity correction -- so the
# opencode seat runs Flash on EVERY phase (see make_opencode_agent below).
FLASH_MODEL = "openrouter/deepseek/deepseek-v4-flash"

# The opencode seat's default model. Flash everywhere (S, 2026-07-05): q-review
# was the bulk of the seat's OpenRouter spend and the Pro grounding edge above
# was too small/non-robust to justify the cost. V4 Pro is still reachable per
# run via CODE_QUORUM_OPENCODE_MODEL; the sandbox pins throughput-sorted routing
# (see _build_opencode_config) to keep a heavy override's first-content gap low.
DEFAULT_MODEL = FLASH_MODEL

# Model for opencode's auxiliary calls -- session-title, summarize, classify --
# NOT the council answer. When `small_model` is unset, opencode auto-selects one
# by matching the provider's model ids against its embedded regex
# /\b(nano|flash|lite|mini|haiku|small|fast)\b/; in the OpenRouter-only sandbox
# that lands on Claude Haiku 4.5, billing an unintended Anthropic call on the
# OpenRouter key once per council run (confirmed via OpenRouter activity
# 2026-06-22: App=OpenCode, anthropic/claude-haiku-4.5, 1254->11 tokens, served
# through Amazon Bedrock as OpenRouter's upstream). Pinning it keeps all council
# spend on one known, cheap model for billing consistency and tracking.
DEFAULT_SMALL_MODEL = FLASH_MODEL

# Idle (no-output) ceiling for a single opencode run. opencode's --format json
# stdout emits events only for tool calls, step boundaries, and final text --
# NOT the reasoning tokens a model streams in between. So a reasoning-heavy
# model (e.g. deepseek-v4-pro) looks idle to us for as long as it "thinks"
# before its next visible action. Measured first-content gaps: ~85s on Kimi,
# but 220-324s when OpenRouter "lowest latency" account routing parked V4 Pro
# on a throughput-poor backend (Parasail) -- the SSE bytes flowed the whole
# time (max 3s inter-chunk gap), so this was a FALSE stall, not a hung stream.
# We pin throughput-sorted routing in the sandbox opencode.json (see
# _build_opencode_config) to keep that gap well under ~85s; 240s is comfortable
# margin above it while still catching a genuinely hung stream. This is an IDLE
# timeout, not a wall-clock cap: a slow run that keeps streaming is never
# killed. The council's 900s AGENT_TIMEOUT_S (council.py) is the wall-clock
# backstop for run_council paths; this idle cap also bounds a direct
# OpenCodeAgent.run() that bypasses run_council.
OPENCODE_IDLE_TIMEOUT_S = 240.0

# Inter-chunk ceiling handed to opencode itself, in MILLISECONDS (opencode's unit
# for this key; ours above are seconds). OPENCODE_IDLE_TIMEOUT_S cannot detect a
# dropped SSE stream promptly when the connection stays established and nothing
# raises. In that state, the idle timer waits the full 240s before killing the
# process with rc 124, which is not retried, and the round loses the seat.
#
# opencode ships the right guard and leaves it OFF: `wrapSSE` is only armed when a
# provider sets `chunkTimeout`, there is no default on the openai-compatible path
# (which OpenRouter uses), and Bun's own fetch timeout is disabled -- so after
# headers arrive, a dead stream blocks forever. Upstream anomalyco/opencode#37580
# (open, 2026-07-18) documents exactly this and names setting chunkTimeout as the
# verified workaround. It also fixes something we cannot: opencode's internal retry
# layers are error-driven, so a silent stall never triggers them -- making the
# stream throw is what lets opencode recover on its own before we ever see it.
#
# 90s: the seat's measured worst-case inter-CHUNK gap is ~3s even during a false
# "stall" (raw SSE bytes keep flowing while a reasoning model thinks -- that gap is
# in our stdout-event view, not in the byte stream), so this is ~30x headroom
# against a false abort while cutting a genuinely dead stream from 240s to 90s.
# Kept well under OPENCODE_IDLE_TIMEOUT_S on purpose: opencode should be the one
# to notice and report, with our idle timer as the backstop it used to be.
OPENROUTER_CHUNK_TIMEOUT_MS = 90_000

# Returncode surfaced when a run is killed for idle stall. 124 is the
# conventional "command timed out" code (GNU timeout), distinct from 2
# (no API key) and 127 (binary missing) used elsewhere in this adapter.
OPENCODE_IDLE_TIMEOUT_RC = 124

# Returncode surfaced when opencode exits cleanly (rc 0) but produced no answer
# text -- the model ended its turn after reasoning/tool calls with only
# whitespace (observed on TraceMark 2026-06-13: 811 output tokens across 31 tool
# calls + 11 reasoning blocks, but the text output was ten " " fragments). This
# used to render as a silent blank and get the agent dropped from later rounds
# invisibly; surfacing a distinct non-zero code keeps the dropout VISIBLE.
OPENCODE_NO_OUTPUT_RC = 125

# An empty/stub completion (rc125) is primarily opencode's `run --format json`
# stdout-drop bug (#31435): the model DID write a full answer, but opencode broke
# its output loop on session-idle before draining the final text/step_finish
# events, so stdout loses it. The PRIMARY fix is DB recovery (_recover_text_from_db
# in _attempt), which pulls the persisted answer back out of opencode.db. This
# retry is a SECONDARY fallback for the residual cases recovery cannot heal -- e.g.
# opencode crashed before persisting the turn, or the db was unreadable. The
# council drops a 125 from later rounds, so the seat silently benches itself;
# surfacing it (and recovering or retrying) keeps the seat in the deliberation. We
# retry ONLY empty output: an idle stall (124) already burned its full window, and
# a real error (no key 2 / missing binary 127 / model or doom-loop 1) will not fix
# itself on retry. 1 retry = 2 attempts total.
OPENCODE_EMPTY_OUTPUT_RETRIES = 1

_SANDBOX_HOME = Path.home() / ".cache" / "code-quorum" / "opencode-sandbox"

# Where a failed/empty opencode run dumps its raw stdout/stderr for diagnosis.
# opencode/openrouter intermittently returns an empty turn (returncode 0, no
# captured text) -- the council then renders a silent blank and drops the agent
# from later rounds, so the failure is invisible at the orchestrator. We discard
# the raw stream today, which is the one artifact that explains WHY the text was
# empty (a text event in an unexpected 1.16 shape, an unparsed error event, or a
# genuinely empty completion). Captures land here, one file per failed run.
_DEBUG_DIR = Path.home() / ".cache" / "code-quorum" / "opencode-debug"
OPENCODE_DEBUG_MAX_FILES = 20

# Opt-out knob for the capture. On by default so the next intermittent failure
# is recorded in normal use; set CODE_QUORUM_OPENCODE_DEBUG=0 to disable.
_DEBUG_ENV_VAR = "CODE_QUORUM_OPENCODE_DEBUG"

# npm's content-addressed cache (.npm/_cacache) under the sandbox HOME never
# garbage-collects on its own: every opencode version bump re-resolves deps and
# orphans the prior version's tarballs, so the cache grows monotonically (~2 GB
# over a month observed, ~96% of it unreferenced orphans). When it crosses this
# size we run `npm cache verify`, which drops content no longer referenced by
# the index -- reclaiming the orphans while keeping the live dependency set warm
# (so the next run doesn't re-download it). Size-gated so the common path pays
# only a cheap directory stat, not an npm subprocess.
_NPM_CACHE_PRUNE_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MiB


def _resolve_openrouter_key(environ: dict[str, str] | None = None) -> str | None:
    """Resolve OPENROUTER_API_KEY from the process environment. Returns None
    if it's absent or empty."""
    env = os.environ if environ is None else environ
    value = env.get("OPENROUTER_API_KEY", "").strip()
    return value or None


_OPENROUTER_PREFIX = "openrouter/"


# Per-model OpenRouter provider routing. Default is throughput-sorted: it keeps
# the council off backends that win OpenRouter's "lowest latency" (first-token)
# ranking but complete slowly -- Parasail served deepseek-v4-pro at 394s total /
# 324s to first content, vs 56s on Fireworks.
#
# V4 Flash is pinned to an explicit order instead. This is an operational
# preference for backends observed to complete quickly, NOT an established
# causal fix -- the original rationale was audited on 2026-08-27 and did not
# survive:
#
#   - The claim that the throughput sort put every turn on SiliconFlow is
#     FALSE. OpenRouter's own activity ledger shows the seat was already
#     71% Novita over the window and 73-100% Novita every day from 08-20;
#     SiliconFlow was a 0-137 req/day minority. The pin can therefore only
#     affect that minority -- it does not move the median turn.
#   - The "SiliconFlow thinks ~20x longer" figure came from a truncated
#     measurement (the probe capped max_tokens, SiliconFlow stopped at the
#     cap, and a floor was compared against other backends' complete runs).
#     It is withdrawn.
#   - Within-day paired comparison, which holds workload roughly constant,
#     finds Novita vs SiliconFlow indistinguishable on reasoning volume
#     (5/10 days, sign p=1.00). The one effect that survives is Parasail
#     reasoning ~40% less than Novita (18/20 days, p=0.0004).
#
# What does hold: reasoning-token volume correlates with step latency in the
# seat's own history, and a controlled probe (2026-08-27, n=5/backend on one
# fixed prompt) found Parasail reasoning ~12x less than Novita with complete
# separation (median 3,173 vs 39,921, exact rank-sum p=0.008) and finishing
# in 54-121s against 318-432s -- agreeing in direction with the ledger. That
# probe is exploratory-tier and did not measure review QUALITY, so the order
# below is left as-is rather than reordered on effort alone. The analysis,
# experiment register, and probe data now live in the private `sdewell/seat-eval`
# repository. Fallbacks stay on so an unavailable backend degrades to the next
# rather than failing the seat.
_PROVIDER_ORDER: dict[str, list[str]] = {
    "deepseek/deepseek-v4-flash": ["novita", "parasail", "siliconflow"],
}


def _provider_routing(model_id: str) -> dict:
    order = _PROVIDER_ORDER.get(model_id)
    if order is None:
        return {"sort": "throughput"}
    return {"order": list(order), "allow_fallbacks": True}


def _build_opencode_config(model: str) -> dict | None:
    """Build the sandbox opencode.json for an OpenRouter model. Two pins:

    1. Provider routing (_provider_routing): an explicit backend order for
       models we have measured, throughput-sorted routing otherwise. opencode
       forwards `options.provider` verbatim as OpenRouter's provider-routing
       object (verified end-to-end). The model id is keyed without the
       "openrouter/" prefix (opencode's per-provider id).
    2. `small_model` (DEFAULT_SMALL_MODEL) so opencode's auxiliary title/
       summarize/classify calls bill a known cheap model instead of the Haiku
       its regex auto-selects in this OpenRouter-only sandbox (see the constant).

    Returns None for non-OpenRouter models -- there is no routing to pin."""
    if not model.startswith(_OPENROUTER_PREFIX):
        return None
    model_id = model[len(_OPENROUTER_PREFIX) :]
    return {
        "$schema": "https://opencode.ai/config.json",
        "small_model": DEFAULT_SMALL_MODEL,
        "provider": {
            "openrouter": {
                "options": {"chunkTimeout": OPENROUTER_CHUNK_TIMEOUT_MS},
                "models": {
                    model_id: {"options": {"provider": _provider_routing(model_id)}}
                },
            }
        },
    }


def _ensure_sandbox_home(
    sandbox: Path = _SANDBOX_HOME, model: str | None = None
) -> Path:
    """Create the sandbox HOME with only our council.md agent. Wipes any
    pre-existing config root so a stale AGENTS.md, opencode.jsonc, or
    legacy-layout agent .md from a prior version of this code cannot
    persist and leak back into opencode's auto-load. opencode 1.15
    scans BOTH `$HOME/.config/opencode/` and the legacy `$HOME/.opencode/`
    layout. Leaves data/cache/state alone (those are runtime artifacts
    opencode writes inside the sandbox). When `model` is an OpenRouter model,
    also writes opencode.json pinning provider routing (see
    _build_opencode_config); model-less callers (test/utility harnesses) get
    council.md only."""
    _ensure_private_dir(sandbox, parents=True)
    config_dir = sandbox / ".config" / "opencode"
    legacy_dir = sandbox / ".opencode"
    for stale in (config_dir, legacy_dir):
        if stale.exists():
            shutil.rmtree(stale)
    agents_dir = config_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "council.md").write_text(COUNCIL_AGENT_MD, encoding="utf-8")
    config = _build_opencode_config(model) if model is not None else None
    if config is not None:
        (config_dir / "opencode.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
    return sandbox


def _dir_size_bytes(path: Path) -> int:
    """Total size in bytes of every file under `path` (0 if absent)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                # A file vanishing mid-walk (a concurrent prune/install) just
                # drops from the tally -- never fatal for a size estimate.
                continue
    return total


def _npm_cache_needs_prune(
    sandbox: Path, threshold: int = _NPM_CACHE_PRUNE_THRESHOLD_BYTES
) -> bool:
    """True when the sandbox's npm cache (.npm) exceeds `threshold` bytes."""
    npm_cache = sandbox / ".npm"
    if not npm_cache.is_dir():
        return False
    return _dir_size_bytes(npm_cache) > threshold


async def _prune_npm_cache(
    sandbox: Path,
    *,
    threshold: int = _NPM_CACHE_PRUNE_THRESHOLD_BYTES,
    binary: str = "npm",
) -> bool:
    """Garbage-collect the sandbox npm cache once it exceeds `threshold`.

    Runs `npm cache verify` against the sandbox's OWN cache dir via --cache
    (never the user's global ~/.npm), which drops content blobs no longer
    referenced by the index. Best-effort: any failure (npm absent, verify
    error) is logged and swallowed so a prune can never break a council run.
    Returns True iff verify was invoked (see _NPM_CACHE_PRUNE_THRESHOLD_BYTES
    for why this is needed at all)."""
    if not _npm_cache_needs_prune(sandbox, threshold):
        return False
    npm_cache = sandbox / ".npm"
    before = _dir_size_bytes(npm_cache)
    env = allowlisted_seat_subprocess_env()
    env["HOME"] = str(sandbox)
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "cache",
            "verify",
            "--cache",
            str(npm_cache),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except asyncio.CancelledError:
        raise
    except OSError as exc:
        logger.warning("opencode npm-cache prune skipped (npm unavailable): %s", exc)
        return False
    reclaimed = before - _dir_size_bytes(npm_cache)
    if proc.returncode != 0:
        logger.warning(
            "opencode npm-cache prune: `npm cache verify` exited %s; "
            "cache may be unchanged",
            proc.returncode,
        )
    logger.info(
        "pruned opencode npm cache: reclaimed %.0f MiB (%.0f -> %.0f MiB)",
        reclaimed / 1024 / 1024,
        before / 1024 / 1024,
        (before - reclaimed) / 1024 / 1024,
    )
    return True


def _build_subprocess_env(
    sandbox_home: Path, api_key: str, base: dict[str, str] | None = None
) -> dict[str, str]:
    """Construct the minimal opencode environment and pin its private HOME."""
    env = allowlisted_seat_subprocess_env(base=base)
    env["HOME"] = str(sandbox_home)
    env["OPENROUTER_API_KEY"] = api_key
    env.update(_PROJECT_DISABLE_ENV)
    return env


def _extract_text_and_error(stdout: bytes) -> tuple[str, str]:
    """Concatenate every 'text' event into output; collect every 'error'
    event into the error string. opencode emits failures (auth, model,
    agent) as type=error JSON events on stdout, not stderr."""
    text_chunks: list[str] = []
    error_chunks: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        raw_part = event.get("part")
        part = raw_part if isinstance(raw_part, dict) else {}
        if kind == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_chunks.append(text)
        elif kind == "error":
            # opencode error shape varies (top-level "error", nested part,
            # structured {name,message}). Pull whichever string we find.
            raw = (
                event.get("error")
                or event.get("message")
                or part.get("error")
                or part.get("message")
            )
            if isinstance(raw, str):
                error_chunks.append(raw)
            elif isinstance(raw, dict):
                # opencode serializes a NamedError as {name, data:{message}}.
                # The actionable text lives in data.message; the top-level
                # `name` is just the class (e.g. "UnknownError"), which on its
                # own is undebuggable. Prefer the first NON-EMPTY STRING among
                # nested data.message, a top-level message, then the name -- a
                # non-string or blank data.message must not suppress a usable
                # message/name (codex review, 2026-06-08).
                data = raw.get("data")
                nested = data.get("message") if isinstance(data, dict) else None
                msg = next(
                    (
                        c
                        for c in (nested, raw.get("message"), raw.get("name"))
                        if isinstance(c, str) and c.strip()
                    ),
                    None,
                )
                error_chunks.append(msg if msg else json.dumps(raw))
    return "".join(text_chunks).strip(), "\n".join(error_chunks).strip()


# opencode 1.16 persists every turn to this SQLite store under its data home
# ($HOME/.local/share/opencode). It superseded the older flat-file layout.
_OPENCODE_DB_REL = Path(".local") / "share" / "opencode" / "opencode.db"


def _session_id_from_stdout(stdout: bytes) -> str | None:
    """The sessionID opencode stamps on every `--format json` event. A run that
    drops its text/step_finish events (opencode #31435) still emits step_start,
    so this is a reliable handle on the just-run session for DB recovery."""
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        sid = event.get("sessionID")
        if isinstance(sid, str) and sid:
            return sid
    return None


def _recover_text_from_db(sandbox_home: Path, session_id: str | None) -> str | None:
    """Recover the assistant's final answer from opencode's SQLite store.

    `opencode run --format json` can break its output loop on session-idle before
    draining the final text/step_finish SSE parts, so stdout loses the answer even
    though opencode persisted it (upstream #31435 / #28955). The full answer lives
    in opencode.db; pull the assistant message's text parts for this session.

    Best-effort and fail-closed: returns None on a missing session id, an absent
    or locked db, schema drift, or whitespace-only text -- the caller keeps its
    stdout-derived result whenever this returns None."""
    if not session_id:
        return None
    db_path = sandbox_home / _OPENCODE_DB_REL
    if not db_path.is_file():
        return None
    try:
        # Read-write (not mode=ro): opencode.db is WAL-mode, and a read-only
        # connection cannot create the -shm index or replay a hot journal, so
        # mode=ro silently fails in exactly the post-crash cases recovery exists
        # for. The sandbox db is private and we only SELECT, so RW is safe and
        # lets SQLite set up WAL / roll back a hot journal. Passing the Path
        # directly also sidesteps file: URI percent-encoding of the path.
        conn = sqlite3.connect(db_path, timeout=2.0)
        try:
            # p.rowid (insertion order) breaks ties on equal time_created so
            # multi-part answers concatenate in opencode's write order; p.id is a
            # ULID-ish TEXT key whose alpha order is not its insertion order.
            rows = conn.execute(
                "SELECT json_extract(p.data, '$.text') "
                "FROM part p JOIN message m ON p.message_id = m.id "
                "WHERE p.session_id = ? "
                "AND json_extract(p.data, '$.type') = 'text' "
                "AND json_extract(m.data, '$.role') = 'assistant' "
                "ORDER BY p.time_created, p.rowid",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return None
    text = "".join(r[0] for r in rows if isinstance(r[0], str)).strip()
    return text or None


def _debug_enabled(environ: dict[str, str] | None = None) -> bool:
    """Whether to write a raw-stream capture on a failed/empty run. On unless
    CODE_QUORUM_OPENCODE_DEBUG is explicitly set to a falsey value."""
    env = os.environ if environ is None else environ
    val = env.get(_DEBUG_ENV_VAR, "1").strip().lower()
    return val not in ("0", "false", "no", "off", "")


def _capture_reason(*, text: str, returncode: int, idle_timed_out: bool) -> str | None:
    """Classify a run as capture-worthy. Returns the reason tag, or None when
    the run succeeded (non-empty text, clean exit) and needs no diagnosis. Empty
    text is checked before the generic non-zero case so a surfaced empty turn
    (returncode OPENCODE_NO_OUTPUT_RC) keeps its informative 'empty_output' tag
    rather than the catch-all 'nonzero_exit'."""
    if idle_timed_out:
        return "idle_timeout"
    if not text:
        return "empty_output"
    if returncode != 0:
        return "nonzero_exit"
    return None


def _maintain_debug_dir(debug_dir: Path, *, create: bool) -> bool:
    """Repair permissions and retain only the newest capture window."""
    if not create and not debug_dir.exists() and not debug_dir.is_symlink():
        return True
    try:
        _ensure_private_dir(debug_dir, parents=True)
        captures = sorted(
            debug_dir.glob("opencode-*.txt"),
            key=lambda candidate: candidate.stat(follow_symlinks=False).st_mtime,
        )
        for stale in captures[:-OPENCODE_DEBUG_MAX_FILES]:
            stale.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("opencode debug maintenance skipped: %s", exc)
        return False
    return True


def _maybe_capture_debug(
    *,
    cmd: list[str],
    model: str,
    cwd: str,
    duration_s: float,
    returncode: int,
    idle_timed_out: bool,
    stdout: bytes,
    stderr: bytes,
    text: str,
    json_error: str,
    proc_pid: int,
    debug_dir: Path = _DEBUG_DIR,
    environ: dict[str, str] | None = None,
    stamp: str | None = None,
) -> Path | None:
    """Write a raw-stream capture iff the run is capture-worthy and the knob is
    on. Returns the file path written, or None. Instrumentation must never break
    a run, so every failure here is swallowed.

    The capture is a self-contained JSON header of run metadata followed by
    the verbatim stdout/stderr. The raw streams are the point -- they reveal
    whether the empty result hid a text event in an unexpected shape, an
    error event we did not parse, or a genuinely empty completion."""
    reason = _capture_reason(
        text=text, returncode=returncode, idle_timed_out=idle_timed_out
    )
    if reason is None or not _debug_enabled(environ):
        return None
    try:
        if not _maintain_debug_dir(debug_dir, create=True):
            return None
        if stamp is None:
            stamp = f"{time.strftime('%Y%m%dT%H%M%S')}-{proc_pid}"
        path = debug_dir / f"opencode-{reason}-{stamp}.txt"
        header = {
            "reason": reason,
            "model": model,
            "cwd": cwd,
            "duration_s": round(duration_s, 1),
            "returncode": returncode,
            "idle_timed_out": idle_timed_out,
            # build_command always places the full prompt last. Preserve the
            # operational argv without writing private plans/diffs a second
            # time in the metadata header.
            "cmd": [*cmd[:-1], "<prompt omitted>"] if cmd else [],
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "extracted_text_len": len(text),
            "json_error": json_error,
        }
        content = (
            json.dumps(header, indent=2)
            + "\n\n===== RAW STDOUT =====\n"
            + stdout.decode("utf-8", errors="replace")
            + "\n===== RAW STDERR =====\n"
            + stderr.decode("utf-8", errors="replace")
            + "\n"
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        _maintain_debug_dir(debug_dir, create=True)
        logger.warning("opencode %s — raw capture written to %s", reason, path)
        return path
    except Exception:
        return None


class OpenCodeAgent(Agent):
    name = "opencode"
    default_role = "neutral"

    def __init__(
        self,
        binary: str = "opencode",
        model: str = DEFAULT_MODEL,
        model_source: str = "shipped",
    ):
        self.binary = binary
        self.model = model
        # Set by make_opencode_agent from resolve_model's (model, source); a
        # direct construction (tests, other callers) defaults to "shipped".
        # Threaded through so a failed run can point at `quorum setup-models`
        # only when the model actually came from a recorded choice (see
        # run()) -- mirrors the gemini seat's model_source precedent.
        self.model_source = model_source

    def build_command(self, prompt: str, cwd: str) -> list[str]:
        # Point --dir at the caller's repo so the read-only council agent can
        # open the files a task references (READMEs, plans, source). A
        # project-local opencode.json(c) / .opencode/agents/council.md could
        # otherwise shadow our sandboxed council -- that is blocked by
        # OPENCODE_DISABLE_PROJECT_CONFIG in the subprocess env (see
        # _build_subprocess_env), which leaves file reads working. Global
        # config contamination (AGENTS.md, instructions[], skills) is handled
        # separately by the sandboxed HOME, since opencode discovers all of
        # those under $HOME.
        return [
            self.binary,
            "run",
            "--agent",
            "council",
            "-m",
            self.model,
            "--format",
            "json",
            "--log-level",
            "WARN",
            "--dir",
            cwd,
            prompt,
        ]

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        # cwd is the caller's repo: --dir points opencode there so the
        # read-only council agent can read the files a task references.
        # HOME stays pinned to the sandbox (see _build_subprocess_env) so the
        # user's global opencode config (AGENTS.md, instructions[], skills)
        # cannot contaminate the council voice or bloat the prompt.
        start = time.monotonic()
        api_key = _resolve_openrouter_key()
        if api_key is None:
            return AgentResult(
                agent=self.name,
                output="",
                error=(
                    "OPENROUTER_API_KEY not found. Export it in your shell "
                    "environment (get a key at https://openrouter.ai)."
                ),
                returncode=2,
                duration_s=time.monotonic() - start,
                unavailable_reason="authentication",
            )
        sandbox_home = _ensure_sandbox_home(model=self.model)
        _maintain_debug_dir(_DEBUG_DIR, create=False)
        # Bound the sandbox npm cache before invoking opencode. Size-gated, so
        # this is a cheap stat on the common path and only fires occasionally.
        await _prune_npm_cache(sandbox_home)
        env = _build_subprocess_env(sandbox_home, api_key)
        cmd = self.build_command(prompt=prompt, cwd=cwd)
        # Retry only an empty completion (OPENCODE_NO_OUTPUT_RC) -- a transient
        # upstream flake that, left alone, drops opencode from later council
        # rounds. See OPENCODE_EMPTY_OUTPUT_RETRIES for why only this case.
        attempts = 1 + OPENCODE_EMPTY_OUTPUT_RETRIES
        result: AgentResult | None = None
        for i in range(attempts):
            result = await self._attempt(
                cmd=cmd, env=env, sandbox_home=sandbox_home, cwd=cwd
            )
            if result.returncode != OPENCODE_NO_OUTPUT_RC or i == attempts - 1:
                break
            logger.warning(
                "opencode produced no answer (attempt %d/%d); retrying -- likely "
                "a transient empty completion from the model/provider.",
                i + 1,
                attempts,
            )
        assert result is not None
        # Report cumulative wall time across attempts, not just the final try,
        # so a retried run reflects its true cost at the orchestrator.
        result.duration_s = time.monotonic() - start
        # Gate the hint away from failure classes that are definitely not
        # model problems: rc 2 (no API key) never reaches here (early return
        # above); rc 124 (idle stall) and rc 127 (binary missing, from
        # _attempt) are excluded explicitly.
        if result.returncode not in (0, OPENCODE_IDLE_TIMEOUT_RC, 127):
            result.error = append_recorded_source_hint(
                result.error, self.model, self.model_source
            )
        return result

    async def _attempt(
        self, *, cmd: list[str], env: dict[str, str], sandbox_home: Path, cwd: str
    ) -> AgentResult:
        """One opencode invocation -> AgentResult. run() calls this up to
        1 + OPENCODE_EMPTY_OUTPUT_RETRIES times, retrying only an empty
        completion. Self-timed; run() overrides duration_s with the cumulative
        wall time so a retried run reports total cost, not just the last try."""
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(sandbox_home),
                env=env,
                # opencode reads non-TTY stdin even when argv carries
                # the prompt. Under `quorum-mcp`, fd 0 is the live
                # JSON-RPC stream from Claude Code, so inheriting it
                # would either block until the MCP pipe closes or
                # consume MCP messages and corrupt the server session.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            return AgentResult(
                agent=self.name,
                output="",
                error=f"{self.binary} not found on PATH",
                returncode=127,
                duration_s=time.monotonic() - start,
                unavailable_reason="not installed",
            )
        stdout_b, stderr_b, idle_timed_out = await communicate_lines_or_kill(
            proc, pgid=proc.pid, idle_timeout=OPENCODE_IDLE_TIMEOUT_S
        )
        text, json_error = _extract_text_and_error(stdout_b)
        stderr_text = stderr_b.decode("utf-8", errors="replace").strip()
        # Prefer parsed JSON error events (richer, structured) over stderr;
        # fall back to stderr when no error event was emitted.
        error = json_error or stderr_text
        if idle_timed_out:
            # The stream stalled and we killed the process group. Surface the
            # stall explicitly -- otherwise the partial output reads as a
            # truncated-but-successful answer. Keep any last diagnostic we did
            # capture appended so a stall with a preceding error stays visible.
            stall = (
                f"opencode produced no output for {OPENCODE_IDLE_TIMEOUT_S:.0f}s "
                "(stalled stream); the process group was terminated."
            )
            error = f"{stall} Last diagnostic: {error}" if error else stall
            returncode = OPENCODE_IDLE_TIMEOUT_RC
            unavailable_reason = "timeout"
        else:
            unavailable_reason = ""
            # opencode can emit a `type: "error"` event AND exit 0. format_rounds
            # only surfaces `error` for nonzero returncode, so synthesize one
            # here to keep the diagnostic visible.
            returncode = proc.returncode or 0
            if returncode == 0 and not json_error:
                # opencode #31435: `run --format json` can break its output loop
                # on session-idle before draining the final text/step_finish
                # events, so stdout loses the answer (only step_start survives)
                # even though opencode persisted the full turn to opencode.db.
                # Prefer the DB copy when it carries more text than stdout did --
                # this heals both the empty (rc125) and truncated-stub cases.
                # Best-effort: a None recovery leaves the stdout result intact.
                # Off the event loop: _recover_text_from_db does blocking SQLite
                # I/O and can wait up to its 2s timeout under lock contention,
                # which would stall every other council seat running concurrently.
                recovered = await asyncio.to_thread(
                    _recover_text_from_db,
                    sandbox_home,
                    _session_id_from_stdout(stdout_b),
                )
                if recovered is not None and len(recovered) > len(text):
                    text = recovered
            if returncode == 0 and json_error:
                returncode = 1
            elif returncode == 0 and not text:
                # Clean exit but no answer: the model ended its turn after
                # reasoning/tool calls with only whitespace. A blank "success"
                # silently vanishes at the orchestrator and gets the agent
                # dropped from later rounds -- surface it as a visible failure.
                returncode = OPENCODE_NO_OUTPUT_RC
                unavailable_reason = "no output"
                error = (
                    "opencode produced no answer: the model ended its turn "
                    "after reasoning/tool calls without writing a final "
                    "response (empty/whitespace-only output)."
                )
        if (
            not unavailable_reason
            and returncode != 0
            and has_usage_limit_diagnostic(error)
        ):
            unavailable_reason = "usage limit"
        duration_s = time.monotonic() - start
        # Dump the raw stream for any failed/empty run. An empty turn (returncode
        # 0, no text) renders as a silent blank at the orchestrator and gets the
        # agent dropped from later rounds; the raw stdout is the only artifact
        # that explains why. Best-effort -- never lets instrumentation break run.
        _maybe_capture_debug(
            cmd=cmd,
            model=self.model,
            cwd=cwd,
            duration_s=duration_s,
            returncode=returncode,
            idle_timed_out=idle_timed_out,
            stdout=stdout_b,
            stderr=stderr_b,
            text=text,
            json_error=json_error,
            proc_pid=proc.pid,
            debug_dir=_DEBUG_DIR,
        )
        return AgentResult(
            agent=self.name,
            output=text,
            error=error,
            returncode=returncode,
            duration_s=duration_s,
            unavailable_reason=unavailable_reason,
        )


# Global override: when set, this model wins over DEFAULT_MODEL (ad-hoc A/B
# without code changes), mirroring CODE_QUORUM_GEMINI_MODEL for the gemini seat.
MODEL_ENV_VAR = "CODE_QUORUM_OPENCODE_MODEL"


def _opencode_version(binary: str = "opencode") -> str | None:
    """Best-effort installed opencode version, or None if it can't be
    determined. Module seam so tests override it without spawning a real
    subprocess; delegates to model_config.probe_cli_version, the probe+cache
    shared by every seat's CLI version check."""
    return probe_cli_version(binary)


def make_opencode_agent() -> OpenCodeAgent:
    """Build the opencode seat. Runs the Flash DEFAULT_MODEL on every council
    phase -- brainstorm (q-brainstorm AND the q-skystorm dream stage), plan,
    validate, AND review -- because a blinded eval found Flash level with Pro
    while cheaper and ~1.75x faster (see DEFAULT_MODEL). Model resolves
    per_run(none here) > env
    CODE_QUORUM_OPENCODE_MODEL > a recorded `quorum setup-models` choice >
    DEFAULT_MODEL. When a recorded choice exists, compares the installed
    opencode version against the version recorded at choice time and warns
    once (never fails) on drift -- the recorded model still runs."""
    model, source = resolve_model(
        "opencode",
        per_run=None,
        # Empty/whitespace-only counts as unset -- matches the pre-Task-2
        # convention (os.environ.get(...).strip() or DEFAULT) so an env var
        # exported empty still falls through to the recorded/shipped model.
        env_value=os.environ.get(MODEL_ENV_VAR, "").strip() or None,
        shipped=DEFAULT_MODEL,
    )
    recorded = recorded_choice("opencode")
    if recorded is not None:
        recorded_version = recorded.get("cli_version")
        if recorded_version is not None:
            installed = _opencode_version()
            if installed is not None and installed != recorded_version:
                warn_version_drift("opencode", installed, recorded_version)
    return OpenCodeAgent(model=model, model_source=source)
