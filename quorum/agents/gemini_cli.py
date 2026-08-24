import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from ..model_config import (
    append_recorded_source_hint,
    probe_cli_version,
    warn_version_drift,
)
from .base import (
    Agent,
    AgentResult,
    _atomic_write_json,
    _ensure_private_dir,
    allowlisted_seat_subprocess_env,
    communicate_or_kill,
)
from .gemini import _non_hidden_workspace

logger = logging.getLogger(__name__)

# Council default model. What we PASS to --model is the DISPLAY NAME, not the
# slug, and that is deliberate: agy's slug resolver SILENTLY MIS-ROUTES this one
# model. Verified live on the installed agy 1.1.8 (2026-07-29) by reading the
# backend label out of agy's own diagnostic log:
#   --model gemini-3.1-pro-high     -> label="Gemini 3.6 Flash (High)"   WRONG
#   --model "Gemini 3.1 Pro (High)" -> label="Gemini 3.1 Pro (High)"     correct
# The wrong label is the FIRST line of `agy models`, i.e. the slug fails to
# resolve and falls back to the catalog default -- while exiting 0 with a normal
# answer, so the council silently reported Flash work as Pro work. It is NOT all
# slugs (gemini-3.5-flash-high and claude-opus-4-6-thinking each resolve to their
# own label), which is exactly why the membership guard below could not see it:
# the Pro slug is still listed by `agy models` and still accepted by --model --
# it just runs a different engine than it names.
#
# The --model contract has now changed shape four times (API slug -> display name
# -> slug id -> slug id that lies), so the seat pins BOTH forms and guards each
# against the thing only it can prove: DEFAULT_MODEL is what routes correctly and
# is what we invoke, DEFAULT_MODEL_CATALOG_SLUG is how `agy models` lists the
# same model. agy accepts either form; it just resolves them differently.
#
# The effort tier is part of the name ("(High)" / -high), so model and reasoning
# budget are chosen together. The council uses 3.1 Pro High -- the strongest the
# AI Pro plan unlocks. Pro spends quota faster than Flash across a multi-round,
# multi-seat council, so dial back per run with
# CODE_QUORUM_GEMINI_MODEL="gemini-3.5-flash-high" (that slug is verified to
# route correctly) or per-instance via GeminiCliAgent(model=...).
DEFAULT_MODEL = "Gemini 3.1 Pro (High)"
DEFAULT_MODEL_CATALOG_SLUG = "gemini-3.1-pro-high"

# Pinning only the seat default would leave every OTHER way in on the bug, and
# those are the likely ways in: `agy models` prints slugs, so a caller reaching
# for CODE_QUORUM_GEMINI_MODEL or the per-run `gemini_model` ask will normally
# hand us the mis-routing form -- and it was this seat's own default until
# 2026-07-29, so it is in shell profiles and notes. The seat therefore rewrites
# the known-bad slug to the form that routes, wherever it arrives from.
# Everything else passes through verbatim. The slugs this seat exposes in its
# own defaults or guidance have been checked against agy's log; the rest of the
# catalog is UNVERIFIED, not known-good, so if another id turns out to mis-route
# the remedy is one more entry here plus a canary case -- never a blanket rewrite.
# Passing an unrecognized id through is also deliberate: agy rejecting a typo by
# name beats this map quietly turning it into some other model.
MODEL_ROUTING_ALIASES = {
    DEFAULT_MODEL_CATALOG_SLUG: DEFAULT_MODEL,
    "gemini-3.5-flash-high": "Gemini 3.5 Flash (High)",
}


def canonical_model(model: str) -> str:
    """The form of `model` that agy routes to the engine it names.

    Trims first: the per-run `gemini_model` ask reaches the seat unstripped, and
    an exact-match alias lookup would let `"gemini-3.1-pro-high "` walk straight
    past the rewrite and back into the mis-routing bug. Trimming here rather than
    at each caller means no future entry point can reopen that path."""
    model = model.strip()
    return MODEL_ROUTING_ALIASES.get(model.lower(), model)


# agy's config + credential locations (it stores data under ~/.gemini, reusing
# the legacy Gemini CLI store; "antigravity-cli/settings.json" is the CLI config).
AGY_SETTINGS_PATH = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
AGY_OAUTH_PATH = Path.home() / ".gemini" / "oauth_creds.json"

# Return codes. Distinct so a silent empty/failed turn reads as failed, not OK.
GEMINI_CLI_NO_BINARY_RC = 127
GEMINI_CLI_NOT_LOGGED_IN_RC = 2
GEMINI_CLI_NOT_READONLY_RC = 3
GEMINI_CLI_TIMEOUT_RC = 124  # exceeded print-timeout + grace (GNU-timeout convention)
GEMINI_CLI_NO_OUTPUT_RC = 125
GEMINI_CLI_NO_SANDBOX_RC = 4  # cannot enforce read-only -> refuse (never run bare)
GEMINI_CLI_RECORDED_MODEL_MISMATCH_RC = 5  # routing mismatch on a recorded choice

# --- Read-only enforcement: a macOS seatbelt sandbox -------------------------
#
# WHY THE KERNEL AND NOT agy's OWN CONFIG. The permissions block was inert in
# --print mode on agy 1.1.2, verified live on 2026-07-14:
# settings.json loads correctly (the log prints the full Deny list), then
# `ApplyProjectPermissionGrants: no grants for project "CLI Project", cleared
# project permissions` WIPES it -- 1.0.12 made project configs outrank the global
# settings, and the CLI's default project carries no grants. toolPermission=
# always-proceed then auto-approves everything. agy wrote a file AND ran a shell
# command (`echo PWNED > /tmp/...`) straight through the deny list. Neither
# toolPermission=strict, nor --sandbox, nor --mode plan, nor putting the grants in
# the project config blocks it. There is no agy-side configuration that restores
# read-only, so the seat stops asking agy to police itself.
#
# The seatbelt is inherited by every child process, so a shell agy spawns is bound
# by it too. Denying TMPDIR writes is load-bearing, not incidental: agy's
# run_command tool stages a temp cert file before executing, so the write-deny
# kills shell execution outright.
#
# This is a STRICTER contract than agy's denylist ever promised: no writes, no
# shell, and no reads inside $HOME beyond the workspace under review plus the three
# paths agy itself needs to function -- ~/.gemini (its own state + oauth), and, in
# ~/Library, its Playwright driver cache and the login keychain it reads for its
# credential (see build_sandbox_profile for why each). Everything else in $HOME --
# ~/.ssh, ~/.aws, ~/Library/Mail, Messages, other apps' tokens, every other repo --
# is fenced off.
SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Read-only tool sets, expressed in agy's permission-grant vocabulary. agy gates
# tools by a small set of coarse ACTIONS, not per tool: verified against v1.0.13 and
# re-verified against v1.1.2 (write a superset deny list, run agy, read settings.json
# back), the ONLY actions its grant store recognizes are read_file, write_file,
# command, execute_url, read_url, unsandboxed, mcp -- any other key is silently
# dropped. So `write_file` covers every file-write tool
# (write_to_file/replace_file_content/...), `command` covers every shell tool
# (run_command/...), and the old per-tool keys (edit_file/create_file/delete_file/
# run_command/execute_command) were phantoms agy stripped -- their removal loses no
# coverage and stops a self-heal rewrite firing on every council run.
#
# SAFETY CONTRACT -- denylist completeness. With toolPermission=always-proceed,
# agy auto-approves any tool NOT covered by a `deny` action, so REQUIRED_DENY must
# name every gateable non-read action. It does: these 6 are the complete set of
# non-read actions agy gates -- v1.0.13, re-verified unchanged on v1.1.2 (read_file
# is the lone allow). The verifier requires exactly this set, so a hand-edit
# removing any one fails closed.
#
# NOT denied, because agy exposes NO grant action for it: web_search (the model's
# built-in Google Vertex grounding -- result URLs are vertexaisearch.cloud.google
# .com/grounding-api-redirect/...); v1.1.2 still drops it from the grant store as
# an unknown action, so there is nothing to deny. Verified v1.0.13: neither a deny
# rule, nor toolPermission=strict, nor --sandbox blocks it -- it runs server-side
# as part of the model turn. It is NOT a new exfiltration boundary (the full
# prompt already goes to Google; a model-generated grounding query stays within
# Google), and it is steered -- not enforced -- by COUNCIL_CLI_PREAMBLE toward
# provided context + the `--research` digest.
_READ_ONLY_ALLOW = ("read_file(*)",)
REQUIRED_DENY = (
    "write_file(*)",
    "command(*)",
    "execute_url(*)",
    "read_url(*)",
    "unsandboxed(*)",
    "mcp(*)",
)

# The agy version the seat's read-only enforcement was last LIVE-VERIFIED
# against. "Verified" means the gated live canaries passed on that installed agy:
# write refused, shell refused, URL fetch does not fire, config survives agy's
# normalization -- ground truth from the filesystem and the wire, never model
# prose. The version guard (check_seat_verified_version) warns when the installed
# agy differs, because the seatbelt's allow-roots and the URL-tool fail-closed
# behavior are empirical facts about a specific agy (it could relocate its driver
# cache, or start honouring URL fetches in --print mode) -- a new version must be
# re-canaried, not assumed.
#
# TO BUMP: run `bash scripts/verify-agy-seat.sh` -- it runs the live canaries
# (CODE_QUORUM_AGY_E2E=1 uv run pytest -m live tests/test_gemini_cli.py) and
# rewrites this constant only when every canary passes. Spends a few real agy
# turns on the default (Pro) model, so it never runs automatically.
#
# That suite now also includes the ROUTING canary (see AGY_EXPECTED_BACKEND_LABEL)
# -- as of 1.1.7/1.1.8 a model id can be listed, accepted, and still reach a
# different engine, so "the seat still works" is no longer evidence that it works
# on the model it claims. Verifying an upgrade means checking both what agy
# refuses (seatbelt) and what it silently substitutes (routing).
#
# 1.1.8 verified 2026-07-29: all 8 live canaries green -- write refused, shell
# exec refused, URL fetch does not fire, denylist survives agy's normalization,
# repo read works, and both seat models route to the engine they name. Note what
# the routing canary proves and does not: 1.1.8 is the version that MIS-ROUTES
# the 3.1-Pro slug, so this pin means "the display-name form we invoke is
# verified correct HERE", not "1.1.8's resolver is sound".
#
# 1.1.12 verified 2026-08-13: all 8 live canaries green (via verify-agy-seat.sh).
# 1.1.13 verified 2026-08-14: all 10 live canaries green (via
# verify-agy-seat.sh), including Pro, the documented Flash override, and the
# Claude quota fallback routing to the engines they name.
SEAT_VERIFIED_AGY_VERSION = "1.1.13"


def _sbpl(path: str | Path) -> str:
    """Quote a path as an SBPL string literal (the profile is Scheme-ish)."""
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def check_sandbox_cwd(cwd: str, home: Path | None = None) -> str | None:
    """Reject a cwd whose recursive read grant would contain all of HOME."""
    h = (home or Path.home()).resolve()
    cwd_p = Path(cwd).resolve()
    try:
        h.relative_to(cwd_p)
    except ValueError:
        return None
    return (
        f"cwd {cwd_p} is the home directory or one of its ancestors; granting "
        "that recursive read would expose private home-directory files. Run "
        "from the repository directory under an allowed project root."
    )


def build_sandbox_profile(
    cwd: str,
    workspace: str | None = None,
    home: Path | None = None,
    binary_path: str | None = None,
) -> str:
    """The seatbelt profile that enforces the council's read-only contract.

    The recursive read allow is anchored on `cwd`, NEVER on the widened --add-dir
    workspace. _non_hidden_workspace widens a hidden cwd to its nearest non-hidden
    ancestor (~/.claude -> ~), so a `(subpath <workspace>)` allow would land right
    after `(deny file-read* (subpath <HOME>))` naming the same path -- the allow
    wins and the entire $HOME fence silently evaporates, re-exposing ~/.ssh and
    every other repo.

    The workspace root still gets a LITERAL allow (plus each intervening ancestor
    of cwd): agy stats and enumerates the root it was handed, and without that it
    cannot resolve files in cwd at all -- verified live, it silently confabulates
    file contents rather than erroring. `literal` grants the directory ENTRY only,
    not its children, so $HOME's other contents stay unreadable.

    Paths are RESOLVED: seatbelt matches the real path, so a cwd reached via a
    symlink (/tmp -> /private/tmp, /var -> /private/var) must be named by its
    target or the allow rule silently never matches.
    """
    h = (home or Path.home()).resolve()
    cwd_p = Path(cwd).resolve()
    cwd_error = check_sandbox_cwd(cwd, h)
    if cwd_error is not None:
        raise ValueError(cwd_error)

    read_roots = [cwd_p]
    # agy reaches into ~/Library for exactly TWO things, both verified with fs_usage
    # on 1.1.2: (1) its bundled Playwright driver under Caches/ms-playwright-go, and
    # (2) login.keychain-db -- a `security` child process (which inherits this
    # sandbox) reads that file to fetch agy's stored credential; deny it and agy
    # reports "not signed in" and falls back to interactive browser OAuth. Grant
    # ONLY these two, so the rest of ~/Library --
    # Mail, Messages, Cookies, Safari, and every other app's Application Support
    # tokens -- stays fenced with the rest of $HOME. A blanket (subpath ~/Library)
    # would re-expose all of that for no functional gain. If agy relocates its driver
    # cache in a future version, the seat fails CLOSED (a visible sandbox denial at
    # spawn), never silently -- re-probe with fs_usage and adjust these two roots.
    read_roots += [
        h / ".gemini",
        h / "Library" / "Caches" / "ms-playwright-go",
    ]

    # Dedupe, keep order stable so the profile is deterministic (and diffable).
    seen: dict[str, None] = {}
    for root in read_roots:
        seen.setdefault(str(root), None)
    allow_reads = "\n".join(f"  (subpath {_sbpl(r)})" for r in seen)

    # Literal-only allows: the login keychain file plus its directory entries,
    # and (when needed) the widened workspace root and ancestors between it and
    # cwd. Directory literals permit traversal but not their children.
    ws = Path(workspace).resolve() if workspace else cwd_p
    keychains = h / "Library" / "Keychains"
    literals: list[Path] = [
        h / "Library",
        keychains,
        keychains / "login.keychain-db",
    ]
    if binary_path:
        binary = Path(binary_path).resolve()
        try:
            binary.relative_to(h)
        except ValueError:
            pass  # outside HOME remains readable through allow-default
        else:
            literals.append(binary)
            node = binary.parent
            while True:
                literals.append(node)
                if node == h:
                    break
                node = node.parent
    if ws != cwd_p and ws in cwd_p.parents:
        node = cwd_p.parent
        while True:
            literals.append(node)
            if node == ws:
                break
            node = node.parent
    allow_literals = "".join(
        f"(allow file-read* (literal {_sbpl(p)}))\n" for p in literals
    )

    # Agy persists council transcripts and ephemeral coordination state, but it
    # must not be able to rewrite its executable/configuration surface. These
    # paths are the state written by observed council runs; settings.json is
    # repaired by Code Quorum before the sandbox starts and needs no write grant.
    agy_root = h / ".gemini" / "antigravity-cli"
    writable_state = (
        "brain",
        "cache",
        "conversations",
        "crashes",
        "implicit",
        "log",
        "presence",
        "scratch",
    )
    allow_state_writes = "\n".join(
        f"  (subpath {_sbpl(agy_root / name)})" for name in writable_state
    )
    oauth_token = agy_root / "antigravity-oauth-token"

    return f"""(version 1)
;; Generated by code-quorum. agy's permission config is defence in depth;
;; read-only does not depend on its version-specific behavior.
(allow default)

;; 1. No writes anywhere -- except agy's observed runtime state.
;;    This also kills agy's shell tool, which must stage a temp file to run.
(deny file-write*)
(allow file-write*
{allow_state_writes}
  (literal {_sbpl(oauth_token)}))
(allow file-write*
  (literal "/dev/null")
  (literal "/dev/tty")
  (subpath "/dev/fd"))

;; 2. No reads inside $HOME except the directory under review and agy's own
;;    files. Fences off ~/.ssh, ~/.aws, and every other repo on the machine.
;;    System paths (certs, dylibs, tzdata) stay readable via (allow default).
(deny file-read* (subpath {_sbpl(h)}))
(allow file-read*
{allow_reads})
;;    Literal entries only (not directory contents): the login keychain plus any
;;    widened --add-dir ancestors needed to traverse down to cwd.
{allow_literals}"""


def _platform() -> str:
    """Module seam for the host platform. Tests override THIS, not `sys.platform`
    -- patching the real sys module mutates interpreter-wide state that anything
    running concurrently (pytest-xdist workers, libraries caching it at import)
    would also see."""
    return sys.platform


def check_sandbox_available() -> str | None:
    """Return None when the read-only sandbox can be enforced, else the reason.

    The seat REFUSES on any non-None result. agy's own config failed to enforce
    this boundary on 1.1.2 and is not trusted as the guarantee, so an unsandboxed
    run has no council read-only contract at all."""
    platform = _platform()
    if platform != "darwin":
        return (
            f"the agy seat enforces read-only with a macOS seatbelt sandbox, and "
            f"{platform!r} has no equivalent wired up. agy's own permission "
            "config is not trusted as the boundary (1.1.2 wrote files and ran "
            "shell commands through its deny list), so the seat "
            "refuses rather than run unsandboxed. Use CODE_QUORUM_GEMINI_BACKEND=sdk."
        )
    if not Path(SANDBOX_EXEC).exists():
        return (
            f"{SANDBOX_EXEC} not found, so the read-only contract cannot be "
            "enforced. The seat refuses rather than run agy unsandboxed."
        )
    try:
        proc = subprocess.run(
            [SANDBOX_EXEC, "-p", "(version 1)\n(allow default)\n", "/usr/bin/true"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (
            f"{SANDBOX_EXEC} could not apply a smoke-test profile: {exc}. "
            "Refusing to run agy unsandboxed."
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
        return (
            f"{SANDBOX_EXEC} could not apply a smoke-test profile: {detail}. "
            "Refusing to run agy unsandboxed."
        )
    return None


def verify_read_only_config(settings_path: Path | None = None) -> str | None:
    """Validate agy's defence-in-depth permission config.

    This checks the expected file shape; it does not certify runtime enforcement.
    Seatbelt remains the read-only boundary. A malformed or surprising config
    returns a refusal string instead of raising.
    """
    path = settings_path or AGY_SETTINGS_PATH
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            f"agy config not found at {path}. Run `agy` once to sign in, then "
            f"run the one-time setup (`quorum setup-agy`)."
        )
    except OSError as exc:
        return f"agy config at {path} cannot be read: {exc}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"agy config at {path} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return f"agy config at {path} must be a JSON object."

    # Google documents false (and omission, whose default is false) as keeping
    # plan access while disabling fallback billing from personal AI credits:
    # https://antigravity.google/docs/cli/credits
    if data.get("useG1Credits", False) is not False:
        return (
            "agy config `useG1Credits` must be false "
            "(prevents spending personal AI credits after plan quota is exhausted)."
        )
    if data.get("toolPermission") != "always-proceed":
        return "agy config `toolPermission` must be `always-proceed`."
    if data.get("allowNonWorkspaceAccess") is True:
        return "agy config has unsafe `allowNonWorkspaceAccess: true`."

    perms = data.get("permissions", {})
    if not isinstance(perms, dict):
        return "agy config `permissions` must be a JSON object."
    # Fail CLOSED on ANY malformed permission list -- not just `deny`. A malformed
    # sibling (`allow`/`ask` as a dict, or any list holding a non-string) could make
    # agy reject the whole permissions block and fall back to always-proceed,
    # nullifying the deny rules. So the safety gate refuses on any non-list[str]
    # permission list, mirroring the builder. (This also makes the set() below
    # crash-safe: deny is now confirmed hashable strings. Never raises.)
    for key in ("allow", "deny", "ask"):
        if key in perms and not _is_str_list(perms[key]):
            return (
                f"agy config `permissions.{key}` must be a JSON array of strings; "
                "refusing to certify read-only. Fix it or run `quorum setup-agy`."
            )
    deny = set(perms.get("deny", []))
    missing = [rule for rule in REQUIRED_DENY if rule not in deny]
    if missing:
        return (
            "agy config is not read-only: permissions.deny is missing "
            + ", ".join(missing)
            + ". Run `quorum setup-agy`."
        )
    return None


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _merge_permissions(existing_perms: dict) -> dict:
    """Union our required allow/deny into existing permissions, preserving the
    user's extra rules and any other keys (e.g. `ask`). Our rules come first;
    duplicates are dropped order-preserving. Every permission list we touch or
    preserve (`allow`/`deny`/`ask`) must be a JSON array of strings -- a malformed
    shape is refused (ValueError) rather than written back unvalidated, and a
    non-string entry is rejected before it can crash dict.fromkeys (unhashable)."""
    for key in ("allow", "deny", "ask"):
        if key in existing_perms and not _is_str_list(existing_perms[key]):
            raise ValueError(
                f"agy settings `permissions.{key}` must be a JSON array of strings"
            )
    merged = dict(existing_perms)  # preserves `ask` + unrelated keys
    for key, required in (("allow", _READ_ONLY_ALLOW), ("deny", REQUIRED_DENY)):
        current = existing_perms.get(key, [])
        merged[key] = list(dict.fromkeys([*required, *current]))
    return merged


def build_read_only_settings(existing: dict) -> dict:
    """Union the read-only block onto `existing`, preserving unrelated keys and
    the user's existing permissions. Pure + idempotent. Raises ValueError on a
    structurally surprising `permissions` shape (we refuse rather than silently
    discard). The untyped-JSON boundary (non-dict top level) is validated by the
    caller, `write_read_only_settings`."""
    existing_perms = existing.get("permissions", {})
    if not isinstance(existing_perms, dict):
        raise ValueError("agy settings `permissions` must be a JSON object")
    merged = dict(existing)
    merged["useG1Credits"] = False
    merged["allowNonWorkspaceAccess"] = False
    merged["toolPermission"] = "always-proceed"
    merged["permissions"] = _merge_permissions(existing_perms)
    return merged


def write_read_only_settings(settings_path: Path | None = None) -> Path:
    """One-time setup: union the read-only block into the existing settings and
    write it back atomically. REFUSES (raises ValueError) on a malformed existing
    file rather than overwriting it -- agy and the user also own this file."""
    path = settings_path or AGY_SETTINGS_PATH
    existing: dict = {}
    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read agy settings at {path}: {exc}") from exc
        if raw.strip():  # empty file is fine; malformed is not
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"agy settings at {path} is not valid JSON ({exc}); fix or "
                    "remove it -- refusing to overwrite."
                ) from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"agy settings at {path} must be a JSON object.")
            existing = loaded
    # settings.json's parent may not exist yet (a fresh agy install that has
    # never run) -- _atomic_write_json (shared, agents/base.py) does not mkdir
    # on the caller's behalf, unlike this function's old private copy did.
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, build_read_only_settings(existing))
    return path


def ensure_read_only_config(settings_path: Path | None = None) -> str | None:
    """Verify the defence-in-depth config and attempt one merge-preserving repair.

    Returns None when the expected file shape is present (possibly after a
    self-heal), else a human-readable refusal reason. Seatbelt remains the
    read-only boundary.

    Why self-heal: agy owns this global file and can reset it on upgrade (observed
    on the v1.0.13 bump -- it dropped the read-only block). Rather than make the
    seat go dark until the user re-runs `setup-agy`, repair it in place. Safety is
    preserved three ways: (1) the repair adds the read-only block without loosening
    it and disables personal-credit fallback; (2) `write_read_only_settings`
    REFUSES (raises) on a malformed/unrecognized config, so a file we don't
    understand is never clobbered -- we keep refusing instead; and (3) the seat
    proceeds only on the post-repair re-verify, never on an assumed result."""
    err = verify_read_only_config(settings_path)
    if err is None:
        return None
    try:
        path = write_read_only_settings(settings_path)
    except (ValueError, OSError) as exc:
        return f"{err} (auto-repair could not run: {exc})"
    logger.warning(
        "agy council config was missing/incomplete and was self-healed "
        "(read-only rules enforced and personal-credit fallback disabled; "
        "merge-preserving rewrite at %s). Original reason: %s",
        path,
        err,
    )
    return verify_read_only_config(settings_path)


def _agy_version(binary: str = "agy") -> str | None:
    """Best-effort installed agy version (e.g. `1.0.13`), or None if it can't be
    determined. Module seam so tests override it without spawning agy;
    delegates to model_config.probe_cli_version, the probe+cache shared by
    every seat's CLI version check (still spawns `agy --version` at most
    once, even though both check_seat_verified_version's own-pin check and
    the recorded-choice drift check below (run()) call this)."""
    return probe_cli_version(binary)


def check_seat_verified_version(installed: str | None = None) -> str | None:
    """Return a warning when the installed agy differs from the version the live
    seatbelt canaries were last run against, else None. The seatbelt's allow-roots
    and the URL tool's fail-closed behavior are empirical facts about a specific
    agy version, so a new version must be re-canaried. Soft by design: the
    sandbox gate (check_sandbox_available) still governs safety, and an
    undeterminable version returns None (no false alarm). Pass `installed`
    explicitly in tests; unset queries agy."""
    version = installed if installed is not None else _agy_version()
    if version is None or version == SEAT_VERIFIED_AGY_VERSION:
        return None
    return (
        f"agy is v{version} but this seat's read-only sandbox was last "
        f"live-verified against v{SEAT_VERIFIED_AGY_VERSION}. The seatbelt "
        f"still applies and fails closed; whether it still contains everything "
        f"this agy can attempt is unproven. Re-verify when ready -- it spends "
        f"real Gemini quota, so it only runs when you choose: "
        f"bash scripts/verify-agy-seat.sh (runs the live canaries, then bumps "
        f"SEAT_VERIFIED_AGY_VERSION on all-pass)."
    )


# Grace added to agy's own --print-timeout for the inner wait_for backstop. The
# layering is: agy self-terminates at --print-timeout (480s); if it wedges and
# ignores that, our inner wait_for fires at 480+30=510s; the council's hard
# AGENT_TIMEOUT_S (600s) is the final cap. Kept under 600 so the inner backstop
# wins for the default timeout.
GEMINI_CLI_TIMEOUT_GRACE_S = 30.0
AGY_AUTH_CHECK_TIMEOUT_S = 20.0

# --- Transient startup-crash retry -------------------------------------------
#
# WHAT THIS ACTUALLY COVERS. The signature below is not specific to one upstream
# bug: it matches ANY fast, output-less death that agy reports only as its
# generic terminal error. Two distinct causes are confirmed live, and treating
# them as one class is deliberate -- both are Google-side transients that clear
# on their own, and agy hides both behind the same stderr line, so the seat
# cannot tell them apart at the point it must decide whether to retry. (The
# distinguishing evidence lives in agy's own log; the quota reflex reads it to
# choose a DIFFERENT model, while this retry just runs the same one again.)
#
#   1. 429 startup race (agy 1.1.5+, observed 2026-07-22, cli log 13:59:51).
#      When its initial loadCodeAssist/fetchAvailableModels call 429s
#      (RESOURCE_EXHAUSTED), agy still constructs the conversation executor from
#      that poisoned fetch -- "failed to construct executor: neither PlanModel
#      nor RequestedModel specified" -- and exits 1 in a few seconds, EVEN
#      THOUGH its own refetch succeeded moments earlier in the same process.
#      Adjacent seat runs in the same window drew the same 429s on background
#      refreshes and completed fine.
#   2. Transient backing-revision 404 (agy 1.1.7, observed 2026-07-29 11:50,
#      cli log code-quorum-tblcb05y.log). A VALID model name resolved correctly
#      to label="Gemini 3.1 Pro (High)", then the turn died at ~4.3s on
#      "NOT_FOUND (code 404): Model not found:
#      models/gemini-v4p1m-rev25-gemdelta-jetski-fast" -- Google's catalog
#      briefly pointing a live label at an internal revision it would not serve.
#      The identical invocation succeeded later, so the 404 was downstream of
#      name resolution and nothing about the seat's config could have prevented
#      it. All four legs hold (rc 1, no stdout, marker, 4.3s < the window), so
#      the existing retry already recovers it -- verified, not assumed.
#
# What must NOT match, and does not: a deterministic bad --model. agy exits 1 on
# an unknown id with "invalid model selection ... is not recognized as a known
# model" and no generic marker (re-verified 2026-07-29 on 1.1.8), so the marker
# leg alone keeps a config error from burning a second turn on every run.
#
# The signature is deliberately narrow -- all four legs must hold:
#   rc 1        -- the generic-failure mapping; auth (2) / timeout (124) / no
#                  output (125) have their own meanings and are never retried.
#   no stdout   -- the run died before producing anything worth keeping.
#   marker      -- agy's generic terminal error, verbatim on stderr. Distinct
#                  failures ("invalid model selection", auth text) don't match.
#   fast fail   -- died within the window below, i.e. at startup. The same
#                  message after a long run means a mid-turn death (e.g. quota
#                  truly exhausted); retrying would double a long wall-clock
#                  loss for nothing. Measured on the agy attempt ALONE --
#                  run()'s preflight (notably the `agy --version` probe with
#                  its own 10s subprocess timeout) is excluded, so a slow
#                  probe cannot eat the window.
# Worst case added by the retry: one crash-window + delay + one full attempt
# (~10+5+510s), still under the council's 600s AGENT_TIMEOUT_S cap.
AGY_TRANSIENT_CRASH_MARKER = "Agent execution terminated due to error"
AGY_STARTUP_CRASH_WINDOW_S = 10.0
AGY_STARTUP_RETRY_DELAY_S = 5.0

# A valid keyring login can still fail before a model turn when its short-lived
# access token needs refreshing and the token endpoint has a transient network
# failure. agy then falls into interactive OAuth, which a headless council seat
# cannot complete. Retry the same attempt once; the failed leg consumed no model
# quota because authentication never completed.
AGY_AUTH_REFRESH_RETRY_DELAY_S = 3.0
_AGY_AUTH_REFRESH_TRANSPORT_MARKER = "token refresh failed due to network error"

# --- Quota reflex: Gemini -> Claude fallback ---------------------------------
#
# gemini-3.1-pro drains the AI Pro plan's Gemini quota fastest of anything the
# council runs. When Google throttles or the bucket is empty, the seat dies
# with a generic terminal error on stderr in older releases; agy 1.1.19 instead
# surfaces "Individual quota reached" directly. In both shapes, the authoritative
# RESOURCE_EXHAUSTED (code 429) evidence is in agy's own cli log. As a result,
# every attempt gets a private --log-file under agy's own log dir (inside the
# seatbelt's ~/.gemini write-allow; TMPDIR
# is write-DENIED by design, so the log cannot live there), and a failed
# attempt is classified by reading that log back. On quota evidence the seat
# re-runs ONCE on AGY_QUOTA_FALLBACK_MODEL -- Claude Opus 4.6 Thinking, a
# separate quota bucket the same AI Pro plan surfaces through agy (>=1.1.7
# listing) -- and labels the answer with the swap so the council transcript
# never changes engines silently. Pinned to `agy models` output exactly like
# DEFAULT_MODEL, guarded by the same zero-quota listing test.
#
# The elapsed gate is a BUDGET, deliberately anchored at run() entry -- unlike
# the startup-crash window, which measures the attempt alone. The
# council kills the seat at AGENT_TIMEOUT_S (600s), and a fallback attempt
# needs print-timeout + grace (510s). Reflexing only under 60s of total
# elapsed keeps the worst case ~570s. The 60s therefore INCLUDES preflight
# (notably the `agy --version` probe, up to 10s) -- that is the point: the
# budget bounds wall clock against the council cap, which is also anchored at
# run() entry. Observed quota deaths are fast (~3s); the gate excludes only
# long mid-turn deaths, where a second full run would blow the cap anyway.
AGY_QUOTA_FALLBACK_MODEL = "claude-opus-4-6-thinking"
AGY_QUOTA_LOG_MARKERS = ("RESOURCE_EXHAUSTED", "(code 429)")
AGY_EXPLICIT_QUOTA_ERROR_MARKER = "Individual quota reached"
AGY_QUOTA_REFLEX_MAX_ELAPSED_S = 60.0
_AGY_FALLBACK_TEMP_ROOT = Path("/tmp")
_AGY_FALLBACK_TEMP_PREFIX = "code-quorum-agy-fallback-"

# Labels a quota-reflex answer for the council transcript (emit site below).
# setup_models.py imports this to detect the same swap in a probe run -- a
# probe answer that came from the fallback model proves the fallback routes,
# not the candidate, so it must fail the probe rather than record a success.
QUOTA_REFLEX_OUTPUT_PREFIX = "[quota reflex:"

# --- Routing contract: which engine a --model string ACTUALLY reaches ---------
#
# The backend label agy must resolve each model the seat can invoke to. This is
# the guard for the failure class that cost us the 3.1 Pro slug (see
# DEFAULT_MODEL): routing is invisible everywhere a caller would normally look
# -- a mis-routed run answers normally, prints nothing unusual, and exits 0 --
# so `agy models` membership cannot prove the model NAMED is the model RUN. The
# only artifact where the truth appears is this line in agy's own diagnostic
# log, which is why the seat asks for one on every attempt anyway (see
# AGY_QUOTA_FALLBACK_MODEL). test_live_seat_models_route_to_expected_backend
# asserts it; a mismatch means agy's resolver moved under us again.
#
# The claude fallback slug is included because the quota reflex promises the
# transcript an honest engine name at the exact moment the seat is already
# degraded -- a silently mis-routed fallback would make that label a lie too.
# Flash is included because the documented quota-saver slug is normalized to its
# display form; every alias target must be live-proven before that rewrite can be
# called safe. (The original two were verified live on agy 1.1.8, 2026-07-29;
# all three were re-verified on 1.1.13, 2026-08-14.)
# COVERAGE BOUNDARY, stated so the next person does not have to infer it: this
# dict is exactly the models whose routing is PROVEN: the primary, the documented
# Flash override, and the automatic quota fallback. Every other configurable
# model is UNPROVEN -- membership can show that agy accepts it, but only this live
# canary detects a silent substitution. Adding an entry costs a real AI Pro turn
# on every version-verification run, so the line stops at the shipped/documented
# surface rather than covering the whole catalog.
AGY_EXPECTED_BACKEND_LABEL = {
    DEFAULT_MODEL: "Gemini 3.1 Pro (High)",
    "Gemini 3.5 Flash (High)": "Gemini 3.5 Flash (High)",
    AGY_QUOTA_FALLBACK_MODEL: "Claude Opus 4.6 (Thinking)",
}

_AGY_BACKEND_LABEL_RE = re.compile(
    r'Propagating selected model override to backend: label="([^"]+)"'
)

# Per-run --log-file target dir: agy's own log dir. A distinct prefix keeps
# these apart from agy's cli-*.log files; uuid names keep concurrent council
# sessions from colliding.
AGY_RUN_LOG_DIR = AGY_SETTINGS_PATH.parent / "log"
AGY_RUN_LOG_MAX_FILES = 20


def _new_run_log_path() -> Path:
    # The dir is NOT guaranteed: `quorum setup-agy` writes settings.json without
    # ever launching agy, so a signed-in fresh install can lack log/ -- and a
    # missing --log-file dir would fail every seat run.
    _ensure_private_dir(AGY_RUN_LOG_DIR, parents=True)
    path = AGY_RUN_LOG_DIR / f"code-quorum-run-{uuid.uuid4().hex}.log"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    logs = sorted(
        AGY_RUN_LOG_DIR.glob("code-quorum-run-*.log"),
        key=lambda candidate: candidate.stat(follow_symlinks=False).st_mtime,
    )
    for stale in logs[:-AGY_RUN_LOG_MAX_FILES]:
        stale.unlink(missing_ok=True)
    return path


def _log_shows_quota_exhaustion(run_log: Path) -> bool:
    """True when an attempt's agy log carries throttling/quota evidence. A
    missing or unreadable log is NO evidence -- the failure reports as itself
    rather than spending a fallback turn on a guess."""
    try:
        text = run_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in text for marker in AGY_QUOTA_LOG_MARKERS)


def _log_shows_auth_refresh_transport_failure(run_log: Path) -> bool:
    """True only for agy's explicit OAuth-refresh network classification."""
    try:
        text = run_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return (
        _AGY_AUTH_REFRESH_TRANSPORT_MARKER in text
        and "oauth2.googleapis.com/token" in text
    )


def resolved_backend_labels(run_log: Path) -> tuple[str, ...]:
    """Every model label agy propagated to the backend during one attempt, in
    order, read back from that attempt's diagnostic log.

    Returns ALL occurrences rather than one, so a caller can assert the whole
    attempt stayed on a single engine instead of trusting the first decision.
    Empty when the log is missing, unreadable, or the run died before resolving
    a model -- so 'no evidence' can never be mistaken for 'wrong model'."""
    try:
        text = run_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    return tuple(_AGY_BACKEND_LABEL_RE.findall(text))


# Runtime routing check. The version pin below proves routing for the models the
# seat defaults to, ON the agy it was last canaried against. This proves it for
# whatever model was ACTUALLY asked for, on EVERY run, on whatever agy happens to
# be installed -- which is the property that matters when the upstream contract
# keeps changing shape (four times so far, and the last one changed silently).
#
# It is free: the seat already writes a per-run diagnostic log and already parses
# it for quota evidence, so the engine agy really used is on disk either way. No
# extra turn, no quota, no network.
#
# Deliberately TABLE-FREE. A map of model id -> expected label would be one more
# thing to maintain and would only ever cover the models someone remembered to
# add. Instead both naming conventions are reduced to letters and digits, which
# collapses them onto the same key:
#   "gemini-3.1-pro-high"     -> "gemini31prohigh"
#   "Gemini 3.1 Pro (High)"   -> "gemini31prohigh"   agree
#   "claude-opus-4-6-thinking"-> "claudeopus46thinking"
#   "Claude Opus 4.6 (Thinking)" -> "claudeopus46thinking"   agree
# and the bug this repo just chased: asked "gemini31prohigh", served
# "gemini36flashhigh" -- caught, without anyone having described that bug to it.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def routing_key(model_or_label: str) -> str:
    """A comparison key that agy's slug ids and its display names both reduce to,
    so 'did we get the model we asked for?' can be answered without a lookup
    table of every model pair."""
    return _NON_ALNUM_RE.sub("", model_or_label.lower())


def routing_complaint(asked: str, labels: tuple[str, ...]) -> str | None:
    """A human-readable complaint when agy served an engine other than `asked`,
    else None.

    No labels is NOT a complaint: an absent or unparsed log is missing evidence,
    not evidence of substitution (the same rule the quota classifier follows).
    That also keeps a change to agy's log format from turning every run into a
    false alarm -- it degrades to the status quo, which is silence."""
    if not labels:
        return None
    want = routing_key(asked)
    wrong = sorted({label for label in labels if routing_key(label) != want})
    if not wrong:
        return None
    served = ", ".join(repr(label) for label in wrong)
    return (
        f"asked agy for {asked!r} but its own log says the turn was served by "
        f"{served}. agy accepts a model id and can silently resolve it to a "
        f"different engine, so this answer is NOT from the model you pinned."
    )


def _apply_version_warning(result: AgentResult, warning: str | None) -> None:
    """Keep agy verification drift visible across the helper-process boundary."""
    if warning is None:
        return
    tagged = f"[agy verification warning: {warning}]"
    if result.returncode == 0:
        result.output = f"{tagged}\n\n{result.output}"
    else:
        result.error = f"{result.error}\n\n{tagged}" if result.error else tagged


def _with_diagnostic_log(message: str, path: Path | None) -> str:
    if path is None:
        return message
    return f"{message}\nagy diagnostic log: {path}"


def _with_quota_evidence(message: str, *, quota_exhausted: bool) -> str:
    if not quota_exhausted:
        return message
    normalized = message.casefold()
    if "quota" in normalized or "resource_exhausted" in normalized:
        return message
    label = "quota exhaustion (RESOURCE_EXHAUSTED/code 429)"
    return f"{message}\n{label}" if message else label


def _surface_quota_evidence(result: AgentResult, *, quota_exhausted: bool) -> None:
    if not quota_exhausted:
        return
    result.error = _with_quota_evidence(result.error, quota_exhausted=True)
    if result.unavailable_reason in ("", "no output"):
        result.unavailable_reason = "usage limit"


def _should_surface_quota_evidence(result: AgentResult) -> bool:
    """Limit private-log quota classification to agy's execution-failure shape."""
    return result.returncode == 1 and not result.output


def _should_quota_reflex(
    result: AgentResult, elapsed_s: float, run_log: Path, model: str
) -> bool:
    """True when a failed attempt should be re-run once on the Claude fallback
    model. Narrow on purpose, like the transient-retry predicate: auth (rc 2),
    timeout (124), and no-output (125) keep their own meanings; distinct
    errors ("invalid model selection") fail as themselves; agy's legacy generic
    terminal error and 1.1.19's explicit "Individual quota reached" shape are
    accepted only with 429 evidence in the attempt log; and the fallback model
    never bounces onto itself. `elapsed_s` is total run() wall time -- a budget
    bound, not a crash signature (see AGY_QUOTA_FALLBACK_MODEL)."""
    return (
        result.returncode == 1
        and not result.output
        and (
            AGY_TRANSIENT_CRASH_MARKER in result.error
            or AGY_EXPLICIT_QUOTA_ERROR_MARKER in result.error
        )
        and elapsed_s < AGY_QUOTA_REFLEX_MAX_ELAPSED_S
        and model != AGY_QUOTA_FALLBACK_MODEL
        and _log_shows_quota_exhaustion(run_log)
    )


@contextlib.contextmanager
def _quota_fallback_handoff(prompt: str) -> Iterator[tuple[str, Path]]:
    """Stage a full fallback request privately and yield a short agy prompt.

    On agy 1.1.19, a non-trivial prompt selected for the Claude fallback can
    still invoke a Gemini-powered intent summarizer before Claude runs. When the
    Gemini bucket is exhausted, that hook returns 429 and strands the otherwise
    available Claude/GPT quota. A short prompt avoids the hook; the Claude model
    then reads the unchanged request through its normal read_file tool.
    """
    with tempfile.TemporaryDirectory(
        prefix=_AGY_FALLBACK_TEMP_PREFIX,
        dir=_AGY_FALLBACK_TEMP_ROOT,
    ) as raw_dir:
        request_path = Path(raw_dir) / "request.md"
        fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(prompt)
        instruction = (
            f"The full council request is stored at {request_path}. Read that file "
            "completely using your available file-reading tool, then follow the "
            "request exactly. Return the complete requested answer; do not summarize "
            "it or ask questions."
        )
        yield instruction, request_path


def _validated_quota_fallback_dir(path: str) -> str:
    """Return the dedicated fallback directory, rejecting broader read scopes."""
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"fallback request directory is unavailable: {path}") from exc
    root = _AGY_FALLBACK_TEMP_ROOT.resolve()
    if (
        not candidate.is_absolute()
        or not resolved.is_dir()
        or resolved.parent != root
        or not resolved.name.startswith(_AGY_FALLBACK_TEMP_PREFIX)
    ):
        raise ValueError(f"invalid fallback request directory: {path}")
    return str(candidate)


def _is_transient_startup_crash(result: AgentResult, attempt_s: float) -> bool:
    """True when a seat result matches agy's transient startup-crash signature
    (see AGY_TRANSIENT_CRASH_MARKER above). Narrow on purpose: anything that
    doesn't match all four legs fails as itself, without a second agy turn.

    `attempt_s` is the elapsed time of the agy attempt alone, measured at the
    call site -- NOT result.duration_s, which is anchored to run() entry and
    includes preflight."""
    primary_error = result.error.partition("\nagy diagnostic log:")[0].strip()
    normalized_error = primary_error.removeprefix("Error: ").rstrip(".")
    return (
        result.returncode == 1
        and not result.output
        and normalized_error == AGY_TRANSIENT_CRASH_MARKER
        and attempt_s < AGY_STARTUP_CRASH_WINDOW_S
    )


# Prepended to the (already stance/grounding-wrapped) council prompt. agy --print
# has no --system flag, so the council read-only persona rides in the prompt.
# Read-only is enforced by config (the deny block), NOT by this text -- this just
# saves the model a wasted turn attempting denied tools and keeps it answering in
# plain text. Single prepend only: anchoring/repeating it does not strengthen the
# safety contract (the deny config does), and the live filesystem canary is the
# real proof. The web-search sentence is a SOFT steer (not enforcement): agy
# exposes no grant action for web_search/Vertex grounding (see REQUIRED_DENY),
# so it cannot be denied -- the council just prefers the supplied context + the
# `--research` digest, and conceptual prior-art queries over pasting repo code.
COUNCIL_CLI_PREAMBLE = (
    "You are a council member running non-interactively via the Antigravity CLI. "
    "Use your read tools (read_file, list_directory, glob) to open any repository "
    "file the prompt refers to before answering. You are read-only: you have no "
    "write, edit, or shell tools -- do not attempt them, and do not treat their "
    "absence as an error. Prefer the context already supplied and any research "
    "digest in the prompt over live web search; if you do search the web, query "
    "conceptually -- how others have approached this problem -- and never paste "
    "repository code into a search. No one can answer follow-up questions, so "
    "state any assumption explicitly and give your best substantive answer -- do "
    "not ask clarifying questions. Respond in plain text or markdown and always "
    "finish by writing your full answer."
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_AUTHORIZATION_URL_RE = re.compile(
    r"https://accounts\.google\.com/o/oauth2/(?:v\d+/)?auth\?\S+"
)
# agy auth/login failures we map to "not logged in" (rc 2) even if exit != 2.
# Phrases are specific on purpose: a bare "token"/"expired" substring also matched
# capacity errors like "context token limit exceeded" or "trial has expired",
# mislabeling them rc 2 "sign in" and hiding the real fix (smaller prompt / quota).
_AUTH_HINTS = (
    "authentication",
    "not logged in",
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


def _is_logged_in() -> bool:
    """Heuristic login gate (module seam so tests can override it without
    monkeypatching Path.exists globally)."""
    return AGY_OAUTH_PATH.exists()


def _looks_like_auth_error(stderr: str) -> bool:
    low = stderr.lower()
    return any(h in low for h in _AUTH_HINTS)


def _redact_auth_error(stderr: str) -> str:
    return _AUTHORIZATION_URL_RE.sub(
        "[authorization URL omitted; run `agy` interactively to sign in]", stderr
    )


def _has_agy_model_row(stdout: str) -> bool:
    for line in stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and all(field.strip() for field in fields):
            return True
    return False


def _sandbox_gate(cwd: str, start: float, purpose: str) -> AgentResult | None:
    error = check_sandbox_available() or check_sandbox_cwd(cwd)
    if error is None:
        return None
    return AgentResult(
        agent="gemini",
        output="",
        error=f"{purpose} refusing to run: {error}",
        returncode=GEMINI_CLI_NO_SANDBOX_RC,
        duration_s=time.monotonic() - start,
        unavailable_reason="sandbox",
    )


def _resolve_binary(binary: str, start: float) -> tuple[str | None, AgentResult | None]:
    path = shutil.which(binary)
    if path is not None:
        return path, None
    return None, AgentResult(
        agent="gemini",
        output="",
        error=f"{binary} not found on PATH",
        returncode=GEMINI_CLI_NO_BINARY_RC,
        duration_s=time.monotonic() - start,
        unavailable_reason="not installed",
    )


class GeminiCliAgent(Agent):
    """Council Gemini seat backed by the Antigravity CLI (`agy --print`), running
    read-only on the user's AI Pro quota. Same council identity as the SDK seat
    (name `gemini`), different engine; selected via CODE_QUORUM_GEMINI_BACKEND."""

    name = "gemini"
    default_role = "architect"

    def __init__(
        self,
        binary: str = "agy",
        model: str = DEFAULT_MODEL,
        print_timeout: float = 480.0,
        model_source: str = "shipped",
        recorded_cli_version: str | None = None,
    ):
        self.binary = binary
        # Normalized here rather than at the call site so every entry point --
        # the default, CODE_QUORUM_GEMINI_MODEL, the per-run ask, a direct
        # construction -- gets the routing fix (see MODEL_ROUTING_ALIASES).
        self.model = canonical_model(model)
        self.print_timeout = print_timeout
        # Set by make_gemini_agent from resolve_model's (model, source); a
        # direct construction (tests, other callers) defaults to "shipped" --
        # never the "recorded" value that enforces a routing-mismatch failure
        # (see run()). Threaded through so the invariant (a recorded choice
        # fails loud rather than silently substituting an engine) can be
        # enforced without this class knowing about models.toml itself.
        self.model_source = model_source
        # The `cli_version` models.toml recorded at the last `quorum
        # setup-models` run for this seat, or None when no recorded choice
        # exists. Drives the version-drift warning in run() -- absent, no
        # drift check runs at all (see _agy_version's cache for why this
        # never costs a second subprocess).
        self.recorded_cli_version = recorded_cli_version

    def build_command(
        self,
        prompt: str,
        cwd: str,
        sandbox_profile: str,
        log_file: str | None = None,
        model: str | None = None,
        extra_add_dir: str | None = None,
    ) -> list[str]:
        # sandbox_profile is REQUIRED, not optional: agy has no working read-only
        # mode of its own, so an un-sandboxed argv is never a valid thing to build.
        # --add-dir widens to a non-hidden workspace root (permission scope); the
        # process cwd (set in run()) stays the original cwd for relative-path
        # anchoring. `log_file` gives the attempt a private agy log (the quota
        # reflex classifies failures from it); `model` overrides the seat model
        # for the fallback attempt only. `extra_add_dir` exposes only the private
        # staged-request directory used by that fallback.
        workspace = _non_hidden_workspace(cwd)
        full_prompt = f"{COUNCIL_CLI_PREAMBLE}\n\n{prompt}"
        cmd = [
            SANDBOX_EXEC,
            "-f",
            sandbox_profile,
            self.binary,
            "--print",
            full_prompt,
            "--add-dir",
            workspace,
        ]
        if extra_add_dir:
            cmd += ["--add-dir", _validated_quota_fallback_dir(extra_add_dir)]
        cmd += [
            "--model",
            canonical_model(model) if model else self.model,
            "--print-timeout",
            f"{int(self.print_timeout)}s",
        ]
        if log_file:
            cmd += ["--log-file", log_file]
        return cmd

    async def check_auth(self, cwd: str) -> AgentResult:
        """Verify agy's saved login through its zero-model-quota listing path."""
        start = time.monotonic()
        gate = _sandbox_gate(cwd, start, "agy authentication check")
        if gate is not None:
            return gate
        binary_path, gate = _resolve_binary(self.binary, start)
        if gate is not None:
            return gate
        assert binary_path is not None
        profile = build_sandbox_profile(
            cwd=cwd,
            workspace=_non_hidden_workspace(cwd),
            binary_path=binary_path,
        )
        with tempfile.NamedTemporaryFile(
            "w",
            prefix="code-quorum-agy-auth-",
            suffix=".sb",
            delete=False,
            encoding="utf-8",
        ) as fh:
            fh.write(profile)
            profile_path = fh.name
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    SANDBOX_EXEC,
                    "-f",
                    profile_path,
                    self.binary,
                    "models",
                    cwd=cwd,
                    env=allowlisted_seat_subprocess_env(),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError:
                return AgentResult(
                    agent=self.name,
                    output="",
                    error=(
                        f"{SANDBOX_EXEC} disappeared before the authentication "
                        "check; refusing to run agy unsandboxed."
                    ),
                    returncode=GEMINI_CLI_NO_SANDBOX_RC,
                    duration_s=time.monotonic() - start,
                    unavailable_reason="sandbox",
                )
            except OSError as exc:
                return AgentResult(
                    agent=self.name,
                    output="",
                    error=f"failed to start agy authentication check: {exc}",
                    returncode=1,
                    duration_s=time.monotonic() - start,
                    unavailable_reason="preflight",
                )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    communicate_or_kill(proc, pgid=proc.pid),
                    timeout=AGY_AUTH_CHECK_TIMEOUT_S,
                )
            except TimeoutError:
                return AgentResult(
                    agent=self.name,
                    output="",
                    error=(
                        "agy authentication check timed out; run `agy` "
                        "interactively to refresh the saved login"
                    ),
                    returncode=GEMINI_CLI_TIMEOUT_RC,
                    duration_s=time.monotonic() - start,
                    unavailable_reason="timeout",
                )
            stdout = _ANSI_RE.sub(
                "", stdout_b.decode("utf-8", errors="replace")
            ).strip()
            stderr = _ANSI_RE.sub(
                "", stderr_b.decode("utf-8", errors="replace")
            ).strip()
            rc = proc.returncode or 0
            if rc != 0 and _looks_like_auth_error(stderr):
                return AgentResult(
                    agent=self.name,
                    output="",
                    error=f"agy auth error: {_redact_auth_error(stderr)}",
                    returncode=GEMINI_CLI_NOT_LOGGED_IN_RC,
                    duration_s=time.monotonic() - start,
                    unavailable_reason="authentication",
                )
            if rc != 0:
                return AgentResult(
                    agent=self.name,
                    output="",
                    error=stderr or f"agy models exited {rc}",
                    returncode=rc,
                    duration_s=time.monotonic() - start,
                    unavailable_reason="preflight",
                )
            if not _has_agy_model_row(stdout):
                return AgentResult(
                    agent=self.name,
                    output="",
                    error="agy models returned no valid model listing",
                    returncode=GEMINI_CLI_NO_OUTPUT_RC,
                    duration_s=time.monotonic() - start,
                    unavailable_reason="no output",
                )
            return AgentResult(
                agent=self.name,
                output="agy authentication is ready",
                duration_s=time.monotonic() - start,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(profile_path)

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        start = time.monotonic()

        # The sandbox IS the read-only contract -- check it before anything else.
        gate = _sandbox_gate(cwd, start, "agy seat")
        if gate is not None:
            return gate
        # Defence in depth, not the boundary: agy 1.1.2 ignored this config in
        # --print mode. Keep it correct across release-specific behavior and catch
        # a tampered config, while Seatbelt supplies the stable guarantee.
        config_err = ensure_read_only_config()
        if config_err is not None:
            return AgentResult(
                agent=self.name,
                output="",
                error=f"agy seat refusing to run: {config_err}",
                returncode=GEMINI_CLI_NOT_READONLY_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="read-only configuration",
            )
        # Offloaded: check_seat_verified_version spawns `agy --version`, and a
        # blocking subprocess here would stall the other council agents (run() is
        # gathered). Non-fatal -- a warning, never a refusal.
        version_warn = await asyncio.get_running_loop().run_in_executor(
            None, check_seat_verified_version
        )
        if version_warn:
            logger.warning("%s", version_warn)
        # Recorded-choice version drift: only when a recorded choice exists
        # for this seat (see __init__). Reuses _agy_version's cache, so this
        # is never a second real subprocess -- check_seat_verified_version's
        # own call above already populated it.
        if self.recorded_cli_version is not None:
            installed = await asyncio.get_running_loop().run_in_executor(
                None, _agy_version
            )
            if installed is not None and installed != self.recorded_cli_version:
                warn_version_drift("gemini", installed, self.recorded_cli_version)
        if not _is_logged_in():
            result = AgentResult(
                agent=self.name,
                output="",
                error=(
                    f"agy not signed in ({AGY_OAUTH_PATH} missing). Run `agy` once "
                    "to sign in with your AI Pro account."
                ),
                returncode=GEMINI_CLI_NOT_LOGGED_IN_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="authentication",
            )
            _apply_version_warning(result, version_warn)
            return result

        # argv[0] is sandbox-exec now, so a missing agy no longer raises
        # FileNotFoundError at spawn (sandbox-exec itself exists and would exec-fail
        # inside the sandbox, surfacing an opaque rc 1). Resolve the binary here so
        # "agy not installed" still reports its own distinct rc 127.
        binary_path, gate = _resolve_binary(self.binary, start)
        if gate is not None:
            _apply_version_warning(gate, version_warn)
            return gate
        assert binary_path is not None

        # sandbox-exec reads the profile at exec time, so it must be a real file
        # that outlives the spawn. Removed once agy has exited (or failed to).
        profile = build_sandbox_profile(
            cwd=cwd,
            workspace=_non_hidden_workspace(cwd),  # literal-only allow, see builder
            binary_path=binary_path,
        )
        with tempfile.NamedTemporaryFile(
            "w",
            prefix="code-quorum-agy-",
            suffix=".sb",
            delete=False,
            encoding="utf-8",
        ) as fh:
            fh.write(profile)
            profile_path = fh.name
        run_logs: list[Path] = []
        keep_logs: set[Path] = set()

        def _next_log() -> Path | None:
            """Reserve a per-attempt log path, or None when it can't be
            created (folds the old standalone _create_diagnostic_log_path's
            try/except in -- it had exactly this one caller)."""
            try:
                path = _new_run_log_path()
            except OSError as exc:
                logger.warning("could not reserve an agy diagnostic log: %s", exc)
                return None
            run_logs.append(path)
            return path

        try:
            attempt_start = time.monotonic()
            active_log = _next_log()
            result = await self._spawn_and_collect(
                prompt,
                cwd,
                profile_path,
                start,
                log_file=str(active_log) if active_log is not None else None,
            )
            attempt_s = time.monotonic() - attempt_start
            retry_initial_log: Path | None = None
            if _is_transient_startup_crash(result, attempt_s):
                logger.warning(
                    "agy crashed at startup (%.1fs, generic terminal error) -- "
                    "known transient 429-poisoned model-config race; retrying "
                    "once in %.0fs.",
                    attempt_s,
                    AGY_STARTUP_RETRY_DELAY_S,
                )
                await asyncio.sleep(AGY_STARTUP_RETRY_DELAY_S)
                # This straight-line code (no loop) is the sole retry bound.
                # Same `start`: duration_s stays the seat's honest total wall
                # time across both attempts. Fresh log: attempt 1's 429 lines
                # must never classify attempt 2 (stale-marker guard).
                retry_initial_log = active_log
                active_log = _next_log()
                result = await self._spawn_and_collect(
                    prompt,
                    cwd,
                    profile_path,
                    start,
                    log_file=str(active_log) if active_log is not None else None,
                )
                if retry_initial_log is not None and (
                    result.returncode != 0 or active_log is None
                ):
                    keep_logs.add(retry_initial_log)
                    note = (
                        "initial failed-attempt diagnostic log kept at "
                        f"{retry_initial_log}"
                    )
                    if result.returncode == 0:
                        result.output = f"{result.output}\n\n[{note}]"
                    else:
                        result.error = (
                            f"{result.error}\n{note}" if result.error else note
                        )
            auth_retry_initial_log: Path | None = None
            if (
                active_log is not None
                and result.returncode == GEMINI_CLI_NOT_LOGGED_IN_RC
                and not result.output
                and _log_shows_auth_refresh_transport_failure(active_log)
            ):
                logger.warning(
                    "agy OAuth access-token refresh hit a transient network "
                    "failure; retrying once in %.0fs.",
                    AGY_AUTH_REFRESH_RETRY_DELAY_S,
                )
                await asyncio.sleep(AGY_AUTH_REFRESH_RETRY_DELAY_S)
                auth_retry_initial_log = active_log
                active_log = _next_log()
                result = await self._spawn_and_collect(
                    prompt,
                    cwd,
                    profile_path,
                    start,
                    log_file=str(active_log) if active_log is not None else None,
                )
                if auth_retry_initial_log is not None and (
                    result.returncode != 0 or active_log is None
                ):
                    keep_logs.add(auth_retry_initial_log)
                    note = (
                        "initial OAuth-refresh network-failure diagnostic log kept "
                        f"at {auth_retry_initial_log}"
                    )
                    if result.returncode == 0:
                        result.output = f"{result.output}\n\n[{note}]"
                    else:
                        result.error = (
                            f"{result.error}\n{note}" if result.error else note
                        )
            if (
                active_log is not None
                and result.returncode == GEMINI_CLI_NOT_LOGGED_IN_RC
                and _log_shows_auth_refresh_transport_failure(active_log)
            ):
                result.unavailable_reason = "network"
                result.error = (
                    "agy OAuth refresh failed after one retry because the Google "
                    "token endpoint had a network error. " + result.error
                )
            elapsed = time.monotonic() - start
            # What the FINAL attempt actually asked agy for -- the reflex below
            # rebinds it. The routing check has to compare against this, not
            # self.model, or a legitimate fallback would read as a mis-route.
            asked_model = self.model
            if active_log is not None and _should_quota_reflex(
                result, elapsed, active_log, self.model
            ):
                # The evidence log outlives the run because it is the only
                # record of why the seat switched engines.
                keep_logs.add(active_log)
                logger.warning(
                    "agy seat: %s died with quota evidence (RESOURCE_EXHAUSTED/"
                    "429 in its run log); re-running once on fallback model %s.",
                    self.model,
                    AGY_QUOTA_FALLBACK_MODEL,
                )
                active_log = _next_log()
                asked_model = AGY_QUOTA_FALLBACK_MODEL
                with _quota_fallback_handoff(prompt) as (
                    fallback_prompt,
                    request_path,
                ):
                    result = await self._spawn_and_collect(
                        fallback_prompt,
                        cwd,
                        profile_path,
                        start,
                        log_file=str(active_log) if active_log is not None else None,
                        model=AGY_QUOTA_FALLBACK_MODEL,
                        extra_add_dir=str(request_path.parent),
                    )
                # Stamp the model that actually answered as soon as the
                # fallback is SELECTED (error paths included) -- run_council
                # preserves a seat-set model, so the transcript roster names
                # the engine behind this result, not the configured one.
                result.model = AGY_QUOTA_FALLBACK_MODEL
                if result.returncode == 0:
                    # Label the engine swap for the council transcript -- the
                    # seat must never change models silently.
                    result.output = (
                        f"{QUOTA_REFLEX_OUTPUT_PREFIX} this seat's {self.model} run "
                        f"hit AI-Pro quota exhaustion (RESOURCE_EXHAUSTED 429); this "
                        f"answer is from {AGY_QUOTA_FALLBACK_MODEL}]\n\n"
                        + result.output
                    )
                else:
                    result.unavailable_reason = "usage limit"
                    result.error = (
                        f"quota reflex: {self.model} hit quota exhaustion and "
                        f"fallback {AGY_QUOTA_FALLBACK_MODEL} also failed: "
                        + (result.error or f"agy exited {result.returncode}")
                    )
            if result.returncode == 0:
                # The seat must never report another engine's work as the pinned
                # model's. Warn rather than fail: a wrong-but-LABELLED answer is
                # still worth something to the council, and refusing would take
                # the seat down over a mere rewording of agy's log line. What is
                # not acceptable is silence -- that is the actual bug.
                if active_log is None:
                    message = (
                        "agy completed but no diagnostic log could be created, so "
                        f"{asked_model!r} could not be verified"
                    )
                    logger.warning("agy seat: %s", message)
                    result.output = (
                        f"[routing unverified: {message}]\n\n{result.output}"
                    )
                else:
                    labels = resolved_backend_labels(active_log)
                    complaint = routing_complaint(asked_model, labels)
                if active_log is not None and complaint:
                    keep_logs.add(active_log)
                    # The invariant enforcement applies only to the PRIMARY
                    # recorded model (asked_model == self.model) -- the agy
                    # quota reflex's fallback attempt is a pre-existing,
                    # named exception to the invariant (never written back to
                    # the record), so a mismatch THERE stays a warning even
                    # when the seat's own model_source is "recorded".
                    if self.model_source == "recorded" and asked_model == self.model:
                        logger.warning(
                            "agy seat routing mismatch on a recorded choice -- "
                            "failing the seat rather than silently substituting "
                            "an engine: %s",
                            complaint,
                        )
                        result = AgentResult(
                            agent=self.name,
                            output="",
                            error=(
                                f"{complaint} This model was recorded via `quorum "
                                "setup-models`; a routing mismatch against a "
                                "recorded choice fails the seat rather than "
                                "silently substituting an engine. Re-run `quorum "
                                "setup-models` to re-verify or record a "
                                "different model."
                            ),
                            returncode=GEMINI_CLI_RECORDED_MODEL_MISMATCH_RC,
                            duration_s=result.duration_s,
                        )
                    else:
                        logger.warning("agy seat routing mismatch: %s", complaint)
                        result.output = (
                            f"[routing warning: {complaint}]\n\n" + result.output
                        )
                elif active_log is not None and not labels:
                    # Staying quiet about a MISSING label is right (it is absent
                    # evidence, not a mis-route) but going quiet about the CHECK
                    # is not: this is the one shape in which the every-run
                    # guarantee disappears -- agy stops logging what it resolved
                    # and every run looks clean forever. So say so, and keep the
                    # log that would otherwise be deleted, since it is the only
                    # thing that shows what agy's format changed to. A normal run
                    # always carries a label, so this must not be routine; if it
                    # ever is, this warning is how we find out immediately.
                    keep_logs.add(active_log)
                    message = (
                        "agy completed but its log carries no routing line, so "
                        f"{asked_model!r} could not be verified; diagnostic log "
                        f"kept at {active_log}"
                    )
                    logger.warning(
                        "agy seat: %s; re-check _AGY_BACKEND_LABEL_RE.", message
                    )
                    result.output = (
                        f"[routing unverified: {message}]\n\n{result.output}"
                    )
            else:
                # Keep the final failed attempt's log because it is the only
                # place agy records why the seat failed. --log-file REPLACES
                # agy's default cli-*.log (live-probed), so deleting it would
                # erase the sole record.
                if active_log is not None:
                    keep_logs.add(active_log)
                # A recorded model that vanished from agy's registry fails
                # here as a plain nonzero exit with no pointer -- unlike the
                # routing-mismatch hard-fail above (RC 5), which already
                # names setup-models. Gated: asked_model == self.model
                # excludes the quota reflex's fallback attempt (a named,
                # pre-existing exception to the invariant -- naming the
                # PRIMARY recorded model there would misattribute the
                # fallback's own failure); rc 127 (binary missing) is
                # excluded too, though unreachable here in practice (agy
                # resolution happens before this attempt loop ever runs).
                if (
                    self.model_source == "recorded"
                    and asked_model == self.model
                    and result.returncode != GEMINI_CLI_NO_BINARY_RC
                    and not result.unavailable_reason
                ):
                    result.error = append_recorded_source_hint(
                        result.error, asked_model, self.model_source
                    )
            if result.returncode != 0:
                quota_exhausted = bool(
                    _should_surface_quota_evidence(result)
                    and active_log is not None
                    and await asyncio.to_thread(_log_shows_quota_exhaustion, active_log)
                )
                if quota_exhausted:
                    _surface_quota_evidence(result, quota_exhausted=True)
                if active_log is not None:
                    result.error = _with_diagnostic_log(result.error, active_log)
            _apply_version_warning(result, version_warn)
            if keep_logs:
                logger.warning(
                    "agy run log(s) kept for diagnosis: %s",
                    ", ".join(str(p) for p in run_logs if p in keep_logs),
                )
            return result
        finally:
            with contextlib.suppress(OSError):
                os.unlink(profile_path)
            for run_log in run_logs:
                if run_log not in keep_logs:
                    with contextlib.suppress(OSError):
                        os.unlink(run_log)

    async def _spawn_and_collect(
        self,
        prompt: str,
        cwd: str,
        sandbox_profile: str,
        start: float,
        log_file: str | None = None,
        model: str | None = None,
        extra_add_dir: str | None = None,
    ) -> AgentResult:
        cmd = self.build_command(
            prompt=prompt,
            cwd=cwd,
            sandbox_profile=sandbox_profile,
            log_file=log_file,
            model=model,
            extra_add_dir=extra_add_dir,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,  # anchor relative paths in the caller's cwd
                env=allowlisted_seat_subprocess_env(),
                stdin=asyncio.subprocess.DEVNULL,  # quorum-mcp fd 0 is JSON-RPC
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            # argv[0] is sandbox-exec, and run() already resolved agy on PATH, so
            # this now means the sandbox binary went missing between the two checks.
            # Still fail closed -- never retry the command without the sandbox.
            return AgentResult(
                agent=self.name,
                output="",
                error=(
                    f"{SANDBOX_EXEC} disappeared before spawn; refusing to run agy "
                    "unsandboxed."
                ),
                returncode=GEMINI_CLI_NO_SANDBOX_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="sandbox",
            )
        except OSError as exc:  # e.g. E2BIG: prompt too large for argv
            return AgentResult(
                agent=self.name,
                output="",
                error=(
                    f"failed to spawn {self.binary}: {exc} (prompt may exceed the OS "
                    "argv limit; the SDK backend has no such limit)"
                ),
                returncode=1,
                duration_s=time.monotonic() - start,
                unavailable_reason="preflight",
            )

        # Bounded inner wait: if agy wedges and ignores its own --print-timeout,
        # wait_for cancels communicate_or_kill (which tears down the process group
        # on CancelledError) and raises TimeoutError -> distinct visible rc 124.
        # CancelledError (the council's outer cap) is NOT TimeoutError, so it
        # propagates through wait_for untouched.
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                communicate_or_kill(proc, pgid=proc.pid),
                timeout=self.print_timeout + GEMINI_CLI_TIMEOUT_GRACE_S,
            )
        except TimeoutError:
            return AgentResult(
                agent=self.name,
                output="",
                error=(
                    f"agy exceeded print-timeout + {int(GEMINI_CLI_TIMEOUT_GRACE_S)}s "
                    "grace; process group killed"
                ),
                returncode=GEMINI_CLI_TIMEOUT_RC,
                duration_s=time.monotonic() - start,
                unavailable_reason="timeout",
            )

        duration = time.monotonic() - start
        output = _ANSI_RE.sub("", stdout_b.decode("utf-8", errors="replace")).strip()
        stderr = _ANSI_RE.sub("", stderr_b.decode("utf-8", errors="replace")).strip()
        rc = proc.returncode or 0

        if rc != 0 and _looks_like_auth_error(stderr):
            return AgentResult(
                agent=self.name,
                output="",
                error=f"agy auth error: {_redact_auth_error(stderr)}",
                returncode=GEMINI_CLI_NOT_LOGGED_IN_RC,
                duration_s=duration,
                unavailable_reason="authentication",
            )
        # Nonzero exit beats empty-output: keep the real error visible. This
        # generic rc-1 mapping is load-bearing for _is_transient_startup_crash
        # (rc 1 + marker = maybe-retry); auth (2), timeout (124), and no-output
        # (125) keep their distinct codes on their own paths and never retry.
        if rc != 0:
            return AgentResult(
                agent=self.name,
                output=output,
                error=stderr or f"agy exited {proc.returncode}",
                returncode=1,
                duration_s=duration,
            )
        if not output:
            return AgentResult(
                agent=self.name,
                output="",
                error=(
                    "agy produced no answer (clean exit, empty/whitespace-only output)."
                ),
                returncode=GEMINI_CLI_NO_OUTPUT_RC,
                duration_s=duration,
                unavailable_reason="no output",
            )
        return AgentResult(
            agent=self.name,
            output=output,
            returncode=0,
            duration_s=duration,
        )
