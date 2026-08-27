import asyncio
import enum
import subprocess
import sys
from pathlib import Path
from typing import Annotated, cast

import typer

from . import setup_models as sm
from .agents.gemini_cli import (
    DEFAULT_MODEL,
    GeminiCliAgent,
    check_seat_verified_version,
    write_read_only_settings,
)
from .agents.seat_helper import (
    default_allowed_roots,
    default_launchagent_path,
    default_project_root,
    default_spool_dir,
    helper_pid,
    helper_state,
    install_launchagent,
    launchagent_installed,
    launchagent_status,
    run_gemini_auth_check_via_helper,
    serve_helper,
)
from .codex_update import CodexUpdateError, perform_codex_update
from .doctor import DOCTOR_HOSTS, run_doctor
from .hosts import HOST_PROFILES, configure_runtime_host, resolve_host
from .orchestration import (
    BRAINSTORM_PROMPT,
    NO_SCOPE_NOTICE,
    PLAN_PROMPT,
    REVIEW_PROMPT,
    VALIDATE_PROMPT,
    append_research,
    format_liveness,
    format_rounds,
    last_round_all_failed,
    no_live_seats_message,
    read_plan_file,
    read_scope_file,
    resolve_doc_path,
    resolve_review_target,
    resolve_rounds,
    run_mode,
    select_agents,
    validate_mode,
)
from .research import (
    DEFAULT_SOURCES,
    RESEARCH_PURPOSES,
    compute_status,
    format_digest,
    research_topic,
    validate_sources,
)
from .roles import ROLES

app = typer.Typer(
    add_completion=False,
    help="Multi-agent council for Claude and Codex hosts.",
    no_args_is_help=True,
)

# StrEnums built from the canonical value tuples so typer renders and
# validates choices natively (choice list in --help, rejection with the
# choices shown) instead of a hand-rolled ', '.join(...) + BadParameter
# round-trip.
HostEnum = enum.StrEnum("HostEnum", list(HOST_PROFILES))
DoctorHostEnum = enum.StrEnum("DoctorHostEnum", list(DOCTOR_HOSTS))
SeatEnum = enum.StrEnum("SeatEnum", list(sm.SEATS))
ModeEnum = enum.StrEnum("ModeEnum", ["revise", "critique"])
AuthSeatEnum = enum.StrEnum("AuthSeatEnum", ["gemini"])


def _bp(fn, *a, **kw):
    """Call `fn(*a, **kw)`, converting a ValueError into typer.BadParameter
    so a bad CLI input surfaces as a clean usage error, not a traceback."""
    try:
        return fn(*a, **kw)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _require_live_seats(chosen, skipped: list[str]) -> None:
    """A default roster that filtered down to zero live seats must be a
    loud, nonzero-exit failure -- never a blank `format_rounds` printed
    against an empty council that then exits 0 and reads as success."""
    if not chosen:
        typer.echo(no_live_seats_message(skipped), err=True)
        raise typer.Exit(1)


def _echo_council_output(rounds_results, skipped: list[str]) -> None:
    """Both surfaces (CLI and MCP) report the same liveness summary + rounds
    matrix, so a disabled seat reads identically as an explicit skip on
    either one instead of silently vanishing on just the CLI."""
    typer.echo(format_liveness(rounds_results, skipped=skipped) + "\n\n", nl=False)
    typer.echo(format_rounds(rounds_results), nl=False)


def _offer_disable(seat: str, exc: sm.ProbeFailure) -> bool:
    """A seat's live probe failed during interactive setup -- most public
    users hold only one or two of the four subscriptions, so a missing seat
    should be a one-time declaration, not per-run noise. Offer to record it
    `disabled` instead, but only when the failure indicates the seat is
    genuinely absent (not logged in, no subscription, binary missing --
    see sm.indicates_absent_seat). A transient failure (quota exhaustion, a
    network blip, a timeout) never gets the offer: one "y" on a bad day
    would durably drop a seat that actually works. Returns True (seat
    handled, doesn't count toward setup-models' exit-1) on a yes; False
    (today's behavior, including the no-offer case) otherwise."""
    typer.echo(f"{seat}: nothing recorded -- {exc}")
    if sm.indicates_absent_seat(str(exc)) and typer.confirm(
        f"{seat}: mark this seat as not used?"
    ):
        sm.record_disabled(seat)
        typer.echo(f"{seat}: recorded as disabled.\n")
        return True
    typer.echo("")
    return False


CwdOption = Annotated[
    Path | None,
    typer.Option(
        "--cwd",
        "-C",
        help="Working directory the agents read from.",
        exists=True,
        file_okay=False,
    ),
]
AgentsOption = Annotated[
    list[str] | None,
    typer.Option(
        "--agent",
        "-a",
        help="Agent to invoke (repeatable). Default: every external seat for the host.",
    ),
]
NoContextOption = Annotated[
    bool,
    typer.Option(
        "--no-context",
        help="Skip the LEARNINGS + git/gh context resolver.",
    ),
]
SkipGhOption = Annotated[
    bool,
    typer.Option(
        "--skip-gh",
        help="Skip the gh PR lookup (still scans LEARNINGS + git).",
    ),
]
# Generated from ROLES (the single source of truth) so the help can never drift
# out of sync with the registered stances, as it did when visionary/pioneer landed.
_ROLE_HELP = (
    "Assign a stance to an agent as 'stance:agent' (repeatable). "
    f"Stances: {', '.join(ROLES)}."
)
RolesOption = Annotated[
    list[str] | None,
    typer.Option("--role", help=_ROLE_HELP),
]
ExtendedOption = Annotated[
    bool,
    typer.Option(
        "--extended",
        help="4-round deliberation with stance rotation at round 3 "
        "(default: 2 rounds, no rotation).",
    ),
]
VerboseOption = Annotated[
    bool,
    typer.Option(
        "--verbose",
        help="Lift the terse output caps (agents write full detail; review "
        "re-lists every held finding). Default: terse.",
    ),
]
GeminiModelOption = Annotated[
    str | None,
    typer.Option(
        "--gemini-model",
        help="Run the gemini seat on this agy model for this invocation only "
        "(an id exactly as `agy models` prints, e.g. claude-opus-4-6-thinking). "
        "Beats CODE_QUORUM_GEMINI_MODEL.",
    ),
]
ResearchOption = Annotated[
    bool,
    typer.Option(
        "--research/--no-research",
        help="Ground the run in a prior-art digest (arXiv + OpenAlex + Europe PMC "
        "published/preprints + Context7 + GitHub + HuggingFace): research runs "
        "first, then the digest is printed and seeded as evidence. Default: on.",
    ),
]
HostOption = Annotated[
    HostEnum | None,
    typer.Option(
        "--host",
        help="Orchestrating host profile for external-seat selection. "
        "Default: CODE_QUORUM_HOST or claude.",
    ),
]


@app.command(name="q-plan")
def q_plan(
    task: Annotated[str, typer.Argument(help="Task description to plan.")],
    cwd: CwdOption = None,
    agents: AgentsOption = None,
    roles: RolesOption = None,
    no_context: NoContextOption = False,
    skip_gh: SkipGhOption = False,
    gemini_model: GeminiModelOption = None,
    host: HostOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Generate alternative plans in parallel from each agent."""
    target_cwd = cwd or Path.cwd()
    chosen, skipped = _bp(
        select_agents, agents, roles, gemini_model=gemini_model, host=host
    )
    _require_live_seats(chosen, skipped)
    rounds_results = asyncio.run(
        run_mode(
            prompt=PLAN_PROMPT.format(task=task),
            cwd=target_cwd,
            agents=chosen,
            context_subject=task,
            phase="plan",
            no_context=no_context,
            skip_gh=skip_gh,
            rounds=1,
            verbose=verbose,
        )
    )
    _echo_council_output(rounds_results, skipped)
    if last_round_all_failed(rounds_results):
        sys.exit(1)


@app.command(name="q-brainstorm")
def q_brainstorm(
    topic: Annotated[str, typer.Argument(help="Topic to brainstorm.")],
    cwd: CwdOption = None,
    agents: AgentsOption = None,
    roles: RolesOption = None,
    no_context: NoContextOption = False,
    skip_gh: SkipGhOption = False,
    research: ResearchOption = True,
    gemini_model: GeminiModelOption = None,
    host: HostOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Generate divergent ideas in parallel from each agent, grounded in a
    prior-art digest by default (research-first; --no-research opts out)."""
    target_cwd = cwd or Path.cwd()
    chosen, skipped = _bp(
        select_agents, agents, roles, gemini_model=gemini_model, host=host
    )
    _require_live_seats(chosen, skipped)
    # Fetch the digest BEFORE the council so round 1 generates from the
    # evidence, then seed it with evidence framing.
    digest_text = ""
    if research:
        try:
            digest = asyncio.run(research_topic(topic, sources=set(DEFAULT_SOURCES)))
        except ValueError as exc:
            # A pre-flight-refused topic must not kill the council -- note the
            # skip and run unseeded.
            typer.echo(f"[research skipped: {exc}]\n")
        except Exception as exc:  # noqa: BLE001 -- research runs BEFORE the
            # council, so ANY crash here (network stack, schema drift) must
            # degrade to an unseeded run and never kill the command.
            typer.echo(f"[research unavailable: {exc}]\n")
        else:
            rendered = format_digest(digest)
            # Always show the digest (verdict included) to the USER; seed the
            # council only when the verdict is not a known whiff. The CLI is
            # not an orchestrator -- it cannot rework-and-retry, so a
            # RETRY-RECOMMENDED digest must not anchor the council as evidence.
            typer.echo(rendered + "\n", nl=False)
            if compute_status(digest).code != "RETRY-RECOMMENDED":
                digest_text = rendered
    rounds_results = asyncio.run(
        run_mode(
            prompt=append_research(BRAINSTORM_PROMPT.format(topic=topic), digest_text),
            cwd=target_cwd,
            agents=chosen,
            context_subject=topic,
            phase="brainstorm",
            no_context=no_context,
            skip_gh=skip_gh,
            rounds=1,
            verbose=verbose,
        )
    )
    _echo_council_output(rounds_results, skipped)
    if last_round_all_failed(rounds_results):
        sys.exit(1)


@app.command(name="q-validate")
def q_validate(
    plan_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the plan file to review.",
        ),
    ],
    cwd: CwdOption = None,
    agents: AgentsOption = None,
    roles: RolesOption = None,
    no_context: NoContextOption = False,
    skip_gh: SkipGhOption = False,
    extended: ExtendedOption = False,
    mode: Annotated[
        ModeEnum,
        typer.Option(
            "--mode",
            "-m",
            help="Round 2+ behavior.",
        ),
    ] = ModeEnum.revise,
    gemini_model: GeminiModelOption = None,
    host: HostOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Have each agent independently review a plan file, then deliberate."""
    target_cwd = cwd or Path.cwd()
    chosen, skipped = _bp(
        select_agents, agents, roles, gemini_model=gemini_model, host=host
    )
    _require_live_seats(chosen, skipped)
    plan_text = _bp(
        lambda: read_plan_file(resolve_doc_path(str(plan_path), target_cwd))
    )
    validated_mode = validate_mode(mode)
    rounds, rotate = resolve_rounds(extended)
    rounds_results = asyncio.run(
        run_mode(
            prompt=VALIDATE_PROMPT.format(plan=plan_text),
            cwd=target_cwd,
            agents=chosen,
            context_subject=plan_text,
            phase="validate",
            no_context=no_context,
            skip_gh=skip_gh,
            rounds=rounds,
            mode=validated_mode,
            rotate=rotate,
            verbose=verbose,
        )
    )
    _echo_council_output(rounds_results, skipped)
    if last_round_all_failed(rounds_results):
        sys.exit(1)


@app.command(name="q-review")
def q_review(
    target: Annotated[
        str,
        typer.Argument(
            help="Target to review. Default: branch vs main (committed + "
            "uncommitted). 'working' = uncommitted only; 'pr:N' or a PR URL; "
            "'A..B'/'A...B' range; 'all' = whole codebase (pair with --scope).",
        ),
    ] = "",
    scope: Annotated[
        Path | None,
        typer.Option(
            "--scope",
            help="Path to a scope doc declaring in/out-of-scope and accepted "
            "risks; embedded verbatim so the council does not converge on "
            "out-of-bounds findings.",
        ),
    ] = None,
    cwd: CwdOption = None,
    agents: AgentsOption = None,
    roles: RolesOption = None,
    no_context: NoContextOption = False,
    skip_gh: SkipGhOption = False,
    extended: ExtendedOption = False,
    mode: Annotated[
        ModeEnum,
        typer.Option(
            "--mode",
            "-m",
            help="Round 2+ behavior.",
        ),
    ] = ModeEnum.revise,
    gemini_model: GeminiModelOption = None,
    host: HostOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Have each agent independently review code changes, then converge."""
    target_cwd = cwd or Path.cwd()
    chosen, skipped = _bp(
        select_agents, agents, roles, gemini_model=gemini_model, host=host
    )
    _require_live_seats(chosen, skipped)
    validated_mode = validate_mode(mode)
    diff_text, context_subject = _bp(
        lambda: asyncio.run(resolve_review_target(target, target_cwd))
    )
    if diff_text == "":
        typer.echo(f"{context_subject} — nothing to review.")
        return
    scope_text = (
        _bp(lambda: read_scope_file(resolve_doc_path(str(scope), target_cwd)))
        if scope
        else ""
    )
    rounds, rotate = resolve_rounds(extended)
    rounds_results = asyncio.run(
        run_mode(
            prompt=REVIEW_PROMPT.format(
                diff=diff_text,
                scope=scope_text or NO_SCOPE_NOTICE,
            ),
            cwd=target_cwd,
            agents=chosen,
            context_subject=context_subject,
            phase="review",
            no_context=no_context,
            skip_gh=skip_gh,
            rounds=rounds,
            mode=validated_mode,
            rotate=rotate,
            verbose=verbose,
        )
    )
    _echo_council_output(rounds_results, skipped)
    if last_round_all_failed(rounds_results):
        sys.exit(1)


@app.command(name="research")
def research(
    topic: Annotated[str, typer.Argument(help="Topic to fetch prior art for.")],
    sources: Annotated[
        list[str] | None,
        typer.Option(
            "--source",
            "-s",
            help="Source to query (repeatable): arxiv, openalex, "
            "europepmc-published, europepmc-preprints, context7, github, "
            "huggingface. Default: all seven.",
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Max results per source.", min=1)
    ] = 5,
    purpose: Annotated[
        str,
        typer.Option(
            "--purpose",
            help=f"Search purpose: {' or '.join(RESEARCH_PURPOSES)}.",
        ),
    ] = "methods",
    query_lanes: Annotated[
        list[str] | None,
        typer.Option(
            "--query-lane",
            help="Semantic query lane (repeatable, up to three).",
        ),
    ] = None,
) -> None:
    """Fetch prior art from publication and artifact sources."""
    chosen = _bp(validate_sources, sources if sources else DEFAULT_SOURCES)
    try:
        digest = asyncio.run(
            research_topic(
                topic,
                sources=chosen,
                limit=limit,
                purpose=purpose,
                query_lanes=tuple(query_lanes) if query_lanes else None,
            )
        )
    except ValueError as exc:
        # A generic/empty topic is refused by the pre-flight lint; surface it as a
        # clean bad-parameter message, not a traceback.
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_digest(digest), nl=False)


@app.command(name="setup-agy")
def setup_agy() -> None:
    """One-time: write the Antigravity CLI (agy) config the `cli` Gemini backend
    expects (deny writes/shell/url-fetch/mcp, disable personal-credit fallback).
    Preserves your existing agy settings; refuses to overwrite a malformed file.

    This config is DEFENCE IN DEPTH, not the read-only guarantee: agy 1.1.2 was
    verified writing through its deny list in --print mode. The council seat is
    read-only because it runs agy under a macOS seatbelt sandbox."""
    try:
        path = write_read_only_settings()  # argless: resolves the path at call time
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote agy config to {path}.")
    typer.echo(
        f"The council's Gemini seat is read-only because it runs agy under a "
        f"seatbelt sandbox (model {DEFAULT_MODEL!r}). Conserve quota with: export "
        "CODE_QUORUM_GEMINI_MODEL=gemini-3.5-flash-high, or opt out with: "
        "export CODE_QUORUM_GEMINI_BACKEND=sdk"
    )
    typer.echo(
        "WARNING: this config does NOT make agy read-only outside the council. "
        "agy 1.1.2 was verified writing files and running shell commands through "
        "the deny list; behavior varies by release, so treat bare `agy` as "
        "write-capable."
    )
    version_warn = check_seat_verified_version()
    if version_warn:
        typer.echo(f"WARNING: {version_warn}")


@app.command(name="auth-check")
def auth_check(
    seat: Annotated[
        AuthSeatEnum,
        typer.Option("--seat", help="Subscription seat whose login to verify."),
    ] = AuthSeatEnum.gemini,
    host: Annotated[
        HostEnum | None,
        typer.Option("--host", help="Environment that will launch the seat."),
    ] = None,
) -> None:
    """Check Gemini login readiness without spending a model turn."""
    cwd = str(Path.cwd())
    profile = _bp(resolve_host, host)
    if profile.name == "codex":
        result = asyncio.run(run_gemini_auth_check_via_helper(cwd=cwd))
    else:
        result = asyncio.run(GeminiCliAgent().check_auth(cwd=cwd))
    if result.returncode == 0:
        typer.echo(result.output)
        return
    typer.echo(f"Error: {result.error}", err=True)
    raise typer.Exit(1)


@app.command(name="seat-helper")
def seat_helper(
    spool_dir: Annotated[
        Path | None,
        typer.Option("--spool-dir", help="File-spool directory for sandboxed clients."),
    ] = None,
    once: Annotated[
        bool,
        typer.Option(
            "--once", help="Process one queued request and exit; intended for tests."
        ),
    ] = False,
    allowed_roots: Annotated[
        list[Path] | None,
        typer.Option(
            "--allowed-root",
            help=(
                "Directory the helper may serve as a cwd (repeatable). "
                "Default: ~/Code, ~/src, and ~/.codex/agent-worktrees."
            ),
        ),
    ] = None,
) -> None:
    """Run Claude and agy requests outside a Codex host sandbox.

    The Claude seat keeps its explicit read-only tool allowlist; the agy seat
    still runs inside its macOS Seatbelt profile. This helper changes where the
    seat starts, not its read-only contract.
    """
    spool = spool_dir or default_spool_dir()
    roots = (
        tuple(root.expanduser().resolve() for root in allowed_roots)
        if allowed_roots
        else None
    )
    typer.echo(f"Starting seat helper on {spool}", err=True)
    try:
        asyncio.run(serve_helper(spool_dir=spool, allowed_roots=roots, once=once))
    except KeyboardInterrupt:
        typer.echo("seat helper stopped", err=True)


@app.command(name="install-seat-helper-launchagent")
def install_seat_helper_launchagent(
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="uv project root for the helper command. Defaults to this checkout.",
            file_okay=False,
        ),
    ] = None,
    allowed_roots: Annotated[
        list[Path] | None,
        typer.Option(
            "--allowed-root",
            help=(
                "Directory the helper may serve as a cwd (repeatable). "
                "Default: ~/Code, ~/src, and ~/.codex/agent-worktrees."
            ),
        ),
    ] = None,
    spool_dir: Annotated[
        Path | None,
        typer.Option("--spool-dir", help="File-spool directory for helper IPC."),
    ] = None,
    no_load: Annotated[
        bool,
        typer.Option(
            "--no-load", help="Write the plist but do not bootstrap it with launchctl."
        ),
    ] = False,
) -> None:
    """Install the shared out-of-sandbox helper as a macOS LaunchAgent."""
    roots = (
        tuple(root.expanduser().resolve() for root in allowed_roots)
        if allowed_roots
        else None
    )
    try:
        plist = install_launchagent(
            project_root=project_root.resolve() if project_root else None,
            spool_dir=spool_dir,
            allowed_roots=roots,
            load=not no_load,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        typer.echo(f"launchctl failed: {detail}", err=True)
        raise typer.Exit(1) from exc
    display_root = project_root.resolve() if project_root else default_project_root()
    typer.echo(f"Installed {plist}")
    typer.echo(f"Project root: {display_root}")
    typer.echo(f"Spool: {spool_dir or default_spool_dir()}")
    typer.echo("Allowed roots:")
    for root in roots or default_allowed_roots():
        typer.echo(f"- {root}")
    typer.echo("LaunchAgent loaded." if not no_load else "LaunchAgent not loaded.")


@app.command(name="update-codex")
def update_codex_command(
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Stable code-quorum Git checkout. Defaults to this checkout.",
            file_okay=False,
        ),
    ] = None,
) -> None:
    """Update the Codex plugin, stable runtime, and seat helper together."""
    root = (
        project_root.expanduser().resolve()
        if project_root is not None
        else default_project_root()
    )
    try:
        version = perform_codex_update(root, progress=typer.echo)
    except CodexUpdateError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Code Quorum {version} is installed for Codex.")
    typer.echo("Fully restart Codex and start a new thread to use it.")


@app.command(name="seat-helper-status")
def seat_helper_status(
    spool_dir: Annotated[
        Path | None,
        typer.Option("--spool-dir", help="File-spool directory for helper IPC."),
    ] = None,
) -> None:
    """Report the shared seat helper's LaunchAgent and process state."""
    spool = spool_dir or default_spool_dir()
    typer.echo(f"LaunchAgent plist: {default_launchagent_path()}")
    typer.echo(f"LaunchAgent installed: {'yes' if launchagent_installed() else 'no'}")
    try:
        typer.echo(f"LaunchAgent state: {launchagent_status()}")
    except (OSError, subprocess.SubprocessError) as exc:
        typer.echo(f"LaunchAgent state: unknown ({exc})")
    pid = helper_pid(spool)
    typer.echo(f"Spool: {spool}")
    typer.echo(f"Helper running: {'yes' if pid is not None else 'no'}")
    if pid is not None:
        typer.echo(f"PID: {pid}")
    state = helper_state(spool)
    if state:
        version = state.get("protocol_version", "missing")
        typer.echo(f"Protocol version: {version}")
        package_version = state.get("code_quorum_version", "missing")
        typer.echo(f"Code Quorum version: {package_version}")
    if state and isinstance(state.get("allowed_roots"), list):
        typer.echo("Allowed roots:")
        for root in state["allowed_roots"]:
            typer.echo(f"- {root}")


@app.command()
def doctor(
    host: Annotated[
        DoctorHostEnum,
        typer.Option("--host", help="Installation profile to check."),
    ] = DoctorHostEnum.claude,
) -> None:
    """Check host binaries, authentication, and read-only boundaries."""
    results = _bp(run_doctor, host)
    for result in results:
        status = "ok" if result.ok else "error"
        typer.echo(f"[{status}] {result.name}: {result.detail}")
    if not all(result.ok for result in results):
        raise typer.Exit(1)


SetupSeatOption = Annotated[
    SeatEnum | None,
    typer.Option(
        "--seat",
        help="Configure one seat non-interactively (requires --model).",
    ),
]
SetupModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        help="Model id to record for --seat (requires --seat; skips prompts).",
    ),
]
SetupEffortOption = Annotated[
    str | None,
    typer.Option(
        "--effort",
        help="Claude/Codex effort to record. Default: current effective.",
    ),
]


@app.command(name="setup-models")
def setup_models(
    seat: SetupSeatOption = None,
    model: SetupModelOption = None,
    effort: SetupEffortOption = None,
    host: Annotated[
        HostEnum | None,
        typer.Option("--host", help="External-seat profile for interactive setup."),
    ] = None,
) -> None:
    """Record each council seat's model choice in
    ~/.config/code-quorum/models.toml, gated by a live probe: nothing is
    recorded unless the candidate model actually routes and answers right
    now. Interactive by default for the selected `--host` profile -- shows the
    current choice + a discovery list per seat, prompts with the current choice
    as default. Scripted use:
    `--seat <s> --model <m> [--effort <e>]` (effort is Claude/Codex-only).

    codex has no discovery command: there is no way to enumerate or rank its
    models, so a vanished codex model is only detectable by a failing probe
    or council run."""
    if host is not None:
        _bp(configure_runtime_host, host)
    cwd = str(Path.cwd())
    if seat is not None or model is not None or effort is not None:
        if seat is None or model is None:
            raise typer.BadParameter("--seat and --model must be given together")
        _bp(sm.validate_noninteractive_flags, seat, effort)
        if seat == "claude" and effort is None:
            effort = sm.current_claude_effort()
            _bp(sm.validate_claude_effort, effort)
        elif seat == "codex" and effort is None:
            effort = sm.current_codex_effort()
            _bp(sm.validate_codex_effort, effort)
        try:
            message = asyncio.run(
                _setup_models_noninteractive(seat, model, effort, cwd)
            )
        except sm.ProbeFailure as exc:
            # The one expected runtime failure: the live probe rejected the
            # candidate model. Every usage error was raised as BadParameter
            # above, before any live work; any other ValueError here is a
            # bug and should crash loudly rather than wear a usage banner.
            typer.echo(f"Error: {exc}")
            raise typer.Exit(1) from exc
        typer.echo(message)
        return
    asyncio.run(_setup_models_interactive(cwd))


async def _setup_models_noninteractive(
    seat: str, model: str, effort: str | None, cwd: str
) -> str:
    prefix = sm.check_malformed_config()
    prefix = f"{prefix}\n" if prefix else ""
    if seat == "claude":
        eff = effort if effort is not None else sm.current_claude_effort()
        await sm.record_claude(model, eff, cwd=cwd)
        return f"{prefix}Recorded claude: model={model!r} effort={eff!r}."
    if seat == "codex":
        eff = effort if effort is not None else sm.current_codex_effort()
        await sm.record_codex(model, eff, cwd=cwd)
        return f"{prefix}Recorded codex: model={model!r} effort={eff!r}."
    if seat == "gemini":
        await sm.record_gemini(model, cwd=cwd)
        return f"{prefix}Recorded gemini: model={model!r}."
    await sm.record_opencode(model, cwd=cwd)
    return f"{prefix}Recorded opencode: model={model!r}."


async def _setup_models_interactive(cwd: str) -> None:
    malformed = sm.check_malformed_config()
    if malformed:
        typer.echo(f"{malformed}\n")
    profile = _bp(resolve_host)
    if "codex" in profile.default_agents:
        typer.echo(
            "codex has no discovery command: there is no way to enumerate or rank "
            "its models. A vanished codex model is only detectable by a failing "
            "probe or council run.\n"
        )
    results = [
        await _setup_seat_interactive(seat, cwd) for seat in profile.default_agents
    ]
    if not all(results):
        raise typer.Exit(code=1)


# Per-seat variation for _setup_seat_interactive: claude/codex additionally
# prompt for effort; gemini/opencode show a live discovery listing first
# (grouped by family for gemini, flat for opencode) instead of an effort
# prompt.
_SETUP_SEAT_SPEC: dict[str, dict] = {
    "claude": {"has_effort": True},
    "codex": {"has_effort": True},
    "gemini": {
        "has_effort": False,
        "listing_fn": "fetch_agy_listing",
        "listing_cmd": "`agy models`",
        "prompt_label": "Model id or slug",
        "grouped": True,
    },
    "opencode": {
        "has_effort": False,
        "listing_fn": "fetch_opencode_listing",
        "listing_cmd": "`opencode models`",
        "prompt_label": "Model id",
        "grouped": False,
    },
}


async def _setup_seat_interactive(seat: str, cwd: str) -> bool:
    spec = _SETUP_SEAT_SPEC[seat]
    model, source = sm.current_choice(seat)
    effort = None
    if spec["has_effort"]:
        effort = (
            sm.current_claude_effort()
            if seat == "claude"
            else sm.current_codex_effort()
        )
        typer.echo(
            f"{seat}: current model={model!r} (source: {source}), effort={effort!r}"
        )
    else:
        typer.echo(f"{seat}: current model={model!r} (source: {source})")
        listing_fn = getattr(sm, spec["listing_fn"])
        try:
            listing = listing_fn()
        except ValueError as exc:
            typer.echo(
                f"{seat}: could not fetch {spec['listing_cmd']} ({exc}); "
                "type an id manually."
            )
            listing = []
        else:
            if seat == "opencode":
                listing = sm.openrouter_only(listing)
        if spec["grouped"]:
            # gemini's listing_fn (fetch_agy_listing) returns
            # list[tuple[str, str]]; the getattr indirection above loses
            # that for the type checker.
            grouped_listing = cast("list[tuple[str, str]]", listing)
            for family, rows in sm.group_gemini_models(grouped_listing).items():
                typer.echo(f"  {family}:")
                for slug, display in rows:
                    typer.echo(f"    {slug}\t{display}")
        else:
            for mid in listing:
                typer.echo(f"  {mid}")
    new_model = typer.prompt(spec.get("prompt_label", "Model id"), default=model)
    new_effort = (
        typer.prompt("Reasoning effort", default=effort) if spec["has_effort"] else None
    )
    record = getattr(sm, f"record_{seat}")
    try:
        if spec["has_effort"]:
            await record(new_model, new_effort, cwd=cwd)
        else:
            await record(new_model, cwd=cwd)
    except sm.ProbeFailure as exc:
        return _offer_disable(seat, exc)
    except ValueError as exc:
        typer.echo(f"{seat}: nothing recorded -- {exc}\n")
        return False
    if spec["has_effort"]:
        typer.echo(f"{seat}: recorded model={new_model!r} effort={new_effort!r}.\n")
    else:
        typer.echo(f"{seat}: recorded model={new_model!r}.\n")
    return True


if __name__ == "__main__":
    app()
