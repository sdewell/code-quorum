"""MCP server exposing quorum modes as a start/await tool pair.

Each `q_<mode>_start` schedules the council run as a background asyncio
task and returns a job_id immediately. The paired `q_await(job_id)`
returns the rounds markdown once the task completes. This split is the
structural anti-bias gate: the Claude or Codex host cannot see agent
output until it commits to its own work between the two calls.

The CLI surface remains synchronous — only the MCP path uses start/await.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from quorum.agents.seat_helper import active_allowed_roots, cwd_allowed_error
from quorum.hosts import HOST_PROFILES, configure_runtime_host
from quorum.orchestration import (
    BRAINSTORM_PROMPT,
    NO_SCOPE_NOTICE,
    PLAN_PROMPT,
    REVIEW_PROMPT,
    VALIDATE_PROMPT,
    append_grounding,
    append_prior_ideas,
    append_research,
    format_liveness,
    format_rounds,
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
from quorum.research import (
    DEFAULT_SOURCES,
    RESEARCH_PURPOSES,
    format_digest,
    research_topic,
)

from .jobs import await_job, start_job

mcp = FastMCP("quorum")


def _resolve_cwd(cwd: str) -> Path:
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("explicit cwd is required for every council start")
    candidate = Path(cwd).expanduser()
    if not candidate.is_absolute():
        raise ValueError("explicit cwd must be absolute for every council start")
    target = candidate.resolve()
    if not target.exists():
        raise ValueError(f"cwd does not exist: {target}")
    if not target.is_dir():
        raise ValueError(f"cwd is not a directory: {target}")
    if error := cwd_allowed_error(str(target), active_allowed_roots()):
        raise ValueError(error)
    return target


def _brainstorm_prompt(
    topic: str,
    prior_ideas: str | None,
    grounding: bool = False,
    research: str | None = None,
) -> str:
    """Build the brainstorm prompt. `research` (a q_research digest) is
    appended FIRST with evidence framing — build-from context, never inside the
    prior-ideas wrap. Then prior_ideas appends
    either the divergence block (diverge past them) or, when grounding is set,
    the grounding block (validation-guide pass over them — skystorm Stage 2)."""
    base = append_research(BRAINSTORM_PROMPT.format(topic=topic), research)
    if grounding:
        return append_grounding(base, prior_ideas)
    return append_prior_ideas(base, prior_ideas)


async def _run_and_format(skipped: list[str] | None = None, **kwargs: Any) -> str:
    """Run a council mode and render the rounds matrix to markdown, prefixed
    with the one-line council liveness summary. This is the unit of work
    submitted as a background task by each _start tool; q_await retrieves
    its return value.

    `skipped` (default-roster seats select_agents left out for being
    disabled -- SeatSelection.skipped) is consumed here, not forwarded to
    run_mode: it only feeds the liveness line so a disabled seat reads as
    an explicit skip rather than vanishing from the roster silently."""
    rounds_results = await run_mode(**kwargs)
    return (
        format_liveness(rounds_results, skipped=skipped)
        + "\n\n"
        + format_rounds(rounds_results)
    )


async def _immediate(message: str) -> str:
    return message


async def _no_live_seats_job(skipped: list[str]) -> dict[str, str]:
    """Every default-roster seat is disabled -- schedule a job whose await
    result is the explicit error (never silently run a council of zero
    members). Keeps the uniform {"job_id"} start/await contract, same
    precedent as q_review_start's empty-diff short-circuit below."""
    job_id = await start_job(_immediate(no_live_seats_message(skipped)))
    return {"job_id": job_id}


@mcp.tool()
async def q_plan_start(
    task: str,
    cwd: str,
    agents: list[str] | None = None,
    roles: list[str] | None = None,
    no_context: bool = False,
    skip_gh: bool = False,
    gemini_model: str | None = None,
    verbose: bool = False,
    host: str | None = None,
) -> dict[str, str]:
    """Start a q-plan run in the background. Returns {"job_id": str}
    immediately. The current host's external seats run in parallel from a
    structurally bias-free starting point. Retrieve results by
    calling `q_await` with the returned job_id.

    Between the start and the await, the caller is expected to form
    its own plan — this is the structural anti-bias gate.

    Optionally assign a cognitive stance per agent via `roles`, a list
    of 'stance:agent' strings (stances: skeptic, architect, security,
    maintainer, analyst, neutral, visionary, pioneer). Unassigned agents
    use their default stance.

    `gemini_model`, when supplied, runs the gemini seat on that agy model for
    this invocation only -- an id exactly as printed by `agy models`, e.g.
    'claude-opus-4-6-thinking' to get a Claude answer from the same AI Pro
    plan when Gemini quota is tight or a different perspective is wanted.

    Expected wall-clock to completion: 30s-4min depending on the agents
    and codebase size."""
    target_cwd = _resolve_cwd(cwd)
    chosen, skipped = select_agents(agents, roles, gemini_model=gemini_model, host=host)
    if not chosen:
        return await _no_live_seats_job(skipped)
    coro = _run_and_format(
        skipped=skipped,
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
    job_id = await start_job(coro)
    return {"job_id": job_id}


@mcp.tool()
async def q_brainstorm_start(
    topic: str,
    cwd: str,
    agents: list[str] | None = None,
    roles: list[str] | None = None,
    no_context: bool = False,
    skip_gh: bool = False,
    prior_ideas: str | None = None,
    grounding: bool = False,
    research: str | None = None,
    gemini_model: str | None = None,
    verbose: bool = False,
    host: str | None = None,
) -> dict[str, str]:
    """Start a q-brainstorm run in the background. Returns {"job_id":
    str} immediately. Each agent contributes 3-5 distinct ideas with
    rationale, trade-offs, and the cheapest test that would give signal;
    no synthesis. Retrieve results by calling `q_await` with the
    returned job_id.

    Between the start and the await, the caller is expected to list its
    own ideas — this is the structural anti-bias gate.

    `research`, when supplied, seeds round 1 with a q_research digest as
    EVIDENCE: agents are told to ground their ideas in it, recombine it, or
    extend past it. Pass the raw digest markdown (verbatim from q_research) --
    never your own summary of it, and never your own ideas; those stay behind
    the anti-bias gate. Distinct from `prior_ideas`, whose do-not-repeat
    framing marks content to diverge AWAY from.

    `prior_ideas`, when supplied, seeds a divergence round: the agents are told
    the listed ideas are already on the table and must not be repeated -- used
    by the `--extended` flow to push past round 1. Composes with `research`
    (evidence stays evidence; ideas stay do-not-repeat).

    `grounding`, with `prior_ideas`, runs a validation-guide pass over the
    listed ideas instead of diverging past them -- used by `q-skystorm`
    Stage 2.

    Optionally assign a cognitive stance per agent via `roles`, a list
    of 'stance:agent' strings (stances: skeptic, architect, security,
    maintainer, analyst, neutral, visionary, pioneer). Unassigned agents
    use their default stance.

    `gemini_model`, when supplied, runs the gemini seat on that agy model for
    this invocation only (an id exactly as printed by `agy models`, e.g.
    'claude-opus-4-6-thinking').

    Expected wall-clock to completion: 30s-4min depending on the agents
    and topic complexity."""
    target_cwd = _resolve_cwd(cwd)
    chosen, skipped = select_agents(agents, roles, gemini_model=gemini_model, host=host)
    if not chosen:
        return await _no_live_seats_job(skipped)
    coro = _run_and_format(
        skipped=skipped,
        prompt=_brainstorm_prompt(topic, prior_ideas, grounding, research),
        cwd=target_cwd,
        agents=chosen,
        context_subject=topic,
        phase="brainstorm",
        no_context=no_context,
        skip_gh=skip_gh,
        rounds=1,
        verbose=verbose,
    )
    job_id = await start_job(coro)
    return {"job_id": job_id}


@mcp.tool()
async def q_validate_start(
    plan_path: str,
    cwd: str,
    agents: list[str] | None = None,
    roles: list[str] | None = None,
    no_context: bool = False,
    skip_gh: bool = False,
    extended: bool = False,
    mode: str = "revise",
    gemini_model: str | None = None,
    verbose: bool = False,
    host: str | None = None,
) -> dict[str, str]:
    """Start a q-validate run in the background. Returns {"job_id":
    str} immediately. Each agent independently reviews the plan file,
    then deliberates across rounds. Round 1 is structurally bias-free;
    later rounds embed each agent's own prior plus peers' priors.
    Retrieve results by calling `q_await` with the returned job_id.

    Between the start and the await, the caller is expected to form its
    own review — this is the structural anti-bias gate.

    `extended` runs 4 rounds with a stance rotation at round 3 (agents
    swap stances and re-examine all priors); the default is 2 rounds
    with no rotation. `mode` is 'revise' (agents soften/strengthen in
    light of peers) or 'critique' (agents attack peer points).

    `verbose` defaults to false: the council writes terse output (no padding,
    `path:line` over pasted code, later rounds omit restating unchanged points).
    Set true only for the full unabridged deliberation — a much larger matrix.

    Optionally assign stances per agent via `roles`, a list of
    'stance:agent' strings (stances: skeptic, architect, security,
    maintainer, analyst, neutral, visionary, pioneer).

    `gemini_model`, when supplied, runs the gemini seat on that agy model for
    this invocation only (an id exactly as printed by `agy models`, e.g.
    'claude-opus-4-6-thinking').

    Expected wall-clock to completion: 1-8min default; 4-15min when
    extended=true. Pick extended deliberately."""
    target_cwd = _resolve_cwd(cwd)
    chosen, skipped = select_agents(agents, roles, gemini_model=gemini_model, host=host)
    if not chosen:
        return await _no_live_seats_job(skipped)
    plan_text = read_plan_file(resolve_doc_path(plan_path, target_cwd))
    validated_mode = validate_mode(mode)
    rounds, rotate = resolve_rounds(extended)
    coro = _run_and_format(
        skipped=skipped,
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
    job_id = await start_job(coro)
    return {"job_id": job_id}


@mcp.tool()
async def q_review_start(
    target: str = "",
    *,
    cwd: str,
    scope_path: str | None = None,
    agents: list[str] | None = None,
    roles: list[str] | None = None,
    no_context: bool = False,
    skip_gh: bool = False,
    extended: bool = False,
    mode: str = "revise",
    gemini_model: str | None = None,
    verbose: bool = False,
    host: str | None = None,
) -> dict[str, str]:
    """Start a q-review run in the background. Returns {"job_id": str}
    immediately. Each agent independently reviews real code changes, then
    converges across rounds. Round 1 is structurally bias-free; later rounds
    embed each agent's own prior plus peers' priors so sustained agreement
    becomes visible. Retrieve results by calling `q_await` with the returned
    job_id.

    Between the start and the await, the caller is expected to form its own
    code review of the diff — this is the structural anti-bias gate.

    `target` selects what to review (default: branch vs main, committed +
    uncommitted). 'working' = uncommitted tracked changes only; 'pr:N' or a
    github PR URL = an open PR (title/body orient the review); 'A..B'/'A...B'
    = an explicit range; 'all' = the whole codebase (agents read cwd — pair
    with `scope_path`). An empty diff (other than 'all') short-circuits: the
    job returns a 'nothing to review' message without running the council.

    `scope_path`, when supplied, points at a scope doc declaring what is
    in/out-of-scope and which risks are accepted; it is embedded verbatim so
    the council does not converge on out-of-bounds findings.

    `extended` runs 4 rounds with a stance rotation at round 3 (agents swap
    stances and re-examine all priors); the default is 2 rounds with no
    rotation. `mode` is 'revise' (agents soften/strengthen in light of peers)
    or 'critique' (agents attack peer points).

    `verbose` defaults to false: the council writes terse output (no padding,
    `path:line` over pasted code, and later rounds collapse each still-held
    finding to one `HELD` line while preserving the agreement count). Set true
    only when you want the full unabridged deliberation — a much larger matrix.

    Optionally assign stances per agent via `roles`, a list of 'stance:agent'
    strings (stances: skeptic, architect, security, maintainer, analyst,
    neutral, visionary, pioneer).

    `gemini_model`, when supplied, runs the gemini seat on that agy model for
    this invocation only (an id exactly as printed by `agy models`, e.g.
    'claude-opus-4-6-thinking').

    Expected wall-clock to completion: 1-8min default; 4-15min when
    extended=true. Pick extended deliberately."""
    target_cwd = _resolve_cwd(cwd)
    chosen, skipped = select_agents(agents, roles, gemini_model=gemini_model, host=host)
    if not chosen:
        return await _no_live_seats_job(skipped)
    diff_text, context_subject = await resolve_review_target(target, target_cwd)
    if diff_text == "":
        # Empty-diff short-circuit — uniform {"job_id"} contract: the SKILL's
        # start→await flow must be unchanged, so we hand q_await an already-
        # resolved coroutine that returns the 'nothing to review' message
        # rather than spawning the council on an empty prompt. The `all`
        # resolver returns an explicit repository-inspection notice, so a blank
        # value here is always a failure to prepare review material.
        message = f"{context_subject} — nothing to review."

        async def _no_changes() -> str:
            return message

        job_id = await start_job(_no_changes())
        return {"job_id": job_id}
    scope_text = (
        read_scope_file(resolve_doc_path(scope_path, target_cwd)) if scope_path else ""
    )
    validated_mode = validate_mode(mode)
    rounds, rotate = resolve_rounds(extended)
    coro = _run_and_format(
        skipped=skipped,
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
    job_id = await start_job(coro)
    return {"job_id": job_id}


@mcp.tool()
async def q_research(
    topic: str,
    sources: list[str] | None = None,
    limit: int = 5,
    mode: str = "grounded",
    purpose: str = "methods",
    query_lanes: list[str] | None = None,
) -> str:
    """Fetch prior art from arXiv, OpenAlex, separate Europe PMC published and
    preprint lanes, Context7, GitHub, and HuggingFace.

    `sources` defaults to all seven source adapters. `europepmc-published`
    searches the non-preprint corpus, prioritizes exact matches, backfills a
    thin result set with labelled MeSH-synonym matches, and returns full-text metadata;
    `europepmc-preprints` covers bioRxiv, medRxiv, Research Square, and similar
    non-peer-reviewed work while excluding arXiv.

    `purpose` is `methods` (default) or `currency`. Methods balances all-time
    OpenAlex relevance candidates with a recent five-year stratum and retains
    the stratum labels after deduplication; currency uses only the recent
    stratum. `query_lanes` accepts 1-3 semantic formulations. Prefer lanes for
    domain + construct, failure/validity, and review/guideline terminology. The
    result limit remains a per-source cap across all lanes.

    Each lane must be short and distinctive -- short is not the same as generic.
    Anchor it in 2+ domain-specific terms. Generic or off-topic lanes should
    re-anchor and call q_research again. For same-domain work, returned titles
    should belong to your domain. For a deliberate cross-domain probe, judge
    structural kinship instead; an off-domain hit can be the intended find.

    The digest opens with `Research status:` and a `Source/lane status` table.
    Each row is ON-TOPIC, THIN, QUERY-COLLISION, SOURCE-MISMATCH,
    INFRASTRUCTURE, or CONFIG. OK means no follow-up action remains. DEGRADED
    means a source exhausted its bounded retry but usable peer evidence remains;
    proceed and disclose it. RETRY-REQUIRED is paired with `Research needs
    action: true` and an exact action/source/lane table. Retry rows carry an
    executable query; a re-anchor row explicitly requires different domain
    terms. Do not fall back on your own knowledge; that is the specific failure
    to avoid.

    Mechanically-fixable failures are retried once inside the tool, with a
    bounded delay where appropriate, and marked with `↻`. CONFIG requires
    fixing the credential or proceeding on peer sources. `mode` is grounded or
    exploratory; exploratory enables the OpenAlex field map for the primary
    query lane and treats cross-domain structural hits as intentional.
    """
    # Reject a bad mode up front, before any backend is queried -- otherwise a
    # typo burns 10-15s of concurrent HTTP and rate limit only to crash in
    # format_digest (compute_status validates too, but that runs post-fetch).
    if mode not in ("grounded", "exploratory"):
        raise ValueError(
            f"Unknown research mode {mode!r}; expected 'grounded' or 'exploratory'."
        )
    if purpose not in RESEARCH_PURPOSES:
        raise ValueError(
            f"Unknown research purpose {purpose!r}; expected one of "
            f"{', '.join(RESEARCH_PURPOSES)}."
        )
    chosen = tuple(sources) if sources else DEFAULT_SOURCES
    digest = await research_topic(
        topic,
        sources=set(chosen),
        limit=limit,
        map_fields=mode == "exploratory",
        purpose=purpose,
        query_lanes=tuple(query_lanes) if query_lanes else None,
    )
    return format_digest(digest, mode=mode)


@mcp.tool()
async def q_await(job_id: str) -> str:
    """Block until the background council run identified by `job_id`
    completes, then return its rounds markdown. One-shot — a job_id
    can only be awaited once.

    This is the blocking completion notification for every council start.
    The orchestrating host must not end its turn with a live job outstanding;
    it calls q_await after its independent work and remains blocked until this
    tool returns a result or error.

    Errors:
    - job_id not found (expired, already retrieved, or invalid) →
      ValueError with the reason.
    - job cancelled by TTL or server shutdown → ValueError.
    - underlying council error → propagated."""
    return await await_job(job_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Code Quorum MCP server.")
    parser.add_argument(
        "--host",
        choices=tuple(HOST_PROFILES),
        default="claude",
        help="Orchestrating host; determines the default external-seat roster.",
    )
    args = parser.parse_args(argv)
    configure_runtime_host(args.host)
    mcp.run()


if __name__ == "__main__":
    main()
