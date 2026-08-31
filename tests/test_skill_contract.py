"""SKILL.md files must match the actual MCP tool surface. These tests
catch drift between the slash-command prose and the tool names/params
exposed by quorum_mcp.server."""

import inspect
from pathlib import Path

import pytest

from quorum.orchestration import OUT_OF_SCOPE_TAG
from quorum_mcp.server import q_await

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Claude plugin MCP tools use `mcp__plugin_<plugin>_<server>__`. Codex tools use
# the generated adapter's distinct `mcp__quorum_codex__` namespace so the root
# Claude `.mcp.json` cannot shadow them in this source checkout.
_PREFIX = "mcp__plugin_code-quorum_quorum__"
_CODEX_PREFIX = "mcp__quorum_codex__"


def _skill(name: str) -> str:
    return (_SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "skill,start_tool",
    [
        ("q-plan", f"{_PREFIX}q_plan_start"),
        ("q-brainstorm", f"{_PREFIX}q_brainstorm_start"),
        ("q-validate", f"{_PREFIX}q_validate_start"),
        ("q-review", f"{_PREFIX}q_review_start"),
    ],
)
def test_skill_references_new_start_tool(skill: str, start_tool: str) -> None:
    """Each skill must reference its start tool by the exact MCP name."""
    assert start_tool in _skill(skill)


@pytest.mark.parametrize(
    "skill", ["q-plan", "q-brainstorm", "q-skystorm", "q-validate", "q-review"]
)
def test_skill_references_q_await(skill: str) -> None:
    """Every skill must instruct the caller to retrieve results via
    q_await — the shared await tool that gates the anti-bias flow."""
    assert f"{_PREFIX}q_await" in _skill(skill)


@pytest.mark.parametrize(
    "skill", ["q-plan", "q-brainstorm", "q-skystorm", "q-validate", "q-review"]
)
def test_council_skills_fail_closed_without_mcp_tools(skill: str) -> None:
    text = _skill(skill)
    flat = " ".join(text.split())

    assert "If either required MCP tool is absent" in flat
    assert "Do not substitute the synchronous CLI" in flat
    assert "structural anti-bias gate" in flat


def test_q_review_no_longer_needs_a_self_hosting_fallback() -> None:
    flat = " ".join(_skill("q-review").split())

    assert "self-hosting exception" not in flat
    assert "uv run quorum q-review" not in flat
    assert _CODEX_PREFIX in flat


def test_codex_skystorm_fails_closed_without_mcp_tools() -> None:
    path = _SKILLS_DIR / "q-skystorm" / "references" / "codex-host.md"
    flat = " ".join(path.read_text(encoding="utf-8").split())

    assert "If either required MCP tool is absent" in flat
    assert "Do not substitute the synchronous CLI" in flat
    assert "structural anti-bias gate" in flat


@pytest.mark.parametrize(
    "relative",
    [
        "q-plan/SKILL.md",
        "q-brainstorm/SKILL.md",
        "q-validate/SKILL.md",
        "q-review/SKILL.md",
        "q-skystorm/SKILL.md",
        "q-skystorm/references/codex-host.md",
    ],
)
def test_council_workflows_use_same_turn_blocking_completion_gate(
    relative: str,
) -> None:
    text = (_SKILLS_DIR / relative).read_text(encoding="utf-8")

    assert "blocking completion notification" in text
    assert "same turn" in text
    assert "Do not end the turn" in text
    assert "check later" in text
    assert "live `job_id` pending" in text


@pytest.mark.parametrize(
    "relative",
    [
        "q-plan/SKILL.md",
        "q-brainstorm/SKILL.md",
        "q-validate/SKILL.md",
        "q-review/SKILL.md",
        "q-skystorm/SKILL.md",
        "q-skystorm/references/codex-host.md",
    ],
)
def test_council_workflows_require_explicit_project_cwd(relative: str) -> None:
    text = (_SKILLS_DIR / relative).read_text(encoding="utf-8")

    assert "Every MCP council start requires an explicit absolute `cwd`" in text
    assert "Never omit it" in text


def test_q_await_tool_doc_is_a_blocking_completion_gate() -> None:
    doc = inspect.getdoc(q_await) or ""

    assert "blocking completion notification" in doc
    assert "must not end its turn" in doc


@pytest.mark.parametrize(
    "skill", ["q-plan", "q-brainstorm", "q-skystorm", "q-validate", "q-review"]
)
def test_skill_documents_verbose(skill: str) -> None:
    """Terse output is the default for every council mode (the spec scopes it to
    all of them, including q-skystorm). Each skill must therefore document the
    `--verbose` escape hatch so a caller can lift the terse caps — otherwise the
    flag silently becomes topic/argument text."""
    text = _skill(skill).lower()
    assert "--verbose" in text
    assert "strip" in text
    assert "`verbose`" in text
    assert "`true` only" in text or "pass `true`" in text


@pytest.mark.parametrize("skill", ["q-plan", "q-brainstorm", "q-validate", "q-review"])
def test_skill_documents_gemini_model(skill: str) -> None:
    """Each council-start skill must document the `--gemini-model` flag (the
    per-run agy-seat model override, e.g. running the gemini seat on Claude) so
    a user's ask maps onto the tool param instead of leaking into topic text --
    and must instruct stripping it from the argument like the other flags."""
    text = _skill(skill).lower()
    assert "--gemini-model" in text
    assert "`gemini_model`" in text
    assert "claude-opus-4-6-thinking" in text


@pytest.mark.parametrize(
    "skill",
    [
        "q-plan",
        "q-brainstorm",
        "q-skystorm",
        "q-validate",
        "q-review",
        "q-research",
    ],
)
def test_skill_documents_both_host_tool_prefixes(skill: str) -> None:
    """One shared skill tree must name Claude's plugin namespace and Codex's
    distinct MCP server namespace explicitly."""
    text = _skill(skill)
    assert _PREFIX in text, f"{skill}: expected the plugin-namespaced tool prefix"
    assert _CODEX_PREFIX in text.replace(_PREFIX, ""), (
        f"{skill}: expected the Codex MCP prefix {_CODEX_PREFIX!r}"
    )


@pytest.mark.parametrize(
    "skill", ["q-plan", "q-brainstorm", "q-skystorm", "q-validate", "q-review"]
)
def test_council_skill_documents_explicit_host_selection(skill: str) -> None:
    text = _skill(skill)
    assert '`host: "claude"`' in text
    assert '`host: "codex"`' in text


@pytest.mark.parametrize(
    "skill",
    [
        "q-plan",
        "q-brainstorm",
        "q-skystorm",
        "q-validate",
        "q-review",
        "q-research",
    ],
)
def test_skill_does_not_reference_old_sync_tool(skill: str) -> None:
    """The pre-start/await synchronous tool names must NOT appear — they
    no longer exist. A skill mentioning them would tell Claude to call a
    non-existent tool."""
    text = _skill(skill)
    # Look for the suffixless form right at the boundary (e.g.
    # `..._quorum__q_plan` followed by a non-_ char), which would be a stale
    # synchronous tool rather than the `_start` form.
    for tool in ("q_plan", "q_brainstorm", "q_validate"):
        bare = f"{_PREFIX}{tool}"
        idx = 0
        while True:
            i = text.find(bare, idx)
            if i == -1:
                break
            after = text[i + len(bare) : i + len(bare) + 1]
            assert after == "_", f"{skill}: bare {bare!r} appears without _start suffix"
            idx = i + 1


def test_q_brainstorm_skill_demands_retry_before_fallback() -> None:
    """The research step must key the retry to the deterministic verdict line and
    keep the fallback escape hatch gated behind an actual reworked retry —
    guarding the 'research backend choked, I'll use what I know' shrug. The
    server now emits the verdict; the skill's job is to make the agent act on
    RETRY-REQUIRED before falling back."""
    text = _skill("q-brainstorm")
    assert "Research status:" in text
    assert "RETRY-REQUIRED" in text
    assert "call `q_research` again" in text
    assert "specific failure to avoid" in text
    assert "Research needs action: true" in text
    assert "### Required research actions" in text


def test_q_brainstorm_skill_routes_required_action_table() -> None:
    """Required actions must distinguish an unchanged infrastructure retry
    from a semantic re-anchor without relying on prose hidden below OK."""
    text = _skill("q-brainstorm")
    assert "Research needs action: true" in text
    assert "### Required research actions" in text
    assert "RETRY-SAME" in text
    assert "RE-ANCHOR" in text
    assert "call `q_research` again" in text


def test_q_brainstorm_skill_demands_domain_anchored_query() -> None:
    """The research step must still steer query FORMATION — short is not the same
    as generic, anchor in 2+ domain-specific terms — even though the tool now
    refuses the most generic queries and shapes each source's query itself."""
    text = _skill("q-brainstorm")
    assert "short is not the same as generic" in text
    assert "domain-specific terms" in text


def test_q_validate_skill_instructs_real_mcp_params() -> None:
    """q-validate exposes `extended: bool`, not a `rounds` integer.
    Instructing the wrong param makes the tool call fail or silently
    do the wrong thing."""
    text = _skill("q-validate")
    assert "`extended`" in text
    assert "`rounds`" not in text


def test_q_review_skill_uses_out_of_scope_tag_literal() -> None:
    """The `[OUT-OF-SCOPE]` tag is a hand-copied literal shared between
    `REVIEW_PROMPT` (Python) and the q-review SKILL.md prose. The SKILL's
    synthesis step must strike rows the agents tagged with this exact
    string, so the literal must match the Python constant verbatim — a
    drift here means Claude looks for the wrong marker."""
    assert OUT_OF_SCOPE_TAG in _skill("q-review")


def test_q_skystorm_skill_uses_brainstorm_engine_with_grounding() -> None:
    """Skystorm has no own start tool — it drives the brainstorm engine. It
    must reference q_brainstorm_start and use the grounding flag for Stage 2."""
    text = _skill("q-skystorm")
    assert f"{_PREFIX}q_brainstorm_start" in text
    assert f"{_PREFIX}q_await" in text
    assert "grounding" in text


def test_q_skystorm_stage1_pins_agents_explicitly() -> None:
    """Stage 1 must pin the agent roster explicitly rather than relying on the
    server's default host roster (resolve_host("claude").default_agents) — a
    roster change could otherwise silently drop opencode or add an
    off-stance neutral agent to the dream pool."""
    assert '["codex", "opencode", "gemini"]' in _skill("q-skystorm")


def test_q_brainstorm_extended_await_relays_liveness() -> None:
    """The --extended flow runs a SECOND q_await whose result also begins with
    the Council: liveness line. The skill must instruct relaying it too, not
    just the first await."""
    extended = _skill("q-brainstorm").split("Extended divergence")[1]
    assert "liveness" in extended.lower() or "Council:" in extended


def test_q_brainstorm_extended_reuses_verbose_setting() -> None:
    """The slash-only second divergence round must not silently fall back to
    terse after the user requested --verbose for round 1. (Round 1's council
    start is Step 3 in the research-first ordering.)"""
    extended = _skill("q-brainstorm").split("Extended divergence")[1].lower()
    assert "`verbose`" in extended
    assert "same" in extended
    assert "step 3" in extended


@pytest.mark.parametrize("skill", ["q-brainstorm", "q-skystorm"])
def test_storm_research_is_default_with_no_research_opt_out(skill: str) -> None:
    """Both storms research unless `--no-research` is passed. The old
    `If \\`--research\\`` opt-in gate must be absent. The opt-out guard must be
    in the same paragraph as the q_research call because Claude evaluates
    conditions per paragraph/list item, so a guard in a heading lets the call
    escape the gate. A character-window check is not enough."""
    text = _skill(skill)
    assert "--no-research" in text
    assert "If `--research`" not in text  # the inverted gate must not survive
    call_paras = [p for p in text.split("\n\n") if f"{_PREFIX}q_research" in p]
    assert call_paras, f"{skill}: no paragraph calls q_research"
    for para in call_paras:
        assert "--no-research" in para, (
            f"{skill}: a q_research call paragraph lacks the --no-research guard"
        )


def test_q_brainstorm_extended_round_orders_own_ideas_before_research() -> None:
    """The extended round must hold the same invariant as round 1. The
    orchestrator must write round-2 ideas before a sharper q_research call."""
    extended = _skill("q-brainstorm").split("Extended divergence")[1]
    own = extended.index("round-2 divergent/synthesis ideas")
    research = extended.index("q_research")
    council = extended.index(f"{_PREFIX}q_brainstorm_start")
    assert own < research < council, (
        "extended round: expected own-ideas -> research -> council ordering"
    )


def test_q_skystorm_analyst_pass_carries_research_param_not_pool_art() -> None:
    """The grounding pool must hold ideas only. Routing prior art through
    `prior_ideas` wraps it in ideas-to-validate framing. The analyst call must
    carry the pivot digest through `research` instead."""
    step5 = _skill("q-skystorm").split("## Step 5")[1].split("## Step 6")[0]
    assert "`research`" in step5  # analyst call passes the digest properly
    assert "key prior art" not in step5  # and the pool holds no prior art


@pytest.mark.parametrize("skill", ["q-brainstorm", "q-skystorm"])
def test_storm_orders_own_ideas_then_research_then_council(skill: str) -> None:
    """The orchestrator writes its own ideas before research, so they are not
    reactions to the digest. It then researches before starting the quorum and
    passes the raw digest through the `research` parameter."""
    text = _skill(skill)
    own = text.index("## Step 1")
    research = text.index(f"{_PREFIX}q_research")
    council = text.index(f"{_PREFIX}q_brainstorm_start")
    assert own < research < council, (
        f"{skill}: expected own-ideas -> research -> council ordering"
    )
    assert "`research`" in text  # the seeding param is named
    assert "verbatim" in text  # raw digest, not a summary


def test_q_skystorm_skill_does_not_gatekeep_cross_domain_hits() -> None:
    """Skystorm's dream-stage research explicitly wants cross-domain
    structural hits (the 'cocktail party effect' framing) -- a brainstorm-style
    sanity-check that rejects anything not from 'your domain' would filter out
    exactly the analogies this stage exists to find."""
    assert "belong to your domain" not in _skill("q-skystorm")


@pytest.mark.parametrize(
    "skill,pool_name",
    [("q-brainstorm", "prior_ideas"), ("q-skystorm", "dream pool")],
)
def test_research_note_never_routed_into_agent_facing_pool(
    skill: str, pool_name: str
) -> None:
    """The research-quality note must never be instructed into the
    prior_ideas/dream pool -- DIVERGENCE_BLOCK and GROUNDING_BLOCK
    (quorum/orchestration.py) wrap the *entire* prior_ideas string in an
    "ideas already on the table, do NOT repeat/generate" frame, so a note
    placed anywhere inside it -- even as a leading "preamble" -- still reads
    as stale content to avoid, not calibration. A Step 2 forward-reference must
    not tell the reader to carry the note into this pool because Step 4/5 says
    to keep it out."""
    text = _skill(skill)
    assert "Keep the research-quality note out of" in text
    assert f"carry it into the `{pool_name}`" not in text
    assert f"carry it forward into the {pool_name}" not in text


def test_q_skystorm_anchor_is_grounded_pivot_is_exploratory() -> None:
    """The anchor query must run in grounded mode, so an off-topic home result
    causes a retry before harvest. Only the pivots run
    exploratory. Applying exploratory to the whole protocol contaminates the
    harvested pivot terms."""
    text = _skill("q-skystorm")
    assert 'mode="grounded"' in text
    assert 'mode="exploratory"' in text
    # The grounded call must be tied to the anchor step, exploratory to the pivot.
    assert 'Anchor (`mode="grounded"`)' in text
    assert 'Pivot (`mode="exploratory"`)' in text


def test_q_skystorm_absence_from_field_map_is_inconclusive() -> None:
    """The field map shows only top-ranked subfields, so an absent
    field is not proof of no connection. The skill must not claim otherwise."""
    text = _skill("q-skystorm")
    assert "genuinely unrelated" not in text
    assert "Absence is inconclusive" in text


def test_q_research_skill_names_sync_tool_and_keeps_retry_protocol() -> None:
    """q-research fronts the synchronous q_research tool directly — no
    start/await pair. It must name the exact plugin-prefixed tool, keep the
    same verdict-keyed action protocol as the storms, and steer query formation.
    This contract prevents drift outside the shared research protocol."""
    text = _skill("q-research")
    # Markdown hard-wraps at ~78 cols, so multi-word phrases may span a
    # newline+indent; normalize before phrase checks.
    flat = " ".join(text.split())
    assert f"{_PREFIX}q_research" in text
    assert "Research status:" in text
    assert "RETRY-REQUIRED" in text
    assert "specific failure to avoid" in flat
    assert "reworked retry has **also** failed" in flat
    assert "Research needs action: true" in text
    assert "### Required research actions" in text
    assert "RETRY-SAME" in text
    assert "RE-ANCHOR" in text
    assert "short is not the same as generic" in flat
    assert "domain-specific terms" in flat


def test_q_research_skill_requires_semantic_lanes_and_per_lane_statuses() -> None:
    text = _skill("q-research")
    flat = " ".join(text.split())

    assert "query_lanes" in text
    assert "purpose" in text
    assert "methods" in text and "currency" in text
    for status in (
        "ON-TOPIC",
        "THIN",
        "QUERY-COLLISION",
        "SOURCE-MISMATCH",
        "INFRASTRUCTURE",
        "CONFIG",
    ):
        assert status in text
    assert "semantic re-anchor" in flat
    assert "mechanical shortening" in flat


def test_q_skystorm_skill_routes_required_action_table() -> None:
    """Skystorm must use the same explicit action routing as q-research and
    q-brainstorm."""
    text = _skill("q-skystorm")
    assert "Research needs action: true" in text
    assert "### Required research actions" in text
    assert "RETRY-SAME" in text
    assert "RE-ANCHOR" in text


def test_claude_skystorm_handles_a_degraded_dream_council() -> None:
    text = _skill("q-skystorm")
    step4 = text.split("## Step 4")[1].split("## Step 5")[0]
    step6 = text.split("## Step 6")[1]

    assert "Council degraded:" in step4
    assert "each unavailable seat and its status" in step4
    assert "available seats" in step4
    assert "never describe the spread as complete or full" in step4
    assert "seat-helper-status" not in step4
    assert "full spread" not in step6
