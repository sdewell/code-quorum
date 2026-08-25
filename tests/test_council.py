import asyncio

import pytest

from quorum.agents import Agent, AgentResult
from quorum.council import (
    AGENT_TIMEOUT_S,
    REVIEW_REVISE_INSTRUCTION,
    REVIEW_REVISE_TERSE_INSTRUCTION,
    REVISE_INSTRUCTION,
    ROTATION_INSTRUCTION,
    _round_instruction,
    build_rotation_prompt,
    build_round_prompt,
    run_council,
)
from quorum.roles import ROLES


def test_default_agent_timeout_allows_fifteen_minute_call() -> None:
    assert AGENT_TIMEOUT_S == 900.0


def test_build_round_prompt_revise_includes_instruction() -> None:
    out = build_round_prompt(
        base_prompt="BASE",
        own_prior="MINE",
        peer_priors={"codex": "C", "gemini": "G"},
        peer_roles={"codex": "skeptic", "gemini": "architect"},
        mode="revise",
    )
    assert "BASE" in out
    assert "MINE" in out
    assert '<peer stance="skeptic">' in out
    assert "C" in out
    assert '<peer stance="architect">' in out
    assert "G" in out
    assert "Revise your answer" in out
    # Peer blocks are anonymized: stance only, never agent identity.
    assert "codex" not in out
    assert "gemini" not in out


def test_build_round_prompt_critique_uses_critique_instruction() -> None:
    out = build_round_prompt(
        base_prompt="BASE",
        own_prior="MINE",
        peer_priors={"x": "X"},
        peer_roles={"x": "neutral"},
        mode="critique",
    )
    assert "Find flaws" in out
    assert "Revise your answer" not in out


class _FakeAgent(Agent):
    """Real Agent ABC implementation used as a deterministic test fixture.
    Captures prompts it was called with so tests can assert on the orchestrator
    behavior without spawning subprocesses."""

    def __init__(self, name: str, outputs: list[str]) -> None:
        self.name = name
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        self.prompts.append(prompt)
        out = self._outputs.pop(0) if self._outputs else ""
        return AgentResult(agent=self.name, output=out, returncode=0)


@pytest.mark.asyncio
async def test_run_council_single_round_returns_one_layer() -> None:
    a = _FakeAgent("a", ["A1"])
    b = _FakeAgent("b", ["B1"])
    rounds = await run_council(
        agents=[a, b], prompt="QUERY", cwd="/", phase="plan", rounds=1
    )
    assert len(rounds) == 1
    assert {r.agent for r in rounds[0]} == {"a", "b"}
    assert a.prompts == ["QUERY"]
    assert b.prompts == ["QUERY"]


@pytest.mark.asyncio
async def test_run_council_stamps_resolved_model_on_results() -> None:
    # Stamped centrally here (not in each seat's run()) so every report can
    # show stance + seat + model without touching agent construction sites.
    a = _FakeAgent("a", ["A1"])
    a.model = "model-x"
    b = _FakeAgent("b", ["B1"])  # no model attr -> stays ""
    rounds = await run_council(
        agents=[a, b], prompt="QUERY", cwd="/", phase="plan", rounds=1
    )
    by_agent = {r.agent: r for r in rounds[0]}
    assert by_agent["a"].model == "model-x"
    assert by_agent["b"].model == ""


class _EngineSwapAgent(_FakeAgent):
    """Seat that stamps its own model mid-run (the agy quota-reflex shape)."""

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        res = await super().run(prompt, cwd)
        res.model = "fallback-model"
        return res


@pytest.mark.asyncio
async def test_run_council_preserves_model_a_seat_stamped_itself() -> None:
    # The agy quota reflex reruns on a fallback engine and stamps it on the
    # result; the central stamp must not overwrite it with the configured one.
    a = _EngineSwapAgent("a", ["A1"])
    a.model = "configured-model"
    rounds = await run_council(
        agents=[a], prompt="QUERY", cwd="/", phase="plan", rounds=1
    )
    assert rounds[0][0].model == "fallback-model"


@pytest.mark.asyncio
async def test_run_council_two_rounds_injects_own_and_peer_priors() -> None:
    a = _FakeAgent("a", ["A1", "A2"])
    b = _FakeAgent("b", ["B1", "B2"])
    rounds = await run_council(
        agents=[a, b],
        prompt="QUERY",
        cwd="/",
        phase="validate",
        rounds=2,
        mode="revise",
    )
    assert len(rounds) == 2
    assert a.prompts[0] == "QUERY"
    a_r2 = a.prompts[1]
    assert "QUERY" in a_r2
    assert "A1" in a_r2
    assert "B1" in a_r2
    assert '<peer stance="neutral">' in a_r2
    # Own output must not reappear as a peer block: exactly one peer.
    assert a_r2.count("<peer ") == 1
    b_r2 = b.prompts[1]
    assert "B1" in b_r2
    assert "A1" in b_r2
    assert '<peer stance="neutral">' in b_r2
    assert b_r2.count("<peer ") == 1
    # Anonymized: agent names never appear in round prompts.
    assert "<peer agent=" not in a_r2
    assert "<peer agent=" not in b_r2


@pytest.mark.asyncio
async def test_run_council_drops_failed_agent_from_later_rounds() -> None:
    class _FailingAgent(Agent):
        name = "bad"

        async def run(self, prompt: str, cwd: str) -> AgentResult:
            return AgentResult(agent=self.name, output="", error="boom", returncode=1)

    good = _FakeAgent("good", ["G1", "G2"])
    bad = _FailingAgent()
    rounds = await run_council(
        agents=[good, bad], prompt="Q", cwd="/", phase="plan", rounds=2
    )
    assert len(rounds) == 2
    r2_agents = [r.agent for r in rounds[1]]
    assert "good" in r2_agents
    assert "bad" not in r2_agents


@pytest.mark.asyncio
async def test_run_council_halts_when_all_fail_in_round_one() -> None:
    class _AllFail(Agent):
        def __init__(self, name: str) -> None:
            self.name = name

        async def run(self, prompt: str, cwd: str) -> AgentResult:
            return AgentResult(agent=self.name, output="", returncode=1)

    rounds = await run_council(
        agents=[_AllFail("x"), _AllFail("y")],
        prompt="Q",
        cwd="/",
        phase="plan",
        rounds=3,
    )
    assert len(rounds) == 1


@pytest.mark.asyncio
async def test_run_council_rejects_zero_rounds() -> None:
    a = _FakeAgent("a", ["A"])
    with pytest.raises(ValueError):
        await run_council(agents=[a], prompt="Q", cwd="/", phase="plan", rounds=0)


def test_build_rotation_prompt_tags_priors_by_stance() -> None:
    out = build_rotation_prompt(
        base_prompt="BASE",
        priors=[("skeptic", "S-OUT"), ("architect", "A-OUT")],
    )
    assert "BASE" in out
    assert '<prior stance="skeptic">' in out
    assert "S-OUT" in out
    assert '<prior stance="architect">' in out
    assert "A-OUT" in out
    assert "your_prior_answer" not in out


@pytest.mark.asyncio
async def test_run_council_applies_role_prefix_round_one() -> None:
    a = _FakeAgent("a", ["A1"])
    a.role = "skeptic"
    rounds = await run_council(
        agents=[a], prompt="QUERY", cwd="/", phase="plan", rounds=1
    )
    assert ROLES["skeptic"]["plan"] in a.prompts[0]
    assert "QUERY" in a.prompts[0]
    assert rounds[0][0].role == "skeptic"


@pytest.mark.asyncio
async def test_run_council_rotates_stances_at_round_three() -> None:
    a = _FakeAgent("a", ["A1", "A2", "A3"])
    a.role = "skeptic"
    b = _FakeAgent("b", ["B1", "B2", "B3"])
    b.role = "architect"
    rounds = await run_council(
        agents=[a, b], prompt="Q", cwd="/", phase="validate", rounds=3, rotate=True
    )
    assert len(rounds) == 3
    assert {res.agent: res.role for res in rounds[2]} == {
        "a": "architect",
        "b": "skeptic",
    }
    a_r3 = a.prompts[2]
    assert ROLES["architect"]["validate"] in a_r3
    assert '<prior stance="skeptic">' in a_r3
    assert '<prior stance="architect">' in a_r3


@pytest.mark.asyncio
async def test_run_council_no_rotation_keeps_stances() -> None:
    a = _FakeAgent("a", ["A1", "A2", "A3"])
    a.role = "skeptic"
    b = _FakeAgent("b", ["B1", "B2", "B3"])
    b.role = "architect"
    rounds = await run_council(
        agents=[a, b], prompt="Q", cwd="/", phase="validate", rounds=3, rotate=False
    )
    assert {res.agent: res.role for res in rounds[2]} == {
        "a": "skeptic",
        "b": "architect",
    }


class _HangingAgent(Agent):
    name = "hang"

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        await asyncio.sleep(30)
        return AgentResult(agent=self.name, output="never", returncode=0)


class _RaisingAgent(Agent):
    name = "boom"

    async def run(self, prompt: str, cwd: str) -> AgentResult:
        raise RuntimeError("argv list too long")


@pytest.mark.asyncio
async def test_run_council_times_out_hung_agent() -> None:
    good = _FakeAgent("good", ["G1"])
    rounds = await run_council(
        agents=[_HangingAgent(), good],
        prompt="Q",
        cwd="/",
        phase="plan",
        rounds=1,
        agent_timeout=0.05,
    )
    results = {r.agent: r for r in rounds[0]}
    assert results["hang"].returncode != 0
    assert "timed out" in results["hang"].error
    assert results["good"].output == "G1"


@pytest.mark.asyncio
async def test_run_council_contains_raising_agent() -> None:
    good = _FakeAgent("good", ["G1"])
    rounds = await run_council(
        agents=[_RaisingAgent(), good], prompt="Q", cwd="/", phase="plan", rounds=1
    )
    results = {r.agent: r for r in rounds[0]}
    assert results["boom"].returncode != 0
    assert "RuntimeError" in results["boom"].error
    assert results["good"].output == "G1"


# --- revision-round discipline (empty-turn dropout fix) ----------------------
# Evidence (TraceMark, 2026-06-13): in round 2 the model re-read the whole repo
# (31 tool calls, heavy reasoning) and ended its turn with only whitespace --
# no revised review. The fix nudges revision rounds away from re-surveying the
# repo and toward ending with a written answer.


def test_revise_instruction_discourages_resurvey_and_demands_written_answer() -> None:
    from quorum.council import REVISE_INSTRUCTION

    low = REVISE_INSTRUCTION.lower()
    assert "re-survey" in low or "re-read" in low or "already" in low
    assert "written" in low or "prose" in low


def test_critique_instruction_discourages_resurvey_and_demands_written_answer() -> None:
    from quorum.council import CRITIQUE_INSTRUCTION

    low = CRITIQUE_INSTRUCTION.lower()
    assert "re-survey" in low or "re-read" in low or "already" in low
    assert "written" in low or "prose" in low


def test_build_round_prompt_carries_revision_discipline() -> None:
    from quorum.council import REVISE_INSTRUCTION

    out = build_round_prompt(
        base_prompt="BASE",
        own_prior="MINE",
        peer_priors={"codex": "C"},
        peer_roles={"codex": "skeptic"},
        mode="revise",
    )
    assert REVISE_INSTRUCTION in out


# --- review-phase revise carve-out (convergence must keep held findings visible) ---
# A review converges by making sustained agreement visible across rounds. Terse (the
# default) keeps that signal by compressing each still-held finding to one line rather
# than restating it in full; --verbose restores today's full re-listing. Either way the
# instruction differs from the shared "do not restate what is unchanged" clause that
# q-plan / q-validate use; every non-review (mode, phase) stays byte-identical.


def test_build_round_prompt_review_revise_terse_by_default() -> None:
    out = build_round_prompt(
        base_prompt="BASE",
        own_prior="MINE",
        peer_priors={"codex": "C"},
        peer_roles={"codex": "skeptic"},
        mode="revise",
        phase="review",
    )
    # Default review deliberation compresses held findings to one compact line
    # each (agreement still countable) rather than restating them in full.
    assert REVIEW_REVISE_TERSE_INSTRUCTION in out
    assert "HELD" in out
    assert "Re-list every finding you still hold" not in out


def test_build_round_prompt_review_revise_verbose_full_relist() -> None:
    out = build_round_prompt(
        base_prompt="BASE",
        own_prior="MINE",
        peer_priors={"codex": "C"},
        peer_roles={"codex": "skeptic"},
        mode="revise",
        phase="review",
        verbose=True,
    )
    # --verbose restores today's full re-listing of every held finding.
    assert REVIEW_REVISE_INSTRUCTION in out
    assert "Re-list every finding you still hold" in out


def test_build_round_prompt_validate_revise_byte_identical_to_today() -> None:
    def build(phase: str) -> str:
        return build_round_prompt(
            base_prompt="BASE",
            own_prior="MINE",
            peer_priors={"codex": "C"},
            peer_roles={"codex": "skeptic"},
            mode="revise",
            phase=phase,
        )

    out = build("validate")
    # The appended round instruction is exactly the unchanged REVISE_INSTRUCTION
    # (byte-for-byte) — not the review variant — for every non-review phase.
    assert out.endswith(f"{REVISE_INSTRUCTION}\n")
    assert REVIEW_REVISE_INSTRUCTION not in out
    for phase in ("plan", "brainstorm", "validate", ""):
        assert build(phase) == out


@pytest.mark.asyncio
async def test_run_council_review_revise_terse_by_default() -> None:
    a = _FakeAgent("codex", ["r1", "r2"])
    b = _FakeAgent("gemini", ["r1", "r2"])
    await run_council(
        agents=[a, b], prompt="Q", cwd="/", phase="review", rounds=2, mode="revise"
    )
    # The round-2 prompt carries the compressed (terse) re-listing instruction.
    assert "HELD" in a.prompts[1]
    assert "Re-list every finding you still hold" not in a.prompts[1]


@pytest.mark.asyncio
async def test_run_council_review_revise_verbose_threads_full_relist() -> None:
    a = _FakeAgent("codex", ["r1", "r2"])
    b = _FakeAgent("gemini", ["r1", "r2"])
    await run_council(
        agents=[a, b],
        prompt="Q",
        cwd="/",
        phase="review",
        rounds=2,
        mode="revise",
        verbose=True,
    )
    assert "Re-list every finding you still hold" in a.prompts[1]


# --- terse compression is a FINAL-round concern; intermediate rounds stay full ---
# Terse HELD lines are lossy: a still-held finding collapses to an <=8-word claim.
# That is fine for the LAST round (its only consumer is the orchestrator counting
# agreement) but ruinous mid-deliberation, because a round's output becomes the
# NEXT round's prior context. If round 2 already collapsed to HELD stubs, round 3
# agents (esp. after stance rotation) critique from stubs, not evidence. So terse
# applies only on the final round; earlier review-revise rounds re-list in full.


def test_round_instruction_review_intermediate_round_relists_in_full() -> None:
    # Non-final review-revise round: keep full detail so the next round's priors
    # carry evidence, not <=8-word HELD stubs.
    assert (
        _round_instruction("revise", "review", verbose=False, is_final_round=False)
        == REVIEW_REVISE_INSTRUCTION
    )
    # Final review-revise round: compress to HELD (output only feeds the verdict).
    assert (
        _round_instruction("revise", "review", verbose=False, is_final_round=True)
        == REVIEW_REVISE_TERSE_INSTRUCTION
    )
    # verbose always re-lists in full, final or not.
    assert (
        _round_instruction("revise", "review", verbose=True, is_final_round=True)
        == REVIEW_REVISE_INSTRUCTION
    )


# --- rotation (round 3 of --extended) must still carry the review-revise contract ---
# The rotation branch used to emit ROTATION_INSTRUCTION alone, bypassing both the
# terse HELD contract and the verbose full re-list. For a review that silently
# dropped the agreement-counting signal in the exact 4-round path the terse work
# targets. Rotation now appends the selected review-revise instruction — but only
# for the review phase; every other phase keeps rotation-only behavior byte-for-byte.


def test_build_rotation_prompt_review_final_carries_terse_contract() -> None:
    out = build_rotation_prompt(
        base_prompt="BASE",
        priors=[("skeptic", "S")],
        mode="revise",
        phase="review",
        is_final_round=True,
    )
    assert ROTATION_INSTRUCTION in out  # rotation semantics preserved
    assert "HELD" in out  # final review round -> terse contract restored
    assert "Re-list every finding you still hold" not in out


def test_build_rotation_prompt_review_intermediate_carries_full_relist() -> None:
    out = build_rotation_prompt(
        base_prompt="BASE",
        priors=[("skeptic", "S")],
        mode="revise",
        phase="review",
        is_final_round=False,
    )
    assert ROTATION_INSTRUCTION in out
    assert "Re-list every finding you still hold" in out  # intermediate -> full
    assert "HELD" not in out


def test_build_rotation_prompt_non_review_is_rotation_only() -> None:
    # Non-review phases keep rotation-only behavior (no review-revise contract).
    out = build_rotation_prompt(
        base_prompt="BASE",
        priors=[("skeptic", "S")],
        mode="revise",
        phase="validate",
        is_final_round=True,
    )
    assert out.endswith(f"{ROTATION_INSTRUCTION}\n")
    assert "HELD" not in out
    assert "Re-list every finding you still hold" not in out


@pytest.mark.asyncio
async def test_run_council_review_multiround_compresses_only_final_round() -> None:
    # 4-round non-rotate review: intermediate rounds re-list in full; only the
    # final round collapses to HELD.
    a = _FakeAgent("codex", ["r1", "r2", "r3", "r4"])
    b = _FakeAgent("gemini", ["r1", "r2", "r3", "r4"])
    await run_council(
        agents=[a, b], prompt="Q", cwd="/", phase="review", rounds=4, mode="revise"
    )
    # round 2 (index 1) is intermediate -> full re-list, not HELD.
    assert "Re-list every finding you still hold" in a.prompts[1]
    assert "HELD" not in a.prompts[1]
    # round 4 (index 3) is final -> terse HELD.
    assert "HELD" in a.prompts[3]
    assert "Re-list every finding you still hold" not in a.prompts[3]


@pytest.mark.asyncio
async def test_run_council_review_extended_rotation_round3_carries_contract() -> None:
    # 4-round rotate review: round 3 (index 2) is the rotation round AND
    # intermediate -> it must carry ROTATION_INSTRUCTION *and* the full re-list
    # contract (not the old rotation-only void).
    a = _FakeAgent("codex", ["r1", "r2", "r3", "r4"])
    a.role = "skeptic"
    b = _FakeAgent("gemini", ["r1", "r2", "r3", "r4"])
    b.role = "architect"
    await run_council(
        agents=[a, b],
        prompt="Q",
        cwd="/",
        phase="review",
        rounds=4,
        mode="revise",
        rotate=True,
    )
    r3 = a.prompts[2]
    assert ROTATION_INSTRUCTION in r3
    assert "Re-list every finding you still hold" in r3
    # round 4 (index 3) final -> terse HELD.
    assert "HELD" in a.prompts[3]


@pytest.mark.asyncio
async def test_run_council_review_3round_rotation_round3_is_terse_final() -> None:
    # 3-round rotate review: round 3 (index 2) is rotation AND final -> terse HELD
    # rides on top of the rotation instruction.
    a = _FakeAgent("codex", ["r1", "r2", "r3"])
    a.role = "skeptic"
    b = _FakeAgent("gemini", ["r1", "r2", "r3"])
    b.role = "architect"
    await run_council(
        agents=[a, b],
        prompt="Q",
        cwd="/",
        phase="review",
        rounds=3,
        mode="revise",
        rotate=True,
    )
    r3 = a.prompts[2]
    assert ROTATION_INSTRUCTION in r3
    assert "HELD" in r3
    assert "Re-list every finding you still hold" not in r3


@pytest.mark.asyncio
async def test_run_council_review_rotation_verbose_full_relist() -> None:
    # --verbose extended rotate review: rotation round re-lists in full.
    a = _FakeAgent("codex", ["r1", "r2", "r3"])
    a.role = "skeptic"
    b = _FakeAgent("gemini", ["r1", "r2", "r3"])
    b.role = "architect"
    await run_council(
        agents=[a, b],
        prompt="Q",
        cwd="/",
        phase="review",
        rounds=3,
        mode="revise",
        rotate=True,
        verbose=True,
    )
    r3 = a.prompts[2]
    assert ROTATION_INSTRUCTION in r3
    assert "Re-list every finding you still hold" in r3
    assert "HELD" not in r3
