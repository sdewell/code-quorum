"""Multi-round orchestration. Each round invokes every agent in parallel via
fresh subprocess calls. Round 1 uses the original prompt; rounds 2+ inject
each agent's own prior answer and the peers' priors so the agent can revise
or critique. Round 3, when `rotate` is set, swaps stances among the active
agents and presents all priors flat, tagged by producing-stance. Each agent's
prompt is prefixed with its current stance. Stateless by design — no session
resumption, no shared memory between rounds beyond the prompt text we build."""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from .agents import Agent, AgentResult
from .roles import apply_role, rotate_roles

Mode = Literal["revise", "critique"]

# Hard ceiling per agent call. Agent CLIs legitimately take minutes to
# review a codebase; this is a safety net against an unbounded hang (e.g.
# a subprocess blocked on an interactive auth prompt), not a tight SLA.
AGENT_TIMEOUT_S = 900.0

# Appended to every revision-round instruction. Round 2+ agents already read
# the repo in round 1; a reasoning model that re-surveys it from scratch can
# burn its whole turn on tool calls and end without writing an answer (observed
# on TraceMark 2026-06-13: 31 reads + heavy reasoning, then a whitespace-only
# turn). Steer them to reason from what they already have and to finish with a
# written response, not a tool call.
_REVISION_DISCIPLINE = (
    " You already examined the repository in the previous round — do not "
    "re-survey it; work from what you and your peers already established, "
    "reading a specific file only if you must verify a precise claim. End with "
    "your written response in prose."
)

REVISE_INSTRUCTION = (
    "Revise your answer. Incorporate valid points from peers and push back "
    "on points you disagree with. Be direct. Do not restate what is "
    "unchanged." + _REVISION_DISCIPLINE
)

# Review-phase variant of REVISE_INSTRUCTION. A review converges by making
# sustained agreement visible across rounds, so — unlike q-plan / q-validate —
# agents must re-list findings they still hold even when unchanged (silence is
# ambiguous between "I retract" and "unchanged"). Composes the shared
# _REVISION_DISCIPLINE; only the trailing "do not restate" clause is swapped.
REVIEW_REVISE_INSTRUCTION = (
    "Revise your answer. Incorporate valid points from peers and push back "
    "on points you disagree with. Be direct. Re-list every finding you still "
    "hold, even unchanged, so sustained agreement is visible." + _REVISION_DISCIPLINE
)

# Terse variant of REVIEW_REVISE_INSTRUCTION, and the review-phase default.
# Sustained agreement must still be visible across rounds, but restating every
# held finding in full is the dominant source of review-matrix bloat. So held
# findings collapse to one compact line -- enough to keep the N-of-M agreement
# count countable -- and only new, severity-changed, or retracted findings get
# full detail. --verbose swaps this back for REVIEW_REVISE_INSTRUCTION.
REVIEW_REVISE_TERSE_INSTRUCTION = (
    "Revise your answer. Incorporate valid points from peers and push back on "
    "points you disagree with. Be direct. For each finding you still hold "
    "unchanged, emit ONE compact line -- `HELD <path:line> | <severity> | claim "
    "in <=8 words>` -- do not restate its detail. Give full detail only for a NEW "
    "finding, one whose severity you are CHANGING (say why), or one you are "
    "RETRACTING. This keeps sustained agreement visible without restating "
    "unchanged findings." + _REVISION_DISCIPLINE
)

CRITIQUE_INSTRUCTION = (
    "Find flaws in the peers' answers. What did they miss? What is wrong? "
    "Be direct. Focus on the critique; do not restate your own answer."
    + _REVISION_DISCIPLINE
)

ROTATION_INSTRUCTION = (
    "Prior analyses from the earlier rounds are below, each tagged with the "
    "stance that produced it. Examine them from your stance — what do they "
    "miss, and what does your stance reveal that theirs did not?"
)


def _round_instruction(
    mode: Mode, phase: str, verbose: bool = False, is_final_round: bool = True
) -> str:
    """Select the revision-round instruction. Review revise rounds keep still-held
    findings visible so convergence is countable. Terse HELD compression is lossy
    (a held finding collapses to an <=8-word claim), so it applies ONLY on the
    final round -- whose output feeds the orchestrator's agreement count and
    nothing else. Earlier review-revise rounds re-list in full, because a round's
    output is the next round's prior context and later agents (esp. after stance
    rotation) must critique from evidence, not stubs. --verbose always re-lists in
    full. Every other (mode, phase) keeps today's instruction byte-for-byte."""
    if mode == "revise" and phase == "review":
        if not verbose and is_final_round:
            return REVIEW_REVISE_TERSE_INSTRUCTION
        return REVIEW_REVISE_INSTRUCTION
    return REVISE_INSTRUCTION if mode == "revise" else CRITIQUE_INSTRUCTION


def build_round_prompt(
    *,
    base_prompt: str,
    own_prior: str,
    peer_priors: dict[str, str],
    peer_roles: dict[str, str],
    mode: Mode,
    phase: str = "",
    verbose: bool = False,
    is_final_round: bool = True,
) -> str:
    # Peer tags carry stance, never agent identity: LLM judges show
    # self/peer-preference bias toward known model names, so withholding
    # them keeps the weighing of peer output on its content. Same convention
    # as build_rotation_prompt. Tag metadata only — a peer's own text may
    # still self-identify; scrubbing names from content would corrupt
    # legitimate mentions (e.g. reviews of code that names the seats).
    # Stances collide only when an opt-in seat doubles a stance (e.g. two
    # neutrals) — accepted: blocks stay separate.
    peer_blocks = "\n".join(
        f'<peer stance="{peer_roles.get(name, "neutral")}">\n{out}\n</peer>'
        for name, out in peer_priors.items()
    )
    return (
        f"{base_prompt}\n\n"
        f"<your_prior_answer>\n{own_prior}\n</your_prior_answer>\n\n"
        f"<peer_answers>\n{peer_blocks}\n</peer_answers>\n\n"
        f"{_round_instruction(mode, phase, verbose, is_final_round)}\n"
    )


def build_rotation_prompt(
    *,
    base_prompt: str,
    priors: list[tuple[str, str]],
    mode: Mode = "revise",
    phase: str = "",
    verbose: bool = False,
    is_final_round: bool = True,
) -> str:
    """Round-3 rotation prompt. Priors are presented flat — no own/peer
    distinction — each tagged by the stance that produced it. `priors` is a
    list of (stance, output) pairs.

    Rotation swaps stances, but a review still needs its convergence contract:
    for the review phase the selected review-revise instruction (terse HELD when
    final, full re-list otherwise / under --verbose) is appended after the
    rotation instruction, so round 3 does not silently drop agreement counting.
    Every other phase keeps rotation-only behavior byte-for-byte."""
    blocks = "\n".join(
        f'<prior stance="{stance}">\n{out}\n</prior>' for stance, out in priors
    )
    instruction = ROTATION_INSTRUCTION
    if mode == "revise" and phase == "review":
        instruction = (
            f"{ROTATION_INSTRUCTION}\n\n"
            f"{_round_instruction(mode, phase, verbose, is_final_round)}"
        )
    return (
        f"{base_prompt}\n\n"
        f"<prior_analyses>\n{blocks}\n</prior_analyses>\n\n"
        f"{instruction}\n"
    )


async def _run_agent_contained(
    agent: Agent, *, prompt: str, cwd: str, timeout: float
) -> AgentResult:
    """Run one agent with a hard timeout and full exception containment.
    A hung subprocess, an oversized-argv OSError, or any other agent-level
    failure must become a failed AgentResult — never hang the council or
    abort peers that succeeded."""
    start = time.monotonic()
    try:
        return await asyncio.wait_for(agent.run(prompt=prompt, cwd=cwd), timeout)
    except TimeoutError:
        return AgentResult(
            agent=agent.name,
            output="",
            error=f"agent timed out after {timeout:.0f}s",
            returncode=124,
            duration_s=time.monotonic() - start,
            role=agent.role,
            unavailable_reason="timeout",
        )
    except Exception as exc:
        return AgentResult(
            agent=agent.name,
            output="",
            error=f"agent crashed: {type(exc).__name__}: {exc}",
            returncode=1,
            duration_s=time.monotonic() - start,
            role=agent.role,
        )


async def run_council(
    *,
    agents: list[Agent],
    prompt: str,
    cwd: str,
    phase: str,
    rounds: int = 1,
    mode: Mode = "revise",
    rotate: bool = False,
    verbose: bool = False,
    agent_timeout: float = AGENT_TIMEOUT_S,
) -> list[list[AgentResult]]:
    """Run `rounds` parallel rounds. Returns a rounds × agents-in-round
    matrix. Each agent's prompt is prefixed with its current stance, expressed
    in the form appropriate to `phase` (`plan` / `brainstorm` / `validate`).
    When `rotate` is set, stances rotate among the active agents at round 3.
    Agents that fail in a round are dropped from later rounds; if no agent
    has a usable prior, the loop halts early."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    all_rounds: list[list[AgentResult]] = []
    prior_outputs: dict[str, str] = {}
    current_roles: dict[str, str] = {a.name: a.role for a in agents}
    prior_roles: dict[str, str] = dict(current_roles)

    for r in range(rounds):
        if r == 0:
            active = list(agents)
        else:
            active = [a for a in agents if prior_outputs.get(a.name)]
            if not active:
                break

        rotation_priors: list[tuple[str, str]] = []
        if r == 2 and rotate:
            current_roles = rotate_roles([a.name for a in active], current_roles)
            rotation_priors = [
                (prior_roles[name], out) for name, out in prior_outputs.items()
            ]

        # Terse HELD compression is a final-round-only concern (see
        # _round_instruction): a non-final round's output feeds the next round's
        # priors and must stay detailed.
        is_final = r == rounds - 1
        prompts: dict[str, str] = {}
        for a in active:
            if r == 0:
                body = prompt
            elif r == 2 and rotate:
                body = build_rotation_prompt(
                    base_prompt=prompt,
                    priors=rotation_priors,
                    mode=mode,
                    phase=phase,
                    verbose=verbose,
                    is_final_round=is_final,
                )
            else:
                body = build_round_prompt(
                    base_prompt=prompt,
                    own_prior=prior_outputs[a.name],
                    peer_priors={n: o for n, o in prior_outputs.items() if n != a.name},
                    peer_roles=current_roles,
                    mode=mode,
                    phase=phase,
                    verbose=verbose,
                    is_final_round=is_final,
                )
            prompts[a.name] = apply_role(current_roles[a.name], body, phase)

        round_results = await asyncio.gather(
            *(
                _run_agent_contained(
                    a, prompt=prompts[a.name], cwd=cwd, timeout=agent_timeout
                )
                for a in active
            )
        )
        for a, res in zip(active, round_results, strict=True):
            res.role = current_roles.get(res.agent, res.role)
            # A seat that swapped engines mid-run (the agy quota reflex) has
            # already stamped the model that actually answered; never
            # overwrite it with the configured one.
            res.model = res.model or a.model
        all_rounds.append(round_results)
        prior_roles = dict(current_roles)
        prior_outputs = {
            res.agent: res.output
            for res in round_results
            if res.returncode == 0 and res.output
        }
    return all_rounds
