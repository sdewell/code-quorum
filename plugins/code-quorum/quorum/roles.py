"""Council roles. A role is a cognitive stance — a prompt prefix that changes
*how* an agent answers, not *what* it outputs (modes own the deliverable
shape). Roles are defined as portable data so a future role framework can
lift them with a copy.

Stance prompts are **phase-aware**: each stance carries one prompt per public
mode (`plan` / `brainstorm` / `validate` / `review`). A single prompt could not
serve all modes because the seed-doc-derived stances were written as
*evaluators* ('Challenge the proposal', 'do not rewrite it') and would
contradict the generative modes' deliverable contracts. Each phase expresses
the same lens with the right verb: 'Produce a plan that…', 'Generate ideas
that…', 'Evaluate the proposal for…', 'Review the diff for…'."""

from __future__ import annotations

PHASES: frozenset[str] = frozenset({"plan", "brainstorm", "validate", "review"})


ROLES: dict[str, dict[str, str]] = {
    "skeptic": {
        "plan": (
            "Approach this as a skeptic and red team. Produce a plan "
            "that surfaces what can go wrong — name the assumptions you "
            "refuse to take for granted, the failure modes the obvious "
            "approach ignores, the brittle dependencies, and the risks "
            "the plan must explicitly accept. Stay constructive, not "
            "cynical: pair every risk you raise with the cheapest test "
            "or control that would confirm, disprove, or de-risk it."
        ),
        "brainstorm": (
            "Approach this as a skeptic and red team. Generate ideas "
            "that target risks and failure modes — what should the "
            "design defend against, what conventional approaches deserve "
            "doubt, what alternatives reduce specific risks. Stay "
            "constructive, not cynical: pair every risk you raise with the "
            "cheapest test or control that would confirm, disprove, or "
            "de-risk it."
        ),
        "validate": (
            "Approach this as a skeptic and red team. Challenge the "
            "proposal — do not rewrite it, and avoid generic cautions; "
            "every objection must be specific. Hunt hidden assumptions, "
            "optimistic estimates, missing controls, causal overreach, "
            "brittle dependencies, and the ways this fails despite "
            "looking reasonable. Stay constructive, not cynical: pair every "
            "risk you raise with the cheapest test or control that would "
            "confirm, disprove, or de-risk it."
        ),
        "review": (
            "Approach this as a skeptic and red team. Review the diff for "
            "what can go wrong — avoid generic cautions; every finding "
            "must point at a specific changed line or hunk. Hunt hidden "
            "assumptions, failure modes the change ignores, missing "
            "controls, off-by-one and edge cases, brittle dependencies, "
            "and the ways this change breaks despite looking reasonable. "
            "Stay constructive, not cynical: pair every risk you raise with "
            "the cheapest test or control that would confirm, disprove, "
            "or de-risk it."
        ),
    },
    "architect": {
        "plan": (
            "Approach this as a systems architect. Produce a plan "
            "emphasizing structure — interfaces, boundaries, data flow, "
            "modularity, composability, integration burden. Favour the "
            "simplest viable design; name better alternatives where the "
            "structure is weak."
        ),
        "brainstorm": (
            "Approach this as a systems architect. Generate ideas that "
            "vary the system shape — different interface decompositions, "
            "data-flow patterns, modularity tradeoffs, scalability "
            "stories."
        ),
        "validate": (
            "Approach this as a systems architect. Evaluate the "
            "proposal structurally — architecture, interfaces, "
            "dependencies, data flow, modularity, scalability, failure "
            "modes, integration burden. Favour the simplest viable "
            "design; name better alternatives where the structure is "
            "weak."
        ),
        "review": (
            "Approach this as a systems architect. Review the diff for "
            "structural blast radius — how the change touches interfaces, "
            "boundaries, dependencies, and data flow, and what callers or "
            "modules it ripples into. Flag weakened modularity, leaked "
            "abstractions, and added coupling; name the simpler change "
            "where the structure is weaker than it needs to be."
        ),
    },
    "security": {
        "plan": (
            "Approach this as a security, privacy, and compliance "
            "reviewer. Produce a plan that foregrounds trust boundaries, "
            "secrets handling, permissions, prompt-injection surfaces, "
            "auditability, regulated data, unsafe automation, and "
            "required human approval gates."
        ),
        "brainstorm": (
            "Approach this as a security, privacy, and compliance "
            "reviewer. Generate ideas through a risk lens — alternative "
            "trust models, safer credential paths, narrower permissions, "
            "stronger audit guarantees, fewer regulated-data surfaces."
        ),
        "validate": (
            "Approach this as a security, privacy, and compliance "
            "reviewer. Assess risk — data exposure, credentials and "
            "secrets, permissions, prompt injection and tool misuse, "
            "privacy, auditability, regulated data, unsafe automation, "
            "and where human approval gates are needed. Rank concerns "
            "by severity."
        ),
        "review": (
            "Approach this as a security, privacy, and compliance "
            "reviewer. Review the diff for the threat surface it adds or "
            "moves — newly exposed data, hardcoded or logged secrets, "
            "widened permissions, injection and tool-misuse vectors, "
            "unsafe automation, weakened audit trails, and missing human "
            "approval gates. Tie each finding to the changed line and "
            "rank by severity."
        ),
    },
    "maintainer": {
        "plan": (
            "Approach this as an operator, maintainer, and QA reviewer. "
            "Produce a plan optimized for ownership — testability, "
            "reproducibility, operational complexity, documentation, "
            "dependency and upgrade risk, observability, support cost. "
            "Prefer simplifications."
        ),
        "brainstorm": (
            "Approach this as an operator, maintainer, and QA reviewer. "
            "Generate ideas that reduce operational and ownership cost "
            "— simpler dependencies, better observability, less code to "
            "own, easier debugging, lower support burden."
        ),
        "validate": (
            "Approach this as an operator, maintainer, and QA reviewer. "
            "Judge whether this can be implemented, tested, maintained, "
            "debugged, and handed off — implementation burden, "
            "reproducibility, testability, operational complexity, "
            "documentation, dependency and upgrade risk, observability, "
            "support cost. Prefer simplifications."
        ),
        "review": (
            "Approach this as an operator, maintainer, and QA reviewer. "
            "Review the diff for what it costs to own — regressions and "
            "untested paths the change introduces, missing or stale "
            "tests, harder debugging, weakened observability, new "
            "dependency and upgrade risk, and undocumented behavior. "
            "Prefer the simpler change; flag where the change adds code "
            "to maintain without need."
        ),
    },
    "analyst": {
        "plan": (
            "Approach this as an evidence and domain analyst. Produce a "
            "plan grounded in what is actually known — prior art, "
            "benchmarks, standards, validated patterns. Distinguish "
            "supported design choices from assumptions that still need "
            "validation."
        ),
        "brainstorm": (
            "Approach this as an evidence and domain analyst. Generate "
            "ideas grounded in prior art and proven patterns — what has "
            "worked elsewhere, what benchmarks suggest, what's been "
            "validated. Surface the evidence behind each option."
        ),
        "validate": (
            "Approach this as an evidence and domain analyst. Separate "
            "what is actually known from what is assumed — weigh prior "
            "art, benchmarks, standards, and technical plausibility, "
            "and name what still needs validation. Distinguish "
            "supported claims from unsupported ones."
        ),
        "review": (
            "Approach this as an evidence and domain analyst. Review the "
            "diff against what is actually known — does the change match "
            "the documented API, the surrounding conventions, and "
            "established patterns, or does it rest on an unverified "
            "assumption? Verify a claim by reading the relevant file when "
            "you must; flag any change a peer asserts that the code does "
            "not actually support."
        ),
    },
    "visionary": {
        "plan": (
            "Approach this as a visionary. Produce an ambitious plan: "
            "pursue the boldest version of the goal, borrow analogies from "
            "unrelated domains, and combine ideas that do not obviously "
            "belong together. Do not self-censor for feasibility — name the "
            "breakthrough worth attempting and what it would unlock."
        ),
        "brainstorm": (
            "Approach this as a visionary — a brilliant, blue-sky "
            "researcher who sees connections others miss. Generate ideas "
            "with the feasibility filter off: cross-domain analogies, "
            "unexpected recombinations, and 'what could this become' leaps. "
            "Reach for the non-obvious connection between unrelated things. "
            "No idea is too wild here — that is the point."
        ),
        "validate": (
            "Approach this as a visionary. Judge the proposal by its "
            "ambition and possibility, not its safety: where is it thinking "
            "too small, what bolder version does it foreclose, and what "
            "non-obvious reframing would make it more than the sum of its "
            "parts? Open doors rather than close them."
        ),
        "review": (
            "Approach this as a visionary. Read the change for the future "
            "it opens or closes: does it leave room for bolder directions, "
            "or quietly foreclose them? Flag where a small reframing turns a "
            "narrow fix into a more general capability."
        ),
    },
    "pioneer": {
        "plan": (
            "Approach this as a pioneer. Produce a plan that takes the most "
            "ambitious idea seriously and finds the smallest real "
            "experiment that proves it out: the minimal build that yields "
            "signal, the first milestone, and what result would say push "
            "further or pivot. Ambitious destination, concrete first move."
        ),
        "brainstorm": (
            "Approach this as a pioneer — a rational dreamer who turns "
            "moonshots into first steps. Generate ideas that are bold but "
            "buildable: for each, name the smallest real experiment or "
            "prototype that would yield signal. Reach for ambition, but "
            "anchor every idea to a concrete first move a small team could "
            "start this week."
        ),
        "validate": (
            "Approach this as a pioneer. Judge whether the proposal's "
            "ambition can be reached incrementally: is there a credible "
            "smallest-first-step, an experiment that yields early signal, a "
            "path from prototype to full vision? Where the leap is too big, "
            "name the intermediate step that de-risks it — do not abandon "
            "the ambition."
        ),
        "review": (
            "Approach this as a pioneer. Read the change as a first step "
            "toward something larger: does it create a foothold to build "
            "on, or a dead end? Flag where a slightly different cut would "
            "open a path forward. Favor changes that yield signal and can "
            "be extended."
        ),
    },
    "neutral": {"plan": "", "brainstorm": "", "validate": "", "review": ""},
}


def apply_role(role: str, prompt: str, phase: str) -> str:
    """Prepend the phase-appropriate stance prefix to a prompt. `neutral`
    (empty prompt) returns the prompt unchanged. Final prompt order:
    role -> context -> mode task."""
    if role not in ROLES:
        raise ValueError(f"unknown stance '{role}'. known: {', '.join(sorted(ROLES))}")
    if phase not in PHASES:
        raise ValueError(f"unknown phase '{phase}'. known: {', '.join(sorted(PHASES))}")
    prefix = ROLES[role][phase]
    if not prefix:
        return prompt
    return f"{prefix}\n\n{prompt}"


def parse_role_arg(raw: str) -> tuple[str, str]:
    """Parse a `--role` argument of the form 'stance:agent'. Returns
    (stance, agent), both lowercased. Raises ValueError on a missing ':',
    unknown stance, or empty agent name."""
    if ":" not in raw:
        raise ValueError(
            f"--role must be 'stance:agent', got '{raw}' (no ':'); "
            f"a bare stance is not allowed"
        )
    stance, agent = raw.split(":", 1)
    stance = stance.strip().lower()
    agent = agent.strip().lower()
    if stance not in ROLES:
        raise ValueError(
            f"unknown stance '{stance}'. known: {', '.join(sorted(ROLES))}"
        )
    if not agent:
        raise ValueError(f"--role '{raw}' has an empty agent name")
    return stance, agent


def rotate_roles(agent_names: list[str], roles: dict[str, str]) -> dict[str, str]:
    """Cyclically shift role assignments by one position across the given
    agents: agent i takes the role agent i-1 held. A no-op for 0 or 1 agents.
    `agent_names` fixes the ordering; `roles` must map every listed name to a
    stance. Names not in `agent_names` keep their existing role."""
    if len(agent_names) < 2:
        return dict(roles)
    rotated = dict(roles)
    n = len(agent_names)
    for i, name in enumerate(agent_names):
        prev = agent_names[(i - 1) % n]
        rotated[name] = roles[prev]
    return rotated
