import pytest

from quorum.roles import (
    PHASES,
    ROLES,
    apply_role,
    parse_role_arg,
    rotate_roles,
)


def test_expected_eight_stances_present() -> None:
    assert set(ROLES) == {
        "skeptic",
        "architect",
        "security",
        "maintainer",
        "analyst",
        "neutral",
        "visionary",
        "pioneer",
    }


def test_expected_four_phases_present() -> None:
    assert PHASES == {"plan", "brainstorm", "validate", "review"}


def test_every_role_has_all_four_phases() -> None:
    for stance, role in ROLES.items():
        assert set(role) == {"plan", "brainstorm", "validate", "review"}, (
            f"{stance} is missing phase entries"
        )


def test_review_prompts_are_diff_voiced() -> None:
    """Each non-neutral stance's review prompt must speak to reviewing a real
    code change — it should mention a review, a diff, or a change."""
    diff_voiced = ("review", "diff", "change")
    for stance, role in ROLES.items():
        if stance == "neutral":
            continue
        text = role["review"].lower()
        assert any(w in text for w in diff_voiced), (
            f"{stance}'s review prompt is not diff/review-voiced"
        )


def test_neutral_review_prompt_is_empty() -> None:
    assert ROLES["neutral"]["review"] == ""


def test_neutral_prompts_are_empty_for_every_phase() -> None:
    for phase in PHASES:
        assert ROLES["neutral"][phase] == ""


def test_apply_role_prepends_prefix() -> None:
    out = apply_role("skeptic", "TASK", "plan")
    assert out.endswith("TASK")
    assert ROLES["skeptic"]["plan"] in out
    assert out != "TASK"


def test_apply_role_uses_the_phase_specific_prompt() -> None:
    plan_out = apply_role("architect", "TASK", "plan")
    validate_out = apply_role("architect", "TASK", "validate")
    assert ROLES["architect"]["plan"] in plan_out
    assert ROLES["architect"]["validate"] in validate_out
    assert plan_out != validate_out


def test_apply_role_neutral_is_noop_for_every_phase() -> None:
    for phase in PHASES:
        assert apply_role("neutral", "TASK", phase) == "TASK"


def test_apply_role_rejects_unknown_stance() -> None:
    with pytest.raises(ValueError):
        apply_role("wizard", "TASK", "plan")


def test_apply_role_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError):
        apply_role("skeptic", "TASK", "synthesize")


def test_plan_prompts_do_not_forbid_rewriting() -> None:
    """Generative-phase regression guard: the review-only 'do not rewrite'
    guardrail must not appear in plan or brainstorm prompts. This is the
    bug the Codex adversarial review found in the single-prompt design."""
    for stance, role in ROLES.items():
        if stance == "neutral":
            continue
        assert "do not rewrite" not in role["plan"].lower()
        assert "do not rewrite" not in role["brainstorm"].lower()


def test_plan_prompts_speak_to_producing_a_plan() -> None:
    for stance, role in ROLES.items():
        if stance == "neutral":
            continue
        assert "plan" in role["plan"].lower(), (
            f"{stance}'s plan prompt does not invoke a plan deliverable"
        )


def test_brainstorm_prompts_speak_to_generating_ideas() -> None:
    for stance, role in ROLES.items():
        if stance == "neutral":
            continue
        text = role["brainstorm"].lower()
        assert "ideas" in text or "generate" in text, (
            f"{stance}'s brainstorm prompt does not invoke ideation"
        )


def test_validate_prompts_are_evaluative() -> None:
    evaluative = ("evaluate", "challenge", "assess", "judge", "separate", "review")
    for stance, role in ROLES.items():
        if stance == "neutral":
            continue
        text = role["validate"].lower()
        assert any(w in text for w in evaluative), (
            f"{stance}'s validate prompt is not evaluative"
        )


def test_skeptic_prompts_are_constructive_not_cynical() -> None:
    """Every skeptic phase must pair criticism with a path forward, not just
    tear down — S's constructive-not-cynical directive."""
    for phase in PHASES:
        text = ROLES["skeptic"][phase].lower()
        assert "constructive" in text, f"skeptic/{phase} lacks the constructive clause"
        assert "de-risk" in text, f"skeptic/{phase} lacks the de-risk pairing"
        # The constructive pairing offers a test or control, never a rewrite —
        # "or change" contradicted validate/review's "challenge it, do not
        # rewrite it" directive, so it must not reappear.
        assert "or change" not in text, f"skeptic/{phase} re-introduced 'or change'"


def test_parse_role_arg_valid() -> None:
    assert parse_role_arg("security:codex") == ("security", "codex")
    assert parse_role_arg("  Architect : Gemini ") == ("architect", "gemini")


def test_parse_role_arg_rejects_missing_colon() -> None:
    with pytest.raises(ValueError):
        parse_role_arg("skeptic")


def test_parse_role_arg_rejects_unknown_stance() -> None:
    with pytest.raises(ValueError):
        parse_role_arg("wizard:codex")


def test_parse_role_arg_rejects_empty_agent() -> None:
    with pytest.raises(ValueError):
        parse_role_arg("skeptic:")


def test_rotate_roles_two_agents_swaps() -> None:
    rotated = rotate_roles(
        ["codex", "gemini"], {"codex": "skeptic", "gemini": "architect"}
    )
    assert rotated == {"codex": "architect", "gemini": "skeptic"}


def test_rotate_roles_single_agent_is_noop() -> None:
    assert rotate_roles(["codex"], {"codex": "skeptic"}) == {"codex": "skeptic"}


def test_rotate_roles_uniform_is_noop() -> None:
    rotated = rotate_roles(
        ["codex", "gemini"], {"codex": "skeptic", "gemini": "skeptic"}
    )
    assert rotated == {"codex": "skeptic", "gemini": "skeptic"}
