import pytest

import quorum_mcp.server as server
from quorum.research import LibraryDoc, ResearchDigest


@pytest.mark.asyncio
async def test_q_research_impl_returns_markdown(monkeypatch):
    async def fake_research_topic(topic, **kw):
        return ResearchDigest(
            topic=topic,
            papers=(),
            libraries=(LibraryDoc("lib", "desc", (), 9.0, "https://context7.com/lib"),),
            errors=(),
        )

    monkeypatch.setattr(server, "research_topic", fake_research_topic)
    out = await server.q_research("diffusion models", None, 5)
    # Opens with the deterministic verdict line, then the prior-art heading.
    assert out.startswith("Research status: ")
    assert "## Prior art for: diffusion models" in out
    assert "**lib**" in out


def test_q_research_docstring_demands_domain_anchored_query():
    # The description must steer query FORMATION away from generic single tokens
    # (which return non-zero but off-topic noise) toward domain-anchored terms.
    doc = server.q_research.__doc__ or ""
    assert "short is not the same as generic" in doc
    assert "domain-specific terms" in doc
    assert "off-topic" in doc.lower()
    # Off-topic (non-zero) results must steer to a refined re-run, not silent
    # fallback -- the same imperative the SKILL gives.
    assert "re-anchor" in doc
    assert "call q_research again" in doc


def test_q_research_docstring_scopes_domain_check_to_same_domain_use():
    # This shared tool docstring travels to every caller, including
    # q-skystorm, which deliberately wants cross-domain structural hits.
    # An unqualified "belong to your domain" check would tell a skystorm
    # caller to discard exactly the hits it needs.
    doc = server.q_research.__doc__ or ""
    assert "belong to your domain" in doc
    assert "cross-domain" in doc
    assert "structural kinship" in doc


def test_q_research_docstring_demands_retry_before_fallback():
    # The tool description travels with the tool to every caller; it must tell
    # the agent to rework-and-retry a query-shaped error rather than treat the
    # first error as terminal and fall back to its own knowledge.
    doc = server.q_research.__doc__ or ""
    assert "retry" in doc.lower()
    assert "reworked retry has ALSO failed" in doc
    assert "fall back on your" in doc.lower()


def test_q_research_docstring_allows_absent_suggested_query():
    # Round-2/3 q-review: the tool now emits RETRY-RECOMMENDED without a
    # suggestion on total backend failure / un-shortenable queries. The docstring
    # travels to every caller: it must not promise a suggestion is always present,
    # AND it must branch the two absent cases (retry-same on infrastructure vs
    # re-anchor on an un-shortenable query) rather than blanket "re-anchor" --
    # blanket re-anchor mutates a fine query on an outage (round-3 contradiction).
    doc = server.q_research.__doc__ or ""
    low = doc.lower()
    assert "usually" in low
    assert "absent" in low
    assert "re-anchor" in low  # the un-shortenable case
    assert "retry the same" in low  # the infrastructure case (opposite move)


@pytest.mark.asyncio
async def test_q_research_impl_rejects_unknown_source():
    # Validation lives in research_topic, so the MCP path rejects bad input too.
    with pytest.raises(ValueError, match="bogus"):
        await server.q_research("diffusion", ["bogus"], 5)


def test_brainstorm_prompt_seeds_prior_ideas():
    base = server._brainstorm_prompt("my topic", None)
    assert "my topic" in base
    assert "do NOT repeat" not in base

    seeded = server._brainstorm_prompt("my topic", "- prior idea A")
    assert "my topic" in seeded
    assert "do NOT repeat" in seeded
    assert "- prior idea A" in seeded


def test_brainstorm_prompt_seeds_research_digest_as_evidence():
    """The digest uses a dedicated `research` slot with evidence framing. It
    must never appear inside the prior_ideas do-NOT-repeat wrap."""
    seeded = server._brainstorm_prompt("my topic", None, research="- paper one")
    assert "my topic" in seeded
    assert "- paper one" in seeded
    assert "do NOT repeat" not in seeded  # evidence, not ideas-to-avoid

    both = server._brainstorm_prompt(
        "my topic", "- prior idea A", research="- paper one"
    )
    # base -> research -> divergence: the digest sits before (outside) the wrap.
    assert both.index("- paper one") < both.index("do NOT repeat")
    assert both.index("do NOT repeat") < both.index("- prior idea A")


@pytest.mark.asyncio
async def test_q_research_impl_rejects_unknown_mode(monkeypatch):
    """An unknown mode must fail visibly, not silently fall back to
    grounded (which would disable the field map without telling the caller).

    The rejection must happen before any backend is queried, so a typo does not
    spend 10-15 seconds of HTTP time or rate limit. The ValueError must occur
    while the fake backend's sentinel is still unset."""
    called = False

    async def fake_research_topic(topic, **kw):
        nonlocal called
        called = True
        from quorum.research import ResearchDigest

        return ResearchDigest(topic=topic, papers=())

    monkeypatch.setattr(server, "research_topic", fake_research_topic)
    with pytest.raises(ValueError, match="mode"):
        await server.q_research("diffusion", None, 5, mode="exploritory")
    assert not called, "research_topic must not run when the mode is invalid"
