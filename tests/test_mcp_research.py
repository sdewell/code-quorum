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
    assert "Research needs action: false" in out
    assert "## Prior art for: diffusion models" in out
    assert "**lib**" in out


@pytest.mark.asyncio
async def test_q_research_passes_purpose_and_semantic_query_lanes(monkeypatch):
    seen = {}

    async def fake_research_topic(topic, **kw):
        seen.update(kw)
        return ResearchDigest(topic=topic, papers=())

    monkeypatch.setattr(server, "research_topic", fake_research_topic)
    await server.q_research(
        "execution provenance",
        None,
        5,
        purpose="currency",
        query_lanes=[
            "computational experiment provenance",
            "minimum information reporting provenance",
        ],
    )

    assert seen["purpose"] == "currency"
    assert seen["mode"] == "grounded"
    assert seen["query_lanes"] == (
        "computational experiment provenance",
        "minimum information reporting provenance",
    )


@pytest.mark.asyncio
async def test_q_research_passes_exploratory_mode_into_retrieval(monkeypatch):
    seen = {}

    async def fake_research_topic(topic, **kw):
        seen.update(kw)
        return ResearchDigest(topic=topic, papers=())

    monkeypatch.setattr(server, "research_topic", fake_research_topic)
    await server.q_research("sinkhorn transport", mode="exploratory")

    assert seen["mode"] == "exploratory"
    assert seen["map_fields"] is True


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
    assert "Research needs" in doc
    assert "exact action/source/lane table" in doc
    assert "Retry rows carry an" in doc
    assert "fall back on your" in doc.lower()


def test_q_research_docstring_routes_exact_required_actions():
    # A RETRY-REQUIRED result may lack a suggested shortening. The exact action
    # table must distinguish retrying an infrastructure-failed query unchanged
    # from semantically re-anchoring an unshortenable query.
    doc = server.q_research.__doc__ or ""
    low = doc.lower()
    assert "retry rows carry an" in low
    assert "executable query" in low
    assert "re-anchor row" in low
    assert "different domain" in low


def test_q_research_docstring_distinguishes_collision_reanchor_from_degraded():
    doc = " ".join((server.q_research.__doc__ or "").split())
    assert "QUERY-COLLISION is semantic failure" in doc
    assert "never mechanically shorten it" in doc
    assert "filtered usable evidence" in doc
    assert "All-collision results are" in doc


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
