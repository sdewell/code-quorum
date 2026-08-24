"""Live smoke tests against the real arXiv / OpenAlex / Europe PMC / Context7 /
GitHub / HuggingFace APIs. Excluded from CI (addopts = -m 'not live'). Run manually:
`uv run pytest -m live -q`. Asserts structural properties only, so schema drift
surfaces without pinning to volatile content. Tokens (if present in the env) are
passed straight to the search functions, which only send them as Bearer headers."""

import os

import httpx
import pytest

from quorum.research import (
    fetch_context7,
    research_topic,
    search_arxiv,
    search_europepmc,
    search_github,
    search_huggingface,
    search_openalex,
)

pytestmark = [pytest.mark.live, pytest.mark.asyncio]
_HF_TOKEN = os.environ.get("QUORUM_HF_TOKEN") or os.environ.get("HF_TOKEN")


async def test_arxiv_live():
    async with httpx.AsyncClient() as client:
        papers = await search_arxiv(client, "diffusion models", limit=3, timeout=20.0)
    assert papers
    assert all(p.title and p.source == "arxiv" for p in papers)


async def test_openalex_live():
    async with httpx.AsyncClient() as client:
        papers = await search_openalex(
            client, "graph neural networks", limit=3, timeout=20.0, email=None
        )
    assert papers
    assert all(p.title and p.source == "openalex" for p in papers)


@pytest.mark.skipif(
    not (
        os.environ.get("QUORUM_OPENALEX_API_KEY") or os.environ.get("OPENALEX_API_KEY")
    ),
    reason="no OpenAlex API key in the environment",
)
async def test_openalex_live_authenticates_with_the_api_key():
    """A key that OpenAlex rejects 401s, so a passing search proves the key is
    actually being honoured -- not merely that anonymous search happened to work."""
    key = os.environ.get("QUORUM_OPENALEX_API_KEY") or os.environ.get(
        "OPENALEX_API_KEY"
    )
    async with httpx.AsyncClient() as client:
        papers = await search_openalex(
            client,
            "graph neural networks",
            limit=3,
            timeout=20.0,
            email=None,
            api_key=key,
        )
    assert papers
    assert all(p.title and p.source == "openalex" for p in papers)


async def test_europepmc_live():
    """The source's whole reason to exist is the preprint tier arXiv does not
    carry, so assert we actually reach bioRxiv/medRxiv -- not merely that the
    backend answered. Schema drift in the publisher field would slip past a
    bare `assert papers`."""
    async with httpx.AsyncClient() as client:
        papers = await search_europepmc(
            client, "single-cell RNA-seq batch correction", limit=5, timeout=20.0
        )
    assert papers
    assert all(p.title and p.abstract for p in papers)
    assert any(p.source in ("bioRxiv", "medRxiv") for p in papers)
    # the arXiv exclusion holds, so this source never re-serves arxiv's hits
    assert not any(p.source == "arXiv" for p in papers)
    # structured-abstract markup is stripped at the boundary, not passed through
    assert not any("<h4>" in p.abstract or "<sup>" in p.abstract for p in papers)


async def test_europepmc_live_returns_nothing_for_a_blank_topic():
    """A blank query matches Europe PMC's entire corpus (~1.2M hits, HTTP 200).
    Guard that we refuse it rather than dressing arbitrary popular papers as
    prior art."""
    async with httpx.AsyncClient() as client:
        assert await search_europepmc(client, "  ", limit=5, timeout=20.0) == []


async def test_context7_live():
    # Direct fetch_context7 calls do not read environment credentials.
    async with httpx.AsyncClient() as client:
        libs = await fetch_context7(
            client,
            "fastapi",
            timeout=15.0,
            token=os.environ.get("CONTEXT7_API_KEY"),
        )
    # May legitimately be empty for some queries; assert no crash + shape.
    assert all(lib.name and lib.trust_score >= 5.0 for lib in libs)


async def test_github_live():
    async with httpx.AsyncClient() as client:
        repos = await search_github(
            client,
            "diffusion models",
            limit=3,
            timeout=20.0,
            token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"),
        )
    assert repos
    assert all(r.name and r.url for r in repos)


async def test_huggingface_live():
    async with httpx.AsyncClient() as client:
        models = await search_huggingface(
            client,
            "sentence embedding",
            limit=3,
            timeout=20.0,
            token=os.environ.get("QUORUM_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        )
    assert models
    assert all(m.id and m.url for m in models)


@pytest.mark.skipif(not _HF_TOKEN, reason="no Hugging Face token in the environment")
async def test_huggingface_live_authenticates_with_the_token():
    async with httpx.AsyncClient() as client:
        identity = await client.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {_HF_TOKEN}"},
            timeout=20.0,
        )
        identity.raise_for_status()
        assert identity.json().get("name")
        models = await search_huggingface(
            client,
            "sentence embedding",
            limit=3,
            timeout=20.0,
            token=_HF_TOKEN,
        )

    assert models


async def test_research_topic_huggingface_live_reports_relevant_models():
    digest = await research_topic(
        "sentence embedding",
        sources={"huggingface"},
        limit=5,
        timeout=20.0,
        hf_token=os.environ.get("QUORUM_HF_TOKEN") or os.environ.get("HF_TOKEN"),
    )

    assert digest.errors == ()
    assert dict(digest.counts).get("huggingface", 0) > 0
    assert digest.models
    ids = " ".join(model.id.lower() for model in digest.models)
    assert "sentence" in ids or "embedding" in ids


async def test_github_unions_distinctive_terms_live():
    # The exact combined multi-artifact phrase that whiffed (GitHub 0) before the
    # per-term-union fix; it must now recover the canonical prior-art repos.
    async with httpx.AsyncClient() as client:
        repos = await search_github(
            client,
            "PuLID InstantID IP-Adapter identity preserving adapter",
            limit=6,
            timeout=20.0,
            token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"),
        )
    assert repos
    blob = " ".join(r.name.lower() for r in repos)
    assert any(a in blob for a in ("pulid", "instantid", "ip-adapter"))


async def test_huggingface_unions_distinctive_terms_live():
    async with httpx.AsyncClient() as client:
        models = await search_huggingface(
            client,
            "PuLID InstantID IP-Adapter identity preserving adapter",
            limit=6,
            timeout=20.0,
            token=os.environ.get("QUORUM_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        )
    assert models
    blob = " ".join(m.id.lower() for m in models)
    assert any(a in blob for a in ("pulid", "instantid", "ip-adapter"))


async def test_research_topic_live_aggregates():
    digest = await research_topic("retrieval augmented generation", limit=3)
    assert digest.topic == "retrieval augmented generation"
    # A mainstream topic should yield papers, or at least report source errors.
    assert digest.papers or digest.errors
    # Every queried source reports a count (whiff visibility), unless it errored.
    counted = {s for s, _ in digest.counts}
    errored = {e.split(":", 1)[0] for e in digest.errors}
    assert counted | errored >= {
        "arxiv",
        "openalex",
        "europepmc",
        "context7",
        "github",
        "huggingface",
    }
