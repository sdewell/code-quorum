import asyncio
import dataclasses
from pathlib import Path

import httpx
import pytest

from quorum.research import (
    _ABSTRACT_MAX_CHARS,
    DEFAULT_SOURCES,
    HFModel,
    LibraryDoc,
    Paper,
    Repo,
    ResearchAction,
    ResearchDigest,
    ResearchStatus,
    SourceLaneStatus,
    _arxiv_id_from_doi,
    _clean_token,
    _compact_count,
    _dedup_papers,
    _distinctive_terms,
    _gh_quote,
    _reconstruct_abstract,
    _redact,
    _retry_hint,
    _source_lane_status,
    _strip_html,
    _truncate,
    compute_status,
    fetch_context7,
    format_digest,
    research_topic,
    search_arxiv,
    search_europepmc_preprints,
    search_europepmc_published,
    search_github,
    search_huggingface,
    search_openalex,
    validate_sources,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "research"


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


_BASE_PAPER = Paper(
    title="A Paper",
    authors=("Ada Lovelace", "Alan Turing", "Grace Hopper", "X"),
    year=2023,
    source="arxiv",
    identifier="2301.1v1",
    url="http://arxiv.org/abs/2301.1v1",
    abstract="An   abstract.",
)


def _paper(**kw) -> Paper:
    return dataclasses.replace(_BASE_PAPER, **kw)


def test_format_digest_renders_papers_libs_and_errors():
    digest = ResearchDigest(
        topic="diffusion",
        papers=(_paper(),),
        libraries=(
            LibraryDoc(
                name="scikit-learn",
                description="ML in Python.",
                snippets=("import sklearn",),
                trust_score=9.5,
                url="https://context7.com/scikit-learn/scikit-learn",
            ),
        ),
        errors=("openalex: TimeoutException: slow",),
    )
    out = format_digest(digest)
    assert "## Prior art for: diffusion" in out
    assert "### Papers (arXiv + OpenAlex + Europe PMC published/preprints)" in out
    assert "**A Paper** (2023, arxiv)" in out
    assert "Ada Lovelace, Alan Turing, Grace Hopper" in out  # capped at 3 authors
    assert "### Library docs (Context7)" in out
    assert "**scikit-learn** (trust 9.5)" in out
    assert "### Sources unavailable" in out
    assert "openalex: TimeoutException: slow" in out


def test_format_digest_empty_is_explicit():
    digest = ResearchDigest(topic="x", papers=(), libraries=(), errors=())
    out = format_digest(digest)
    assert "_No prior art found._" in out


def test_truncate_long_text_gets_ellipsis():
    out = _truncate("x " * 300)
    assert out.endswith("…")
    assert len(out) <= _ABSTRACT_MAX_CHARS


def test_format_digest_none_year_and_no_authors():
    digest = ResearchDigest(
        topic="x",
        papers=(_paper(year=None, authors=()),),
        libraries=(),
        errors=(),
    )
    out = format_digest(digest)
    assert "(n.d., arxiv)" in out
    assert "— unknown." in out


def test_format_digest_errors_only_omits_not_found():
    digest = ResearchDigest(topic="x", papers=(), libraries=(), errors=("arxiv: boom",))
    out = format_digest(digest)
    assert "### Sources unavailable" in out
    assert "arxiv: boom" in out
    assert "_No prior art found._" not in out


def test_retry_hint_query_shaped_vs_transient():
    # A surfaced arXiv 400 has already exhausted the in-tool mechanical rework,
    # so another shortening would repeat the failed recovery.
    q = _retry_hint("arxiv: HTTPStatusError: Client error '400 Bad Request' for url x")
    assert "could not be mechanically repaired" in q.lower()
    assert "semantic" in q.lower()
    assert "re-anchor" in q.lower()
    assert "shorten" not in q.lower()
    # Anything else (timeout/flake) is transient -> retry as-is (shorten only
    # as a fallback "if it persists", never the lead instruction).
    t = _retry_hint("openalex: TimeoutException: slow")
    assert "transient" in t.lower()
    assert "retry the same query" in t.lower()
    assert "before falling back" not in t.lower()


def test_format_digest_errors_carry_inline_retry_hint():
    # The hint must appear at the decision point (inline under the failed
    # source), so the consumer reworks-and-retries instead of falling back to
    # internal knowledge -- the comfyui shrug this guards against.
    digest = ResearchDigest(
        topic="x",
        papers=(),
        libraries=(),
        errors=(
            "arxiv: HTTPStatusError: Client error '400 Bad Request' for url x",
            "openalex: TimeoutException: slow",
        ),
    )
    out = format_digest(digest)
    assert "↳" in out
    # arXiv 400 -> the internal shortening failed; semantically re-anchor.
    assert "semantically re-anchor" in out.lower()
    # OpenAlex timeout -> plain retry.
    assert "transient" in out.lower()


@pytest.mark.asyncio
async def test_search_arxiv_parses_entries():
    xml = (_FIXTURES / "arxiv_sample.xml").read_text()

    def handler(request):
        assert request.url.host == "export.arxiv.org"
        return httpx.Response(200, text=xml)

    async with _client(handler) as client:
        papers = await search_arxiv(client, "diffusion", limit=5, timeout=5.0)

    assert len(papers) == 2
    first = papers[0]
    assert first.title == "Diffusion Models for Tabular Data"
    assert first.identifier == "2301.12345v1"
    assert first.year == 2023
    assert first.authors == ("Ada Lovelace", "Alan Turing")
    assert first.source == "arxiv"
    assert first.research_source == "arxiv"
    assert first.url == "http://arxiv.org/abs/2301.12345v1"
    assert (
        first.abstract
        == "We present a diffusion approach to synthesizing tabular records."
    )


@pytest.mark.asyncio
async def test_search_arxiv_rate_limit_200_raises_instead_of_zero_results():
    """arXiv signals its rate limit as HTTP 200 with a ~14-byte plain-text body
    ("Rate exceeded.") -- a success status, not an error. Parsed as a feed that
    body has no entries, which would silently report 'no papers found'; it must
    raise a loud, classifiable refusal instead, carrying the body as evidence."""

    def handler(request):
        return httpx.Response(200, text="Rate exceeded.")

    async with _client(handler) as client:
        with pytest.raises(RuntimeError, match="rate refusal.*Rate exceeded"):
            await search_arxiv(client, "diffusion", limit=5, timeout=5.0)


@pytest.mark.asyncio
async def test_search_arxiv_non_feed_200_without_the_marker_is_not_rate():
    """A 200 body that is neither a feed nor the documented 'Rate exceeded'
    marker (a proxy or block page) must NOT wear the rate label -- it raises
    the distinct non-feed error, which is classified as transient."""

    def handler(request):
        return httpx.Response(200, text="<html>Service Maintenance</html>")

    async with _client(handler) as client:
        with pytest.raises(RuntimeError, match="non-feed body"):
            await search_arxiv(client, "diffusion", limit=5, timeout=5.0)


def test_reconstruct_abstract_orders_by_position():
    assert _reconstruct_abstract({"b": [1], "a": [0], "c": [2]}) == "a b c"
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""
    # a word recurring at multiple positions is emitted once per position
    assert (
        _reconstruct_abstract({"the": [0, 5], "cat": [1], "sat": [2]})
        == "the cat sat the"
    )


def test_arxiv_id_from_doi():
    assert (
        _arxiv_id_from_doi("https://doi.org/10.48550/arXiv.2401.00001") == "2401.00001"
    )
    assert _arxiv_id_from_doi("https://doi.org/10.1109/TPAMI.2022.123") is None
    # a stray 'arxiv.' in a non-arXiv DOI suffix must not yield a bogus id
    assert _arxiv_id_from_doi("https://doi.org/10.9999/not-arxiv.paper") is None


@pytest.mark.asyncio
async def test_search_openalex_parses_results():
    data = (_FIXTURES / "openalex_sample.json").read_text()

    def handler(request):
        assert request.url.host == "api.openalex.org"
        return httpx.Response(200, text=data)

    async with _client(handler) as client:
        papers = await search_openalex(
            client, "diffusion", limit=5, timeout=5.0, email=None
        )

    assert len(papers) == 2
    first = papers[0]
    assert first.title == "Tabular Diffusion at Scale"
    assert first.year == 2024
    assert first.abstract == "We scale diffusion."
    assert first.identifier == "2401.00001"  # arXiv id derived from DOI
    assert first.source == "openalex"
    assert papers[1].identifier == "https://doi.org/10.1109/TPAMI.2022.123"


@pytest.mark.asyncio
async def test_research_topic_methods_balances_openalex_strata():
    seen: list[dict[str, str]] = []

    def work(title: str, doi: str, year: int) -> dict:
        return {
            "title": title,
            "publication_year": year,
            "doi": doi,
            "authorships": [],
            "abstract_inverted_index": {"reproducibility": [0]},
            "primary_topic": None,
        }

    def handler(request):
        params = dict(request.url.params)
        seen.append(params)
        if "filter" in params:
            results = [
                work("Canonical method", "https://doi.org/10.1/canonical", 2020),
                work("Recent method", "https://doi.org/10.1/recent", 2026),
            ]
        else:
            results = [work("Canonical method", "https://doi.org/10.1/canonical", 2020)]
        return httpx.Response(200, json={"results": results})

    async with _client(handler) as client:
        digest = await research_topic(
            "computational reproducibility",
            sources={"openalex"},
            purpose="methods",
            limit=3,
            client=client,
        )

    assert len(seen) == 2
    assert all(params["sort"] == "relevance_score:desc" for params in seen)
    assert any("filter" not in params for params in seen)
    assert any("from_publication_date:" in params.get("filter", "") for params in seen)
    papers = {paper.title: paper for paper in digest.papers}
    assert papers["Canonical method"].strata == ("all-time", "recent")
    assert papers["Recent method"].strata == ("recent",)


@pytest.mark.asyncio
async def test_research_topic_currency_keeps_openalex_five_year_filter():
    seen: list[dict[str, str]] = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        await research_topic(
            "computational reproducibility",
            sources={"openalex"},
            purpose="currency",
            client=client,
        )

    assert len(seen) == 1
    assert "from_publication_date:" in seen[0]["filter"]


def test_format_digest_omits_empty_url_link():
    digest = ResearchDigest(
        topic="x", papers=(_paper(url=""),), libraries=(), errors=()
    )
    out = format_digest(digest)
    assert "<>" not in out
    assert "A Paper" in out


@pytest.mark.asyncio
async def test_fetch_context7_filters_low_trust_and_collects_snippets():
    search = (_FIXTURES / "context7_search.json").read_text()
    docs = (_FIXTURES / "context7_docs.txt").read_text()

    def handler(request):
        assert request.url.host == "context7.com"
        if request.url.path.endswith("/search"):
            return httpx.Response(200, text=search)
        return httpx.Response(200, text=docs)

    async with _client(handler) as client:
        libs = await fetch_context7(client, "scikit-learn", timeout=5.0)

    assert len(libs) == 1  # low-trust filtered out
    assert all(lib.name != "low-trust-lib" for lib in libs)  # filtered, not capped
    lib = libs[0]
    assert lib.name == "scikit-learn"
    assert lib.trust_score == 9.5
    assert lib.url == "https://context7.com/scikit-learn/scikit-learn"
    assert lib.snippets[0] == "import sklearn"
    assert len(lib.snippets) == 3


@pytest.mark.asyncio
async def test_fetch_context7_skips_idless_and_nonnumeric_trust():
    # high trust but no id (unusable), and a non-numeric trustScore (below gate);
    # only the well-formed entry survives.
    search = (
        '{"results": ['
        '{"title": "no-id", "trustScore": 9.9},'
        '{"id": "/x/y", "title": "bad-trust", "trustScore": "high"},'
        '{"id": "/ok/ok", "title": "ok", "trustScore": 8.0}'
        "]}"
    )

    def handler(request):
        if request.url.path.endswith("/search"):
            return httpx.Response(200, text=search)
        return httpx.Response(200, text="snippet")

    async with _client(handler) as client:
        libs = await fetch_context7(client, "q", timeout=5.0)

    assert {lib.name for lib in libs} == {"ok"}


@pytest.mark.asyncio
async def test_fetch_context7_sends_bearer_token_only_in_header():
    seen = {"auth": [], "urls": []}

    def handler(request):
        seen["auth"].append(request.headers.get("authorization"))
        seen["urls"].append(str(request.url))
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "/fastapi/fastapi",
                            "title": "FastAPI",
                            "trustScore": 9.0,
                        }
                    ]
                },
            )
        return httpx.Response(200, text="FastAPI snippet")

    async with _client(handler) as client:
        await fetch_context7(client, "fastapi", timeout=5.0, token="ctx7-secret")

    assert seen["auth"] == ["Bearer ctx7-secret", "Bearer ctx7-secret"]
    assert all("ctx7-secret" not in url for url in seen["urls"])


@pytest.mark.asyncio
async def test_fetch_context7_dedups_variants_keeps_highest_trust():
    # One library indexed under several version ids (same title) must collapse
    # to a single highest-trust entry, not consume every slot with redundant
    # copies of itself -- freeing the budget for a distinct library.
    search = (
        '{"results": ['
        '{"id": "/pytorch/pytorch", "title": "PyTorch", "trustScore": 8.4},'
        '{"id": "/websites/pytorch_2_12", "title": "PyTorch", "trustScore": 10.0},'
        '{"id": "/websites/pytorch_2_11", "title": "PyTorch", "trustScore": 9.0},'
        '{"id": "/numpy/numpy", "title": "NumPy", "trustScore": 9.5}'
        "]}"
    )

    def handler(request):
        if request.url.path.endswith("/search"):
            return httpx.Response(200, text=search)
        return httpx.Response(200, text="snippet")

    async with _client(handler) as client:
        libs = await fetch_context7(client, "pytorch", timeout=5.0, max_libs=3)

    names = [lib.name for lib in libs]
    assert names.count("PyTorch") == 1  # collapsed, not 3 redundant variants
    assert "NumPy" in names  # the freed slot went to a distinct library
    pt = next(lib for lib in libs if lib.name == "PyTorch")
    assert pt.trust_score == 10.0  # kept the highest-trust variant


def test_dedup_papers_by_normalized_title():
    a = _paper(title="Same Title!", source="arxiv", identifier="2301.1v1")
    b = _paper(title="same  title", source="openalex", identifier="10.1/x")
    assert len(_dedup_papers([a, b])) == 1


def test_dedup_papers_by_identifier():
    a = _paper(title="One", identifier="2301.1")
    b = _paper(title="Two", identifier="2301.1")
    assert len(_dedup_papers([a, b])) == 1


def test_dedup_papers_merges_richer_metadata():
    sparse = _paper(title="Same", identifier="10.1/x", source="arxiv")
    rich = dataclasses.replace(
        _paper(title="Same", identifier="10.1/x", source="Journal"),
        field="Bioinformatics",
        strata=("recent",),
        full_text_available=True,
        full_text_url="https://europepmc.org/articles/PMC1",
        mesh_terms=("Reproducibility of Results",),
        query_lanes=("lane-2",),
    )

    [paper] = _dedup_papers([sparse, rich])

    assert paper.field == "Bioinformatics"
    assert paper.source == "Journal"
    assert paper.strata == ("recent",)
    assert paper.full_text_available is True
    assert paper.full_text_url == "https://europepmc.org/articles/PMC1"
    assert paper.mesh_terms == ("Reproducibility of Results",)
    assert paper.query_lanes == ("lane-2",)


def test_dedup_papers_fills_sparse_identity_without_overclaiming_full_text():
    sparse = dataclasses.replace(
        _paper(title="Same", identifier="", source="arxiv"),
        authors=(),
        year=None,
        full_text_available=None,
    )
    rich = dataclasses.replace(
        _paper(title="Same", identifier="10.1/x", source="Journal"),
        authors=("Ada Lovelace",),
        year=2024,
        full_text_available=False,
    )

    [paper] = _dedup_papers([sparse, rich])

    assert paper.identifier == "10.1/x"
    assert paper.authors == ("Ada Lovelace",)
    assert paper.year == 2024
    assert paper.full_text_available is False
    rendered = format_digest(ResearchDigest(topic="same", papers=(paper,)))
    assert "no open full text identified by Europe PMC" in rendered
    assert "full text unavailable" not in rendered


def test_dedup_papers_keeps_distinct_punctuation_titles():
    # all-punctuation titles normalize to "" and must NOT collapse together
    a = _paper(title="???", identifier="id-a")
    b = _paper(title="!!!", identifier="id-b")
    assert len(_dedup_papers([a, b])) == 2


_GITHUB_JSON = (
    '{"items": [{"full_name": "owner/repo", "description": "A repo.",'
    ' "stargazers_count": 1, "html_url": "https://github.com/owner/repo",'
    ' "language": "Python"}]}'
)
_HF_JSON = (
    '[{"id": "org/model", "downloads": 1, "likes": 1,'
    ' "pipeline_tag": "text-generation", "library_name": "transformers"}]'
)


def _all_sources_handler(arxiv_xml, openalex_json, c7_search, c7_docs, *, fail=None):
    europepmc_preprint_json = (_FIXTURES / "europepmc_sample.json").read_text()
    europepmc_published_json = (
        _FIXTURES / "europepmc_published_sample.json"
    ).read_text()

    def handler(request):
        host = request.url.host
        if host == "export.arxiv.org":
            if fail == "arxiv":
                raise httpx.TimeoutException("simulated", request=request)
            return httpx.Response(200, text=arxiv_xml)
        if host == "api.openalex.org":
            return httpx.Response(200, text=openalex_json)
        if host == "www.ebi.ac.uk":
            payload = (
                europepmc_preprint_json
                if "SRC:PPR" in request.url.params["query"]
                and "NOT SRC:PPR" not in request.url.params["query"]
                else europepmc_published_json
            )
            return httpx.Response(200, text=payload)
        if host == "context7.com":
            if request.url.path.endswith("/search"):
                return httpx.Response(200, text=c7_search)
            return httpx.Response(200, text=c7_docs)
        if host == "api.github.com":
            return httpx.Response(200, text=_GITHUB_JSON)
        if host == "huggingface.co":
            return httpx.Response(200, text=_HF_JSON)
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_research_topic_aggregates_all_sources():
    handler = _all_sources_handler(
        (_FIXTURES / "arxiv_sample.xml").read_text(),
        (_FIXTURES / "openalex_sample.json").read_text(),
        (_FIXTURES / "context7_search.json").read_text(),
        (_FIXTURES / "context7_docs.txt").read_text(),
    )
    async with _client(handler) as client:
        digest = await research_topic("diffusion", client=client)

    assert digest.topic == "diffusion"
    assert len(digest.papers) == 7
    assert any(paper.source == "Briefings in Bioinformatics" for paper in digest.papers)
    assert len(digest.libraries) == 1
    assert len(digest.repos) == 1
    assert len(digest.models) == 1
    assert digest.errors == ()
    # counts cover every queried source (whiff visibility), in query order
    assert dict(digest.counts) == {
        "arxiv": 2,
        "openalex": 2,
        "europepmc-published": 1,
        "europepmc-preprints": 2,
        "context7": 1,
        "github": 1,
        "huggingface": 1,
    }


@pytest.mark.asyncio
async def test_research_topic_routes_each_credential_only_to_its_provider():
    seen: dict[str, set[str | None]] = {}
    fixture_handler = _all_sources_handler(
        (_FIXTURES / "arxiv_sample.xml").read_text(),
        (_FIXTURES / "openalex_sample.json").read_text(),
        (_FIXTURES / "context7_search.json").read_text(),
        (_FIXTURES / "context7_docs.txt").read_text(),
    )

    def handler(request):
        seen.setdefault(request.url.host, set()).add(
            request.headers.get("authorization")
        )
        return fixture_handler(request)

    async with _client(handler) as client:
        digest = await research_topic(
            "diffusion",
            openalex_api_key="openalex-only-token",
            context7_api_key="context7-only-token",
            github_token="github-only-token",
            hf_token="huggingface-only-token",
            client=client,
        )

    assert digest.errors == ()
    assert seen == {
        "export.arxiv.org": {None},
        "api.openalex.org": {"Bearer openalex-only-token"},
        "www.ebi.ac.uk": {None},
        "context7.com": {"Bearer context7-only-token"},
        "api.github.com": {"Bearer github-only-token"},
        "huggingface.co": {"Bearer huggingface-only-token"},
    }


@pytest.mark.asyncio
async def test_research_topic_isolates_a_failing_source():
    handler = _all_sources_handler(
        "",
        (_FIXTURES / "openalex_sample.json").read_text(),
        (_FIXTURES / "context7_search.json").read_text(),
        (_FIXTURES / "context7_docs.txt").read_text(),
        fail="arxiv",
    )
    async with _client(handler) as client:
        digest = await research_topic("diffusion", client=client)

    assert any(e.startswith("arxiv:") for e in digest.errors)
    # a dead arxiv sinks neither of its peer PAPER sources
    assert len(digest.papers) == 5  # 2 OpenAlex + 3 Europe PMC hits survived
    assert len(digest.libraries) == 1  # context7 survived


def test_validate_sources_accepts_known():
    assert validate_sources(("arxiv", "openalex")) == {"arxiv", "openalex"}
    assert validate_sources(set(DEFAULT_SOURCES)) == set(DEFAULT_SOURCES)


def test_validate_sources_rejects_unknown():
    with pytest.raises(ValueError) as exc:
        validate_sources({"arxiv", "arixv"})
    msg = str(exc.value)
    assert "arixv" in msg  # names the offender
    assert "arxiv" in msg  # lists the valid set


@pytest.mark.asyncio
async def test_research_topic_rejects_unknown_source():
    # Fails fast on bad input -- before any network client is created.
    with pytest.raises(ValueError, match="paper"):
        await research_topic("x", sources={"paper"})


@pytest.mark.asyncio
async def test_research_topic_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="limit"):
        await research_topic("x", limit=0)
    with pytest.raises(ValueError, match="limit"):
        await research_topic("x", limit=-3)


@pytest.mark.asyncio
async def test_fetch_context7_fetches_docs_concurrently():
    # Three high-trust libraries whose docs endpoint blocks until all three doc
    # requests are simultaneously in flight. Concurrent fetches release the
    # barrier; a sequential loop would wait on #1 forever (never firing #2/#3)
    # and the per-request wait would time out.
    search = (
        '{"results": ['
        '{"id": "/a/a", "title": "a", "trustScore": 9.0},'
        '{"id": "/b/b", "title": "b", "trustScore": 9.0},'
        '{"id": "/c/c", "title": "c", "trustScore": 9.0}'
        "]}"
    )
    n_libs = 3
    all_in_flight = asyncio.Event()
    arrived = 0

    async def handler(request):
        nonlocal arrived
        if request.url.path.endswith("/search"):
            return httpx.Response(200, text=search)
        arrived += 1
        if arrived >= n_libs:
            all_in_flight.set()
        await asyncio.wait_for(all_in_flight.wait(), timeout=2.0)
        return httpx.Response(200, text="snippet")

    async with _client(handler) as client:
        libs = await fetch_context7(client, "q", timeout=5.0)

    assert {lib.name for lib in libs} == {"a", "b", "c"}
    assert all(lib.snippets == ("snippet",) for lib in libs)


@pytest.mark.asyncio
async def test_research_topic_threads_limit_into_context7():
    # Four high-trust libraries available; with limit=4 all four come back,
    # proving `limit` is honored (the old hard-coded max_libs=3 would cap at 3).
    search = (
        '{"results": ['
        '{"id": "/a/a", "title": "a", "trustScore": 9.0},'
        '{"id": "/b/b", "title": "b", "trustScore": 9.0},'
        '{"id": "/c/c", "title": "c", "trustScore": 9.0},'
        '{"id": "/d/d", "title": "d", "trustScore": 9.0}'
        "]}"
    )

    def handler(request):
        if request.url.path.endswith("/search"):
            return httpx.Response(200, text=search)
        return httpx.Response(200, text="diffusion snippet")

    async with _client(handler) as client:
        digest = await research_topic(
            "diffusion", sources={"context7"}, limit=4, client=client
        )

    assert {lib.name for lib in digest.libraries} == {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_research_topic_reads_context7_api_key_from_environment(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"results": []})

    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7-environment-key")
    async with _client(handler) as client:
        digest = await research_topic(
            "fastapi authentication",
            sources={"context7"},
            client=client,
        )

    assert digest.errors == ()
    assert seen["auth"] == "Bearer ctx7-environment-key"


@pytest.mark.asyncio
async def test_research_topic_redacts_context7_api_key_from_errors(monkeypatch):
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7-secret-key")

    def handler(request):
        raise httpx.ConnectError("failed with ctx7-secret-key", request=request)

    async with _client(handler) as client:
        digest = await research_topic(
            "fastapi authentication",
            sources={"context7"},
            client=client,
        )

    assert digest.errors
    assert "ctx7-secret-key" not in format_digest(digest)
    assert "<redacted>" in digest.errors[0]


def test_distinctive_terms_extracts_artifact_names():
    # The artifact names (the only thing GitHub repo-search and HF model-id search
    # can match) are kept; generic English is dropped.
    terms = _distinctive_terms("PuLID InstantID identity preserving diffusion adapter")
    assert terms == ["PuLID", "InstantID"]


def test_distinctive_terms_keeps_hyphen_digit_and_acronym_names():
    terms = _distinctive_terms("compare IP-Adapter and FLUX.2 and SDXL on faces")
    assert terms == ["IP-Adapter", "FLUX.2", "SDXL"]


def test_distinctive_terms_empty_for_generic_topic():
    # A concept topic (no artifact names) yields nothing -> caller falls back to
    # the raw phrase, which is what the paper backends want anyway.
    assert _distinctive_terms("flow matching diffusion preference optimization") == []


def test_distinctive_terms_skips_boolean_ops_and_short_tokens():
    assert _distinctive_terms("PuLID OR InstantID") == ["PuLID", "InstantID"]
    assert _distinctive_terms("AI ML on GPU") == []  # 2-char tokens dropped


def test_distinctive_terms_dedupes_and_caps():
    q = "PuLID PuLID InstantID IP-Adapter FLUX.2 PhotoMaker SDXL"
    # deduped (one PuLID) and capped at 4, first-seen order
    assert _distinctive_terms(q) == ["PuLID", "InstantID", "IP-Adapter", "FLUX.2"]


def test_distinctive_terms_strips_natural_and_markdown_punctuation():
    # `?` `!` backtick and wrapping parens must not glue onto a term -- a glued
    # term whiffs on both backends, the exact failure this whole change fixes.
    terms = _distinctive_terms("compare IP-Adapter? with `FLUX.2`! and (InstantID)")
    assert terms == ["IP-Adapter", "FLUX.2", "InstantID"]


def test_distinctive_terms_drops_pure_numeric_tokens():
    # A bare year/version is generic -> dropped, so the topic falls back to the
    # raw phrase instead of searching just "2024" / "3.5".
    assert _distinctive_terms("2024 diffusion survey") == []
    assert _distinctive_terms("SD 3.5 large") == []
    assert _distinctive_terms("FLUX.2 in 2024") == ["FLUX.2"]


def test_distinctive_terms_keeps_short_named_models():
    # Real 2-char names carrying a digit survive (the old len<3 gate dropped them);
    # generic short words are still filtered by the distinctiveness predicates.
    assert _distinctive_terms("T5 and FLAN for summarization") == ["T5", "FLAN"]


def test_distinctive_terms_drops_plural_generic_acronyms():
    # LLMs/GPUs are the plural of the generic acronyms the len>=4 gate suppresses;
    # the trailing -s must not smuggle them past it and eat a term slot.
    assert _distinctive_terms("InstantID for LLMs on GPUs") == ["InstantID"]


def test_gh_quote_passes_alnum_and_quotes_separators():
    assert _gh_quote("PuLID") == "PuLID"  # plain alnum -> no-op
    assert _gh_quote("IP-Adapter") == '"IP-Adapter"'  # hyphen -> quoted
    assert _gh_quote("FLUX.2") == '"FLUX.2"'  # dot -> quoted


@pytest.mark.asyncio
async def test_search_github_parses_repos():
    data = (
        '{"items": ['
        '{"full_name": "huggingface/diffusers", "description": "SOTA diffusion.",'
        ' "stargazers_count": 27000, "html_url": "https://github.com/huggingface/diffusers",'
        ' "language": "Python"},'
        '{"full_name": "bare/repo", "description": null,'
        ' "stargazers_count": 5,'
        ' "html_url": "https://github.com/bare/repo", "language": null}'
        "]}"
    )

    def handler(request):
        assert request.url.host == "api.github.com"
        return httpx.Response(200, text=data)

    async with _client(handler) as client:
        repos = await search_github(
            client, "diffusion", limit=5, timeout=5.0, token=None
        )

    assert len(repos) == 2
    assert repos[0].name == "huggingface/diffusers"
    assert repos[0].stars == 27000
    assert repos[0].language == "Python"
    assert repos[0].url == "https://github.com/huggingface/diffusers"
    assert repos[1].description == ""  # null -> empty string
    assert repos[1].language is None  # null -> None


@pytest.mark.asyncio
async def test_search_github_sends_bearer_token_only_in_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, text='{"items": []}')

    async with _client(handler) as client:
        await search_github(client, "q", limit=3, timeout=5.0, token="ghp_fake")

    assert seen["auth"] == "Bearer ghp_fake"  # token used, as a header
    assert "ghp_fake" not in seen["url"]  # never leaks into the URL


@pytest.mark.asyncio
async def test_search_github_constrains_to_public_repos():
    # An authenticated token can otherwise surface private repos it can see in
    # search results, which would leak their names/descriptions into the digest
    # fed to third-party council LLMs. The query must pin the search to public
    # repos regardless of token -- and prior-art research wants public code anyway.
    seen = {}

    def handler(request):
        seen["q"] = request.url.params.get("q")
        return httpx.Response(200, text='{"items": []}')

    async with _client(handler) as client:
        await search_github(
            client, "transformer", limit=3, timeout=5.0, token="ghp_fake"
        )

    assert "is:public" in seen["q"]  # private repos excluded
    assert "transformer" in seen["q"]  # original query preserved


@pytest.mark.asyncio
async def test_search_github_unions_distinctive_terms():
    # A multi-artifact topic: GitHub ANDs space-joined terms, so the combined
    # phrase matches nothing. Each distinctive term is queried on its own and the
    # results unioned + deduped by full_name, ranked by stars.
    by_term = {
        "PuLID": [("ToTheBeginning/PuLID", 3500)],
        '"IP-Adapter"': [
            ("tencent-ailab/IP-Adapter", 6600),
            ("ToTheBeginning/PuLID", 3500),  # also surfaced here -> must dedup
        ],
    }
    calls = []

    def handler(request):
        q = request.url.params.get("q")
        calls.append(q)
        repos = next((v for k, v in by_term.items() if k in q), [])
        items = [
            {
                "full_name": n,
                "stargazers_count": s,
                "description": "",
                "html_url": f"https://github.com/{n}",
                "language": "Python",
            }
            for n, s in repos
        ]
        return httpx.Response(200, json={"items": items})

    async with _client(handler) as client:
        repos = await search_github(
            client, "PuLID IP-Adapter identity", limit=5, timeout=5.0, token=None
        )

    assert len(calls) == 2  # one request per distinctive term ("identity" dropped)
    assert any('"IP-Adapter"' in q for q in calls)  # hyphenated term quoted
    assert all("is:public" in q for q in calls)  # every sub-query pinned public
    # union + dedup by name, sorted by stars desc
    assert [r.name for r in repos] == [
        "tencent-ailab/IP-Adapter",
        "ToTheBeginning/PuLID",
    ]


@pytest.mark.asyncio
async def test_search_github_falls_back_to_raw_phrase_when_no_distinctive_terms():
    # A generic topic has no artifact names -> a single raw-phrase request, kept
    # UNquoted so it stays a loose AND (quoting would force an exact-phrase match).
    # The phrase draws real hits here, so no escalation follows -- precision is
    # preserved and no extra API calls are made.
    calls = []

    def handler(request):
        calls.append(request.url.params.get("q"))
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "full_name": "atong01/conditional-flow-matching",
                        "stargazers_count": 400,
                        "description": "",
                        "html_url": "https://github.com/atong01/conditional-flow-matching",
                        "language": "Python",
                    }
                ]
            },
        )

    async with _client(handler) as client:
        repos = await search_github(
            client, "flow matching diffusion", limit=5, timeout=5.0, token=None
        )

    assert len(calls) == 1
    assert calls[0] == "flow matching diffusion is:public"  # raw, unquoted
    assert [r.name for r in repos] == ["atong01/conditional-flow-matching"]


@pytest.mark.asyncio
async def test_search_github_escalates_to_word_union_when_phrase_draws_zero():
    # A realistic full-sentence concept topic: no distinctive artifact-name terms,
    # and GitHub ANDs the whole raw phrase so it draws a clean 0. The zero must
    # not read as "no prior art" -- escalate once to a per-content-word union.
    calls = []
    topic = "Korean morphological analyzers for tokenization in NLP pipelines"

    def handler(request):
        q = request.url.params.get("q")
        calls.append(q)
        if q == f"{topic} is:public":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "full_name": f"someone/{q.split()[0]}-repo",
                        "stargazers_count": 10,
                        "description": "",
                        "html_url": "https://github.com/someone/repo",
                        "language": "Python",
                    }
                ]
            },
        )

    async with _client(handler) as client:
        repos = await search_github(client, topic, limit=5, timeout=5.0, token=None)

    assert calls[0] == f"{topic} is:public"  # raw phrase tried first
    # Escalation fires: one additional sub-query per scorable content word
    # (stopwords "for"/"in" dropped), capped at _MAX_DISTINCTIVE_TERMS.
    assert len(calls) == 1 + 4
    assert repos  # per-word union produced hits, not a silent zero


@pytest.mark.asyncio
async def test_search_github_does_not_escalate_when_distinctive_terms_present():
    # A topic WITH distinctive terms never falls to the raw-phrase path at all,
    # so a zero from every term-query must not trigger the word-union escalation.
    calls = []

    def handler(request):
        calls.append(request.url.params.get("q"))
        return httpx.Response(200, json={"items": []})

    async with _client(handler) as client:
        repos = await search_github(
            client, "PuLID IP-Adapter identity", limit=5, timeout=5.0, token=None
        )

    assert len(calls) == 2  # only the two distinctive-term sub-queries
    assert repos == []


@pytest.mark.asyncio
async def test_search_github_partial_subquery_failure_keeps_successes():
    # One term's sub-query 500s; the other term's repos must still come back --
    # the fan-out isolates per-term failures instead of discarding everything.
    def handler(request):
        q = request.url.params.get("q")
        if "PuLID" in q:
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "full_name": "ToTheBeginning/PuLID",
                            "stargazers_count": 3500,
                            "description": "",
                            "html_url": "https://github.com/ToTheBeginning/PuLID",
                            "language": "Python",
                        }
                    ]
                },
            )
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        repos = await search_github(
            client, "PuLID IP-Adapter", limit=5, timeout=5.0, token=None
        )

    assert [r.name for r in repos] == ["ToTheBeginning/PuLID"]


@pytest.mark.asyncio
async def test_search_github_raises_only_when_every_subquery_fails():
    def handler(request):
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await search_github(
                client, "PuLID IP-Adapter", limit=5, timeout=5.0, token=None
            )


@pytest.mark.asyncio
async def test_search_github_raw_phrase_failure_raises_without_escalating():
    # A generic topic whose single raw-phrase request errors out entirely must
    # raise, same as before -- a real failure is not an empty result, so the
    # escalation must not fire and mask it as a silent empty union.
    calls = []

    def handler(request):
        calls.append(request.url.params.get("q"))
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await search_github(
                client, "flow matching diffusion", limit=5, timeout=5.0, token=None
            )

    assert len(calls) == 1  # no escalation attempted after a real failure


@pytest.mark.asyncio
async def test_search_github_skips_escalation_when_identical_to_raw_phrase():
    # A single generic-word topic has no distinctive terms, so the raw-phrase
    # fallback IS "diffusion" -- and the escalation's own per-word union
    # (_scorable_terms("diffusion")) would just resubmit that exact word.
    # Escalating would guarantee the same zero and burn a call from GitHub's
    # scarce rate-limit budget for nothing; skip it.
    calls = []

    def handler(request):
        calls.append(request.url.params.get("q"))
        return httpx.Response(200, json={"items": []})

    async with _client(handler) as client:
        repos = await search_github(
            client, "diffusion", limit=5, timeout=5.0, token=None
        )

    assert len(calls) == 1  # no repeat request for the identical escalation
    assert repos == []


@pytest.mark.asyncio
async def test_search_github_escalation_ranks_multi_term_above_single_term():
    # A popular repo that only matches ONE broad escalation word must not
    # outrank a less-popular repo that matches several -- multi-term overlap
    # is stronger evidence of relevance for a broadened query than star count.
    topic = "Korean morphological analyzers for tokenization in NLP pipelines"
    items = [
        {
            "full_name": "someone/popular-unrelated",
            "stargazers_count": 50000,
            "description": "A big general repo that also supports Korean text",
            "html_url": "https://github.com/someone/popular-unrelated",
            "language": None,
        },
        {
            "full_name": "real/morphological-analyzer",
            "stargazers_count": 100,
            "description": "Morphological analyzers for Korean and Japanese",
            "html_url": "https://github.com/real/morphological-analyzer",
            "language": "Python",
        },
    ]

    def handler(request):
        q = request.url.params.get("q")
        if q == f"{topic} is:public":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(200, json={"items": items})

    async with _client(handler) as client:
        repos = await search_github(client, topic, limit=5, timeout=5.0, token=None)

    assert [r.name for r in repos] == [
        "real/morphological-analyzer",  # 3-term match (korean/morphological/analyzers)
        "someone/popular-unrelated",  # 1-term match (korean), despite ★50000
    ]
    assert all(r.broadened for r in repos)  # both tagged as broadened-query hits


@pytest.mark.asyncio
async def test_search_github_raw_zero_then_rate_limit_does_not_amplify_requests():
    # Raw phrase draws a clean zero -> escalation fires. If every escalation
    # sub-query would then hit 429, the fan-out must stop at the FIRST
    # refusal instead of burning the rest of the budget on requests that
    # would hit the same rate wall -- and the rate error must still surface,
    # not be masked as an empty union.
    topic = "Korean morphological analyzers for tokenization in NLP pipelines"
    calls = []

    def handler(request):
        q = request.url.params.get("q")
        calls.append(q)
        if q == f"{topic} is:public":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(429, text="rate limited")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await search_github(client, topic, limit=5, timeout=5.0, token=None)

    assert exc_info.value.response.status_code == 429
    # 1 raw-phrase request + exactly 1 escalation request (stopped at the
    # first 429, not all 4 escalation terms).
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_search_huggingface_parses_models():
    data = (
        '[{"id": "sentence-transformers/all-MiniLM-L6-v2", "downloads": 1500000,'
        ' "likes": 900, "pipeline_tag": "sentence-similarity",'
        ' "library_name": "sentence-transformers"},'
        '{"id": "bare/model", "downloads": 0, "likes": 0}]'
    )

    def handler(request):
        assert request.url.host == "huggingface.co"
        return httpx.Response(200, text=data)

    async with _client(handler) as client:
        models = await search_huggingface(
            client, "embedding", limit=5, timeout=5.0, token=None
        )

    assert len(models) == 2
    assert models[0].id == "sentence-transformers/all-MiniLM-L6-v2"
    assert models[0].downloads == 1500000
    assert models[0].pipeline_tag == "sentence-similarity"
    assert (
        models[0].url == "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
    )
    assert models[1].pipeline_tag is None  # missing -> None
    assert models[1].library_name is None


@pytest.mark.asyncio
async def test_search_huggingface_sends_bearer_token_only_in_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, text="[]")

    async with _client(handler) as client:
        await search_huggingface(client, "q", limit=3, timeout=5.0, token="hf_fake")

    assert seen["auth"] == "Bearer hf_fake"
    assert "hf_fake" not in seen["url"]


@pytest.mark.asyncio
async def test_search_huggingface_unions_distinctive_terms():
    # HF `search` substring-matches a model id, so a combined phrase matches
    # nothing. Each distinctive term is queried on its own (no quoting -- it is a
    # param value, not a query language) and unioned + deduped by id, by downloads.
    by_term = {
        "PuLID": [("guozinan/PuLID", 5000)],
        "IP-Adapter": [
            ("h94/IP-Adapter-FaceID", 90000),
            ("guozinan/PuLID", 5000),  # surfaced twice -> must dedup
        ],
    }
    calls = []

    def handler(request):
        s = request.url.params.get("search")
        calls.append(s)
        models = by_term.get(s, [])
        data = [{"id": mid, "downloads": dl, "likes": 1} for mid, dl in models]
        return httpx.Response(200, json=data)

    async with _client(handler) as client:
        models = await search_huggingface(
            client, "PuLID IP-Adapter identity", limit=5, timeout=5.0, token=None
        )

    assert len(calls) == 2
    assert set(calls) == {"PuLID", "IP-Adapter"}  # raw terms, no quoting
    assert [m.id for m in models] == [
        "h94/IP-Adapter-FaceID",
        "guozinan/PuLID",
    ]  # union, deduped, sorted by downloads desc


@pytest.mark.asyncio
async def test_search_huggingface_falls_back_to_raw_phrase_when_it_draws_hits():
    # A generic topic has no artifact names -> a single raw-phrase request. It
    # draws a real hit here, so no escalation follows -- precision is preserved
    # and no extra API calls are made.
    calls = []

    def handler(request):
        calls.append(request.url.params.get("search"))
        return httpx.Response(
            200, json=[{"id": "some/flow-matching-model", "downloads": 100, "likes": 1}]
        )

    async with _client(handler) as client:
        models = await search_huggingface(
            client, "flow matching diffusion", limit=5, timeout=5.0, token=None
        )

    assert len(calls) == 1
    assert calls[0] == "flow matching diffusion"  # raw phrase, unquoted
    assert [m.id for m in models] == ["some/flow-matching-model"]


@pytest.mark.asyncio
async def test_search_huggingface_escalates_to_word_union_when_phrase_draws_zero():
    # HF `search` substring-matches a model id, so ANY multi-word phrase reliably
    # matches 0 -- that must not read as "no prior art". Escalate once to a
    # per-content-word union.
    calls = []
    topic = "Korean morphological analyzers for tokenization in NLP pipelines"

    def handler(request):
        s = request.url.params.get("search")
        calls.append(s)
        if s == topic:
            return httpx.Response(200, json=[])
        return httpx.Response(
            200, json=[{"id": f"someone/{s}-model", "downloads": 10, "likes": 1}]
        )

    async with _client(handler) as client:
        models = await search_huggingface(
            client, topic, limit=5, timeout=5.0, token=None
        )

    assert calls[0] == topic  # raw phrase tried first
    # Escalation fires: one additional sub-query per scorable content word
    # (stopwords "for"/"in" dropped), capped at _MAX_DISTINCTIVE_TERMS.
    assert len(calls) == 1 + 4
    assert models  # per-word union produced hits, not a silent zero


@pytest.mark.asyncio
async def test_search_huggingface_does_not_escalate_when_distinctive_terms_present():
    # A topic WITH distinctive terms never falls to the raw-phrase path at all,
    # so a zero from every term-query must not trigger the word-union escalation.
    calls = []

    def handler(request):
        calls.append(request.url.params.get("search"))
        return httpx.Response(200, json=[])

    async with _client(handler) as client:
        models = await search_huggingface(
            client, "PuLID IP-Adapter identity", limit=5, timeout=5.0, token=None
        )

    assert len(calls) == 2  # only the two distinctive-term sub-queries
    assert models == []


@pytest.mark.asyncio
async def test_search_huggingface_raw_phrase_failure_raises_without_escalating():
    # A generic topic whose single raw-phrase request errors out entirely must
    # raise -- a real failure is not an empty result, so the escalation must not
    # fire and mask it as a silent empty union.
    calls = []

    def handler(request):
        calls.append(request.url.params.get("search"))
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await search_huggingface(
                client, "flow matching diffusion", limit=5, timeout=5.0, token=None
            )

    assert len(calls) == 1  # no escalation attempted after a real failure


@pytest.mark.asyncio
async def test_search_huggingface_skips_escalation_when_identical_to_raw_phrase():
    # A single generic-word topic has no distinctive terms, so the raw-phrase
    # fallback IS "diffusion" -- and the escalation's own per-word union
    # (_scorable_terms("diffusion")) would just resubmit that exact word.
    # Escalating would guarantee the same zero and burn a call for nothing.
    calls = []

    def handler(request):
        calls.append(request.url.params.get("search"))
        return httpx.Response(200, json=[])

    async with _client(handler) as client:
        models = await search_huggingface(
            client, "diffusion", limit=5, timeout=5.0, token=None
        )

    assert len(calls) == 1  # no repeat request for the identical escalation
    assert models == []


@pytest.mark.asyncio
async def test_search_huggingface_escalation_ranks_multi_term_above_single_term():
    # A popular model whose id only matches ONE broad escalation word must not
    # outrank a less-popular model whose id matches several -- multi-term
    # overlap is stronger evidence of relevance than download count here.
    topic = "Korean morphological analyzers for tokenization in NLP pipelines"
    data = [
        {
            "id": "hf-internal-testing/tiny-korean-bert",
            "downloads": 900000,
            "likes": 5,
        },
        {
            "id": "someone/korean-morphological-analyzers",
            "downloads": 50,
            "likes": 1,
        },
    ]

    def handler(request):
        s = request.url.params.get("search")
        if s == topic:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=data)

    async with _client(handler) as client:
        models = await search_huggingface(
            client, topic, limit=5, timeout=5.0, token=None
        )

    assert [m.id for m in models] == [
        "someone/korean-morphological-analyzers",  # 3-term match
        "hf-internal-testing/tiny-korean-bert",  # 1-term match, despite ↓900k
    ]
    assert all(m.broadened for m in models)  # both tagged as broadened-query hits


@pytest.mark.asyncio
async def test_search_huggingface_raw_zero_then_rate_limit_does_not_amplify_requests():
    # Raw phrase draws a clean zero -> escalation fires. If every escalation
    # term would then hit 429, the fan-out must stop at the FIRST refusal
    # instead of burning the rest of the budget on requests that would hit
    # the same rate wall -- and the rate error must still surface, not be
    # masked as an empty union.
    topic = "Korean morphological analyzers for tokenization in NLP pipelines"
    calls = []

    def handler(request):
        s = request.url.params.get("search")
        calls.append(s)
        if s == topic:
            return httpx.Response(200, json=[])
        return httpx.Response(429, text="rate limited")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await search_huggingface(client, topic, limit=5, timeout=5.0, token=None)

    assert exc_info.value.response.status_code == 429
    # 1 raw-phrase request + exactly 1 escalation request (stopped at the
    # first 429, not all 4 escalation terms).
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_search_huggingface_treats_non_list_success_payload_as_empty():
    async with _client(
        lambda request: httpx.Response(200, json={"error": "unexpected payload"})
    ) as client:
        models = await search_huggingface(
            client, "sentence embedding", limit=5, timeout=5.0, token=None
        )

    assert models == []


def test_clean_token_strips_and_drops_malformed():
    assert _clean_token("  ghp_clean\n") == "ghp_clean"  # surrounding ws stripped
    assert _clean_token(None) is None
    assert _clean_token("   ") is None  # whitespace-only -> absent
    assert _clean_token("ghp_\nbad") is None  # embedded newline -> dropped
    assert _clean_token("ghp_\x00bad") is None  # embedded NUL -> dropped
    assert _clean_token("ghp_\x7fbad") is None  # DEL -> dropped


def test_redact_replaces_every_secret():
    out = _redact("err ghp_aaa and hf_bbb", ["ghp_aaa", "hf_bbb", ""])
    assert out == "err <redacted> and <redacted>"  # both secrets gone, empty skipped
    assert _redact("no secrets here", []) == "no secrets here"


def test_redact_replaces_longest_overlapping_secret_first():
    assert _redact("err tokenAB", ["tokenA", "tokenAB"]) == "err <redacted>"


@pytest.mark.asyncio
async def test_search_github_strips_token_whitespace():
    # The common footgun: `export GH_TOKEN=$(cat file)` leaves a trailing newline.
    # It must be stripped so a clean header is sent -- never handed to the
    # transport raw (which would raise an h11 error echoing the secret).
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, text='{"items": []}')

    async with _client(handler) as client:
        await search_github(client, "q", limit=3, timeout=5.0, token="  ghp_clean\n")

    assert seen["auth"] == "Bearer ghp_clean"


@pytest.mark.asyncio
async def test_search_github_drops_token_with_control_chars():
    # A token that is still malformed after stripping (embedded control char) is
    # dropped entirely rather than sent -- so the transport never raises an error
    # whose string would contain the secret.
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, text='{"items": []}')

    async with _client(handler) as client:
        await search_github(client, "q", limit=3, timeout=5.0, token="ghp_\nLEAK")

    assert seen["auth"] is None  # malformed token never sent


@pytest.mark.asyncio
async def test_search_huggingface_drops_token_with_control_chars():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, text="[]")

    async with _client(handler) as client:
        await search_huggingface(client, "q", limit=3, timeout=5.0, token="hf_\nLEAK")

    assert seen["auth"] is None


@pytest.mark.asyncio
async def test_search_huggingface_excludes_private_models():
    # An authenticated token can return private models it can read; their
    # id/metadata must never reach the digest (which feeds third-party LLMs).
    data = (
        '[{"id": "org/public-model", "downloads": 100, "private": false},'
        '{"id": "org/secret-internal", "downloads": 50, "private": true}]'
    )

    def handler(request):
        return httpx.Response(200, text=data)

    async with _client(handler) as client:
        models = await search_huggingface(
            client, "q", limit=5, timeout=5.0, token="hf_fake"
        )

    ids = [m.id for m in models]
    assert "org/public-model" in ids
    assert "org/secret-internal" not in ids  # private model excluded


@pytest.mark.asyncio
async def test_research_topic_redacts_token_from_error_output():
    # Regression: a token must never reach a rendered digest, whatever error path
    # produced the message. Simulate a transport error whose string echoes the
    # Bearer token (as h11 does for a malformed header value) and assert the
    # aggregation redacts it from both digest.errors and the formatted output.
    secret = "ghp_SUPERSECRETVALUE"

    def handler(request):
        raise httpx.ConnectError(f"boom Bearer {secret}", request=request)

    async with _client(handler) as client:
        digest = await research_topic(
            "diffusion", sources={"github"}, github_token=secret, client=client
        )

    assert digest.errors  # the github source failed
    assert all(secret not in e for e in digest.errors)  # token redacted
    assert "<redacted>" in digest.errors[0]
    assert secret not in format_digest(digest)


def test_format_digest_renders_repos_and_models():
    digest = ResearchDigest(
        topic="x",
        papers=(),
        libraries=(),
        errors=(),
        repos=(
            Repo(
                name="owner/repo",
                description="A repo.",
                stars=27000,
                language="Python",
                url="https://github.com/owner/repo",
            ),
        ),
        models=(
            HFModel(
                id="org/model",
                downloads=1500000,
                likes=900,
                pipeline_tag="sentence-similarity",
                library_name="sentence-transformers",
                url="https://huggingface.co/org/model",
            ),
        ),
    )
    out = format_digest(digest)
    assert "### Repositories (GitHub)" in out
    assert "**owner/repo** (★27k, Python) — A repo." in out
    assert "### Models (HuggingFace)" in out
    assert "**org/model** (↓1.5M, sentence-similarity) [sentence-transformers]" in out
    assert "_No prior art found._" not in out  # repos/models count as found


def test_format_digest_counts_footer_makes_whiffs_visible():
    digest = ResearchDigest(
        topic="x",
        papers=(_paper(),),
        libraries=(),
        errors=(),
        counts=(("arxiv", 1), ("openalex", 0), ("context7", 0)),
    )
    out = format_digest(digest)
    assert "Sources queried:" in out
    assert "arXiv 1" in out
    assert "OpenAlex 0" in out  # a whiffed source is now visible
    assert "Context7 0" in out


def test_format_digest_counts_footer_notes_dropped_duplicates():
    # arxiv 3 + openalex 3 found, but only 4 distinct papers survive -> 2 dropped
    papers = tuple(_paper(title=f"t{i}", identifier=str(i)) for i in range(4))
    digest = ResearchDigest(
        topic="x",
        papers=papers,
        libraries=(),
        errors=(),
        counts=(("arxiv", 3), ("openalex", 3)),
    )
    out = format_digest(digest)
    assert "2 duplicate paper(s) dropped" in out


def test_compact_count():
    assert _compact_count(900) == "900"
    assert _compact_count(27000) == "27k"
    assert _compact_count(32567) == "32.6k"
    assert _compact_count(1500000) == "1.5M"
    assert _compact_count(2000000) == "2M"


# --- Europe PMC (the bioRxiv/medRxiv preprint tier) -------------------------


@pytest.mark.asyncio
async def test_search_europepmc_published_uses_full_corpus_metadata():
    payload = (_FIXTURES / "europepmc_published_sample.json").read_text()
    seen = {}

    def handler(request):
        seen["query"] = request.url.params["query"]
        seen["result_type"] = request.url.params["resultType"]
        seen["synonym"] = request.url.params["synonym"]
        return httpx.Response(200, text=payload)

    async with _client(handler) as client:
        papers = await search_europepmc_published(
            client,
            "computational reproducibility",
            limit=5,
            timeout=5.0,
            expand_synonyms=True,
        )

    assert "NOT SRC:PPR" in seen["query"]
    assert "AND (NOT SRC:PPR)" not in seen["query"]
    assert ") NOT SRC:PPR" in seen["query"]
    assert seen["result_type"] == "core"
    assert seen["synonym"] == "true"
    assert len(papers) == 1
    paper = papers[0]
    assert paper.source == "Briefings in Bioinformatics"
    assert paper.research_source == "europepmc-published"
    assert paper.full_text_available is True
    assert paper.full_text_url == "https://europepmc.org/articles/PMC1234567"
    assert paper.mesh_terms == (
        "Reproducibility of Results",
        "Computational Biology",
    )


@pytest.mark.asyncio
async def test_published_synonyms_only_backfill_a_thin_exact_search():
    calls: list[tuple[str, str]] = []

    def row(identifier: str, title: str, abstract: str) -> dict:
        return {
            "id": identifier,
            "source": "MED",
            "title": title,
            "abstractText": abstract,
            "pubYear": "2024",
        }

    exact = [
        row(
            f"exact-{index}",
            f"Computational provenance {index}",
            "Computational provenance supports reproducible experiments.",
        )
        for index in range(2)
    ]
    expanded = exact + [
        row(
            f"expanded-{index}",
            f"Unrelated biology {index}",
            "A zebrafish gene expression assay.",
        )
        for index in range(3)
    ]

    def handler(request):
        synonym = request.url.params["synonym"]
        calls.append((synonym, request.url.params["pageSize"]))
        rows = expanded if synonym == "true" else exact
        return httpx.Response(200, json={"resultList": {"result": rows}})

    async with _client(handler) as client:
        papers = await search_europepmc_published(
            client,
            "computational provenance",
            limit=5,
            timeout=5.0,
            expand_synonyms=True,
        )

    assert calls == [("false", "5"), ("true", "10")]
    assert [paper.identifier for paper in papers[:2]] == ["exact-0", "exact-1"]
    assert all(paper.strata == ("synonym-expanded",) for paper in papers[2:])
    status = _source_lane_status(
        "europepmc-published",
        "lane-1",
        "computational provenance",
        papers,
        None,
    )
    assert status.code == "THIN"
    assert status.detail == (
        "only 2 exact-match candidate(s); the rest are synonym-expanded"
    )


@pytest.mark.asyncio
async def test_published_synonym_duplicate_enriches_the_exact_record():
    exact = {
        "source": "MED",
        "title": "Computational provenance standard",
        "abstractText": "Computational provenance for reproducible experiments.",
    }
    enriched = {
        **exact,
        "doi": "10.1000/provenance",
        "meshHeadingList": {
            "meshHeading": [{"descriptorName": "Reproducibility of Results"}]
        },
    }
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        rows = [exact] if calls == 1 else [enriched]
        return httpx.Response(200, json={"resultList": {"result": rows}})

    async with _client(handler) as client:
        [paper] = await search_europepmc_published(
            client,
            "computational provenance",
            limit=2,
            timeout=5.0,
            expand_synonyms=True,
        )

    assert paper.identifier == "10.1000/provenance"
    assert paper.mesh_terms == ("Reproducibility of Results",)
    assert paper.strata == ()


@pytest.mark.asyncio
async def test_search_europepmc_published_does_not_claim_subscription_link_is_free():
    payload = {
        "resultList": {
            "result": [
                {
                    "id": "paid-1",
                    "source": "MED",
                    "title": "A subscription article",
                    "hasPDF": "Y",
                    "isOpenAccess": "N",
                    "inEPMC": "N",
                    "fullTextUrlList": {
                        "fullTextUrl": [
                            {
                                "availability": "Subscription required",
                                "availabilityCode": "S",
                                "url": "https://publisher.example/paid",
                            }
                        ]
                    },
                }
            ]
        }
    }
    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        [paper] = await search_europepmc_published(
            client, "subscription article", limit=1, timeout=5.0
        )

    assert paper.full_text_available is False
    assert paper.full_text_url == ""
    rendered = format_digest(
        ResearchDigest(topic="subscription article", papers=(paper,))
    )
    assert "[full text](" not in rendered
    assert "full text unavailable" not in rendered
    assert "no open full text identified by Europe PMC" in rendered


@pytest.mark.asyncio
async def test_published_full_text_url_rejects_markdown_delimiters():
    payload = {
        "resultList": {
            "result": [
                {
                    "id": "unsafe-1",
                    "source": "MED",
                    "title": "An article with an unsafe full-text URL",
                    "isOpenAccess": "Y",
                    "fullTextUrlList": {
                        "fullTextUrl": [
                            {
                                "availabilityCode": "OA",
                                "url": "https://example.org/paper) injected",
                            }
                        ]
                    },
                }
            ]
        }
    }
    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        [paper] = await search_europepmc_published(
            client, "unsafe full text", limit=1, timeout=5.0
        )

    assert paper.full_text_url == ""
    assert "injected" not in format_digest(
        ResearchDigest(topic="unsafe full text", papers=(paper,))
    )


@pytest.mark.asyncio
async def test_published_full_text_availability_is_unknown_when_flags_are_absent():
    payload = {
        "resultList": {
            "result": [
                {
                    "id": "unknown-1",
                    "source": "MED",
                    "title": "An article with incomplete availability metadata",
                }
            ]
        }
    }
    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        [paper] = await search_europepmc_published(
            client, "availability metadata", limit=1, timeout=5.0
        )

    assert paper.full_text_available is None


@pytest.mark.asyncio
async def test_europepmc_sources_do_not_submit_query_stripped_to_empty():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"resultList": {"result": []}})

    async with _client(handler) as client:
        published = await search_europepmc_published(client, "()", limit=5, timeout=5.0)
        preprints = await search_europepmc_preprints(client, '""', limit=5, timeout=5.0)

    assert published == []
    assert preprints == []
    assert calls == 0


@pytest.mark.asyncio
async def test_europepmc_sources_strip_boolean_and_field_syntax_from_user_query():
    queries: list[str] = []

    def handler(request):
        queries.append(request.url.params["query"])
        return httpx.Response(200, json={"resultList": {"result": []}})

    async with _client(handler) as client:
        await search_europepmc_published(
            client, "gene editing NOT CRISPR SRC:MED", limit=5, timeout=5.0
        )
        await search_europepmc_preprints(
            client, "gene editing NOT CRISPR SRC:MED", limit=5, timeout=5.0
        )

    user_groups = [query.split(")", 1)[0] for query in queries]
    assert all("NOT" not in group for group in user_groups)
    assert all(":" not in group for group in user_groups)


@pytest.mark.asyncio
async def test_search_europepmc_parses_results():
    payload = (_FIXTURES / "europepmc_sample.json").read_text()

    def handler(request):
        assert request.url.host == "www.ebi.ac.uk"
        return httpx.Response(200, text=payload)

    async with _client(handler) as client:
        papers = await search_europepmc_preprints(
            client, "batch correction", limit=5, timeout=5.0
        )

    assert len(papers) == 2
    first = papers[0]
    assert first.title.startswith("Probing as a new technique")
    assert first.authors == ("Codicè F", "Fariselli P", "Raimondi D")
    assert first.year == 2025
    assert first.identifier == "10.1101/2025.05.12.653389"
    assert first.url == "https://doi.org/10.1101/2025.05.12.653389"
    assert first.abstract.startswith("Single-cell RNA sequencing")


@pytest.mark.asyncio
async def test_search_europepmc_reports_the_preprint_server_as_the_source():
    """A hit's weight depends on it being a non-peer-reviewed bioRxiv preprint, so
    the per-paper source carries the server, not the backend that found it."""
    payload = (_FIXTURES / "europepmc_sample.json").read_text()

    async with _client(lambda r: httpx.Response(200, text=payload)) as client:
        papers = await search_europepmc_preprints(
            client, "batch correction", limit=5, timeout=5.0
        )

    assert {p.source for p in papers} == {"bioRxiv"}


@pytest.mark.asyncio
async def test_search_europepmc_strips_html_from_abstracts():
    """Europe PMC serves structured-abstract markup ('<h4>Motivation</h4>') that
    arXiv/OpenAlex never emit; raw tags would otherwise reach the council verbatim."""
    payload = (_FIXTURES / "europepmc_sample.json").read_text()

    async with _client(lambda r: httpx.Response(200, text=payload)) as client:
        papers = await search_europepmc_preprints(
            client, "batch correction", limit=5, timeout=5.0
        )

    html_abstract = papers[1].abstract
    assert "<h4>" not in html_abstract and "</h4>" not in html_abstract
    assert html_abstract.startswith("Motivation Batch effect correction is essential")


@pytest.mark.asyncio
async def test_search_europepmc_pins_query_to_preprints_excluding_arxiv():
    """This source exists to cover the preprint tier arXiv does NOT serve. Without
    the SRC:PPR pin, MEDLINE (a ~20x larger stratum) buries every preprint; without
    the arXiv exclusion it re-finds papers the arxiv source already returned."""
    seen = {}

    def handler(request):
        seen["query"] = request.url.params["query"]
        return httpx.Response(200, json={"resultList": {"result": []}})

    async with _client(handler) as client:
        await search_europepmc_preprints(
            client, "protein folding", limit=5, timeout=5.0
        )

    assert "protein folding" in seen["query"]
    assert "SRC:PPR" in seen["query"]
    assert 'NOT PUBLISHER:"arXiv"' in seen["query"]


@pytest.mark.asyncio
async def test_search_europepmc_does_not_query_on_a_blank_topic():
    """A blank query is not an error at Europe PMC -- it matches the whole corpus
    (~1.2M hits, HTTP 200), so an empty topic would silently return arbitrary
    popular papers instead of nothing."""
    called = False

    def handler(request):
        nonlocal called
        called = True
        return httpx.Response(200, json={"resultList": {"result": []}})

    async with _client(handler) as client:
        assert (
            await search_europepmc_preprints(client, "   ", limit=5, timeout=5.0) == []
        )

    assert not called


def test_strip_html_preserves_inequalities():
    """The tag stripper must not treat a '<...>' span as markup. Biology abstracts
    are full of inequalities ('p < 0.05', '> 2-fold'); a naive `<[^>]+>` eats
    everything between the first '<' and the next '>', which does not merely
    truncate -- it fabricates a plausible, WRONG sentence ('p 2-fold') and feeds it
    to the council as a finding. Measured: 7/50 abstracts on an ordinary bio query
    carry such a span."""
    text = "Expression was significant (p < 0.05) and enrichment was > 2-fold."
    assert _strip_html(text) == text


def test_strip_html_still_removes_real_markup():
    assert (
        _strip_html("<h4>Motivation</h4> Batch effects <i>confound</i> x<sub>1</sub>.")
        == "Motivation Batch effects confound x 1 ."
    )


def test_dedup_papers_normalizes_doi_identifiers_across_sources():
    """OpenAlex keeps the resolver prefix ('https://doi.org/10.1101/X'); Europe PMC
    returns the bare DOI ('10.1101/X'). Unnormalized, the SAME preprint fails
    identifier-dedup, and title-dedup only rescues it when the titles match
    character-for-character -- one word of drift and the duplicate reaches the
    council twice."""
    oa = _paper(
        title="Probing to assess batch correction",
        source="openalex",
        identifier="https://doi.org/10.1101/2025.05.12.653389",
    )
    ep = _paper(
        title="Probing: a new technique to assess batch correction",  # title drifted
        source="bioRxiv",
        identifier="10.1101/2025.05.12.653389",
    )
    assert len(_dedup_papers([oa, ep])) == 1


@pytest.mark.asyncio
async def test_search_europepmc_strips_query_syntax_that_could_restructure_the_pin():
    """A topic carrying `)` or a quote can close our grouping early and rewrite the
    boolean tree around the SRC:PPR pin. Europe PMC never 400s, so that silently
    changes what is searched instead of erroring. Neutralize the metacharacters at
    the boundary; the pin must not depend on the server's operator precedence."""
    seen = {}

    def handler(request):
        seen["query"] = request.url.params["query"]
        return httpx.Response(200, json={"resultList": {"result": []}})

    async with _client(handler) as client:
        await search_europepmc_preprints(
            client, 'cancer) OR (SRC:MED "x', limit=5, timeout=5.0
        )

    q = seen["query"]
    # exactly one group around the topic, and the pin is its own group
    assert q == '(cancer SRC MED x) AND (SRC:PPR NOT PUBLISHER:"arXiv")'


@pytest.mark.asyncio
async def test_search_europepmc_survives_null_json_fields():
    """Boundary discipline: a null `resultList` (key present, value null) raises
    AttributeError under `.get(k, {})`, and a null `author` raises TypeError -- either
    would drop the whole source rather than degrade to empty."""

    def handler(request):
        return httpx.Response(200, json={"resultList": None})

    async with _client(handler) as client:
        assert await search_europepmc_preprints(client, "x", limit=5, timeout=5.0) == []

    def null_authors(request):
        return httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [{"title": "A preprint", "authorList": {"author": None}}]
                }
            },
        )

    async with _client(null_authors) as client:
        papers = await search_europepmc_preprints(client, "x", limit=5, timeout=5.0)

    assert len(papers) == 1 and papers[0].authors == ()


# --- OpenAlex API key ------------------------------------------------------


@pytest.mark.asyncio
async def test_search_openalex_sends_the_key_as_a_bearer_header_never_in_the_url():
    """OpenAlex validates the key BOTH as an `api_key` query param and as a Bearer
    header (verified live: a bogus key 401s either way). We must use the header:
    httpx puts the full URL in its HTTPStatusError text, so a key in the query
    string lands verbatim in digest.errors -- which is rendered into the digest and
    fed to the council's third-party LLMs. Same reasoning as GH_TOKEN/HF_TOKEN."""
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        await search_openalex(
            client, "x", limit=5, timeout=5.0, email=None, api_key="SECRET-KEY"
        )

    assert seen["auth"] == "Bearer SECRET-KEY"
    assert "api_key" not in seen["url"]
    assert "SECRET-KEY" not in seen["url"]


@pytest.mark.asyncio
async def test_search_openalex_sends_no_authorization_without_a_key():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        await search_openalex(client, "x", limit=5, timeout=5.0, email=None)

    assert seen["auth"] is None


@pytest.mark.asyncio
async def test_research_topic_reads_the_openalex_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "ENV-KEY")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        await research_topic("diffusion", sources={"openalex"}, client=client)

    assert seen["auth"] == "Bearer ENV-KEY"


@pytest.mark.asyncio
async def test_research_topic_redacts_the_openalex_key_from_errors(monkeypatch):
    """Backstop: whatever error path fires, the key must never reach the digest."""
    monkeypatch.setenv("OPENALEX_API_KEY", "ENV-KEY")

    def handler(request):
        # an error whose text embeds the secret, the way httpx's own would
        raise httpx.ConnectError("boom for url ...api_key=ENV-KEY...")

    async with _client(handler) as client:
        digest = await research_topic("diffusion", sources={"openalex"}, client=client)

    assert digest.errors
    assert "ENV-KEY" not in format_digest(digest)
    assert "<redacted>" in digest.errors[0]


def test_retry_hint_for_paused_anonymous_search_says_set_a_key_not_retry():
    """OpenAlex load-sheds anonymous search with a 503 telling you to use an API
    key. Calling that 'transient -- retry the same query' sends the council into a
    pointless retry loop against a source that is deliberately refusing it."""
    hint = _retry_hint(
        "openalex: HTTPStatusError: Server error '503 Service Unavailable' for url x"
    )
    assert "OPENALEX_API_KEY" in hint
    assert "retry the same query" not in hint


def test_openalex_409_without_a_key_is_a_config_error():
    from quorum.research import _error_class

    error = (
        "openalex: HTTPStatusError: Client error '409 Conflict' for url "
        "https://api.openalex.org/works?search=diffusion"
    )

    assert _error_class(error) == "config"
    hint = _retry_hint(error)
    assert "OPENALEX_API_KEY" in hint
    assert "retry the same query" not in hint


def test_retry_hint_for_a_rejected_key_says_the_key_is_bad():
    hint = _retry_hint(
        "openalex: HTTPStatusError: Client error '401 Unauthorized' for url x"
    )
    assert "key" in hint.lower()
    assert "shorten" not in hint.lower()  # not a query-shape problem


def test_retry_hint_for_arxiv_rate_refusal_says_wait_not_key():
    """An arXiv rate refusal clears on its own -- the hint must say wait/lean on
    peers, never blame a key (there is none) or the query's shape."""
    hint = _retry_hint(
        "arxiv: RuntimeError: arXiv rate refusal: HTTP 200 with body ('Rate exceeded.')"
    )
    assert "wait" in hint.lower()
    assert "key" not in hint.lower()
    assert "shorten" not in hint.lower()


def test_retry_hint_and_class_for_a_429_say_wait_from_any_source():
    """A standard 429 is the same self-clearing refusal shape from any source:
    rate class, wait hint, and no key or query blame."""
    from quorum.research import _error_class

    err = "openalex: HTTPStatusError: Client error '429 Too Many Requests' for url x"
    assert _error_class(err) == "rate"
    hint = _retry_hint(err)
    assert "wait" in hint.lower()


def test_retry_hint_and_class_for_github_403_rate_limit_say_wait():
    """GitHub's search API signals its rate limit as '403 rate limit exceeded'
    (observed live in the q_research false-zero session; READINESS follow-up)
    -- not a 429. Same self-clearing refusal: rate class, wait hint, no
    key/query blame."""
    from quorum.research import _error_class

    err = (
        "github: HTTPStatusError: Client error '403 rate limit exceeded' "
        "for url https://api.github.com/search/repositories?q=x"
    )
    assert _error_class(err) == "rate"
    hint = _retry_hint(err)
    assert "wait" in hint.lower()
    assert "key" not in hint.lower()
    assert "key" not in hint.lower()


def test_error_class_rate_refusal_even_when_body_echoes_a_status_phrase():
    """The refusal body is server-controlled text embedded in the error string.
    A status phrase inside it ('401 unauthorized') must not flip the class to
    config -- the rate and non-feed predicates key on our own fixed phrases and
    win first (same discipline as the numbers-in-the-query test)."""
    from quorum.research import _error_class

    e = "arxiv: RuntimeError: arXiv rate refusal: HTTP 200 with body ('Rate exceeded.')"
    assert _error_class(e) == "rate"
    e2 = (
        "arxiv: RuntimeError: arXiv rate refusal: "
        "HTTP 200 with body ('401 unauthorized')"
    )
    assert _error_class(e2) == "rate"
    # the non-refusal non-feed shape is transient -- and its echoed snippet
    # cannot masquerade as a rejected key either
    e3 = (
        "arxiv: RuntimeError: arXiv answered HTTP 200 with a "
        "non-feed body ('401 unauthorized')"
    )
    assert _error_class(e3) == "transient"


def test_error_class_source_prefix_is_colon_anchored():
    """A future source such as 'arxiv-sanity' must not inherit arXiv's classes
    through a bare startswith('arxiv') prefix."""
    from quorum.research import _error_class

    e = "arxiv-sanity: RuntimeError: arXiv rate refusal: HTTP 200 with body ('x')"
    assert _error_class(e) == "transient"


def test_claim_rate_slot_serializes_concurrent_refusals(monkeypatch):
    """Two coroutines that hit the rate wall together must not wake in
    lockstep. Each retry claims a slot one polite gap after the previous claim."""
    import quorum.research as research_mod

    monkeypatch.setattr(research_mod, "_RATE_PAUSE_S", 3.0)
    monkeypatch.setattr(research_mod, "_rate_next_free", 0.0)
    assert research_mod._claim_rate_slot(100.0) == 103.0
    assert research_mod._claim_rate_slot(100.0) == 106.0
    # once the wall clears, a later claim is paced from 'now', not history
    assert research_mod._claim_rate_slot(200.0) == 203.0


def test_claim_rate_slot_refuses_unbounded_queue_debt(monkeypatch):
    import quorum.research as research_mod

    monkeypatch.setattr(research_mod, "_RATE_PAUSE_S", 3.0)
    monkeypatch.setattr(research_mod, "_MAX_RATE_WAIT_S", 9.0)
    monkeypatch.setattr(research_mod, "_rate_next_free", 0.0)

    assert [research_mod._claim_rate_slot(100.0) for _ in range(4)] == [
        103.0,
        106.0,
        109.0,
        None,
    ]


def test_europepmc_sources_split_published_work_from_preprints():
    assert "europepmc-published" in DEFAULT_SOURCES
    assert "europepmc-preprints" in DEFAULT_SOURCES
    assert "europepmc" not in DEFAULT_SOURCES
    assert validate_sources(["europepmc-published", "europepmc-preprints"]) == {
        "europepmc-published",
        "europepmc-preprints",
    }


def test_legacy_europepmc_source_names_both_replacements():
    with pytest.raises(
        ValueError,
        match=r"europepmc was split into europepmc-published and europepmc-preprints",
    ):
        validate_sources(["europepmc"])


@pytest.mark.asyncio
async def test_research_topic_queries_both_europepmc_sources():
    queries: list[str] = []

    def handler(request):
        queries.append(request.url.params["query"])
        return httpx.Response(200, json={"resultList": {"result": []}})

    async with _client(handler) as client:
        digest = await research_topic(
            "protein folding",
            sources={"europepmc-published", "europepmc-preprints"},
            client=client,
        )

    assert dict(digest.counts) == {
        "europepmc-published": 0,
        "europepmc-preprints": 0,
    }
    assert any("NOT SRC:PPR" in query for query in queries)
    assert any(
        "SRC:PPR" in query and 'NOT PUBLISHER:"arXiv"' in query for query in queries
    )


def test_format_digest_counts_europepmc_when_reporting_dropped_duplicates():
    """Both Europe PMC lanes are paper sources for duplicate accounting."""
    out = format_digest(
        ResearchDigest(
            topic="t",
            papers=(_paper(),),
            libraries=(),
            errors=(),
            counts=(
                ("arxiv", 1),
                ("openalex", 1),
                ("europepmc-published", 1),
                ("europepmc-preprints", 1),
            ),
        )
    )
    assert "Europe PMC published 1" in out
    assert "Europe PMC preprints 1" in out
    assert "3 duplicate paper(s) dropped" in out


# --- Query verdict: on-topic scoring (Layer 3) --------------------------------


def test_on_topic_fraction_flags_off_topic_collision():
    """Texts carrying the required share of the query's scorable terms score
    1.0 (here 2 of the 3 terms in `DanceGRPO flow matching`); texts that carry
    too few -- the author-surname / generic-survey collision -- drag the
    fraction down. See `_required_matches` for the bar."""
    from quorum.research import _on_topic_fraction

    query = "DanceGRPO flow matching"
    on = ["DanceGRPO improves flow matching for video", "A flow matching survey"]
    off = ["Unrelated results on B-meson decay", "ATLAS detector calibration"]
    assert _on_topic_fraction(on, query) == 1.0
    assert _on_topic_fraction(on + off, query) == 0.5
    assert _on_topic_fraction(off, query) == 0.0


def test_on_topic_fraction_rejects_single_generic_word_matches():
    """A hit must carry a SHARE of the query's terms, not just one of them.
    The failing case from the field: a classical-statistics query whose common
    words ('control', 'design', 'validation') appear in unrelated planetary and
    physics work, which the one-shared-term rule counted as on-topic and stamped
    OK. Each off-topic text below shares 1-2 of the six terms; none is on-topic."""
    from quorum.research import _on_topic_fraction

    query = "positive control negative control experimental design validation"
    off = [
        "Autonomous LiDAR control for regolith excavation. We design a rover loop.",
        "Quantum Monte Carlo for correlated electrons, with validation against ED.",
        "Design of an adaptive control system for UAV swarms in GPS-denied flight.",
    ]
    on = [
        "A positive and negative control design for experimental validation of "
        "assay pipelines, with replication.",
    ]
    assert _on_topic_fraction(off, query) == 0.0
    assert _on_topic_fraction(on, query) == 1.0


def test_on_topic_fraction_short_queries_stay_scoreable():
    """The floor of 2 matched terms is capped at the query's own term count, so
    a one-term query is not made unscoreable by it."""
    from quorum.research import _on_topic_fraction

    assert (
        _on_topic_fraction(["a sinkhorn solver for optimal transport"], "sinkhorn")
        == 1.0
    )
    assert _on_topic_fraction(["unrelated B-meson decay results"], "sinkhorn") == 0.0


def _papers(*title_abstract: tuple[str, str]):
    """Minimal Paper rows for verdict tests -- only title and abstract carry
    meaning to `compute_status`. The identifier is required by the dataclass
    and made unique per row so the rows stay distinguishable; `_dedup_papers`
    does not run on this path (it lives in `research_topic`), so nothing here
    depends on it."""
    from quorum.research import Paper

    return tuple(
        Paper(
            source="arxiv",
            title=title,
            abstract=abstract,
            url="u",
            identifier=f"id{i}",
            year=2024,
            authors=(),
        )
        for i, (title, abstract) in enumerate(title_abstract)
    )


def test_compute_status_off_topic_collision_is_retry():
    """End-to-end at the verdict layer: papers that share only generic words with
    the query no longer reach OK -- the defect Specimen 8 recorded."""
    from quorum.research import ResearchDigest, compute_status

    topic = "positive control negative control experimental design validation"
    digest = ResearchDigest(
        topic=topic,
        papers=_papers(
            (
                "LiDAR control for regolith excavation design",
                "We design a control loop for a lunar rover.",
            ),
            (
                "Quantum Monte Carlo validation study",
                "Validation against exact diagonalization.",
            ),
            (
                "Adaptive control design for UAV swarms",
                "A control system design for GPS-denied flight.",
            ),
        ),
        libraries=(),
        errors=(),
        counts=(("arxiv", 3),),
    )
    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert status.suggested_query == ""
    assert status.required_actions == (
        ResearchAction(
            "RE-ANCHOR",
            "All sources",
            "lane-1",
            "<supply different domain-specific terms>",
        ),
    )


def test_compute_status_ignores_abstract_less_papers():
    """An abstract-less paper whose bare title MISSES the coverage bar is
    dropped rather than counted as off-topic: a title has far less text to
    carry the required query terms, so scoring the miss would read a metadata
    gap as a bad query. (The hit case is the asymmetric half -- see
    `test_compute_status_counts_abstract_less_paper_whose_title_clears_the_bar`.)
    Here four abstract-less papers, whose titles carry too few query terms to
    clear the bar, sit beside three scoreable on-topic ones. Scored, they would
    drag the fraction to 3/7 and fire a RETRY on a good query; dropped, the
    verdict is the 3/3 the real evidence supports."""
    from quorum.research import ResearchDigest, compute_status

    topic = "positive control negative control experimental design validation"
    scoreable = _papers(
        *[
            (
                "Control design for assay validation",
                "A positive and negative control design for experimental validation.",
            )
        ]
        * 3
    )
    bare = _papers(*[("A control loop for lunar rovers", "")] * 4)
    digest = ResearchDigest(
        topic=topic,
        papers=scoreable + bare,
        libraries=(),
        errors=(),
        counts=(("arxiv", 7),),
    )
    assert compute_status(digest).code == "OK"


def test_compute_status_counts_abstract_less_paper_whose_title_clears_the_bar():
    """The exclusion is asymmetric. A bare title that carries the required query
    terms ANYWAY is strong evidence -- it cleared the bar on far less text --
    so it counts, and dropping it would fire RETRY on a good query. Here three
    on-topic abstract-less papers sit beside three off-topic ones that do have
    abstracts; scoring only the abstract-carrying rows would give 0/3."""
    from quorum.research import ResearchDigest, compute_status

    digest = ResearchDigest(
        topic="positive control negative control experimental design validation",
        papers=(
            _papers(
                *[("A positive and negative control design for validation", "")] * 3
            )
            + _papers(*[("LiDAR regolith survey", "A rover control loop design.")] * 3)
        ),
        libraries=(),
        errors=(),
        counts=(("arxiv", 6),),
    )
    assert compute_status(digest).code == "OK"


def test_compute_status_leaves_all_abstract_less_digest_unjudged():
    """When no paper carries an abstract AND no bare title clears the bar,
    nothing is judgeable: the overlap gate is skipped (the scoreable set is
    empty, so `_MIN_SCORING_HITS` is never met) rather than firing a RETRY on
    absent metadata."""
    from quorum.research import ResearchDigest, compute_status

    digest = ResearchDigest(
        topic="positive control negative control experimental design validation",
        papers=_papers(*[("LiDAR regolith excavation", "")] * 4),
        libraries=(),
        errors=(),
        counts=(("arxiv", 4),),
    )
    assert compute_status(digest).code == "OK"


def test_compute_status_all_paper_sources_zero_is_retry():
    """All paper sources queried but empty, on a clean query -> RETRY-REQUIRED
    with a server-suggested shorter query. This is the 'don't give up' hinge:
    a zero here is a suspected bad query, not 'no prior art'."""
    from quorum.research import ResearchDigest, compute_status

    digest = ResearchDigest(
        topic="satellite imagery equity return prediction alpha signal",
        papers=(),
        libraries=(),
        errors=(),
        counts=(
            ("arxiv", 0),
            ("openalex", 0),
            ("europepmc-preprints", 0),
        ),
    )
    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert status.suggested_query  # a concrete reworked query to resubmit


def test_compute_status_legitimate_zero_is_ok():
    """Europe PMC 0 while arXiv/OpenAlex hit = a non-biology topic, a real
    answer, not a whiff. GitHub/HF 0 alongside paper hits = non-software. OK."""
    from quorum.research import ResearchDigest, compute_status

    digest = ResearchDigest(
        topic="diffusion flow matching",
        papers=(_paper(title="Flow matching for diffusion", abstract="flow matching"),),
        libraries=(),
        errors=(),
        counts=(
            ("arxiv", 1),
            ("openalex", 1),
            ("europepmc-preprints", 0),
            ("github", 0),
        ),
    )
    status = compute_status(digest)
    assert status.code == "OK"
    assert not status.needs_action


def test_compute_status_config_error_outranks_a_bad_query():
    """A rejected key / OpenAlex anonymous load-shed is CONFIG even when the
    query also whiffed -- a retry can't fix it, so it must not read as a bad
    query the model should rework."""
    from quorum.research import ResearchDigest, compute_status

    digest = ResearchDigest(
        topic="diffusion flow matching",
        papers=(),
        libraries=(),
        errors=(
            "openalex: HTTPStatusError: Client error '401 Unauthorized' for url x",
        ),
        counts=(("arxiv", 0),),
    )
    assert compute_status(digest).code == "CONFIG"


def test_compute_status_low_overlap_splits_by_mode():
    """Off-topic hits: grounded mode calls it a defect (RETRY); exploratory mode
    reports LOW-OVERLAP but never forces a retry -- a deliberate cross-domain
    probe is expected to read off-domain."""
    from quorum.research import ResearchDigest, compute_status

    off = tuple(
        _paper(title=t, abstract=t, identifier=f"id{i}", url=f"u{i}")
        for i, t in enumerate(
            ["B-meson decay measurement", "ATLAS calorimeter", "Higgs boson search"]
        )
    )
    digest = ResearchDigest(
        topic="DanceGRPO flow matching video",
        papers=off,
        libraries=(),
        errors=(),
        counts=(("arxiv", 3),),
    )
    assert compute_status(digest, mode="grounded").code == "RETRY-REQUIRED"
    exploratory = dataclasses.replace(digest, mode="exploratory")
    assert compute_status(exploratory, mode="exploratory").code == "LOW-OVERLAP"


def test_format_digest_opens_with_research_status_line():
    """The verdict is the FIRST line so the consumer sees it before any results
    and can't skip past it. Mode flows through to the verdict."""
    off = tuple(
        _paper(title=t, abstract=t, identifier=f"id{i}", url=f"u{i}")
        for i, t in enumerate(["B-meson decay", "ATLAS run", "Higgs search"])
    )
    grounded = format_digest(
        ResearchDigest(
            topic="DanceGRPO flow matching", papers=off, counts=(("arxiv", 3),)
        )
    )
    assert grounded.splitlines()[0].startswith("Research status: RETRY-REQUIRED")
    explor = format_digest(
        ResearchDigest(
            topic="DanceGRPO flow matching",
            papers=off,
            counts=(("arxiv", 3),),
            mode="exploratory",
        ),
        mode="exploratory",
    )
    assert explor.splitlines()[0].startswith("Research status: LOW-OVERLAP")


def test_explicit_mode_override_must_match_digest_mode():
    digest = ResearchDigest(topic="sinkhorn transport", papers=(), mode="exploratory")

    assert compute_status(digest, mode="exploratory") == compute_status(digest)
    with pytest.raises(ValueError, match="does not match digest mode"):
        compute_status(digest, mode="grounded")
    with pytest.raises(ValueError, match="does not match digest mode"):
        format_digest(digest, mode="grounded")


def test_exploratory_combined_status_accepts_cross_domain_lane_overlap():
    digest = ResearchDigest(
        topic="sinkhorn transport",
        papers=(),
        mode="exploratory",
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="sinkhorn transport",
                code="QUERY-COLLISION",
                detail="cross-domain vocabulary",
                count=5,
            ),
        ),
    )

    status = compute_status(digest, mode="exploratory")

    assert status.code == "LOW-OVERLAP"


def test_grounded_mixed_on_topic_and_collision_is_degraded_without_retry_action():
    digest = ResearchDigest(
        topic="execution provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="computational reproducibility",
                code="ON-TOPIC",
                detail="candidate vocabulary matches the query lane",
                count=3,
            ),
            SourceLaneStatus(
                source="europepmc-published",
                lane="lane-2",
                query="minimum information reporting provenance",
                code="QUERY-COLLISION",
                detail="no candidate vocabulary matches the query lane",
                count=3,
            ),
        ),
    )

    rendered = format_digest(digest)

    assert rendered.startswith("Research status: DEGRADED")
    assert "Research needs action: false" in rendered
    assert "RETRY-SHORTENED" not in rendered


def test_zero_count_query_rejection_with_on_topic_peer_requires_reanchor():
    digest = ResearchDigest(
        topic="execution provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="computational reproducibility",
                code="ON-TOPIC",
                detail="candidate vocabulary matches the query lane",
                count=3,
            ),
            SourceLaneStatus(
                source="arxiv",
                lane="lane-1",
                query="execution provenance",
                code="QUERY-COLLISION",
                detail="could not be mechanically repaired",
                count=0,
            ),
        ),
    )

    status = compute_status(digest)

    assert status.code == "RETRY-REQUIRED"
    assert status.required_actions == (
        ResearchAction(
            "RE-ANCHOR",
            "arxiv",
            "lane-1",
            "<supply different domain-specific terms>",
        ),
    )


def test_artifact_hit_cannot_mask_an_all_collision_paper_stratum():
    digest = ResearchDigest(
        topic="execution provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="github",
                lane="lane-1",
                query="computational provenance tooling",
                code="ON-TOPIC",
                detail="candidate vocabulary matches the query lane",
                count=3,
            ),
            SourceLaneStatus(
                source="arxiv",
                lane="lane-1",
                query="computational provenance literature",
                code="QUERY-COLLISION",
                detail="no candidate vocabulary matches the query lane",
                count=3,
            ),
        ),
    )

    status = compute_status(digest)

    assert status.code == "RETRY-REQUIRED"
    assert status.needs_action
    assert {action.source for action in status.required_actions} == {"arxiv"}


def test_grounded_all_collisions_require_semantic_reanchors_not_shortening():
    digest = ResearchDigest(
        topic="execution provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="minimum information experimental reporting provenance",
                code="QUERY-COLLISION",
                detail="no candidate vocabulary matches the query lane",
                count=3,
            ),
            SourceLaneStatus(
                source="europepmc-published",
                lane="lane-2",
                query="traceability protocol validation reporting",
                code="QUERY-COLLISION",
                detail="no candidate vocabulary matches the query lane",
                count=3,
            ),
        ),
    )

    status = compute_status(digest)

    assert status.code == "RETRY-REQUIRED"
    assert status.suggested_query == ""
    assert {action.kind for action in status.required_actions} == {"RE-ANCHOR"}
    assert {action.query for action in status.required_actions} == {
        "<supply different domain-specific terms>"
    }


def test_collision_and_infrastructure_without_evidence_keep_both_actions():
    digest = ResearchDigest(
        topic="execution provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="minimum information reporting provenance",
                code="QUERY-COLLISION",
                detail="no candidate vocabulary matches the query lane",
                count=3,
            ),
            SourceLaneStatus(
                source="github",
                lane="lane-1",
                query="computational provenance tooling",
                code="INFRASTRUCTURE",
                detail="source failed after its bounded retry",
                count=0,
            ),
        ),
    )

    status = compute_status(digest)

    assert status.code == "RETRY-REQUIRED"
    assert {(action.source, action.kind) for action in status.required_actions} == {
        ("openalex", "RE-ANCHOR"),
        ("github", "RETRY-SAME"),
    }
    assert "infrastructure rows retry unchanged" in status.detail


def test_paper_rework_keeps_multiple_domain_terms_instead_of_artifact_name():
    from quorum.research import _query_action

    action = _query_action(
        "arxiv", "lane-1", "FMCW radar vital signs signal processing"
    )

    assert action.kind == "RETRY-SHORTENED"
    assert action.query == "FMCW radar vital"


def test_paper_rework_preserves_distinctive_names_before_domain_terms():
    from quorum.research import _query_action

    action = _query_action(
        "arxiv", "lane-1", "IP-Adapter FLUX.2 identity preserving adapter survey"
    )

    assert action.kind == "RETRY-SHORTENED"
    assert action.query == "IP-Adapter FLUX.2"


def test_paper_rework_does_not_pad_two_named_terms_with_a_filler():
    from quorum.research import _query_action

    action = _query_action(
        "arxiv", "lane-1", "compare IP-Adapter with FLUX.2 performance"
    )

    assert action.kind == "RETRY-SHORTENED"
    assert action.query == "IP-Adapter FLUX.2"


@pytest.mark.parametrize("query", ("contrastive learning", "diffusion models"))
def test_paper_rework_requires_two_scorable_domain_terms(query):
    from quorum.research import _query_action

    action = _query_action("arxiv", "lane-1", query)

    assert action.kind == "RE-ANCHOR"
    assert action.query == "<supply different domain-specific terms>"


def test_artifact_rework_keeps_distinctive_names():
    from quorum.research import _query_action

    action = _query_action(
        "github", "lane-1", "PuLID InstantID identity preserving adapter"
    )

    assert action.kind == "RETRY-SHORTENED"
    assert action.query == "PuLID InstantID"


def test_aggregate_rework_keeps_distinctive_artifact_names():
    from quorum.research import _query_action

    action = _query_action(
        "All sources", "lane-1", "IP-Adapter FLUX.2 identity preserving adapter"
    )

    assert action.kind == "RETRY-SHORTENED"
    assert action.query == "IP-Adapter FLUX.2"


def test_legacy_all_zero_paper_recovery_uses_paper_rework_with_all_sources_action():
    digest = ResearchDigest(
        topic="FMCW radar vital signs signal processing",
        papers=(),
        counts=(("arxiv", 0),),
    )

    status = compute_status(digest)

    assert status.suggested_query == "FMCW radar vital"
    assert status.required_actions == (
        ResearchAction("RETRY-SHORTENED", "All sources", "lane-1", "FMCW radar vital"),
    )


def test_legacy_all_zero_paper_recovery_preserves_distinctive_artifact_tokens():
    digest = ResearchDigest(
        topic="IP-Adapter FLUX.2 identity preserving adapter survey",
        papers=(),
        counts=(("arxiv", 0),),
    )

    status = compute_status(digest)

    assert status.required_actions == (
        ResearchAction(
            "RETRY-SHORTENED",
            "All sources",
            "lane-1",
            "IP-Adapter FLUX.2",
        ),
    )


@pytest.mark.asyncio
async def test_arxiv_400_retry_retains_multiple_domain_terms():
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.params["search_query"])
        return httpx.Response(400, text="bad request")

    async with _client(handler) as client:
        await research_topic(
            "FMCW radar vital signs signal processing",
            sources={"arxiv"},
            client=client,
        )

    assert len(calls) == 2
    assert "vital" in calls[1]
    assert "radar" in calls[1]
    assert "fmcw" in calls[1].lower()


@pytest.mark.asyncio
async def test_arxiv_400_retry_preserves_distinctive_artifact_tokens_and_shortens():
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.params["search_query"])
        return httpx.Response(400, text="bad request")

    async with _client(handler) as client:
        await research_topic(
            "IP-Adapter FLUX.2 identity preserving adapter survey",
            sources={"arxiv"},
            client=client,
        )

    assert len(calls) == 2
    assert "IP-Adapter" in calls[1]
    assert "FLUX.2" in calls[1]
    assert len(calls[1]) < len(calls[0])


@pytest.mark.asyncio
async def test_research_topic_rejects_reanchor_placeholder_before_dispatch():
    dispatched = False

    def handler(request):
        nonlocal dispatched
        dispatched = True
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        with pytest.raises(ValueError, match="re-anchor placeholder"):
            await research_topic(
                "<supply different domain-specific terms>",
                sources={"openalex"},
                client=client,
            )

    assert not dispatched


@pytest.mark.asyncio
async def test_research_topic_rejects_reanchor_placeholder_lane_before_dispatch():
    dispatched = False

    def handler(request):
        nonlocal dispatched
        dispatched = True
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        with pytest.raises(ValueError, match="re-anchor placeholder"):
            await research_topic(
                "execution provenance",
                query_lanes=("<supply different domain-specific terms>",),
                sources={"openalex"},
                client=client,
            )

    assert not dispatched


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    (
        "`<SUPPLY DIFFERENT DOMAIN-SPECIFIC TERMS>`",
        "| <supply different domain-specific terms> |",
        '"<supply different domain-specific terms>"',
        "**<supply different domain-specific terms>**",
        "<supply different domain-specific terms>.",
        "<supply different domain-specific terms> radar",
    ),
)
async def test_research_topic_rejects_wrapped_reanchor_placeholder_before_dispatch(
    query,
):
    dispatched = False

    def handler(request):
        nonlocal dispatched
        dispatched = True
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        with pytest.raises(ValueError, match="re-anchor placeholder"):
            await research_topic(query, sources={"openalex"}, client=client)

    assert not dispatched


@pytest.mark.asyncio
async def test_query_lanes_report_status_per_source_without_hiding_good_lane():
    def rows(prefix: str, abstract: str) -> list[dict]:
        return [
            {
                "id": f"{prefix}-{index}",
                "source": "MED",
                "title": f"{prefix} {index}",
                "abstractText": abstract,
                "pubYear": "2024",
            }
            for index in range(3)
        ]

    def handler(request):
        query = request.url.params["query"]
        if "computational reproducibility" in query:
            result = rows(
                "Computational reproducibility method",
                "A computational reproducibility method for experiments.",
            )
        else:
            result = rows(
                "Cancer biomarker",
                "A clinical oncology biomarker validation study.",
            )
        return httpx.Response(200, json={"resultList": {"result": result}})

    async with _client(handler) as client:
        digest = await research_topic(
            "execution provenance",
            query_lanes=(
                "computational reproducibility method",
                "minimum information reporting provenance",
            ),
            sources={"europepmc-published"},
            client=client,
        )

    statuses = {(row.lane, row.code) for row in digest.source_statuses}
    assert statuses == {
        ("lane-1", "ON-TOPIC"),
        ("lane-2", "QUERY-COLLISION"),
    }
    assert len(digest.papers) == 3
    assert all("Cancer biomarker" not in paper.title for paper in digest.papers)
    rendered = format_digest(digest)
    assert rendered.startswith("Research status: DEGRADED")
    assert "Research needs action: false" in rendered
    assert "### Required research actions" not in rendered
    assert "### Source/lane status" in rendered
    assert "| Europe PMC published | lane-1 | ON-TOPIC |" in rendered
    assert "| Europe PMC published | lane-2 | QUERY-COLLISION |" in rendered
    assert "only 0% of candidates share lane vocabulary" in rendered


@pytest.mark.asyncio
async def test_artifact_collision_keeps_diagnostics_but_excludes_repositories():
    repos = {
        "items": [
            {
                "full_name": f"org/cancer-biomarker-{index}",
                "description": "clinical oncology biomarker validation",
                "stargazers_count": 10 - index,
                "html_url": f"https://github.com/org/cancer-biomarker-{index}",
            }
            for index in range(3)
        ]
    }

    async with _client(lambda request: httpx.Response(200, json=repos)) as client:
        digest = await research_topic(
            "execution provenance",
            sources={"github"},
            client=client,
        )

    assert digest.repos == ()
    assert digest.source_statuses == (
        SourceLaneStatus(
            source="github",
            lane="lane-1",
            query="execution provenance",
            code="QUERY-COLLISION",
            detail="only 0% of candidates share lane vocabulary",
            count=3,
        ),
    )
    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert status.required_actions[0].kind == "RE-ANCHOR"
    rendered = format_digest(digest)
    assert "| GitHub | lane-1 | QUERY-COLLISION | 3 |" in rendered
    assert (
        "GitHub 0 shown after dedup/cap; 3 lane-result candidate(s) rejected "
        "before dedup/cap" in rendered
    )
    assert "_No prior art found._" not in rendered


@pytest.mark.asyncio
async def test_collision_retains_individually_on_topic_papers_and_counts_exclusions():
    result = [
        {
            "id": f"on-topic-{index}",
            "source": "MED",
            "title": f"Execution provenance method {index}",
            "abstractText": "Execution provenance for reproducible experiments.",
            "pubYear": "2024",
        }
        for index in range(2)
    ] + [
        {
            "id": f"off-topic-{index}",
            "source": "MED",
            "title": f"Cancer biomarker {index}",
            "abstractText": "Clinical oncology biomarker validation study.",
            "pubYear": "2024",
        }
        for index in range(3)
    ]

    async with _client(
        lambda request: httpx.Response(200, json={"resultList": {"result": result}})
    ) as client:
        digest = await research_topic(
            "execution provenance",
            sources={"europepmc-published"},
            client=client,
        )

    assert digest.source_statuses[0].code == "QUERY-COLLISION"
    assert digest.source_statuses[0].count == 5
    assert [paper.identifier for paper in digest.papers] == ["on-topic-0", "on-topic-1"]
    assert digest.excluded_collision_counts == (("europepmc-published", 3),)
    rendered = format_digest(digest)
    assert (
        "Europe PMC published 2 shown after dedup/cap; 3 lane-result candidate(s) "
        "rejected before dedup/cap" in rendered
    )


@pytest.mark.asyncio
async def test_collision_retains_papers_not_used_to_grade_the_lane(monkeypatch):
    import quorum.research as research_mod

    papers = [
        _paper(
            title=f"Cancer biomarker {index}",
            abstract="Clinical oncology biomarker validation study.",
            identifier=f"off-topic-{index}",
        )
        for index in range(3)
    ] + [
        _paper(
            title="Expanded terminology result",
            abstract="Alternate vocabulary supplied by the source.",
            identifier="synonym-expanded",
            strata=("synonym-expanded",),
        ),
        _paper(
            title="Record without abstract metadata",
            abstract="",
            identifier="abstract-missing",
        ),
    ]

    async def fake_arxiv(client, query, *, limit, timeout):
        return papers

    monkeypatch.setattr(research_mod, "search_arxiv", fake_arxiv)
    async with _client(lambda request: httpx.Response(500)) as client:
        digest = await research_topic(
            "execution provenance", sources={"arxiv"}, client=client
        )

    assert digest.source_statuses[0].code == "QUERY-COLLISION"
    assert {paper.identifier for paper in digest.papers} == {
        "synonym-expanded",
        "abstract-missing",
    }
    assert digest.excluded_collision_counts == (("arxiv", 3),)


@pytest.mark.asyncio
async def test_context7_collision_retains_individually_on_topic_libraries(monkeypatch):
    import quorum.research as research_mod

    libraries = [
        LibraryDoc(
            name=f"execution-provenance-{index}",
            description="execution provenance for reproducible experiments",
            snippets=(),
            trust_score=9.0,
            url=f"https://context7.com/on-topic-{index}",
        )
        for index in range(2)
    ] + [
        LibraryDoc(
            name=f"cancer-biomarker-{index}",
            description="clinical oncology biomarker validation",
            snippets=(),
            trust_score=9.0,
            url=f"https://context7.com/off-topic-{index}",
        )
        for index in range(3)
    ]

    async def fake_context7(client, query, *, timeout, max_libs, token):
        return libraries

    monkeypatch.setattr(research_mod, "fetch_context7", fake_context7)
    async with _client(lambda request: httpx.Response(500)) as client:
        digest = await research_topic(
            "execution provenance", sources={"context7"}, client=client
        )

    assert [library.name for library in digest.libraries] == [
        "execution-provenance-0",
        "execution-provenance-1",
    ]
    assert digest.excluded_collision_counts == (("context7", 3),)


@pytest.mark.asyncio
async def test_exploratory_collision_retains_cross_domain_candidates():
    result = [
        {
            "id": f"cancer-{index}",
            "source": "MED",
            "title": f"Cancer biomarker {index}",
            "abstractText": "A clinical oncology biomarker validation study.",
            "pubYear": "2024",
        }
        for index in range(3)
    ]

    async with _client(
        lambda request: httpx.Response(200, json={"resultList": {"result": result}})
    ) as client:
        digest = await research_topic(
            "execution provenance",
            sources={"europepmc-published"},
            mode="exploratory",
            client=client,
        )

    assert len(digest.papers) == 3
    assert digest.source_statuses[0].code == "QUERY-COLLISION"
    assert digest.mode == "exploratory"
    assert compute_status(digest).code == "LOW-OVERLAP"
    rendered = format_digest(digest)
    assert rendered.startswith("Research status: LOW-OVERLAP")
    assert "rejected before dedup/cap" not in rendered


def test_usable_peer_evidence_with_exhausted_infrastructure_is_degraded():
    """A source that survives its bounded in-tool retry must prevent OK, but
    usable peer evidence means there is no remaining mechanical action for the
    caller and therefore no need for cross-call retry state."""
    digest = ResearchDigest(
        topic="computational reproducibility",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="computational reproducibility",
                code="ON-TOPIC",
                detail="candidate vocabulary matches the query lane",
                count=5,
            ),
            SourceLaneStatus(
                source="arxiv",
                lane="lane-1",
                query="computational reproducibility",
                code="INFRASTRUCTURE",
                detail="source failed after its bounded retry",
                count=0,
            ),
        ),
    )

    status = compute_status(digest)

    assert status.code == "DEGRADED"
    assert not status.needs_action
    assert "bounded retry" in status.detail


def test_usable_evidence_collision_and_infrastructure_is_degraded_with_outage_detail():
    digest = ResearchDigest(
        topic="computational reproducibility",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="computational reproducibility",
                code="ON-TOPIC",
                detail="candidate vocabulary matches the query lane",
                count=3,
            ),
            SourceLaneStatus(
                source="europepmc-published",
                lane="lane-1",
                query="minimum reporting provenance",
                code="QUERY-COLLISION",
                detail="no candidate vocabulary matches the query lane",
                count=3,
            ),
            SourceLaneStatus(
                source="arxiv",
                lane="lane-1",
                query="computational reproducibility",
                code="INFRASTRUCTURE",
                detail="source failed after its bounded retry",
                count=0,
            ),
        ),
    )

    status = compute_status(digest)

    assert status.code == "DEGRADED"
    assert not status.needs_action
    assert status.required_actions == ()
    assert "bounded retry" in status.detail


@pytest.mark.parametrize(
    ("peer_code", "peer_count", "mode"),
    (
        ("THIN", 2, "grounded"),
        ("QUERY-COLLISION", 5, "exploratory"),
    ),
)
def test_any_mode_usable_peer_evidence_with_infrastructure_is_degraded(
    peer_code, peer_count, mode
):
    digest = ResearchDigest(
        topic="cross-domain method",
        papers=(),
        mode=mode,
        source_statuses=(
            SourceLaneStatus(
                source="github",
                lane="lane-1",
                query="cross-domain method",
                code=peer_code,
                detail="peer evidence",
                count=peer_count,
            ),
            SourceLaneStatus(
                source="context7",
                lane="lane-1",
                query="cross-domain method",
                code="INFRASTRUCTURE",
                detail="source failed after its bounded retry",
                count=0,
            ),
        ),
    )

    status = compute_status(digest, mode=mode)

    assert status.code == "DEGRADED"
    assert not status.needs_action


def test_retry_required_status_cannot_omit_exact_actions():
    with pytest.raises(ValueError, match="required_actions"):
        ResearchStatus("RETRY-REQUIRED", "unfinished research")


def test_empty_artifact_rows_render_an_exact_required_action():
    digest = ResearchDigest(
        topic="computational provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="context7",
                lane="lane-1",
                query="computational provenance",
                code="THIN",
                detail="only 0 candidates",
                count=0,
            ),
            SourceLaneStatus(
                source="github",
                lane="lane-2",
                query="experimental traceability",
                code="THIN",
                detail="only 0 candidates",
                count=0,
            ),
        ),
    )

    rendered = format_digest(digest)

    assert rendered.startswith("Research status: RETRY-REQUIRED")
    assert "Research needs action: true" in rendered
    assert "### Required research actions" in rendered
    status = compute_status(digest)
    assert {(action.source, action.lane) for action in status.required_actions} == {
        ("context7", "lane-1"),
        ("github", "lane-2"),
    }


def test_reanchor_action_does_not_present_the_rejected_query_as_executable():
    rejected = "sinkhorn transport"
    digest = ResearchDigest(
        topic=rejected,
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query=rejected,
                code="QUERY-COLLISION",
                detail="collision",
                count=3,
            ),
        ),
    )

    action = compute_status(digest).required_actions[0]

    assert action.kind == "RE-ANCHOR"
    assert action.query != rejected
    assert "different domain" in action.query


def test_empty_paper_stratum_retries_only_failed_sources():
    """A peer source that completed with a valid zero must not be queried again;
    only the source whose result is unknown needs the unchanged retry."""
    digest = ResearchDigest(
        topic="computational provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="arxiv",
                lane="lane-1",
                query="computational provenance",
                code="INFRASTRUCTURE",
                detail="source failed after its bounded retry",
                count=0,
            ),
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="computational provenance",
                code="THIN",
                detail="only 0 candidates",
                count=0,
            ),
        ),
    )

    status = compute_status(digest)

    assert status.code == "RETRY-REQUIRED"
    assert status.required_actions == (
        ResearchAction("RETRY-SAME", "arxiv", "lane-1", "computational provenance"),
    )


def test_empty_paper_stratum_reworks_every_source_lane():
    digest = ResearchDigest(
        topic="computational provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="computational provenance",
                code="THIN",
                detail="only 0 candidates",
                count=0,
            ),
            SourceLaneStatus(
                source="openalex",
                lane="lane-2",
                query="experimental traceability",
                code="THIN",
                detail="only 0 candidates",
                count=0,
            ),
        ),
    )

    status = compute_status(digest)

    assert status.code == "RETRY-REQUIRED"
    assert {(action.source, action.lane) for action in status.required_actions} == {
        ("openalex", "lane-1"),
        ("openalex", "lane-2"),
    }


@pytest.mark.asyncio
async def test_empty_peer_source_is_reported_as_source_mismatch():
    works = {
        "results": [
            {
                "title": f"Computational reproducibility method {index}",
                "publication_year": 2024,
                "doi": f"https://doi.org/10.1/{index}",
                "authorships": [],
                "abstract_inverted_index": {
                    "computational": [0],
                    "reproducibility": [1],
                },
                "primary_topic": None,
            }
            for index in range(3)
        ]
    }

    def handler(request):
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json=works)
        return httpx.Response(200, json={"resultList": {"result": []}})

    async with _client(handler) as client:
        digest = await research_topic(
            "computational reproducibility method",
            sources={"openalex", "europepmc-preprints"},
            client=client,
        )

    statuses = {row.source: row.code for row in digest.source_statuses}
    assert statuses["openalex"] == "ON-TOPIC"
    assert statuses["europepmc-preprints"] == "SOURCE-MISMATCH"


@pytest.mark.asyncio
async def test_artifact_hit_cannot_hide_empty_paper_stratum():
    repos = {
        "items": [
            {
                "full_name": f"org/computational-provenance-{index}",
                "description": "computational provenance artifacts",
                "stargazers_count": 10 - index,
                "html_url": f"https://github.com/org/repo-{index}",
            }
            for index in range(3)
        ]
    }

    def handler(request):
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json=repos)

    async with _client(handler) as client:
        digest = await research_topic(
            "computational provenance artifacts",
            sources={"openalex", "github"},
            client=client,
        )

    statuses = {row.source: row.code for row in digest.source_statuses}
    assert statuses["github"] == "ON-TOPIC"
    assert statuses["openalex"] == "THIN"
    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert status.needs_action
    assert status.required_actions
    assert status.required_actions[0].source == "openalex"
    assert status.required_actions[0].lane == "lane-1"


@pytest.mark.asyncio
async def test_colliding_short_lane_requires_semantic_reanchor():
    result = [
        {
            "id": f"cancer-{index}",
            "source": "MED",
            "title": f"Cancer biomarker {index}",
            "abstractText": "A clinical oncology biomarker validation study.",
            "pubYear": "2024",
        }
        for index in range(3)
    ]

    async with _client(
        lambda request: httpx.Response(200, json={"resultList": {"result": result}})
    ) as client:
        digest = await research_topic(
            "minimum information reporting provenance",
            sources={"europepmc-published"},
            client=client,
        )

    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert status.suggested_query == ""
    assert "semantically re-anchor" in status.detail


@pytest.mark.asyncio
async def test_failed_arxiv_shortening_is_not_suggested_again():
    async with _client(lambda request: httpx.Response(400, text="bad query")) as client:
        digest = await research_topic(
            "computational experiment provenance reproducibility artifacts",
            sources={"arxiv"},
            client=client,
        )

    assert digest.source_statuses[0].query == ("computational experiment provenance")
    assert "could not be mechanically repaired" in digest.source_statuses[0].detail
    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert status.suggested_query == ""


def test_collision_action_names_the_colliding_lane_without_shortening():
    digest = ResearchDigest(
        topic="execution provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-2",
                query="minimum information experimental reporting provenance",
                code="QUERY-COLLISION",
                detail="collision",
                count=3,
            ),
        ),
    )

    rendered = format_digest(digest)

    assert "· try lane-2:" not in rendered
    assert "| RE-ANCHOR | OpenAlex | lane-2 |" in rendered


def test_collision_actions_keep_their_own_lanes_with_multiple_collisions():
    digest = ResearchDigest(
        topic="execution provenance",
        papers=(),
        source_statuses=(
            SourceLaneStatus(
                source="openalex",
                lane="lane-1",
                query="sinkhorn transport",
                code="QUERY-COLLISION",
                detail="collision",
                count=3,
            ),
            SourceLaneStatus(
                source="openalex",
                lane="lane-2",
                query="minimum information experimental reporting provenance",
                code="QUERY-COLLISION",
                detail="collision",
                count=3,
            ),
        ),
    )

    rendered = format_digest(digest)

    assert "· try lane-2:" not in rendered
    assert "| RE-ANCHOR | OpenAlex | lane-1 |" in rendered
    assert "| RE-ANCHOR | OpenAlex | lane-2 |" in rendered


def test_errors_without_usable_evidence_require_retry_instead_of_degrading():
    digest = ResearchDigest(
        topic="diffusion flow matching",
        papers=(),
        errors=("context7: TimeoutException: slow",),
    )

    status = compute_status(digest)

    assert status.code == "RETRY-REQUIRED"
    assert status.needs_action


def test_legacy_errors_deduplicate_required_source_actions():
    digest = ResearchDigest(
        topic="diffusion flow matching",
        papers=(),
        errors=(
            "context7: TimeoutException: first lane",
            "context7: TimeoutException: second lane",
        ),
    )

    status = compute_status(digest)

    assert [action.source for action in status.required_actions] == ["context7"]


@pytest.mark.asyncio
async def test_small_limit_is_thin_but_not_forced_to_retry():
    result = [
        {
            "id": f"method-{index}",
            "source": "MED",
            "title": f"Computational reproducibility method {index}",
            "abstractText": "A computational reproducibility method.",
            "pubYear": "2024",
        }
        for index in range(2)
    ]
    async with _client(
        lambda request: httpx.Response(200, json={"resultList": {"result": result}})
    ) as client:
        digest = await research_topic(
            "computational reproducibility method",
            sources={"europepmc-published"},
            limit=2,
            client=client,
        )

    assert digest.source_statuses[0].code == "THIN"
    assert compute_status(digest).code == "OK"


def test_unscoreable_papers_are_thin_not_on_topic():
    status = _source_lane_status(
        "europepmc-published",
        "lane-1",
        "computational reproducibility",
        [
            dataclasses.replace(_paper(title=f"Unrelated title {index}"), abstract="")
            for index in range(3)
        ],
        None,
    )

    assert status.code == "THIN"


def test_huggingface_model_ids_are_thin_not_query_collisions():
    status = _source_lane_status(
        "huggingface",
        "lane-1",
        "protein language model",
        [
            HFModel(
                id=f"org/model-{index}",
                downloads=100,
                likes=1,
                pipeline_tag="feature-extraction",
                library_name="transformers",
                url=f"https://huggingface.co/org/model-{index}",
            )
            for index in range(3)
        ],
        None,
    )

    assert status.code == "THIN"


def test_descriptionless_repositories_are_thin_not_query_collisions():
    status = _source_lane_status(
        "github",
        "lane-1",
        "computational provenance",
        [
            Repo(
                name=f"org/provenance-{index}",
                description="",
                stars=10,
                language=None,
                url=f"https://github.com/org/provenance-{index}",
            )
            for index in range(3)
        ],
        None,
    )

    assert status.code == "THIN"


def test_library_snippets_supply_scoreable_status_text():
    status = _source_lane_status(
        "context7",
        "lane-1",
        "computational provenance",
        [
            LibraryDoc(
                name=f"library-{index}",
                description="",
                snippets=("Computational provenance records execution artifacts.",),
                trust_score=9.0,
                url=f"https://context7.com/library-{index}",
            )
            for index in range(3)
        ],
        None,
    )

    assert status.code == "ON-TOPIC"


@pytest.mark.asyncio
async def test_artifact_sources_only_use_the_primary_semantic_lane():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "full_name": "org/provenance-tool",
                        "description": "computational provenance artifacts",
                        "stargazers_count": 10,
                        "html_url": "https://github.com/org/provenance-tool",
                    }
                ]
            },
        )

    async with _client(handler) as client:
        digest = await research_topic(
            "execution provenance",
            query_lanes=(
                "computational provenance artifacts",
                "raw data traceability",
                "minimum information reporting",
            ),
            sources={"github"},
            client=client,
        )

    assert calls == 1
    assert len(digest.source_statuses) == 1
    assert len(digest.repos) == 1


@pytest.mark.asyncio
async def test_sources_run_concurrently_while_each_sources_lanes_stay_sequential(
    monkeypatch,
):
    import quorum.research as research_mod

    second_openalex_lane_started = asyncio.Event()
    calls: dict[str, list[str]] = {"arxiv": [], "openalex": []}

    async def fake_arxiv(client, query, *, limit, timeout):
        calls["arxiv"].append(query)
        if len(calls["arxiv"]) == 1:
            await second_openalex_lane_started.wait()
        return []

    async def fake_openalex(client, query, *, limit, timeout, email, api_key, purpose):
        calls["openalex"].append(query)
        if len(calls["openalex"]) == 2:
            second_openalex_lane_started.set()
        return []

    monkeypatch.setattr(research_mod, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(research_mod, "search_openalex", fake_openalex)
    monkeypatch.setattr(research_mod, "_RATE_PAUSE_S", 0.0)

    async with _client(lambda request: httpx.Response(500)) as client:
        await asyncio.wait_for(
            research_topic(
                "execution provenance",
                query_lanes=(
                    "computational provenance",
                    "reporting traceability",
                ),
                sources={"arxiv", "openalex"},
                client=client,
            ),
            timeout=0.5,
        )

    assert calls == {
        "arxiv": ["computational provenance", "reporting traceability"],
        "openalex": ["computational provenance", "reporting traceability"],
    }


@pytest.mark.asyncio
async def test_arxiv_semantic_lanes_observe_the_polite_gap(monkeypatch):
    import quorum.research as research_mod

    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def fake_arxiv(client, query, *, limit, timeout):
        return []

    monkeypatch.setattr(research_mod, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(research_mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(research_mod, "_RATE_PAUSE_S", 3.0)

    async with _client(lambda request: httpx.Response(500)) as client:
        await research_topic(
            "execution provenance",
            query_lanes=(
                "computational provenance",
                "reporting traceability",
            ),
            sources={"arxiv"},
            client=client,
        )

    assert sleeps == [3.0]


@pytest.mark.asyncio
async def test_query_lanes_do_not_bypass_generic_topic_lint():
    with pytest.raises(ValueError, match="Research topic 'data'"):
        await research_topic(
            "data",
            query_lanes=("computational provenance reproducibility",),
            sources={"openalex"},
        )


@pytest.mark.asyncio
async def test_research_topic_normalizes_topic_before_rendering():
    async with _client(lambda request: httpx.Response(500)) as client:
        digest = await research_topic(
            "execution provenance\n\n## fabricated evidence",
            sources=set(),
            client=client,
        )

    rendered = format_digest(digest)
    assert digest.topic == "execution provenance ## fabricated evidence"
    assert "\n## fabricated evidence" not in rendered


@pytest.mark.asyncio
async def test_research_topic_rejects_limit_above_provider_maximum():
    async with _client(lambda request: httpx.Response(500)) as client:
        with pytest.raises(ValueError, match=r"limit must be <= 20"):
            await research_topic(
                "execution provenance",
                sources=set(),
                limit=21,
                client=client,
            )


@pytest.mark.asyncio
async def test_multi_lane_papers_are_balanced_and_capped_per_source():
    def handler(request):
        lane = "primary" if "computational" in request.url.params["query"] else "review"
        rows = [
            {
                "id": f"{lane}-{index}",
                "source": "MED",
                "title": f"{lane} provenance {index}",
                "abstractText": f"{lane} provenance reproducibility",
                "pubYear": "2024",
            }
            for index in range(3)
        ]
        return httpx.Response(200, json={"resultList": {"result": rows}})

    async with _client(handler) as client:
        digest = await research_topic(
            "execution provenance",
            query_lanes=(
                "computational provenance reproducibility",
                "reporting provenance review",
            ),
            sources={"europepmc-published"},
            limit=3,
            client=client,
        )

    assert len(digest.papers) == 3
    assert {paper.query_lanes for paper in digest.papers} == {
        ("lane-1",),
        ("lane-2",),
    }


@pytest.mark.asyncio
async def test_research_topic_auto_retries_arxiv_400_with_shorter_query():
    """An arXiv 400 (query too long / has operators) is mechanically fixable, so
    research_topic reworks to scorable domain terms and resubmits ONCE inside the
    tool -- the orchestrator never sees the first-attempt failure it would use as
    an excuse to fall back on its own knowledge. The rework is surfaced as a note."""
    xml = (_FIXTURES / "arxiv_sample.xml").read_text()
    calls: list[str] = []

    def handler(request):
        sq = request.url.params.get("search_query", "")
        calls.append(sq)
        if "survey" in sq:  # the long first-attempt query
            return httpx.Response(400, text="bad request")
        return httpx.Response(200, text=xml)  # the shortened retry

    async with _client(handler) as client:
        digest = await research_topic(
            "FLUX.2 identity preserving adapter survey",
            sources={"arxiv"},
            client=client,
        )

    assert digest.papers  # the retry succeeded, so papers came back
    assert len(calls) == 2  # exactly one failure + one retry (not a loop)
    assert not digest.errors  # a repaired source is not reported as unavailable
    assert any("FLUX.2 identity preserving" in n for n in digest.notes)


@pytest.mark.asyncio
async def test_research_topic_does_not_retry_arxiv_400_for_case_only_rework():
    """Round-3 review: when an arXiv 400's only available 'rework' is a case- or
    whitespace-normalized copy of the same query, _contained must NOT burn a
    second request on a semantically identical query. 'Diffusion Flow Matching'
    -> _suggest_query 'diffusion flow matching' is a no-op rework, so the source
    is called exactly once and the error is surfaced."""
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.params.get("search_query", ""))
        return httpx.Response(400, text="bad request")

    async with _client(handler) as client:
        digest = await research_topic(
            "Diffusion Flow Matching", sources={"arxiv"}, client=client
        )

    assert len(calls) == 1  # no wasted retry on a case-only rework
    assert digest.errors  # the 400 is still surfaced, not silently dropped


@pytest.mark.asyncio
async def test_research_topic_does_not_retry_a_config_error():
    """A 401 is the key, not the query -- retrying is futile and just doubles the
    latency. The source is called exactly once and the error is surfaced."""
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.params.get("search_query", ""))
        return httpx.Response(401, text="unauthorized")

    async with _client(handler) as client:
        digest = await research_topic(
            "diffusion flow matching", sources={"arxiv"}, client=client
        )

    assert len(calls) == 1  # no retry on a config-class error
    assert digest.errors and "401" in digest.errors[0]
    assert not digest.notes
    assert "retry will not fix" in format_digest(digest).lower()


def test_format_digest_renders_retry_notes():
    out = format_digest(
        ResearchDigest(
            topic="t",
            papers=(_paper(),),
            notes=('arxiv: retried after auto-shortening query to "FLUX.2"',),
        )
    )
    assert "↻" in out
    assert "auto-shortening" in out


@pytest.mark.asyncio
async def test_research_topic_retries_transient_with_same_query(monkeypatch):
    """A timeout is transient -- retry ONCE with the SAME query after a
    bounded delay, rather than immediately repeating the same failing call."""
    import quorum.research as research_mod

    xml = (_FIXTURES / "arxiv_sample.xml").read_text()
    calls: list[str] = []
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    def handler(request):
        calls.append(request.url.params.get("search_query", ""))
        if len(calls) == 1:
            raise httpx.TimeoutException("slow")
        return httpx.Response(200, text=xml)

    monkeypatch.setattr(research_mod, "_TRANSIENT_RETRY_DELAY_S", 0.75)
    monkeypatch.setattr(research_mod.asyncio, "sleep", fake_sleep)

    async with _client(handler) as client:
        digest = await research_topic(
            "diffusion flow matching", sources={"arxiv"}, client=client
        )

    assert len(calls) == 2
    assert calls[0] == calls[1]  # same query on the retry
    assert sleeps == [0.75]
    assert digest.papers and not digest.errors
    assert any("transient" in n for n in digest.notes)


@pytest.mark.asyncio
async def test_research_topic_retries_arxiv_rate_refusal_after_pause(monkeypatch):
    """The 200-'Rate exceeded' refusal clears on its own, so retry ONCE with the
    SAME query after a polite pause (patched to 0 here), and surface the repair
    as a note naming the refusal -- not a generic 'transient failure'."""
    import quorum.research as research_mod

    monkeypatch.setattr(research_mod, "_RATE_PAUSE_S", 0.0)
    xml = (_FIXTURES / "arxiv_sample.xml").read_text()
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.params.get("search_query", ""))
        if len(calls) == 1:
            return httpx.Response(200, text="Rate exceeded.")
        return httpx.Response(200, text=xml)

    async with _client(handler) as client:
        digest = await research_topic(
            "diffusion flow matching", sources={"arxiv"}, client=client
        )

    assert len(calls) == 2
    assert calls[0] == calls[1]  # same query -- the query was never the problem
    assert digest.papers and not digest.errors
    assert any("rate" in n.lower() for n in digest.notes)


@pytest.mark.asyncio
async def test_research_topic_surfaces_persistent_arxiv_rate_refusal(monkeypatch):
    """A refusal that survives the paused retry is surfaced as an error carrying
    the body text, with a wait hint -- never as 'arxiv found 0 papers'."""
    import quorum.research as research_mod

    monkeypatch.setattr(research_mod, "_RATE_PAUSE_S", 0.0)
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.params.get("search_query", ""))
        return httpx.Response(200, text="Rate exceeded.")

    async with _client(handler) as client:
        digest = await research_topic(
            "diffusion flow matching", sources={"arxiv"}, client=client
        )

    assert len(calls) == 2  # one paused retry, not a loop
    assert digest.errors and "rate exceeded" in digest.errors[0].lower()
    assert "arxiv" not in dict(digest.counts)  # errored, not counted as 0 hits
    out = format_digest(digest)
    assert "wait" in out.lower()


@pytest.mark.asyncio
async def test_research_topic_refuses_a_generic_query():
    """A query with no distinctive terms (empty, or only generic/stopword
    tokens) is refused up front with an actionable message -- a tool that won't
    run is far harder to shrug off than a footnote. Refusal happens before any
    network call (no client needed)."""
    import pytest as _pytest

    for bad in ("data", "the model", "a survey of methods", "   "):
        with _pytest.raises(ValueError, match="distinctive"):
            await research_topic(bad, sources={"arxiv"})


@pytest.mark.asyncio
async def test_map_openalex_fields_returns_subfield_distribution():
    """The topology engine: one group_by call turns a method query into a
    ranked subfield distribution -- the fields the method actually spans. Blank
    'unknown' buckets are dropped."""
    import json as _json

    from quorum.research import map_openalex_fields

    payload = {
        "group_by": [
            {"key": "a", "key_display_name": "Artificial Intelligence", "count": 8239},
            {"key": "b", "key_display_name": "Geophysics", "count": 1547},
            {"key": "c", "key_display_name": "", "count": 10},
        ]
    }

    def handler(request):
        assert request.url.host == "api.openalex.org"
        assert request.url.params.get("group_by") == "primary_topic.subfield.id"
        return httpx.Response(200, text=_json.dumps(payload))

    async with _client(handler) as client:
        fields = await map_openalex_fields(
            client, "sparse signal anomaly detection", timeout=5.0
        )

    assert fields[0] == ("Artificial Intelligence", 8239)
    assert ("Geophysics", 1547) in fields
    assert all(name for name, _ in fields)  # blank display-name bucket dropped


@pytest.mark.asyncio
async def test_research_topic_populates_field_map_when_requested():
    """map_fields=True runs the extra group_by call and fills digest.field_map;
    the render shows the spread. A non-openalex-only run leaves it empty."""
    import json as _json

    xml = (_FIXTURES / "arxiv_sample.xml").read_text()
    group = {
        "group_by": [
            {"key": "a", "key_display_name": "Geophysics", "count": 1547},
            {"key": "b", "key_display_name": "Oceanography", "count": 1295},
        ]
    }

    def handler(request):
        if request.url.host == "api.openalex.org":
            if request.url.params.get("group_by"):
                return httpx.Response(200, text=_json.dumps(group))
            return httpx.Response(200, text=_json.dumps({"results": []}))
        return httpx.Response(200, text=xml)

    async with _client(handler) as client:
        digest = await research_topic(
            "sparse signal anomaly detection",
            sources={"arxiv", "openalex"},
            map_fields=True,
            mode="exploratory",
            client=client,
        )

    assert ("Geophysics", 1547) in digest.field_map
    out = format_digest(digest, mode="exploratory")
    assert "### Field map" in out
    assert "Geophysics 1.5k" in out


@pytest.mark.asyncio
async def test_research_topic_skips_field_map_by_default():
    """Default (grounded/brainstorm): no extra group_by call, empty field_map."""
    import json as _json

    def handler(request):
        assert request.url.params.get("group_by") is None  # never grouped
        return httpx.Response(200, text=_json.dumps({"results": []}))

    async with _client(handler) as client:
        digest = await research_topic(
            "diffusion flow matching", sources={"openalex"}, client=client
        )
    assert digest.field_map == ()


@pytest.mark.asyncio
async def test_search_openalex_annotates_paper_field():
    """OpenAlex papers carry their subfield/field (from primary_topic) so the
    skystorm harvest step can see which field each hit sits in."""
    import json as _json

    payload = {
        "results": [
            {
                "title": "Sparse anomaly detection in seismic streams",
                "publication_year": 2024,
                "doi": "https://doi.org/10.1/x",
                "authorships": [],
                "abstract_inverted_index": None,
                "primary_topic": {
                    "field": {"display_name": "Earth and Planetary Sciences"},
                    "subfield": {"display_name": "Geophysics"},
                },
            }
        ]
    }

    def handler(request):
        assert "primary_topic" in request.url.params.get("select", "")
        return httpx.Response(200, text=_json.dumps(payload))

    async with _client(handler) as client:
        papers = await search_openalex(
            client, "sparse anomaly seismic", limit=3, timeout=5.0, email=None
        )

    assert papers[0].field == "Geophysics"


def test_format_digest_annotates_field_in_exploratory_mode_only():
    """Field annotation is skystorm signal -- shown in exploratory mode, kept out
    of the grounded (brainstorm) render to avoid clutter."""
    p = _paper(source="openalex", field="Geophysics")
    explor = format_digest(
        ResearchDigest(topic="t", papers=(p,), mode="exploratory"),
        mode="exploratory",
    )
    grounded = format_digest(ResearchDigest(topic="t", papers=(p,)), mode="grounded")
    assert "[field: Geophysics]" in explor
    assert "[field: Geophysics]" not in grounded


def test_format_digest_shows_openalex_strata_and_full_text_availability():
    paper = dataclasses.replace(
        _paper(source="Briefings in Bioinformatics"),
        research_source="europepmc-published",
        strata=("all-time", "recent"),
        full_text_available=True,
        full_text_url="https://europepmc.org/articles/PMC123",
        query_lanes=("lane-1", "lane-2"),
    )

    rendered = format_digest(ResearchDigest(topic="reproducibility", papers=(paper,)))

    assert "[strata: all-time, recent]" in rendered
    assert "[source: Europe PMC published]" in rendered
    assert "[lanes: lane-1, lane-2]" in rendered
    assert "[full text] <https://europepmc.org/articles/PMC123>" in rendered


def test_compute_status_all_paper_sources_errored_is_not_ok():
    """When every paper source errors, none reach digest.counts. The verdict
    must be RETRY-REQUIRED, not 'results look on-topic'."""
    from quorum.research import ResearchDigest, compute_status

    digest = ResearchDigest(
        topic="diffusion flow matching",
        papers=(),
        errors=(
            "arxiv: TimeoutException: slow",
            "openalex: TimeoutException: slow",
            "europepmc-preprints: TimeoutException: slow",
        ),
        counts=(),
    )
    assert compute_status(digest).code == "RETRY-REQUIRED"


def test_compute_status_omits_suggested_query_when_it_equals_original():
    """If the reworked query is identical to the original, the skill's
    mandated 'retry with the suggested query' would loop forever. Omit
    the suggestion and tell the agent to re-anchor with its own judgment."""
    from quorum.research import ResearchDigest, compute_status

    # A short query that _suggest_query returns unchanged, all paper sources empty.
    digest = ResearchDigest(
        topic="diffusion flow matching",
        papers=(),
        counts=(("arxiv", 0), ("openalex", 0)),
    )
    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert status.suggested_query == ""  # no identical-query suggestion
    assert "different" in status.detail.lower()  # steer to judgment, not a loop


def test_retry_hint_and_error_class_agree_on_config_errors():
    """Round-2 review: the comment claims _retry_hint and _error_class share
    predicates so they 'never drift'. Pin it: every error _error_class calls
    'config' must get a non-retry (key/refusal) hint from _retry_hint, never the
    'shorten and resubmit' arXiv hint that implies the query is at fault."""
    from quorum.research import _error_class, _retry_hint

    config_errors = [
        "openalex: HTTPStatusError: Client error '401 Unauthorized' for url ...",
        "github: HTTPStatusError: Client error '403 Forbidden' for url ...",
        "openalex: HTTPStatusError: Server error '503 Service Unavailable' for url ...",
    ]
    for err in config_errors:
        assert _error_class(err) == "config", err
        hint = _retry_hint(err).lower()
        assert "shorten" not in hint  # not the query-shortening (arXiv) hint
        assert "key" in hint or "anonymous" in hint  # the config remedy


def test_compute_status_mixed_error_and_zero_is_not_labeled_all_zero():
    """Round-2 review: when one paper source ERRORS and another returns 0, the
    stratum total is 0 but it is NOT 'every paper source returned 0'. That label
    masks the infrastructure failure and steers the agent to mutate a query that
    may have been fine. The mixed state must name the failure and must NOT offer a
    reworked query."""
    from quorum.research import ResearchDigest, compute_status

    digest = ResearchDigest(
        topic="diffusion flow matching",
        papers=(),
        counts=(("openalex", 0),),  # succeeded, 0 hits
        errors=("arxiv: TimeoutException: timed out",),  # errored
    )
    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert "returned 0 on a valid query" not in status.detail
    assert "failed" in status.detail.lower()  # the infra failure is surfaced
    assert "arxiv" in status.detail.lower()  # and names WHICH source to investigate
    assert status.suggested_query == ""  # do not mutate a possibly-fine query


def test_compute_status_omits_suggested_query_for_case_only_rework():
    """Round-2 review: _suggest_query lowercases, so a mixed-case topic like
    'Diffusion Flow Matching' would rework to 'diffusion flow matching' -- a
    cosmetically-different string the agent would resubmit for one wasted,
    identical retry. A case-only difference is not a meaningful rework."""
    from quorum.research import ResearchDigest, compute_status

    digest = ResearchDigest(
        topic="Diffusion Flow Matching",
        papers=(),
        counts=(("arxiv", 0), ("openalex", 0)),
    )
    status = compute_status(digest)
    assert status.code == "RETRY-REQUIRED"
    assert status.suggested_query == ""  # case-only rework is not offered
    assert "different" in status.detail.lower()


def test_error_class_is_not_fooled_by_numbers_in_the_query():
    """A 500 error whose URL echoes a query containing '4000' must NOT match
    the '400' path. Classification keys on the HTTP status phrase, not
    bare digits."""
    from quorum.research import _error_class

    e = (
        "arxiv: HTTPStatusError: Server error '500 Internal Server Error' "
        "for url ...search=RTX 4000"
    )
    assert _error_class(e) == "transient"
    e2 = (
        "openalex: HTTPStatusError: Server error '500' "
        "for url ...search=Area 401 mapping"
    )
    assert _error_class(e2) == "transient"


def test_error_class_scopes_query_shorten_to_arxiv():
    """A 400 is arXiv's 'query too long/operators' signal; a 400 from GitHub or
    OpenAlex means something else and must NOT trigger query-mutation retry."""
    from quorum.research import _error_class

    assert (
        _error_class("arxiv: HTTPStatusError: Client error '400 Bad Request' for url x")
        == "query"
    )
    assert (
        _error_class(
            "github: HTTPStatusError: Client error '400 Bad Request' for url x"
        )
        == "transient"
    )


def test_error_class_config_needs_status_phrase():
    from quorum.research import _error_class

    assert (
        _error_class(
            "openalex: HTTPStatusError: Client error '401 Unauthorized' for url x"
        )
        == "config"
    )
    assert (
        _error_class(
            "openalex: HTTPStatusError: Server error "
            "'503 Service Unavailable' for url x"
        )
        == "config"
    )


def test_scorable_terms_drops_single_char_tokens():
    """A single character ('x') must not be scorable because it matches almost
    every text and produces a false OK. Lint then refuses 'x' alone."""
    from quorum.research import _scorable_terms

    assert _scorable_terms("x") == []
    assert _scorable_terms("flow x matching") == ["flow", "matching"]


def test_scorable_terms_keeps_single_letter_compound():
    """Round-2/3 review: the hyphen split + single-char drop combined to reject
    'Q-learning' ('q' dropped + stopword 'learning' -> empty -> pre-flight refuses
    a valid term). A single-letter compound is now kept as a JOINED canonical
    token ('qlearning'), not a bare 'q' -- the bare letter false-matched unrelated
    compounds like 'Q-factor'. A bare single char is still dropped (#6)."""
    from quorum.research import _scorable_terms

    assert _scorable_terms("Q-learning")  # topic is not refused
    assert "q" not in _scorable_terms("Q-learning")  # no bare-letter token
    assert _scorable_terms("x") == []  # bare single char still dropped


def test_iter_tokens_single_letter_compound_no_false_match():
    """Round-3 review: 'Q-learning' represented as bare 'q' false-matched the
    unrelated 'Q-factor' (both -> 'q'). The joined canonical token matches its own
    compound form but NOT an unrelated single-letter compound."""
    from quorum.research import _match_tokens

    assert _match_tokens("deep Q-learning") & _match_tokens("Q-learning applied")
    assert not (_match_tokens("Q-learning") & _match_tokens("Q-factor deposition"))


def test_on_topic_fraction_normalizes_hyphens():
    """'flow-matching' must match papers that say 'flow matching' because
    a hyphen/slash is a token boundary, not an off-topic signal."""
    from quorum.research import _on_topic_fraction

    assert _on_topic_fraction(["a flow matching study"], "flow-matching") == 1.0
    assert _on_topic_fraction(["optimal transport / sinkhorn"], "sinkhorn") == 1.0


def test_format_digest_no_prior_art_excludes_field_map_case():
    """A digest that carries only a field map must not print
    '_No prior art found._' because the field map is a result."""
    d = ResearchDigest(
        topic="t", papers=(), field_map=(("Geophysics", 1547),), mode="exploratory"
    )
    out = format_digest(d, mode="exploratory")
    assert "_No prior art found._" not in out
    assert "Geophysics" in out
