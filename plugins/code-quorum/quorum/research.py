"""Prior-art research for grounded brainstorming. Queries arXiv, OpenAlex,
Europe PMC published/preprints, Context7 (library docs), GitHub (repos), and HuggingFace
(models) over HTTP and returns a bounded digest. Standalone -- no council
coupling. Mirrors the agent runtime's ethos: async, per-call timeouts, failures
surfaced into the digest rather than swallowed or allowed to sink peer sources.

Europe PMC is split into a published lane (full-corpus metadata, MeSH, and
full-text availability) and a preprint lane (bioRxiv, medRxiv, Research Square,
excluding arXiv)."""

from __future__ import annotations

import asyncio
import datetime
import functools
import os
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import TypeVar, cast
from xml.etree import ElementTree as ET

import httpx

DEFAULT_LIMIT = 5
_MAX_LIMIT = 20
RESEARCH_PURPOSES = ("methods", "currency")
_BOOLEAN_OPS = frozenset({"OR", "AND", "NOT"})
# Sources whose results land in digest.papers. Kept as one constant because the
# renderer's duplicate-drop arithmetic must agree with research_topic's routing:
# a paper source missing from here silently under-reports dropped duplicates.
_PAPER_SOURCES = frozenset(
    {"arxiv", "openalex", "europepmc-published", "europepmc-preprints"}
)
_ABSTRACT_MAX_CHARS = 240
_MAX_SNIPPETS = 3
# Display labels for the per-source count footer (whiff visibility).
_SOURCE_LABELS = {
    "arxiv": "arXiv",
    "openalex": "OpenAlex",
    "europepmc-published": "Europe PMC published",
    "europepmc-preprints": "Europe PMC preprints",
    "context7": "Context7",
    "github": "GitHub",
    "huggingface": "HuggingFace",
}
DEFAULT_SOURCES: tuple[str, ...] = tuple(_SOURCE_LABELS)


@dataclass(frozen=True)
class Paper:
    title: str
    authors: tuple[str, ...]
    year: int | None
    # Human-facing venue: arxiv/openalex, a published journal, or the preprint
    # server itself (bioRxiv, medRxiv, ...). Backend identity is kept separately
    # in `research_source` for per-source status reporting.
    source: str
    identifier: str  # arXiv id or DOI
    url: str
    abstract: str
    # OpenAlex subfield (from primary_topic), e.g. "Geophysics" -- the skystorm
    # field annotation. None for sources without a taxonomy (arXiv, Europe PMC).
    field: str | None = None
    # Backend identity is distinct from the human-facing venue in `source`.
    # Europe PMC preprints display their server (bioRxiv, medRxiv, ...), while
    # status reporting still needs to attribute the hit to its explicit lane.
    research_source: str = ""
    full_text_available: bool | None = None
    full_text_url: str = ""
    mesh_terms: tuple[str, ...] = ()
    strata: tuple[str, ...] = ()
    query_lanes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LibraryDoc:
    name: str
    description: str
    snippets: tuple[str, ...]
    trust_score: float
    url: str


@dataclass(frozen=True)
class Repo:
    name: str  # "owner/repo"
    description: str
    stars: int
    language: str | None
    url: str
    # True when this hit came from the per-content-word escalation (a broadened
    # query), not the precise distinctive-term/raw-phrase query -- so the
    # renderer can mark it for a reader to discount relative to a precise hit.
    broadened: bool = False


@dataclass(frozen=True)
class HFModel:
    id: str  # "org/model"
    downloads: int
    likes: int
    pipeline_tag: str | None  # task, e.g. "sentence-similarity"
    library_name: str | None
    url: str
    # Same escalation marker as Repo.broadened.
    broadened: bool = False


@dataclass(frozen=True)
class SourceLaneStatus:
    source: str
    lane: str
    query: str
    code: str
    detail: str
    count: int


@dataclass(frozen=True)
class ResearchDigest:
    topic: str
    papers: tuple[Paper, ...]
    libraries: tuple[LibraryDoc, ...] = ()
    errors: tuple[str, ...] = ()
    repos: tuple[Repo, ...] = ()
    models: tuple[HFModel, ...] = ()
    # (source, n_results_retained) per queried, non-errored source after the
    # per-source cap. A 0 surfaces a whiffed query the consumer should rework.
    counts: tuple[tuple[str, int], ...] = ()
    # One line per source that succeeded only after an in-tool auto-retry (arXiv
    # 400 reworked to a shorter query, or a transient failure re-tried) -- so a
    # silently-repaired source is still visible to the consumer.
    notes: tuple[str, ...] = ()
    # (subfield, count) pairs from OpenAlex group_by -- the skystorm 'field map'
    # showing which fields the query spans. Populated only when map_fields=True
    # (exploratory/skystorm); empty otherwise.
    field_map: tuple[tuple[str, int], ...] = ()
    # One deterministic quality row per source/query-lane attempt. Empty on
    # legacy/manual digests, whose combined status still uses the aggregate
    # fallback in `compute_status`.
    source_statuses: tuple[SourceLaneStatus, ...] = ()
    # Candidates removed from grounded collision lanes, grouped by source.
    excluded_collision_counts: tuple[tuple[str, int], ...] = ()
    # The retrieval mode determines whether lexical collisions are rejected
    # (grounded) or retained as deliberate cross-domain evidence (exploratory).
    mode: str = "grounded"


def _truncate(text: str, limit: int = _ABSTRACT_MAX_CHARS) -> str:
    """Collapse whitespace and cap at `limit` Unicode codepoints (not bytes),
    appending an ellipsis when truncated."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _clean_token(token: str | None) -> str | None:
    """Make a token safe to place in an Authorization header. Strips surrounding
    whitespace (a trailing newline from `export TOK=$(cat file)` is the common
    case) and drops the token entirely if it isn't all-printable-ASCII -- so a
    malformed token is never handed to the transport, whose protocol error would
    otherwise echo the secret value into digest.errors."""
    if not token:
        return None
    token = token.strip()
    if not token or not (token.isascii() and token.isprintable()):
        return None
    return token


def _redact(text: str, secrets: Iterable[str]) -> str:
    """Replace any secret value with <redacted> -- a backstop so a token can
    never surface in a rendered digest, whatever error path produced the text."""
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def _compact_count(n: int) -> str:
    """Human-compact a count: 1500000 -> '1.5M', 32567 -> '33k', 900 -> '900'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.1f}k".replace(".0k", "k")
    return str(n)


# Error classification is shared by _retry_hint (the inline hint text) and
# _error_class (the retry/verdict decision) -- enumerate-writers: both parse the
# SAME error string, so the phrase predicates live here once and BOTH callers
# compose from them (a config error therefore always yields a config hint -- they
# cannot drift). They key on the HTTP status PHRASE ('400 bad request'), never a
# bare digit, so a query that echoes a status-like number in the URL ('RTX 4000',
# 'Area 401') cannot flip the classification. Fixed phrases that this module
# emits ('rate refusal:', 'non-feed body') also cannot flip it. The 400/query
# case is scoped to arXiv, the only backend for which a 400 means 'query too
# long / has operators'. A 400 from any other source is left transient rather
# than mutating a query that was valid.
def _is_auth_error(low: str) -> bool:
    # Deliberately NOT scoped to a source prefix (unlike the arXiv/OpenAlex
    # predicates): a 401/403 from ANY source means a rejected credential that a
    # retry cannot fix, so classing it 'config' (no retry) is right regardless of
    # which backend raised it -- even a keyless source hitting a proxy/CDN 401.
    return "401 unauthorized" in low or "403 forbidden" in low


def _is_openalex_shedding(low: str) -> bool:
    return low.startswith("openalex:") and "503 service" in low


def _is_openalex_key_required(low: str) -> bool:
    return low.startswith("openalex:") and "409 conflict" in low


def _is_config_error(low: str) -> bool:
    return (
        _is_auth_error(low)
        or _is_openalex_shedding(low)
        or _is_openalex_key_required(low)
    )


def _is_arxiv_query_error(low: str) -> bool:
    return low.startswith("arxiv:") and "400 bad request" in low


# The source prefixes above and below are colon-anchored to the error-string
# format `f"{name}: ..."`. A bare startswith("arxiv") would let a future
# source such as 'arxiv-sanity' inherit arXiv's classes.


def _is_arxiv_rate_refusal(low: str) -> bool:
    # search_arxiv recognizes arXiv's documented refusal (HTTP 200 with a bare
    # 'Rate exceeded' body) and emits the fixed phrase 'rate refusal:' -- OUR
    # text, not the server's. Both consumers check this and _is_arxiv_non_feed
    # FIRST: the body snippet after the phrase is server-controlled and could
    # echo a status phrase ('401 unauthorized') that would otherwise flip the
    # class.
    return low.startswith("arxiv:") and "rate refusal:" in low


def _is_arxiv_non_feed(low: str) -> bool:
    # A 200 body that is neither a feed nor the documented refusal marker (a
    # proxy or block page): transient, not rate -- and pinned before the
    # config predicates for the same snippet-echo reason as above.
    return low.startswith("arxiv:") and "non-feed body" in low


def _is_rate_limited_status(low: str) -> bool:
    # A standard 429 from ANY source is the same shape of deliberate,
    # self-clearing refusal. It is keyed by a status phrase, like the auth
    # predicate, never a bare digit. GitHub's search API signals the
    # same refusal as "403 rate limit exceeded" (observed live), so both
    # phrases wear the rate label.
    return "429 too many requests" in low or "403 rate limit exceeded" in low


_TRANSIENT_HINT = (
    "likely transient (timeout/flake) — retry the same query; shorten it if it persists"
)


def _retry_hint(error: str) -> str:
    """Map one source error string to a one-line, actionable retry instruction,
    rendered inline under the failed source so the consumer reworks-and-retries
    at the decision point instead of treating the error as a dead backend and
    falling back to its own knowledge.

    The cases are distinguished because the right move differs, and telling the
    consumer to 'retry' when retrying cannot work sends it into a loop:
    - arXiv 200 with a bare 'Rate exceeded' body, or a 429 from any source --
      a rate refusal that clears on its own. Wait, then retry the same query;
      checked first (see _is_arxiv_rate_refusal for why).
    - arXiv 200 with any OTHER non-feed body (proxy/block page) -- transient,
      but pinned before the config cases so its snippet cannot flip the class.
    - arXiv 400 bad request -- the in-tool mechanical query rework was either
      impossible or also rejected. Semantically re-anchor rather than shortening
      again.
    - 401 / 403 -- the KEY, not the query or the backend. Retrying is futile.
    - 409 on OpenAlex -- anonymous demo credits are exhausted; a key is required.
    - 503 on OpenAlex -- it load-sheds ANONYMOUS search under load and says so,
      pointing at a free API key. Deliberate refusal, not a flake.
    - anything else (timeout, flake, a non-arXiv 400) -- transient; retry as-is."""
    low = error.lower()
    if _is_arxiv_rate_refusal(low):
        return (
            "arXiv refused the request — its rate limit arrives as HTTP 200 "
            "with a bare 'Rate exceeded' body, and it clears on its own; wait "
            "a few seconds and call q_research again, or lean on the other "
            "paper sources rather than your own knowledge"
        )
    if _is_rate_limited_status(low):
        return (
            "the source rate-limited the request — it clears on its "
            "own; wait a few seconds and retry the same query, or lean on "
            "the other sources rather than your own knowledge"
        )
    if _is_arxiv_non_feed(low):
        return _TRANSIENT_HINT
    if _is_arxiv_query_error(low):
        return (
            "the query could not be mechanically repaired — semantically re-anchor "
            "with different domain terminology before calling q_research again"
        )
    if _is_auth_error(low):
        return (
            "the API key was rejected, not the query — retrying is futile; check "
            "the source's key (OPENALEX_API_KEY / CONTEXT7_API_KEY / GH_TOKEN / "
            "HF_TOKEN) and, if it cannot be fixed here, proceed on the other "
            "sources and say so"
        )
    if _is_openalex_key_required(low):
        return (
            "OpenAlex requires a free API key for normal search — retrying the "
            "same query without one is futile; set OPENALEX_API_KEY and, until "
            "then, proceed on the other paper sources and say so"
        )
    if _is_openalex_shedding(low):
        return (
            "OpenAlex pauses ANONYMOUS search under load — this is a deliberate "
            "refusal, not a flake, so retrying the same query will not clear it; "
            "set OPENALEX_API_KEY (free) to get an uninterrupted pool, and until "
            "then lean on the other paper sources rather than your own knowledge"
        )
    return _TRANSIENT_HINT


def format_digest(digest: ResearchDigest, *, mode: str | None = None) -> str:
    """Render a digest as prompt-safe markdown. Opens with a deterministic
    `Research status:` verdict line (see `compute_status`) so the consumer meets
    the quality judgment before any result and can act on RETRY-REQUIRED
    before falling back. Failed sources appear under 'Sources unavailable' --
    each with an inline retry hint -- and a per-source count footer makes a
    queried-but-empty source visible. When omitted, `mode` is the retrieval
    mode stored on the digest; an explicit value asserts that the caller's
    expected mode matches the retrieval mode."""
    mode = digest.mode if mode is None else mode
    status = compute_status(digest, mode=mode)
    status_line = f"Research status: {status.code} — {status.detail}"
    if status.suggested_query:
        lane = f" {status.suggested_lane}" if status.suggested_lane else ""
        status_line += f' · try{lane}: "{status.suggested_query}"'
    lines: list[str] = [
        status_line,
        f"Research needs action: {str(status.needs_action).lower()}",
        "",
    ]
    if status.required_actions:
        lines.extend(
            (
                "### Required research actions",
                "| Action | Source | Lane | Query |",
                "|---|---|---|---|",
            )
        )
        for action in status.required_actions:
            query = action.query.replace("|", "\\|")
            lines.append(
                f"| {action.kind} | {_SOURCE_LABELS.get(action.source, action.source)} "
                f"| {action.lane} | {query} |"
            )
        lines.append("")
    lines.extend((f"## Prior art for: {digest.topic}", ""))
    if digest.source_statuses:
        lines.extend(
            (
                "### Source/lane status",
                "| Source | Lane | Status | Candidates | Query | Detail |",
                "|---|---|---|---:|---|---|",
            )
        )
        for row in digest.source_statuses:
            query = row.query.replace("|", "\\|")
            detail = row.detail.replace("|", "\\|")
            lines.append(
                f"| {_SOURCE_LABELS.get(row.source, row.source)} | {row.lane} | "
                f"{row.code} | {row.count} | {query} | {detail} |"
            )
        lines.append("")
    if digest.papers:
        lines.append("### Papers (arXiv + OpenAlex + Europe PMC published/preprints)")
        for p in digest.papers:
            who = ", ".join(p.authors[:3]) or "unknown"
            yr = p.year if p.year is not None else "n.d."
            link = f" <{p.url}>" if p.url else ""
            # Field annotation is skystorm signal (which field the hit sits in,
            # for the harvest/pivot judgment); kept out of the grounded render.
            tag = f" [field: {p.field}]" if mode == "exploratory" and p.field else ""
            strata = f" [strata: {', '.join(p.strata)}]" if p.strata else ""
            source = (
                f" [source: {_SOURCE_LABELS.get(p.research_source, p.research_source)}]"
                if p.research_source and p.research_source != p.source
                else ""
            )
            lanes = f" [lanes: {', '.join(p.query_lanes)}]" if p.query_lanes else ""
            if p.full_text_available is True:
                full_text = (
                    f" [full text] <{p.full_text_url}>"
                    if p.full_text_url
                    else " [full text available]"
                )
            elif p.full_text_available is False:
                full_text = " [no open full text identified by Europe PMC]"
            else:
                full_text = ""
            lines.append(
                f"- **{p.title}** ({yr}, {p.source}){tag}{strata}{source}{lanes}"
                f"{full_text} — {who}. {_truncate(p.abstract)}{link}"
            )
        lines.append("")
    if digest.libraries:
        lines.append("### Library docs (Context7)")
        for lib in digest.libraries:
            lines.append(
                f"- **{lib.name}** (trust {lib.trust_score:.1f}) — "
                f"{_truncate(lib.description)} <{lib.url}>"
            )
            for snip in lib.snippets[:_MAX_SNIPPETS]:
                lines.append(f"    - {_truncate(snip)}")
        lines.append("")
    if digest.repos:
        lines.append("### Repositories (GitHub)")
        for r in digest.repos:
            lang = f", {r.language}" if r.language else ""
            body = _truncate(r.description) if r.description else ""
            link = f" <{r.url}>" if r.url else ""
            # A broadened-query hit is marked so a reader can discount it
            # relative to a precise distinctive-term/raw-phrase hit.
            broad = " [broadened]" if r.broadened else ""
            lines.append(
                f"- **{r.name}** (★{_compact_count(r.stars)}{lang}){broad}"
                + (f" — {body}" if body else "")
                + link
            )
        lines.append("")
    if digest.models:
        lines.append("### Models (HuggingFace)")
        for m in digest.models:
            tag = f", {m.pipeline_tag}" if m.pipeline_tag else ""
            lib = f" [{m.library_name}]" if m.library_name else ""
            broad = " [broadened]" if m.broadened else ""
            link = f" <{m.url}>" if m.url else ""
            lines.append(
                f"- **{m.id}** (↓{_compact_count(m.downloads)}{tag}){lib}{broad}{link}"
            )
        lines.append("")
    if digest.errors:
        lines.append("### Sources unavailable")
        for err in digest.errors:
            lines.append(f"- {err}")
            lines.append(f"  ↳ {_retry_hint(err)}")
        lines.append("")
    excluded_collision_counts = dict(digest.excluded_collision_counts)
    if (
        not any(
            (
                digest.papers,
                digest.libraries,
                digest.repos,
                digest.models,
                digest.errors,
                digest.field_map,
            )
        )
        and not excluded_collision_counts
    ):
        lines.append("_No prior art found._")
        lines.append("")
    if digest.field_map:
        lines.append("### Field map (OpenAlex — subfields this query spans)")
        spread = " · ".join(
            f"{name} {_compact_count(n)}" for name, n in digest.field_map
        )
        lines.append(spread)
        lines.append(
            "_Presence/ranking readout, not a normalized metric (counts are "
            "corpus-sized). A field here is connected to the query by published "
            "work; one absent is not._"
        )
        lines.append("")
    for note in digest.notes:
        lines.append(f"_↻ {note}._")
    if digest.notes:
        lines.append("")
    if digest.counts:
        parts = []
        for source, retained in digest.counts:
            label = _SOURCE_LABELS.get(source, source)
            excluded = excluded_collision_counts.get(source, 0)
            if excluded:
                parts.append(
                    f"{label} {retained} shown after dedup/cap; {excluded} "
                    "lane-result candidate(s) rejected before dedup/cap"
                )
            else:
                parts.append(f"{label} {retained}")
        paper_found = sum(n for s, n in digest.counts if s in _PAPER_SOURCES)
        dropped = paper_found - len(digest.papers)
        footer = "_Sources queried: " + ", ".join(parts)
        if dropped > 0:
            footer += f" · {dropped} duplicate paper(s) dropped"
        lines.append(footer + "._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_ARXIV_URL = "https://export.arxiv.org/api/query"
# arXiv's published etiquette is one query every ~3 seconds; a rate refusal
# retried sooner just re-hits the wall (tests patch this to 0).
_RATE_PAUSE_S = 3.0
_MAX_RATE_WAIT_S = 15.0
# A non-rate transient gets one delayed retry. Keeping the delay here, rather
# than in each adapter, makes the bounded policy identical across sources.
_TRANSIENT_RETRY_DELAY_S = 1.0
# Monotonic time before which no rate-classed retry may fire. Concurrent
# refusals must not sleep the same fixed pause and wake in lockstep, which
# would recreate the burst that triggered the refusal. Each retry claims the
# next polite slot instead. Per-process best effort;
# separate processes still race.
_rate_next_free = 0.0


def _claim_rate_slot(now: float) -> float | None:
    """Return when this rate retry may fire, or None when the bounded queue is
    full. Successful claims push the next slot one polite gap further out.
    Atomic on the event loop (no await between read and write), and pure
    bookkeeping so serialization is testable without sleeping."""
    global _rate_next_free
    fire_at = max(now + _RATE_PAUSE_S, _rate_next_free)
    if fire_at > now + _MAX_RATE_WAIT_S:
        return None
    _rate_next_free = fire_at + _RATE_PAUSE_S
    return fire_at


_ATOM = "{http://www.w3.org/2005/Atom}"


async def search_arxiv(
    client: httpx.AsyncClient, query: str, *, limit: int, timeout: float
) -> list[Paper]:
    resp = await client.get(
        _ARXIV_URL,
        params={
            "search_query": f"all:{query}",
            "max_results": limit,
            "sortBy": "relevance",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    if "<feed" not in resp.text:
        # arXiv signals its rate limit as HTTP 200 with a tiny plain-text body
        # ("Rate exceeded.") -- a success status, not an error. Parsed as a
        # feed that body has no entries, which would silently read as 'no
        # papers found'; raise a loud error instead, carrying the body as
        # evidence. Only the documented refusal marker gets the rate label --
        # any other non-feed 200 (a proxy page, a block page) must NOT be
        # assumed to be a rate limit. The fixed phrases 'rate refusal:' and
        # 'non-feed body' are the classification anchors
        # (_is_arxiv_rate_refusal / _is_arxiv_non_feed).
        snippet = " ".join(resp.text.split())[:80]
        if "rate exceeded" in resp.text.lower():
            raise RuntimeError(f"arXiv rate refusal: HTTP 200 with body ({snippet!r})")
        raise RuntimeError(
            f"arXiv answered HTTP 200 with a non-feed body ({snippet!r})"
        )
    root = ET.fromstring(resp.text)
    papers: list[Paper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip()
        if not title:
            continue
        summary = (entry.findtext(f"{_ATOM}summary") or "").strip()
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        published = (entry.findtext(f"{_ATOM}published") or "").strip()
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = tuple(
            name
            for a in entry.findall(f"{_ATOM}author")
            if (name := (a.findtext(f"{_ATOM}name") or "").strip())
        )
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=year,
                source="arxiv",
                identifier=arxiv_id,
                url=raw_id,
                abstract=summary,
                research_source="arxiv",
            )
        )
    return papers


_OPENALEX_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _arxiv_id_from_doi(doi: str) -> str | None:
    """Extract the arXiv id from arXiv's canonical DOI prefix (10.48550/arXiv.<id>).
    Anchored on the full prefix so a stray 'arxiv.' in some other publisher's DOI
    suffix can't yield a bogus identifier."""
    marker = "10.48550/arxiv."
    low = doi.lower()
    if marker in low:
        return doi[low.index(marker) + len(marker) :]
    return None


_C7_BASE = "https://context7.com/api/v1"
_C7_MIN_TRUST = 5.0


async def _fetch_snippets(
    client: httpx.AsyncClient,
    lib_id: str,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Fetch one library's doc text and split it into up to _MAX_SNIPPETS
    blocks. A failed fetch yields no snippets rather than sinking the library."""
    try:
        doc = await client.get(
            f"{_C7_BASE}{lib_id}",
            params={"type": "txt", "tokens": 2000},
            headers=headers,
            timeout=timeout,
        )
        doc.raise_for_status()
    except httpx.HTTPError:
        return ()
    return tuple(s.strip() for s in doc.text.split("\n\n") if s.strip())[:_MAX_SNIPPETS]


async def fetch_context7(
    client: httpx.AsyncClient,
    query: str,
    *,
    timeout: float,
    max_libs: int = 3,
    token: str | None = None,
) -> list[LibraryDoc]:
    """Search Context7 and fetch snippets for the highest-trust libraries.

    `token`, when present, is sent only as a Bearer header on both search and
    snippet requests. Anonymous requests remain supported but can be rate-limited.
    """
    headers = {}
    token = _clean_token(token)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = await client.get(
        f"{_C7_BASE}/search",
        params={"query": query},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    # Collapse version-variants of one library (same normalized title) to a
    # single highest-trust entry, so a popular term (e.g. "pytorch", indexed
    # under many versions) doesn't fill every slot with redundant copies of
    # itself and crowd out distinct libraries. Dict insertion order preserves
    # Context7's relevance ranking across distinct names; cap after the dedup.
    best: dict[str, tuple[dict, str, float]] = {}
    for item in resp.json().get("results", []):
        try:
            trust = float(item.get("trustScore") or 0.0)
        except (TypeError, ValueError):
            trust = 0.0
        if trust < _C7_MIN_TRUST:
            continue
        lib_id = item.get("id") or item.get("libraryId") or ""
        if not lib_id:
            continue  # no id => no docs URL; unusable for grounding
        name = item.get("title") or lib_id
        key = "".join(ch for ch in name.lower() if ch.isalnum()) or lib_id.lower()
        existing = best.get(key)
        if existing is None or trust > existing[2]:
            best[key] = (item, lib_id, trust)
    selected = list(best.values())[:max_libs]  # (item, lib_id, trust) per library
    # Fetch each library's docs concurrently so total time is bounded by a
    # single request, not the sum over libraries.
    snippets = await asyncio.gather(
        *(
            _fetch_snippets(client, lib_id, timeout, headers)
            for _, lib_id, _ in selected
        )
    )
    return [
        LibraryDoc(
            name=item.get("title") or lib_id,
            description=(item.get("description") or "").strip(),
            snippets=snips,
            trust_score=trust,
            url=f"https://context7.com{lib_id}",
        )
        for (item, lib_id, trust), snips in zip(selected, snippets, strict=True)
    ]


_OPENALEX_SUBFIELD_GROUP = "primary_topic.subfield.id"
_OPENALEX_FIELD_MAP_TOP = 10


async def map_openalex_fields(
    client: httpx.AsyncClient,
    query: str,
    *,
    timeout: float,
    email: str | None = None,
    api_key: str | None = None,
) -> list[tuple[str, int]]:
    """Return the OpenAlex subfield distribution for `query` as (subfield, count)
    pairs, highest count first -- the skystorm 'field map'. One `group_by` call
    answers "which fields does this method actually span?": a method-term query
    surfaces the geoscience / bio / finance tail alongside the CS/ML mass
    (probe-verified). Subfield level, not field level -- field level is dominated
    by the CS/Engineering mass and hides the interesting tail. Counts are
    corpus-sized (big fields publish more), so this is a presence/ranking
    readout, not a normalized metric. Empty/'unknown' buckets are dropped."""
    headers = {"User-Agent": f"code-quorum (mailto:{email})"} if email else {}
    api_key = _clean_token(api_key)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = await client.get(
        _OPENALEX_URL,
        params={
            "search": query,
            "group_by": _OPENALEX_SUBFIELD_GROUP,
            "per-page": 200,
        },
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    out: list[tuple[str, int]] = []
    for group in resp.json().get("group_by", []):
        name = (group.get("key_display_name") or "").strip()
        if not name or name.lower() == "unknown":
            continue
        out.append((name, int(group.get("count") or 0)))
    return out[:_OPENALEX_FIELD_MAP_TOP]


async def search_openalex(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    timeout: float,
    email: str | None,
    api_key: str | None = None,
    purpose: str = "methods",
) -> list[Paper]:
    """Search OpenAlex works. `api_key` is required for normal use. OpenAlex
    permits only a small anonymous demo allowance before returning 409, and it
    can also load-shed anonymous search with a 503.

    The key goes in a Bearer header, never the `api_key` query param OpenAlex also
    accepts. Both authenticate (verified live -- a bogus key 401s either way), but
    httpx embeds the full URL in its HTTPStatusError text, so a key in the query
    string would land verbatim in digest.errors -- which is rendered into the
    digest and handed to the council's third-party LLMs. Same rule as GH/HF."""
    if purpose not in RESEARCH_PURPOSES:
        raise ValueError(
            f"Unknown research purpose {purpose!r}; expected one of "
            f"{', '.join(RESEARCH_PURPOSES)}."
        )
    from_date = (datetime.date.today() - datetime.timedelta(days=365 * 5)).isoformat()
    headers = {"User-Agent": f"code-quorum (mailto:{email})"} if email else {}
    api_key = _clean_token(api_key)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def search_stratum(name: str, *, recent: bool) -> list[Paper]:
        params = {
            "search": query,
            "per-page": limit,
            "sort": "relevance_score:desc",
            "select": (
                "title,publication_year,doi,authorships,"
                "abstract_inverted_index,primary_topic"
            ),
        }
        if recent:
            params["filter"] = f"from_publication_date:{from_date}"
        resp = await client.get(
            _OPENALEX_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        papers: list[Paper] = []
        for work in resp.json().get("results", []):
            title = (work.get("title") or "").strip()
            if not title:
                continue
            doi = work.get("doi") or ""
            authors = tuple(
                author_name
                for authorship in work.get("authorships", [])
                if (
                    author_name := (
                        (authorship.get("author") or {}).get("display_name") or ""
                    ).strip()
                )
            )
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
            subfield = ((work.get("primary_topic") or {}).get("subfield") or {}).get(
                "display_name"
            ) or None
            arxiv_id = _arxiv_id_from_doi(doi)
            url = (
                doi
                if doi.startswith("http")
                else (f"https://doi.org/{doi}" if doi else "")
            )
            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    year=work.get("publication_year"),
                    source="openalex",
                    identifier=arxiv_id or doi,
                    url=url,
                    abstract=abstract,
                    field=subfield,
                    research_source="openalex",
                    strata=(name,),
                )
            )
        return papers

    if purpose == "currency":
        return await search_stratum("recent", recent=True)

    all_time, recent = await asyncio.gather(
        search_stratum("all-time", recent=False),
        search_stratum("recent", recent=True),
    )
    merged: dict[str, Paper] = {}
    order: list[str] = []
    for index in range(max(len(all_time), len(recent))):
        for papers in (all_time, recent):
            if index >= len(papers):
                continue
            paper = papers[index]
            key = _normalize_identifier(paper.identifier) or "".join(
                char for char in paper.title.lower() if char.isalnum()
            )
            if key in merged:
                existing = merged[key]
                merged[key] = replace(
                    existing,
                    strata=tuple(dict.fromkeys(existing.strata + paper.strata)),
                )
                continue
            merged[key] = paper
            order.append(key)
    return [merged[key] for key in order[:limit]]


_EUROPEPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Pin the query to the preprint stratum, minus arXiv. Europe PMC indexes MEDLINE
# (SRC:MED) too -- ~20x larger for a given topic -- which would bury every
# preprint and merely re-cover ground OpenAlex already holds; and it carries
# arXiv preprints, which the `arxiv` source already returns. What is left is the
# tier nothing else in the panel reaches: bioRxiv, medRxiv, Research Square.
_EUROPEPMC_FILTER = 'SRC:PPR NOT PUBLISHER:"arXiv"'
_EUROPEPMC_PUBLISHED_FILTER = "NOT SRC:PPR"
# Anchored on a tag NAME (a letter or a closing slash right after the '<'). A bare
# `<[^>]+>` also matches an inequality span -- "p < 0.05 ... > 2-fold" -- and eats
# the text between them, which does not merely truncate the abstract: it fabricates
# a plausible, wrong sentence ("p 2-fold") and hands it to the council as evidence.
# Biology abstracts are dense with such spans (measured: 7/50 on an ordinary query).
_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
# Metacharacters that let a topic restructure the boolean tree we build below.
_EUROPEPMC_META = str.maketrans({"(": " ", ")": " ", '"': " ", ":": " "})


def _strip_html(text: str) -> str:
    """Drop the structured-abstract markup Europe PMC serves ('<h4>Motivation</h4>
    ...') while leaving inequalities intact. arXiv and OpenAlex return plain text,
    so this is Europe-PMC-specific; left in, the tags reach the council verbatim."""
    return " ".join(_TAG_RE.sub(" ", text).split())


def _europepmc_safe_query(query: str) -> str:
    """Remove syntax that could restructure the Europe PMC boolean group."""
    translated = query.translate(_EUROPEPMC_META).split()
    return " ".join(token for token in translated if token.upper() not in _BOOLEAN_OPS)


def _safe_http_url(value: str) -> str:
    """Return a prompt-safe absolute HTTP(S) URL, or an empty string."""
    value = value.strip()
    if not value.startswith(("http://", "https://")):
        return ""
    if any(char.isspace() or char in "()<>" for char in value):
        return ""
    return value


def _parse_europepmc_published_item(item: dict) -> Paper | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None
    doi = (item.get("doi") or "").strip()
    source_id = (item.get("source") or "").strip()
    ext_id = (item.get("extId") or item.get("id") or "").strip()
    authors = tuple(
        name
        for author in ((item.get("authorList") or {}).get("author") or [])
        if (name := (author.get("fullName") or "").strip())
    )
    year = item.get("pubYear")
    year_value = int(str(year)) if year is not None and str(year).isdigit() else None
    full_text_rows = (item.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    full_text_url = next(
        (
            url
            for row in full_text_rows
            if (url := _safe_http_url(row.get("url") or ""))
            and (
                (row.get("availabilityCode") or "").upper() in {"OA", "F"}
                or (row.get("availability") or "").lower() in {"open access", "free"}
            )
        ),
        "",
    )
    access_flags = [
        item.get(field) for field in ("isOpenAccess", "inEPMC") if field in item
    ]
    availability_rows = [
        row.get("availabilityCode") or row.get("availability")
        for row in full_text_rows
        if row.get("availabilityCode") or row.get("availability")
    ]
    if full_text_url or any(flag == "Y" for flag in access_flags):
        full_text_available: bool | None = True
    elif access_flags or availability_rows:
        full_text_available = False
    else:
        full_text_available = None
    mesh_terms = tuple(
        term
        for row in ((item.get("meshHeadingList") or {}).get("meshHeading") or [])
        if (term := (row.get("descriptorName") or "").strip())
    )
    if doi:
        url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    elif source_id and ext_id:
        url = f"https://europepmc.org/article/{source_id}/{ext_id}"
    else:
        url = full_text_url
    return Paper(
        title=title,
        authors=authors,
        year=year_value,
        source=(item.get("journalTitle") or "Europe PMC").strip(),
        identifier=doi or ext_id,
        abstract=_strip_html(item.get("abstractText") or ""),
        url=url,
        research_source="europepmc-published",
        full_text_available=full_text_available,
        full_text_url=full_text_url,
        mesh_terms=mesh_terms,
    )


async def search_europepmc_published(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    timeout: float,
    expand_synonyms: bool = False,
) -> list[Paper]:
    """Search Europe PMC's published literature lane.

    The lane excludes preprints but otherwise uses the full Europe PMC corpus.
    `core` results carry abstracts, MeSH headings, and full-text availability.
    When requested, synonym expansion only backfills an exact search that did
    not fill the result limit. Exact candidates keep priority; expansion-only
    candidates are labelled so lexical status scoring does not mistake changed
    vocabulary for a query collision.
    """
    if not query.strip():
        return []
    safe_query = _europepmc_safe_query(query)
    if not safe_query:
        return []

    async def fetch(*, synonyms: bool) -> list[Paper]:
        resp = await client.get(
            _EUROPEPMC_URL,
            params={
                "query": f"({safe_query}) {_EUROPEPMC_PUBLISHED_FILTER}",
                "format": "json",
                "pageSize": limit * 2 if synonyms else limit,
                "resultType": "core",
                "synonym": str(synonyms).lower(),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        result = (resp.json().get("resultList") or {}).get("result") or []
        return [
            paper
            for item in result
            if (paper := _parse_europepmc_published_item(item)) is not None
        ]

    papers = _dedup_papers(await fetch(synonyms=False))[:limit]
    if not expand_synonyms or len(papers) >= limit:
        return papers
    for paper in await fetch(synonyms=True):
        deduped = _dedup_papers([*papers, paper])
        if len(deduped) == len(papers):
            papers = deduped
            continue
        papers.append(
            replace(
                paper,
                strata=tuple(dict.fromkeys(paper.strata + ("synonym-expanded",))),
            )
        )
        if len(papers) >= limit:
            break
    return papers


async def search_europepmc_preprints(
    client: httpx.AsyncClient, query: str, *, limit: int, timeout: float
) -> list[Paper]:
    """Search the bioRxiv/medRxiv preprint tier via Europe PMC.

    bioRxiv's own API cannot do this: it serves date-range and DOI lookups only,
    and silently ignores a search parameter (verified -- an unknown `?search=`
    returns a byte-identical payload), so it can never answer "papers about X".
    Europe PMC indexes those same preprints and does keyword search over them.

    Unlike arXiv, Europe PMC never rejects a query -- boolean metacharacters and
    an unbalanced paren all return 200 -- so there is no 400 to react to. Both
    hazards therefore run silently. (a) A blank query matches the ENTIRE corpus
    (~1.2M hits), so an empty topic would return arbitrary popular papers dressed
    as prior art; refuse it rather than search on it. (b) A topic carrying `)` or
    a quote can close our grouping early and rewrite the boolean tree around the
    pin -- `foo) OR (bar` becomes `(foo) OR (bar) AND <pin>`, whose meaning then
    rests on the server's operator precedence rather than on us. Europe PMC today
    associates left-to-right, so the pin does still bind (verified live: no
    crafted topic leaked a MEDLINE record) -- but that is a property of their
    parser, not a guarantee. Strip the metacharacters so correctness does not
    depend on it, and parenthesize the pin as its own group."""
    if not query.strip():
        return []
    safe_query = _europepmc_safe_query(query)
    if not safe_query:
        return []
    resp = await client.get(
        _EUROPEPMC_URL,
        params={
            "query": f"({safe_query}) AND ({_EUROPEPMC_FILTER})",
            "format": "json",
            "pageSize": limit,
            "resultType": "core",  # 'core' is what carries abstractText
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    # `.get(k, default)` returns None when the key EXISTS with a null value, so a
    # null resultList/author would crash the source rather than degrade to empty.
    result = (resp.json().get("resultList") or {}).get("result") or []
    papers: list[Paper] = []
    for item in result:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        doi = (item.get("doi") or "").strip()
        authors = tuple(
            name
            for a in ((item.get("authorList") or {}).get("author") or [])
            if (name := (a.get("fullName") or "").strip())
        )
        year = item.get("pubYear")
        # The preprint server ("bioRxiv"), not the backend that found it.
        publisher = (
            (item.get("bookOrReportDetails") or {}).get("publisher")
            or item.get("publisher")
            or "europepmc-preprints"
        ).strip()
        # Prefer the DOI: it resolves to the preprint's own page. Fall back to the
        # Europe PMC record when a preprint has not been assigned one yet.
        pmc_id = (item.get("id") or "").strip()
        if doi:
            # already-absolute guard, same as search_openalex
            url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        else:
            url = f"https://europepmc.org/article/PPR/{pmc_id}" if pmc_id else ""
        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=int(year) if str(year).isdigit() else None,
                source=publisher,
                identifier=doi or pmc_id,
                abstract=_strip_html(item.get("abstractText") or ""),
                url=url,
                research_source="europepmc-preprints",
            )
        )
    return papers


_MAX_DISTINCTIVE_TERMS = 4
# Punctuation stripped only from a token's EDGES (wrapping quotes/parens/commas,
# trailing ?/!, markdown backticks/asterisks from a pasted prompt); internal
# separators (the - . / _ in IP-Adapter, FLUX.2) are what make a term distinctive,
# so they must survive.
_EDGE_PUNCT = "\"'`*(),.;:!?[]{}"


def _distinctive_terms(
    query: str, *, max_terms: int = _MAX_DISTINCTIVE_TERMS
) -> list[str]:
    """Pull the artifact-name terms out of a free-form topic -- the tokens the
    keyword backends can actually match. GitHub repo-search ANDs space-joined
    terms (over name/description/topics) and HuggingFace `search` substring-matches
    a model id, so a combined multi-word phrase ("PuLID InstantID identity
    preserving adapter") matches nothing on either; the distinctive names within
    it (PuLID, InstantID, IP-Adapter, FLUX.2) each match plenty. A term is
    distinctive when it carries a digit, an internal separator (-_./), or
    internal/non-Title capitalization -- the shape of a proper artifact name, not
    a dictionary word. Returns at most `max_terms`, first-seen order, deduped
    case-insensitively. Empty when the topic has none: the caller then falls back
    to the raw phrase, so concept topics (better served by the paper backends) are
    unaffected."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in query.split():
        term = raw.strip(_EDGE_PUNCT)
        # Drop only the empty string and boolean operators here -- length is NOT a
        # filter (it would lose real 2-char names like T5/S3); the distinctiveness
        # predicates below already reject short dictionary words.
        if not term or term.upper() in _BOOLEAN_OPS:
            continue
        # A pure number / punctuation token (a bare year "2024", "3.5") is generic;
        # require at least one letter so it falls to the raw-phrase fallback.
        if not any(c.isalpha() for c in term):
            continue
        has_digit = any(c.isdigit() for c in term)
        has_sep = any(c in "-_./" for c in term)
        # A trailing plural -s on an otherwise all-caps acronym (LLMs, GPUs, APIs)
        # would dodge the all-caps gate below; gate on the singular stem.
        stem = term[:-1] if term[-1:] == "s" and term[:-1].isupper() else term
        internal_caps = stem != stem.lower() and stem != stem.title()
        # A bare all-caps acronym (GPU, API, GAN) is too generic to be a useful
        # search term and would let its high-star repos bury the real artifact;
        # keep all-caps only at length >= 4 (SDXL, CLIP). Mixed-case names (PuLID,
        # LoRA) and anything with a digit/separator are always distinctive.
        caps_ok = internal_caps and (len(stem) >= 4 or not stem.isupper())
        if not (has_digit or has_sep or caps_ok):
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def _gh_quote(term: str) -> str:
    """Quote a GitHub query term unless it is purely alphanumeric. GitHub repo
    search treats `-` as the exclusion prefix, so an unquoted `IP-Adapter` parses
    as `IP NOT Adapter` (verified live: it drags in EtherNet/IP repos); quoting
    forces an exact-token match. Plain alphanumerics (PuLID) are a no-op when
    quoted, so only non-alnum terms are wrapped to keep queries readable."""
    return term if term.isalnum() else f'"{term}"'


_GITHUB_URL = "https://api.github.com/search/repositories"
_HF_MODELS_URL = "https://huggingface.co/api/models"


def _is_rate_error(exc: httpx.HTTPError) -> bool:
    """Whether `exc` is a 429 refusal -- shared by the union fan-outs below so
    a fan-out stops issuing further sub-queries the moment one comes back
    rate-limited, rather than burning the rest of its budget on requests that
    would hit the same wall (see `_is_rate_limited_status`, the string-based
    twin used on the aggregated error text `_contained` sees)."""
    return _is_rate_limited_status(str(exc).lower())


_T = TypeVar("_T")


async def _search_union(
    client: httpx.AsyncClient,
    sub_queries: list[str],
    *,
    url: str,
    params: Callable[[str], dict],
    items_of: Callable[[httpx.Response], Iterable[dict]],
    parse_item: Callable[[dict, bool], tuple[str, _T] | None],
    timeout: float,
    headers: dict[str, str],
    broadened: bool = False,
) -> tuple[dict[str, _T], list[httpx.HTTPError]]:
    """Run each sub-query against `url` and union the hits by the key
    `parse_item` returns (full_name for GitHub, id for HuggingFace). Isolates
    per-sub-query failures: one term 500-ing or timing out must not discard
    the others' hits -- the caller decides whether to raise from the returned
    errors. Stops early on a 429: the rest of the sub-queries would just hit
    the same rate wall, and burning them anyway is what turns one escalation
    into a double-digit request spike when `_contained`'s outer retry re-runs
    the whole fan-out afterward. `broadened` is passed through to `parse_item`
    so it can tag every resulting item (set by the caller for an escalation
    call, never for the precise query)."""
    found: dict[str, _T] = {}
    errors: list[httpx.HTTPError] = []
    for sub in sub_queries:
        try:
            resp = await client.get(
                url, params=params(sub), headers=headers, timeout=timeout
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            errors.append(exc)
            if _is_rate_error(exc):
                break
            continue
        for item in items_of(resp):
            parsed = parse_item(item, broadened)
            if parsed is None:
                continue
            key, obj = parsed
            if key in found:
                continue
            found[key] = obj
    return found, errors


async def _escalate_union(
    query: str,
    distinctive: list[str],
    sub_queries: list[str],
    found: dict[str, _T],
    errors: list[httpx.HTTPError],
    run_union: Callable[..., Awaitable[tuple[dict[str, _T], list[httpx.HTTPError]]]],
    *,
    quote: Callable[[str], str],
    match_text: Callable[[_T], str],
    popularity: Callable[[_T], float],
    limit: int,
) -> list[_T]:
    """Shared escalation for `search_github`/`search_huggingface`. A topic
    with no distinctive terms falls back to a single raw-phrase request --
    good precision when it hits, but a plain-English topic reliably matches 0
    on both backends, and a genuine 0 there would read as "no prior art
    exists" -- a false negative, not a finding. So when the raw phrase draws a
    clean (error-free) 0, escalate once to a per-content-word union (via
    `_scorable_terms`, capped like the distinctive-term union). Escalated hits
    are ranked by how many distinct escalation terms they actually match,
    THEN by popularity -- a broad single-word match must not outrank a hit
    that satisfies several words just because it is popular -- and are tagged
    `broadened=True` so the renderer can mark them for a reader to discount
    relative to a precise hit. Skipped when the escalation would just
    resubmit the identical query (a single generic-word topic) -- that is
    guaranteed to return the same zero and would only burn a call from the
    source's scarce rate-limit budget."""
    escalation_terms: list[str] = []
    if not distinctive and not found and not errors:
        # Raw phrase matched nothing and every request succeeded (a genuine
        # failure is left alone below, not masked as an empty union).
        candidate_terms = _scorable_terms(query)[:_MAX_DISTINCTIVE_TERMS]
        candidate_queries = [quote(t) for t in candidate_terms]
        if candidate_queries and {q.lower() for q in candidate_queries} != {
            q.lower() for q in sub_queries
        }:
            escalation_terms = candidate_terms
            found, errors = await run_union(candidate_queries, broadened=True)
    # Per-sub-query errors are isolated inside the union call above and only
    # surface here when the final union produced nothing to show for them.
    if errors and not found:
        raise errors[0]
    if escalation_terms:
        return sorted(
            found.values(),
            key=lambda x: (
                -_term_match_count(escalation_terms, match_text(x)),
                -popularity(x),
            ),
        )[:limit]
    return sorted(found.values(), key=popularity, reverse=True)[:limit]


def _gh_parse(item: dict, broadened: bool) -> tuple[str, Repo] | None:
    name = (item.get("full_name") or "").strip()
    if not name:
        return None
    return name, Repo(
        name=name,
        description=(item.get("description") or "").strip(),
        stars=int(item.get("stargazers_count") or 0),
        language=item.get("language") or None,
        url=item.get("html_url") or "",
        broadened=broadened,
    )


async def search_github(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    timeout: float,
    token: str | None,
) -> list[Repo]:
    """Search public GitHub repos by stars. `token` (if set) is sent only as a
    Bearer header to raise the rate limit -- never logged, never in the URL or
    errors -- and the search is pinned to public repos so a private-repo token
    can only lift the rate limit, never widen what is returned. `limit` is
    both the per-sub-query page size and the final cap, so the candidate pool
    a multi-term fan-out draws from scales with the number of sub-queries.

    GitHub repo-search ANDs space-joined terms, so a combined multi-artifact
    topic matches nothing. We query each distinctive term on its own (quoted
    to defuse the `-` exclusion footgun) and union the results, deduped by
    full_name and re-sorted by stars. See `_escalate_union` for the raw-phrase
    fallback and zero-result escalation."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "code-quorum"}
    token = _clean_token(token)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    terms = _distinctive_terms(query)
    sub_queries = [_gh_quote(t) for t in terms] if terms else [query]
    # Pinned to `is:public` here (the single request boundary -- never remove
    # this without another public-only guard replacing it: an authenticated
    # token would otherwise widen results to private repos, whose
    # names/descriptions would leak into the digest and on to the council's
    # third-party LLMs).
    run_union = functools.partial(
        _search_union,
        client,
        url=_GITHUB_URL,
        params=lambda sub: {
            "q": f"{sub} is:public",
            "sort": "stars",
            "order": "desc",
            "per_page": limit,
        },
        items_of=lambda r: r.json().get("items", []),
        parse_item=_gh_parse,
        timeout=timeout,
        headers=headers,
    )
    found, errors = await run_union(sub_queries)
    return await _escalate_union(
        query,
        terms,
        sub_queries,
        found,
        errors,
        run_union,
        quote=_gh_quote,
        match_text=lambda r: f"{r.name} {r.description}",
        popularity=lambda r: r.stars,
        limit=limit,
    )


def _hf_parse(item: dict, broadened: bool) -> tuple[str, HFModel] | None:
    if not isinstance(item, dict):
        return None
    # Exclude private models: an authenticated token can read the user's own
    # private models, whose id/metadata would otherwise leak into the digest
    # (and on to the council's third-party LLMs). Prior-art research wants
    # public models anyway. (gated != private -- gated stays public.)
    if item.get("private"):
        return None
    mid = (item.get("id") or item.get("modelId") or "").strip()
    if not mid:
        return None
    return mid, HFModel(
        id=mid,
        downloads=int(item.get("downloads") or 0),
        likes=int(item.get("likes") or 0),
        pipeline_tag=item.get("pipeline_tag") or None,
        library_name=item.get("library_name") or None,
        url=f"https://huggingface.co/{mid}",
        broadened=broadened,
    )


async def search_huggingface(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    timeout: float,
    token: str | None,
) -> list[HFModel]:
    """Search public HuggingFace models by downloads. `token` (if set) is sent
    only as a Bearer header to raise the rate limit -- never logged, never in the
    URL or errors -- and private models the token can read are filtered out, so a
    private-repo token can only lift the rate limit, never widen what is returned.
    `limit` is both the per-term page size and the final cap, so the candidate
    pool a multi-term fan-out draws from scales with the number of terms.

    HF `search` substring-matches a model id, so a combined multi-artifact
    topic matches nothing. We query each distinctive term on its own (no
    quoting -- it is a param value, not a query language) and union the
    results, deduped by id and re-sorted by downloads. See `_escalate_union`
    for the raw-phrase fallback and zero-result escalation."""
    headers = {}
    token = _clean_token(token)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    distinctive = _distinctive_terms(query)
    terms = distinctive or [query]
    run_union = functools.partial(
        _search_union,
        client,
        url=_HF_MODELS_URL,
        params=lambda term: {
            "search": term,
            "sort": "downloads",
            "direction": -1,
            "limit": limit,
        },
        items_of=lambda r: r.json(),
        parse_item=_hf_parse,
        timeout=timeout,
        headers=headers,
    )
    found, errors = await run_union(terms)
    return await _escalate_union(
        query,
        distinctive,
        terms,
        found,
        errors,
        run_union,
        quote=lambda t: t,
        match_text=lambda m: m.id,
        popularity=lambda m: m.downloads,
        limit=limit,
    )


# Query-quality scoring (Layer 3). Stopwords cover articles/prepositions PLUS
# the generic single-token words that collide with unrelated work when used as a
# whole query (author surnames, "A Survey of ..." titles, stray docs). A term is
# scorable -- counts toward on-topic matching and, alone, is too generic to be a
# valid query -- only when it survives this set.
_STOPWORDS = frozenset(
    """
    a an the of for and or to in on at with by from as is are be using via over
    """.split()
    # generic content tokens that keyword-match everything
    + """
    data model models network networks signal signals system systems learning
    agent agents method methods approach analysis survey framework algorithm
    algorithms
    """.split()
)


def _iter_tokens(text: str):
    """Yield lowercased tokens (length >= 2) in order. A hyphen or slash is a
    token boundary, not part of the word, so `flow-matching` and
    `optimal/transport` tokenize the same as their spaced forms -- otherwise an
    orthographic variant reads as off-topic. A bare single-character token is
    dropped because a lone `x` matches almost any text and would inflate the
    on-topic score.

    A compound whose fragments are all >= 2 chars splits normally. A compound
    with a single-letter fragment (`Q-learning`, `X-ray`) is a special case:
    splitting drops the letter, and if the remainder is a stopword (`learning`)
    the whole topic vanishes and the pre-flight lint refuses a valid term
    (round-2 review). Such a compound is yielded JOINED into one canonical token
    (`qlearning`) rather than as a bare letter -- the bare `q` false-matched
    unrelated single-letter compounds like `Q-factor` (round-3 review); the
    joined form matches only the same compound. (Trade-off: the joined token no
    longer matches the fully-spaced `Q learning`, but that is rare and far better
    than a false positive.)"""
    for word in text.split():
        parts = [
            p.strip(_EDGE_PUNCT).lower()
            for p in word.replace("-", " ").replace("/", " ").split()
        ]
        parts = [p for p in parts if p]
        if not parts:
            continue
        if len(parts) > 1 and any(len(p) < 2 for p in parts):
            joined = "".join(parts)
            if len(joined) >= 2:
                yield joined
        else:
            for tok in parts:
                if len(tok) >= 2:
                    yield tok


def _match_tokens(text: str) -> set[str]:
    """Token set of `text` for lexical overlap tests (order irrelevant here)."""
    return set(_iter_tokens(text))


def _scorable_terms(query: str) -> list[str]:
    """Query tokens that carry topical weight -- every non-stopword, non-boolean
    token of length >= 2, deduped in FIRST-SEEN order (so `_suggest_query`'s cap
    is deterministic). Broader than `_distinctive_terms` (which pulls only
    artifact-name tokens): scoring needs the ordinary domain words too (`flow`,
    `matching`), since an off-topic hit shares few of them while an on-topic one
    shares several (`_required_matches` sets the bar)."""
    out: list[str] = []
    seen: set[str] = set()
    for term in _iter_tokens(query):
        if term in _STOPWORDS or term.upper() in _BOOLEAN_OPS or term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def _term_match_count(terms: Iterable[str], text: str) -> int:
    """How many DISTINCT `terms` appear in `text` (token-set overlap, same
    normalization as `_match_tokens`). Used to rank a broadened per-word-union
    escalation: a hit that satisfies several escalation terms is stronger
    evidence of relevance than one that only grazed a single broad word, no
    matter how popular that hit is -- so this is the PRIMARY sort key for
    escalated results, stars/downloads only breaking ties. `terms` are
    expected already-lowercase (as `_scorable_terms` returns them)."""
    matched = _match_tokens(text)
    return sum(1 for t in terms if t.lower() in matched)


_ON_TOPIC_COVERAGE = 2, 5  # a hit must carry 2/5 of the query's scorable terms


def _required_matches(term_count: int) -> int:
    """How many DISTINCT query terms one hit must carry to count as on-topic:
    `_ON_TOPIC_COVERAGE` of the query's scorable terms, floored at 2 and capped
    at the term count. The floor is the point of the rule -- one shared word is
    no evidence of topic when that word is ordinary ("control", "design",
    "validation" put LiDAR and quantum-Monte-Carlo hits above the OK bar for a
    classical-statistics query). The cap keeps a one-term query scoreable.
    Consequence at two terms: both are required, so a two-word query is scored
    as a phrase -- deliberate, since half of a two-word query IS one word."""
    numerator, denominator = _ON_TOPIC_COVERAGE
    ceil_share = (term_count * numerator + denominator - 1) // denominator
    return min(max(2, ceil_share), term_count)


def _on_topic_fraction(texts: Iterable[str], query: str) -> float | None:
    """Fraction of `texts` carrying enough of `query`'s scorable terms to count
    as on-topic (see `_required_matches`) -- a deterministic lexical proxy for
    "did this query hit its domain, or collide with unrelated work?". Token-set
    overlap (not substring), so `flow` does not match `flower`, but hyphen/slash
    variants DO match (see `_match_tokens`).
    Returns None when the query has no scorable terms or `texts` is empty
    (nothing to score); the caller decides what an unscoreable result means. Does
    NOT gate on hit count -- that (the min-hits floor before a low score is
    actionable) is the verdict layer's call, kept out so this stays a pure
    measurement."""
    terms = set(_scorable_terms(query))
    texts = list(texts)
    if not terms or not texts:
        return None
    required = _required_matches(len(terms))
    return sum(
        1 for text in texts if len(_match_tokens(text) & terms) >= required
    ) / len(texts)


def _candidate_is_on_topic(
    candidate: Paper | LibraryDoc | Repo | HFModel, query: str
) -> bool:
    """Apply the lane's coverage rule to one candidate before grounded filtering."""
    terms = set(_scorable_terms(query))
    if not terms:
        return False
    if isinstance(candidate, Paper):
        if "synonym-expanded" in candidate.strata or not candidate.abstract.strip():
            return True
        text = f"{candidate.title} {candidate.abstract}"
    elif isinstance(candidate, LibraryDoc):
        text = " ".join((candidate.name, candidate.description, *candidate.snippets))
    elif isinstance(candidate, Repo):
        text = f"{candidate.name} {candidate.description}"
    else:
        text = candidate.id
    return len(_match_tokens(text) & terms) >= _required_matches(len(terms))


_ON_TOPIC_MIN = 0.5  # below this fraction of on-topic hits, a query is suspect
_MIN_SCORING_HITS = 3  # too few hits to judge overlap; don't flag on 1-2 papers


def _scoreable_texts(papers: Iterable[Paper], query: str) -> list[str]:
    """The texts `compute_status` may judge a query by. A paper with an abstract
    contributes title+abstract and is scored either way. A paper whose source
    returned no abstract (OpenAlex omits `abstract_inverted_index` on some
    records) is treated asymmetrically: its title counts when it CLEARS the
    coverage bar, and is dropped when it does not.

    The asymmetry is the point. A bare title carries far less text than an
    abstract to find a fixed number of query terms in, so a title that clears
    the bar anyway is strong evidence the paper is on-topic, while a title that
    misses is no evidence either way -- there was never enough text to judge it
    on. Scoring the misses as off-topic reads a metadata gap as a collision and
    fires RETRY on a good query. Dropping the hits throws away a title that
    already proved itself. Dropping the unjudgeable ones from the count as well
    means a mostly-abstract-less digest falls under `_MIN_SCORING_HITS` and is
    left unjudged rather than judged on too little text.

    Bound on the asymmetry: admitting a bare title can only move the fraction
    toward the verdict that scoring every text would already have produced --
    the same tokens score the same whether they arrive as a title or inside an
    abstract. So this cannot invent a false OK that the pre-existing scorer did
    not also give. An off-topic paper whose title happens to carry the query's
    vocabulary ("Positive and negative control design for LiDAR calibration"
    against a design-of-experiments query) does read as on-topic here -- that
    is the standing limit of lexical scoring, not a defect of the asymmetry,
    and the design accepts it rather than reaching for a semantic model."""
    terms = set(_scorable_terms(query))
    required = _required_matches(len(terms)) if terms else 0
    out: list[str] = []
    for paper in papers:
        if paper.abstract.strip():
            out.append(f"{paper.title} {paper.abstract}")
        elif terms and len(_match_tokens(paper.title) & terms) >= required:
            out.append(paper.title)
    return out


def _scoreable_artifact_texts(
    items: Iterable[LibraryDoc | Repo], query: str
) -> list[str]:
    """Keep artifact metadata that can support a lexical verdict.

    Descriptions and documentation snippets provide enough context to score.
    A bare name is retained only when the name itself already clears the query
    coverage bar; otherwise missing metadata degrades to THIN, not collision.
    """
    terms = set(_scorable_terms(query))
    required = _required_matches(len(terms)) if terms else 0
    texts: list[str] = []
    for item in items:
        if isinstance(item, LibraryDoc):
            supporting = " ".join((item.description, *item.snippets)).strip()
            name = item.name
        else:
            supporting = item.description.strip()
            name = item.name
        if supporting:
            texts.append(f"{name} {supporting}")
        elif terms and len(_match_tokens(name) & terms) >= required:
            texts.append(name)
    return texts


@dataclass(frozen=True)
class ResearchAction:
    """One exact source/lane operation required before research may proceed."""

    kind: str
    source: str
    lane: str
    query: str


_REANCHOR_QUERY = "<supply different domain-specific terms>"


def _is_reanchor_placeholder(query: str) -> bool:
    def words(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())

    return words(_REANCHOR_QUERY) in words(query)


@dataclass(frozen=True)
class ResearchStatus:
    """The deterministic quality verdict rendered as the digest's first line.
    `code` is one of OK / DEGRADED / RETRY-REQUIRED / CONFIG / LOW-OVERLAP;
    `detail` is a one-line human reason. `required_actions` preserves exact
    source/lane/query work instead of asking the caller to infer it from rows."""

    code: str
    detail: str
    suggested_query: str = ""
    suggested_lane: str = ""
    required_actions: tuple[ResearchAction, ...] = ()

    def __post_init__(self) -> None:
        if self.code == "RETRY-REQUIRED" and not self.required_actions:
            raise ValueError("RETRY-REQUIRED status must include required_actions")

    @property
    def needs_action(self) -> bool:
        return self.code == "RETRY-REQUIRED"


def _required_actions(rows: Iterable[SourceLaneStatus]) -> tuple[ResearchAction, ...]:
    actions: list[ResearchAction] = []
    for row in rows:
        if row.code == "INFRASTRUCTURE":
            actions.append(
                ResearchAction("RETRY-SAME", row.source, row.lane, row.query)
            )
            continue
        if row.code != "QUERY-COLLISION":
            continue
        actions.append(
            ResearchAction("RE-ANCHOR", row.source, row.lane, _REANCHOR_QUERY)
        )
    return tuple(actions)


def _query_action(source: str, lane: str, query: str) -> ResearchAction:
    reworked = _rework(source, query)
    return ResearchAction(
        "RETRY-SHORTENED" if reworked else "RE-ANCHOR",
        source,
        lane,
        reworked or _REANCHOR_QUERY,
    )


def _query_actions(rows: Iterable[SourceLaneStatus]) -> tuple[ResearchAction, ...]:
    actions: list[ResearchAction] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.source, row.lane, row.query)
        if key in seen:
            continue
        seen.add(key)
        if row.code == "QUERY-COLLISION":
            actions.append(
                ResearchAction("RE-ANCHOR", row.source, row.lane, _REANCHOR_QUERY)
            )
        else:
            actions.append(_query_action(row.source, row.lane, row.query))
    return tuple(actions)


def _source_lane_status(
    source: str,
    lane: str,
    query: str,
    value: list,
    error: str | None,
) -> SourceLaneStatus:
    """Classify one source/query-lane attempt without consulting peer sources."""
    if error is not None:
        error_class = _error_class(error)
        if error_class == "config":
            code = "CONFIG"
            detail = "credential or source configuration rejected the request"
        elif error_class == "query":
            code = "QUERY-COLLISION"
            detail = (
                "query could not be mechanically repaired; use a semantic re-anchor"
            )
        else:
            code = "INFRASTRUCTURE"
            detail = "source failed after its bounded retry"
        return SourceLaneStatus(source, lane, query, code, detail, 0)

    count = len(value)
    if count < _MIN_SCORING_HITS:
        return SourceLaneStatus(
            source,
            lane,
            query,
            "THIN",
            f"only {count} candidate(s); too few for a relevance verdict",
            count,
        )
    if value and isinstance(value[0], HFModel):
        return SourceLaneStatus(
            source,
            lane,
            query,
            "THIN",
            "model identifiers lack enough descriptive text for lexical grading",
            count,
        )
    exact_papers: list[Paper] | None = None
    if value and isinstance(value[0], Paper):
        exact_papers = [
            paper for paper in value if "synonym-expanded" not in paper.strata
        ]
        texts = _scoreable_texts(exact_papers, query)
    elif value and isinstance(value[0], LibraryDoc):
        texts = _scoreable_artifact_texts(value, query)
    elif value and isinstance(value[0], Repo):
        texts = _scoreable_artifact_texts(value, query)
    else:
        texts = []
    if len(texts) < _MIN_SCORING_HITS:
        if (
            exact_papers is not None
            and len(exact_papers) < len(value)
            and len(exact_papers) < _MIN_SCORING_HITS
        ):
            detail = (
                f"only {len(exact_papers)} exact-match candidate(s); the rest "
                "are synonym-expanded"
            )
        else:
            detail = f"only {len(texts)} candidate(s) carry scoreable text"
        return SourceLaneStatus(
            source,
            lane,
            query,
            "THIN",
            detail,
            count,
        )
    fraction = _on_topic_fraction(texts, query)
    if (
        fraction is not None
        and len(texts) >= _MIN_SCORING_HITS
        and fraction < _ON_TOPIC_MIN
    ):
        return SourceLaneStatus(
            source,
            lane,
            query,
            "QUERY-COLLISION",
            f"only {round(fraction * 100)}% of candidates share lane vocabulary",
            count,
        )
    return SourceLaneStatus(
        source,
        lane,
        query,
        "ON-TOPIC",
        "candidate vocabulary matches the query lane",
        count,
    )


def _error_class(error: str) -> str:
    """Classify one source error string as 'rate' (arXiv's 200-with-'Rate
    exceeded' refusal, or a 429 from any source -- clears on its own, retry
    the same query after a pause), 'config' (key rejected / OpenAlex
    load-shedding anonymous search -- retrying is futile), 'query' (arXiv 400
    -- the query is malformed), or 'transient' (timeout/flake, or any other
    4xx/5xx -- worth one bounded retry). Shares the phrase predicates with
    `_retry_hint` so the two never drift, and keys on status phrases so a
    status-like number in the query cannot flip the class.
    'rate' and the non-feed case are checked first because their error strings
    embed a server-controlled body snippet (see _is_arxiv_rate_refusal)."""
    low = error.lower()
    if _is_arxiv_rate_refusal(low) or _is_rate_limited_status(low):
        return "rate"
    if _is_arxiv_non_feed(low):
        return "transient"
    if _is_config_error(low):
        return "config"
    if _is_arxiv_query_error(low):
        return "query"
    return "transient"


_REWORK_FILLERS = frozenset({"compare", "overview", "survey", "towards", "using"})


def _rework(source: str, query: str) -> str:
    """A resubmittable reworked query, or "" when reworking would just return the
    original. The skill MANDATES retrying with a non-empty suggestion.
    Returning the identical query would loop forever, so an empty result tells
    the caller to re-anchor with its own judgment instead. INVARIANT: the
    comparison folds case and whitespace. A case- or spacing-only difference
    ('Diffusion' -> 'diffusion', 'a  b' -> 'a b') is not a meaningful rework and
    must not cost a retry (round-2/3 review); comparing the raw strings would
    loop forever on such a query."""
    # Paper searches need enough domain vocabulary to remain anchored, while
    # artifact-like terms retain their spelling so name-indexed backends can
    # still find them. Aggregate paper recovery uses the same compact shape.
    paper_rework = source in _PAPER_SOURCES or source == "All sources"
    scorable = _scorable_terms(query)
    if paper_rework:
        if len(scorable) < 2:
            return ""
        distinctive = _distinctive_terms(query)
        terms: list[str] = list(distinctive)
        covered: set[str] = set()
        for term in distinctive:
            covered.update(_match_tokens(term))
        if len(distinctive) < 2:
            for term in scorable:
                if term not in covered and term not in _REWORK_FILLERS:
                    terms.append(term)
                    covered.add(term)
        shorter = " ".join(terms[:3])
        if len(_scorable_terms(shorter)) < 2:
            return ""
    else:
        terms = _distinctive_terms(query) or scorable
        shorter = " ".join(terms[:_MAX_DISTINCTIVE_TERMS])

    def canonical(q: str) -> str:
        return " ".join(q.lower().split())

    return shorter if shorter and canonical(shorter) != canonical(query) else ""


def compute_status(
    digest: ResearchDigest, *, mode: str | None = None
) -> ResearchStatus:
    """Derive the quality verdict from a completed digest. `mode` is 'grounded'
    (brainstorm: a low on-topic score is a defect -> RETRY-REQUIRED) or
    'exploratory' (skystorm: a low score is reported as LOW-OVERLAP but never
    forces a retry, because a deliberate cross-domain probe is expected to read
    off-domain). When supplied, `mode` asserts the digest's retrieval mode; it
    does not reinterpret retained or filtered candidates.

    Precedence: a CONFIG error (rejected key / anonymous load-shed) outranks
    everything -- a retry cannot fix it, so it must not be mistaken for a bad
    query. Then a paper stratum that produced NO evidence -- whether every paper
    source errored or every one returned 0 -- is RETRY-REQUIRED,
    never OK. Then low lexical overlap. Otherwise OK. The zero-legitimacy cases
    Source mismatches never trip a retry when a peer source found on-topic work;
    they are explicit rows rather than hidden in the aggregate."""
    if digest.mode not in ("grounded", "exploratory"):
        raise ValueError(
            f"Unknown research mode {digest.mode!r}; expected 'grounded' or "
            "'exploratory'."
        )
    if mode is not None and mode != digest.mode:
        raise ValueError(
            f"Explicit mode {mode!r} does not match digest mode {digest.mode!r}."
        )
    mode = digest.mode
    if digest.source_statuses:
        counts: dict[str, int] = {}
        for row in digest.source_statuses:
            counts[row.code] = counts.get(row.code, 0) + 1
        summary = ", ".join(
            f"{count} {code.lower()}" for code, count in sorted(counts.items())
        )
        if counts.get("CONFIG"):
            return ResearchStatus(
                "CONFIG",
                summary
                + "; retry will not fix the rejected credential or access policy -- "
                "set the source key or proceed on the other sources and say so",
            )
        collisions = tuple(
            row for row in digest.source_statuses if row.code == "QUERY-COLLISION"
        )
        paper_rows = [
            row for row in digest.source_statuses if row.source in _PAPER_SOURCES
        ]
        if paper_rows and all(row.count == 0 for row in paper_rows):
            if any(row.code == "INFRASTRUCTURE" for row in paper_rows):
                return ResearchStatus(
                    "RETRY-REQUIRED",
                    summary
                    + "; the paper stratum produced no evidence and at least one "
                    "paper source failed -- retry each failed source/lane without "
                    "changing terms",
                    required_actions=_required_actions(paper_rows),
                )
            actions = _query_actions(paper_rows)
            suggested_action = next(
                (action for action in actions if action.kind == "RETRY-SHORTENED"),
                None,
            )
            return ResearchStatus(
                "RETRY-REQUIRED",
                summary + "; the paper stratum produced no evidence",
                suggested_query=suggested_action.query if suggested_action else "",
                suggested_lane=suggested_action.lane if suggested_action else "",
                required_actions=actions,
            )
        on_topic_rows = [
            row for row in digest.source_statuses if row.code == "ON-TOPIC"
        ]
        usable_on_topic = (
            any(row.code == "ON-TOPIC" for row in paper_rows)
            if paper_rows
            else bool(on_topic_rows)
        )
        if (
            collisions
            and all(row.count > 0 for row in collisions)
            and usable_on_topic
            and mode == "grounded"
        ):
            return ResearchStatus(
                "DEGRADED",
                summary + "; rejected colliding source/lane attempts while retaining "
                "the usable on-topic evidence"
                + (
                    "; one or more sources exhausted their bounded retry"
                    if counts.get("INFRASTRUCTURE")
                    else ""
                ),
            )
        if collisions and mode == "grounded":
            actions = _required_actions(
                row
                for row in digest.source_statuses
                if row.code in ("QUERY-COLLISION", "INFRASTRUCTURE")
            )
            detail = (
                summary
                + "; semantic collisions or unrepaired query rejections require "
                "the caller to semantically re-anchor with different domain "
                "terminology"
            )
            if counts.get("INFRASTRUCTURE"):
                detail += "; infrastructure rows retry unchanged as listed"
            return ResearchStatus(
                "RETRY-REQUIRED",
                detail,
                required_actions=actions,
            )
        if counts.get("INFRASTRUCTURE"):
            if any(row.count > 0 for row in digest.source_statuses):
                return ResearchStatus(
                    "DEGRADED",
                    summary + "; one or more sources exhausted their bounded retry -- "
                    "proceed with the available evidence and disclose the outage",
                )
            return ResearchStatus(
                "RETRY-REQUIRED",
                summary + "; retry the failed source/lane without changing terms",
                required_actions=_required_actions(digest.source_statuses),
            )
        if counts.get("ON-TOPIC"):
            return ResearchStatus("OK", summary)
        if collisions:
            return ResearchStatus(
                "LOW-OVERLAP",
                summary + "; acceptable for a deliberate cross-domain probe",
            )
        if any(row.count > 0 for row in digest.source_statuses):
            return ResearchStatus(
                "OK",
                summary + "; thin rows require manual relevance review",
            )
        actions = _query_actions(digest.source_statuses)
        suggested_action = next(
            (action for action in actions if action.kind == "RETRY-SHORTENED"),
            None,
        )
        return ResearchStatus(
            "RETRY-REQUIRED",
            summary + "; broaden or semantically re-anchor the thin lanes",
            suggested_query=suggested_action.query if suggested_action else "",
            suggested_lane=suggested_action.lane if suggested_action else "",
            required_actions=actions,
        )

    config = next((e for e in digest.errors if _error_class(e) == "config"), None)
    if config:
        return ResearchStatus(
            "CONFIG",
            "a source rejected its key or is load-shedding anonymous access; a "
            "retry will not fix it -- set the source key or proceed on the "
            "other sources and say so",
        )
    counts = dict(digest.counts)
    # A paper source counts as attempted whether it succeeded (in counts, maybe
    # with 0 hits) or errored (in errors) -- so an all-errored stratum, which
    # never reaches counts, is still caught rather than falling through to OK.
    paper_succeeded = [s for s in counts if s in _PAPER_SOURCES]
    paper_failed = {
        e.split(":", 1)[0].strip().lower() for e in digest.errors
    } & _PAPER_SOURCES
    paper_attempted = set(paper_succeeded) | paper_failed
    paper_total = sum(counts[s] for s in paper_succeeded)
    if paper_attempted and paper_total == 0:
        if paper_failed and not paper_succeeded:
            # Every attempted paper source errored (already retried once in-tool).
            # Infrastructure, not the query -- so no reworked query to suggest.
            return ResearchStatus(
                "RETRY-REQUIRED",
                "every paper source failed after an in-tool retry -- retry once "
                "more, or proceed on the other evidence and say so; do not fall "
                "back on your own knowledge silently",
                required_actions=tuple(
                    ResearchAction("RETRY-SAME", source, "lane-1", digest.topic)
                    for source in sorted(paper_failed)
                ),
            )
        if paper_failed:
            # Mixed: some paper sources errored while the rest returned 0. The
            # stratum total is 0, but calling that 'every source returned 0' would
            # hide the failure and steer a mutation of a query that may be fine --
            # the errored source could have had hits. Name the failure; suggest no
            # rework (do not mutate a possibly-fine query).
            failed = ", ".join(sorted(paper_failed))
            return ResearchStatus(
                "RETRY-REQUIRED",
                f"some paper sources failed ({failed}) and the rest returned 0 -- "
                "the empty result may be incomplete; retry, and if it persists "
                "proceed on the other evidence and say so, do not fall back "
                "silently",
                required_actions=tuple(
                    ResearchAction("RETRY-SAME", source, "lane-1", digest.topic)
                    for source in sorted(paper_failed)
                ),
            )
        reworked = _rework("All sources", digest.topic)
        detail = "every paper source returned 0 on a valid query -- " + (
            "resubmit the suggested query before falling back to your own knowledge"
            if reworked
            else "re-anchor with DIFFERENT domain-specific terms (your judgment); "
            "do not resubmit the same query, and do not fall back silently"
        )
        action = _query_action("All sources", "lane-1", digest.topic)
        return ResearchStatus(
            "RETRY-REQUIRED",
            detail,
            suggested_query=reworked,
            suggested_lane="lane-1" if reworked else "",
            required_actions=(action,),
        )
    texts = _scoreable_texts(digest.papers, digest.topic)
    frac = _on_topic_fraction(texts, digest.topic)
    if frac is not None and len(texts) >= _MIN_SCORING_HITS and frac < _ON_TOPIC_MIN:
        pct = round(frac * 100)
        if mode == "exploratory":
            return ResearchStatus(
                "LOW-OVERLAP",
                f"{pct}% of hits share query vocabulary -- acceptable if this "
                "was a deliberate cross-domain probe; otherwise re-anchor with "
                "more domain context",
            )
        detail = (
            f"only {pct}% of hits are on-topic -- the query likely collided with "
            "unrelated work; re-anchor with DIFFERENT domain-specific terms "
            "(your judgment) and resubmit -- do not mechanically shorten it"
        )
        action = ResearchAction("RE-ANCHOR", "All sources", "lane-1", _REANCHOR_QUERY)
        return ResearchStatus(
            "RETRY-REQUIRED",
            detail,
            required_actions=(action,),
        )
    if digest.errors:
        failed_sources = sorted({e.split(":", 1)[0] for e in digest.errors})
        failed = ", ".join(failed_sources)
        if not any((digest.papers, digest.libraries, digest.repos, digest.models)):
            actions = tuple(
                ResearchAction("RETRY-SAME", source, "lane-1", digest.topic)
                for source in failed_sources
            )
            return ResearchStatus(
                "RETRY-REQUIRED",
                f"sources failed after their bounded retry ({failed}) and no usable "
                "evidence remains",
                required_actions=actions,
            )
        return ResearchStatus(
            "DEGRADED",
            f"sources failed after their bounded retry ({failed}) -- proceed with "
            "the available evidence and disclose the outage",
        )
    return ResearchStatus("OK", "results look on-topic")


_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")


def _normalize_identifier(identifier: str) -> str:
    """Reduce a DOI to its bare form so the same paper matches across sources.
    OpenAlex returns a resolver URL (`https://doi.org/10.1101/X`) while Europe PMC
    returns the bare DOI (`10.1101/X`); compared raw they never match, so the SAME
    preprint reaches the council twice whenever its two titles differ by so much as
    a word (title-dedup is the only other key)."""
    key = identifier.lower().strip()
    for prefix in _DOI_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


_GENERIC_PAPER_VENUES = frozenset(
    {"arxiv", "openalex", "Europe PMC", "europepmc-preprints"}
)


def _dedup_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers while preserving richer metadata from later sources."""
    index_by_id: dict[str, int] = {}
    index_by_title: dict[str, int] = {}
    out: list[Paper] = []
    for p in papers:
        key_id = _normalize_identifier(p.identifier)
        key_title = "".join(ch for ch in p.title.lower() if ch.isalnum())
        existing_index = index_by_id.get(key_id) if key_id else None
        if existing_index is None and key_title:
            existing_index = index_by_title.get(key_title)
        if existing_index is not None:
            existing = out[existing_index]
            if existing.full_text_available is True or p.full_text_available is True:
                full_text_available: bool | None = True
            elif (
                existing.full_text_available is False or p.full_text_available is False
            ):
                full_text_available = False
            else:
                full_text_available = None
            out[existing_index] = replace(
                existing,
                identifier=existing.identifier or p.identifier,
                authors=existing.authors or p.authors,
                year=existing.year if existing.year is not None else p.year,
                source=(
                    p.source
                    if existing.source in _GENERIC_PAPER_VENUES
                    and p.source not in _GENERIC_PAPER_VENUES
                    else existing.source
                ),
                url=existing.url or p.url,
                abstract=existing.abstract or p.abstract,
                field=existing.field or p.field,
                research_source=existing.research_source or p.research_source,
                full_text_available=full_text_available,
                full_text_url=existing.full_text_url or p.full_text_url,
                mesh_terms=tuple(dict.fromkeys(existing.mesh_terms + p.mesh_terms)),
                strata=tuple(dict.fromkeys(existing.strata + p.strata)),
                query_lanes=tuple(dict.fromkeys(existing.query_lanes + p.query_lanes)),
            )
            if key_id:
                index_by_id[key_id] = existing_index
            if key_title:
                index_by_title[key_title] = existing_index
            continue
        index = len(out)
        out.append(p)
        if key_id:
            index_by_id[key_id] = index
        if key_title:
            index_by_title[key_title] = index
    return out


def validate_sources(sources: Iterable[str]) -> set[str]:
    """Normalize requested sources to a set, rejecting any not in
    DEFAULT_SOURCES. Fails fast so a typo'd source surfaces as an error
    rather than a silently empty digest."""
    chosen = set(sources)
    unknown = chosen - set(DEFAULT_SOURCES)
    if unknown:
        migration = (
            " europepmc was split into europepmc-published and europepmc-preprints."
            if "europepmc" in unknown
            else ""
        )
        raise ValueError(
            f"Unknown research source(s): {', '.join(sorted(unknown))}. "
            f"Valid sources: {', '.join(DEFAULT_SOURCES)}.{migration}"
        )
    return chosen


async def _contained(name: str, factory, query: str):
    """Run one source's query with a bounded single retry, converting any final
    failure into an error string so a dead source never sinks its peers.
    `factory` builds the source coroutine from a query string, so the retry can
    resubmit a *reworked* query. Returns (name, value|None, error|None,
    note|None, actual_query); `actual_query` is the query used by the final
    attempt, so status scoring never grades shortened-query results against the
    original longer lane. `note` is set only when a retry succeeded, so a
    repaired source is visible. CancelledError (BaseException) propagates.

    The retry is deliberate, not a blanket loop -- the class of the first error
    decides (see `_error_class`):
    - 'query' (arXiv 400): the query is malformed. Resubmit ONCE with the
      auto-shortened scorable-domain-term query -- the same fix the retry hint asks
      the consumer to make, done here so a first-attempt 400 never reaches the
      orchestrator as an excuse to fall back on its own knowledge.
    - 'rate' (arXiv's 200-with-'Rate exceeded' refusal, or a 429): resubmit
      ONCE with the SAME query after claiming the next polite slot -- an
      immediate retry just re-hits the wall, and concurrent refusals must not
      wake in lockstep (see _claim_rate_slot). A saturated retry queue surfaces
      the failure instead of accumulating unbounded wait debt.
    - 'transient' (timeout/flake): resubmit ONCE with the SAME query.
    - 'config' (rejected key / anonymous load-shed): NOT retried -- a retry
      cannot fix it; surface it so the verdict can label it CONFIG."""
    attempt_query = query
    try:
        return name, await factory(attempt_query), None, None, attempt_query
    except Exception as exc:  # noqa: BLE001 -- surface, don't sink peers
        err = f"{name}: {type(exc).__name__}: {exc}"
        cls = _error_class(err)
        if cls == "config":
            return name, None, err, None, attempt_query
        if cls == "query":
            reworked = _rework(name, query)
            if not reworked:
                # Nothing left to rework -- `_rework` folds case and whitespace,
                # so a case-/spacing-only 'shorter' query never triggers a
                # wasted, semantically identical retry (round-3 review).
                return name, None, err, None, attempt_query
            attempt_query = reworked
        elif cls == "rate":
            fire_at = _claim_rate_slot(time.monotonic())
            if fire_at is None:
                return name, None, err, None, attempt_query
            await asyncio.sleep(max(0.0, fire_at - time.monotonic()))
        else:
            await asyncio.sleep(_TRANSIENT_RETRY_DELAY_S)
    try:
        value = await factory(attempt_query)
    except Exception as exc:  # noqa: BLE001 -- retry also failed; surface it
        return (
            name,
            None,
            f"{name}: {type(exc).__name__}: {exc}",
            None,
            attempt_query,
        )
    if attempt_query != query:
        note = f'retried after auto-shortening query to "{attempt_query}"'
    elif cls == "rate":
        note = "retried after a rate-limit refusal cleared"
    else:
        note = "retried after a transient failure"
    return name, value, None, note, attempt_query


async def research_topic(
    topic: str,
    *,
    query_lanes: tuple[str, ...] | list[str] | None = None,
    sources: set[str] | tuple[str, ...] = DEFAULT_SOURCES,
    limit: int = DEFAULT_LIMIT,
    timeout: float = 15.0,
    email: str | None = None,
    openalex_api_key: str | None = None,
    context7_api_key: str | None = None,
    github_token: str | None = None,
    hf_token: str | None = None,
    map_fields: bool = False,
    mode: str = "grounded",
    purpose: str = "methods",
    client: httpx.AsyncClient | None = None,
) -> ResearchDigest:
    """Query the selected sources concurrently and return a deduped digest. Each
    source is failure-isolated; errors land in digest.errors. Grounded mode
    excludes candidates from lexically colliding attempts; exploratory mode
    retains them as possible cross-domain evidence. When `map_fields` is set and
    OpenAlex is a source, one extra concurrent group_by call fills digest.field_map
    with the subfield distribution (the skystorm topology readout); a failure
    there degrades to an empty map, never sinks the digest.
    Pass `client` (for example, a MockTransport-backed one) for testing;
    otherwise a short-lived AsyncClient is created and closed here. Context7,
    GitHub, and Hugging Face tokens are read from the environment when not
    passed. All provider tokens are sent only as Bearer headers."""
    sources = validate_sources(sources)
    if mode not in ("grounded", "exploratory"):
        raise ValueError(
            f"Unknown research mode {mode!r}; expected 'grounded' or 'exploratory'."
        )
    if purpose not in RESEARCH_PURPOSES:
        raise ValueError(
            f"Unknown research purpose {purpose!r}; expected one of "
            f"{', '.join(RESEARCH_PURPOSES)}."
        )
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}.")
    if limit > _MAX_LIMIT:
        raise ValueError(f"limit must be <= {_MAX_LIMIT}, got {limit}.")
    topic = " ".join(topic.split())
    if _is_reanchor_placeholder(topic):
        raise ValueError(
            "Research topic is a re-anchor placeholder; supply different "
            "domain-specific terms before calling research."
        )
    if not _scorable_terms(topic):
        raise ValueError(
            f"Research topic {topic!r} has no distinctive terms to search on -- "
            "anchor it in 2+ domain-specific terms (the field PLUS the specific "
            "method or concept) and call again."
        )
    raw_lanes = tuple(query_lanes) if query_lanes is not None else (topic,)
    if not raw_lanes or len(raw_lanes) > 3:
        raise ValueError("query_lanes must contain between 1 and 3 queries.")
    lanes: list[tuple[str, str]] = []
    seen_lanes: set[str] = set()
    for raw_query in raw_lanes:
        query = " ".join(raw_query.split())
        if _is_reanchor_placeholder(query):
            raise ValueError(
                "Research query lane is a re-anchor placeholder; supply different "
                "domain-specific terms before calling research."
            )
        canonical = query.lower()
        if not _scorable_terms(query):
            raise ValueError(
                f"Research query lane {raw_query!r} has no distinctive terms to "
                "search on -- anchor it in 2+ domain-specific terms (the field "
                "PLUS the specific method or concept) and call again."
            )
        if canonical in seen_lanes:
            continue
        seen_lanes.add(canonical)
        lanes.append((f"lane-{len(lanes) + 1}", query))
    if not lanes:
        raise ValueError("query_lanes must contain at least one distinct query.")
    if email is None:
        email = os.environ.get("QUORUM_OPENALEX_EMAIL")
    if openalex_api_key is None:
        openalex_api_key = os.environ.get("QUORUM_OPENALEX_API_KEY") or os.environ.get(
            "OPENALEX_API_KEY"
        )
    if context7_api_key is None:
        context7_api_key = os.environ.get("CONTEXT7_API_KEY")
    if github_token is None:
        github_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if hf_token is None:
        hf_token = os.environ.get("QUORUM_HF_TOKEN") or os.environ.get("HF_TOKEN")
    # Secret values that must never surface in a rendered digest -- a backstop for
    # the error path (the search fns also refuse to send a malformed token, which
    # is what closes the demonstrated h11 "illegal header value" leak at source).
    secrets = [
        t
        for t in (
            _clean_token(openalex_api_key),
            _clean_token(context7_api_key),
            _clean_token(github_token),
            _clean_token(hf_token),
        )
        if t
    ]
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()
    # Each source is a query-taking factory so `_contained` can resubmit a
    # reworked query on retry. `q` is the (possibly shortened) retry query; all
    # other args are fixed per call. Built unconditionally, then filtered to
    # `sources` -- dict order matches DEFAULT_SOURCES, so filtering preserves
    # iteration order regardless of what order `sources` was requested in.
    all_factories = {
        "arxiv": lambda q: search_arxiv(client, q, limit=limit, timeout=timeout),
        "openalex": lambda q: search_openalex(
            client,
            q,
            limit=limit,
            timeout=timeout,
            email=email,
            api_key=openalex_api_key,
            purpose=purpose,
        ),
        "europepmc-published": lambda q: search_europepmc_published(
            client,
            q,
            limit=limit,
            timeout=timeout,
            expand_synonyms=True,
        ),
        "europepmc-preprints": lambda q: search_europepmc_preprints(
            client, q, limit=limit, timeout=timeout
        ),
        # Context7's default timeout (10.0s) has always differed from the rest
        # (15.0s) and is never overridden by any caller -- hardcoded here rather
        # than threaded to the shared `timeout` param so that default is preserved.
        "context7": lambda q: fetch_context7(
            client,
            q,
            timeout=10.0,
            max_libs=limit,
            token=context7_api_key,
        ),
        "github": lambda q: search_github(
            client, q, limit=limit, timeout=timeout, token=github_token
        ),
        "huggingface": lambda q: search_huggingface(
            client, q, limit=limit, timeout=timeout, token=hf_token
        ),
    }
    factories = {name: fn for name, fn in all_factories.items() if name in sources}

    async def run_lane(
        lane: str, query: str, name: str, factory: Callable
    ) -> tuple[str, str, str, list | None, str | None, str | None]:
        source, value, error, note, actual_query = await _contained(
            name, factory, query
        )
        return lane, actual_query, source, value, error, note

    async def run_source(name: str, factory: Callable) -> list:
        results: list = []
        for lane_index, (lane, query) in enumerate(lanes):
            if lane_index > 0 and name not in _PAPER_SOURCES:
                break
            if lane_index > 0 and name == "arxiv":
                await asyncio.sleep(_RATE_PAUSE_S)
            results.append(await run_lane(lane, query, name, factory))
        return results

    async def run_lanes() -> list:
        by_source = await asyncio.gather(
            *(run_source(name, factory) for name, factory in factories.items())
        )
        return [result for source_results in by_source for result in source_results]

    async def _safe_field_map() -> list[tuple[str, int]]:
        # Failure-isolated like every source: a group_by hiccup degrades the map
        # to empty, it never sinks the digest. Only runs when requested and
        # OpenAlex is in play (it is the only backend with the taxonomy).
        try:
            return await map_openalex_fields(
                client,
                lanes[0][1],
                timeout=timeout,
                email=email,
                api_key=openalex_api_key,
            )
        except Exception:  # noqa: BLE001 -- an optional annotation, never fatal
            return []

    want_map = map_fields and "openalex" in sources
    try:
        if want_map:
            results, field_map = await asyncio.gather(run_lanes(), _safe_field_map())
        else:
            results, field_map = await run_lanes(), []
    finally:
        if owns_client:
            await client.aclose()
    papers: list[Paper] = []
    libraries: list[LibraryDoc] = []
    repos: list[Repo] = []
    models: list[HFModel] = []
    errors: list[str] = []
    count_by_source = {name: 0 for name in factories}
    successful_sources: set[str] = set()
    notes: list[str] = []
    source_statuses: list[SourceLaneStatus] = []
    excluded_collision_by_source: dict[str, int] = {}
    paper_groups: dict[str, list[tuple[str, list[Paper]]]] = {}
    for lane, query, name, value, err, note in results:
        if note:
            notes.append(_redact(f"{name} [{lane}]: {note}", secrets))
        row = _source_lane_status(name, lane, query, value or [], err)
        source_statuses.append(row)
        if err is not None:
            labelled = err.replace(f"{name}:", f"{name}: [{lane}]", 1)
            errors.append(_redact(labelled, secrets))
            continue
        if value is None:
            continue
        successful_sources.add(name)
        if row.code == "QUERY-COLLISION" and mode == "grounded":
            retained = [
                candidate
                for candidate in value
                if _candidate_is_on_topic(candidate, query)
            ]
            excluded = len(value) - len(retained)
            if excluded:
                excluded_collision_by_source[name] = (
                    excluded_collision_by_source.get(name, 0) + excluded
                )
            value = retained
        if name == "context7":
            libraries.extend(cast(list[LibraryDoc], value))
        elif name == "github":
            repos.extend(cast(list[Repo], value))
        elif name == "huggingface":
            models.extend(cast(list[HFModel], value))
        else:
            tagged = [
                replace(
                    paper,
                    research_source=paper.research_source or name,
                    query_lanes=tuple(dict.fromkeys(paper.query_lanes + (lane,))),
                )
                for paper in cast(list[Paper], value)
            ]
            paper_groups.setdefault(name, []).append((lane, tagged))
    for name in factories:
        groups = paper_groups.get(name)
        if not groups:
            continue
        candidates: list[Paper] = []
        for index in range(max(len(rows) for _, rows in groups)):
            for _, rows in groups:
                if index < len(rows):
                    candidates.append(rows[index])
        selected = _dedup_papers(candidates)[:limit]
        papers.extend(selected)
        count_by_source[name] = len(selected)

    def dedup(items: list, key: Callable) -> list:
        found: dict[str, object] = {}
        for item in items:
            found.setdefault(key(item), item)
        return list(found.values())[:limit]

    libraries = cast(
        list[LibraryDoc], dedup(libraries, lambda item: item.url or item.name)
    )
    repos = cast(list[Repo], dedup(repos, lambda item: item.name))
    models = cast(list[HFModel], dedup(models, lambda item: item.id))
    for name, values in (
        ("context7", libraries),
        ("github", repos),
        ("huggingface", models),
    ):
        if name in successful_sources:
            count_by_source[name] = len(values)

    def source_group(source: str) -> str:
        return "paper" if source in _PAPER_SOURCES else "artifact"

    on_topic_groups = {
        (row.lane, source_group(row.source))
        for row in source_statuses
        if row.code == "ON-TOPIC"
    }
    source_statuses = [
        replace(
            row,
            code="SOURCE-MISMATCH",
            detail="this source returned 0 while a peer source found on-topic work",
        )
        if (
            (row.lane, source_group(row.source)) in on_topic_groups
            and row.code == "THIN"
            and row.count == 0
        )
        else row
        for row in source_statuses
    ]
    counts = [
        (name, count_by_source[name])
        for name in factories
        if name in successful_sources
    ]
    return ResearchDigest(
        topic=topic,
        papers=tuple(_dedup_papers(papers)),
        libraries=tuple(libraries),
        errors=tuple(errors),
        repos=tuple(repos),
        models=tuple(models),
        counts=tuple(counts),
        notes=tuple(notes),
        field_map=tuple(field_map),
        source_statuses=tuple(source_statuses),
        excluded_collision_counts=tuple(excluded_collision_by_source.items()),
        mode=mode,
    )
