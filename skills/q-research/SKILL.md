---
name: q-research
description: Fetch publications and prior art for a topic — papers from arXiv, OpenAlex, and Europe PMC, plus library docs (Context7), GitHub repos, and HuggingFace models — returned as one markdown digest with links. Use when the user wants literature, publications, papers, or prior art pulled on a subject, without running the council. Pass the topic as the argument; add repeatable --source flags to restrict sources, --limit for max results per source, or --exploratory for cross-domain analogy hunting.
---

# Q-Research — standalone prior-art digest

One synchronous call, no council, no `job_id`/`q_await`. This fronts the same
engine the storm modes use for research-first grounding; use it when the
digest itself is the deliverable.

## Arguments

`$ARGUMENTS` is the research topic, optionally with flags:
- `--source <s>` / `-s <s>` (repeatable) — restrict to a subset of `arxiv`,
  `openalex`, `europepmc`, `context7`, `github`, `huggingface`. Default: all
  six. "Publications/literature only" means
  `-s arxiv -s openalex -s europepmc`.
- `--limit <n>` — max results per source (default 5). Below 3 the automatic
  on-topic verdict cannot fire — see Step 3.
- `--exploratory` — cross-domain mode: the user is hunting structural
  analogies in *other* fields, so low topic overlap is expected (reported as
  `LOW-OVERLAP`, not a retry demand) and the digest adds a `### Field map`
  of which OpenAlex subfields the query spans.

Strip the flags from the topic text. If the topic (after removing flags) is
empty, ask the user what they want researched.

## Step 1 — Form the query

**Short *and distinctive*; short is not the same as generic.** Anchor it in
**two or more domain-specific terms** (the field *plus* the specific
method/concept) so it can only match the intended domain. A bare common token —
`data`, `model`, `network`, `signal` — or a word that doubles as an author
surname (`Sun`, `Li`) collides and returns **confident-looking noise** (author
names, generic "A Survey of …" titles); the tool refuses the most generic
queries outright. Example: ❌ `data` / `alternative data` →
✓ `alternative data equity return prediction`. Don't hand-tune per backend —
the tool shapes each source's query and strips arXiv's boolean operators for
you. Not a full paragraph either.

## Step 2 — Call the tool

Call `mcp__plugin_code-quorum_quorum__q_research` in Claude Code or
`mcp__quorum_codex__q_research` in Codex with:
- `topic`: the topic text
- `sources`: the list from `--source`, only if the user restricted it
- `limit`: the `--limit` value, only if given
- `mode`: `"exploratory"` only under `--exploratory` (default is grounded)

No-MCP fallback (headless, another agent, plugin not loaded): from a
code-quorum checkout,
`uv run quorum research "<topic>" [-s <source>]... [--limit <n>]` — same
engine, grounded mode only (the CLI has no exploratory flag; an
`--exploratory` pull needs the MCP tool).

## Step 3 — Act on the `Research status:` line before anything else

The digest **opens** with a deterministic quality verdict:

- `OK` — results look on-topic; proceed. The overlap check needs at least 3
  paper hits to fire — with fewer (a small `--limit`, a thin topic), `OK`
  only means nothing errored, so eyeball the returned titles for relevance
  yourself before treating them as grounding.
- `RETRY-RECOMMENDED` — the query whiffed or collided with unrelated work. It
  **usually** carries a suggested shorter query (`· try: "…"`) — resubmit that
  verbatim before concluding "no prior art". When the suggestion is **absent**,
  the detail text after the dash names the move, and the two cases need
  opposite ones: a **backend/infrastructure failure** says *retry* — resubmit
  the **same** query (changing terms cannot fix an outage); an
  **un-shortenable query** says *re-anchor* — resubmit with DIFFERENT
  domain-specific terms of your own. Shrugging the result off and answering
  from your own knowledge — *"the research backend choked, I'll use what I
  know"* — is the **specific failure to avoid**; only after a reworked retry
  has **also** failed may you note a source unavailable and continue.
- `CONFIG` — a key/anonymous-access problem no retry fixes; proceed on the
  other sources and say so.
- `LOW-OVERLAP` (exploratory mode only) — expected for a deliberate
  cross-domain pivot; judge hits by structural kinship to the problem, not
  literal subject-matter overlap.

Mechanically-fixable failures (an arXiv 400, its 200-with-`Rate exceeded`
rate refusal, a transient flake) are **already retried inside the tool** — a
`↻` note marks a source that was repaired, not
hidden. A **domain-legitimate 0** in the per-source footer (Europe PMC on a
non-biology topic, GitHub/HuggingFace on a non-software one, Context7 with no
matching library) is a real answer, not a whiff to rework.

## Step 4 — Present

- Open with the `Research status:` line quoted, plus one line on per-source
  signal strength — so a domain-legitimate whiff is never mistaken for missing
  prior art, and noise is never mistaken for confirmed grounding.
- Then the findings, keeping the digest's links: papers (title, year, source,
  URL), then docs/repos/models if queried. Don't pad — the digest is already
  ranked and deduped.
- Europe PMC hits labelled by preprint server ("bioRxiv", "Research Square")
  are **not peer-reviewed** — say so wherever one carries weight.
- Before stating what any hit *says*, apply **Verify cited evidence** (below).

## Verify cited evidence

The digest hands you titles, abstracts, ranking lines, and links — it is a
**search result, not a read**. Never state what a paper or repo *says* on the
strength of its title or abstract; that is where confident-sounding
misattribution enters. How far to go depends on what the citation is holding up:

- **Load-bearing** — a decision, a number, a method choice, a baseline, or a
  "this is (or isn't) already solved" claim would rest on it. The test: *if this
  source said the opposite, would anything change?* If yes: fetch the artifact
  itself — the arXiv PDF or HTML, the Europe PMC full text, the repo's actual
  files at a pinned commit, the model card — and read the parts that decide
  whether it transfers (method, the results table with its N and spread,
  limitations), then confirm it says what's claimed. Existence is **not**
  confirmation: a real paper is easy to cite for something it never said.
- **Idea-seeding** — the citation gestures at a concept and nothing rests on it
  yet. Confirming the **concept** is real and roughly as described is enough.
  Say that's the tier; never let a skim pass as grounding.

Anything that can't be confirmed at its own tier is down-weighted and marked
`[unverified]`. Never pass a flaky citation through as settled fact.

A failed fetch is a refusal until proven otherwise: registries refuse as well
as answer — arXiv signals its rate limit as HTTP 200 with a bare `Rate
exceeded` body, not an error status — so an identifier that won't resolve is
`[unverified]`, never evidence of fabrication on its own. Only a well-formed
"no such record" answer from the registry counts against it.

## Notes

- Not subject to the council's anti-bias gate — this is external prior art,
  not peer output. No own-answer-first ceremony; call it directly.
- For council ideation grounded in this digest, use `q-brainstorm`; for
  forward-research with a validation pass, `q-skystorm`.
