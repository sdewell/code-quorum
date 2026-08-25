---
name: q-research
description: Fetch publications and prior art for a topic — papers from arXiv, OpenAlex, and separate published/preprint Europe PMC lanes, plus library docs (Context7), GitHub repos, and HuggingFace models — returned as one markdown digest with links. Use when the user wants literature, publications, papers, or prior art pulled on a subject, without running the council. Pass the topic as the argument; add repeatable --source or --query-lane flags, --purpose methods|currency, --limit for max results per source, or --exploratory for cross-domain analogy hunting.
---

# Q-Research — standalone prior-art digest

One synchronous call, no council, no `job_id`/`q_await`. This fronts the same
engine the storm modes use for research-first grounding; use it when the
digest itself is the deliverable.

## Arguments

`$ARGUMENTS` is the research topic, optionally with flags:
- `--source <s>` / `-s <s>` (repeatable) — restrict to a subset of `arxiv`,
  `openalex`, `europepmc-published`, `europepmc-preprints`, `context7`,
  `github`, `huggingface`. Default: all seven. "Publications/literature only"
  means `-s arxiv -s openalex -s europepmc-published -s europepmc-preprints`.
- `--limit <n>` — max results per source (default 5, maximum 20). Below 3 the
  automatic on-topic verdict cannot fire — see Step 3. The cap stays fixed
  across lanes, so extra lanes broaden coverage without growing the digest.
- `--purpose methods|currency` — `methods` is the default: OpenAlex balances
  all-time canonical/relevance candidates with a recent five-year stratum and
  retains the stratum labels after deduplication. `currency` queries only the
  recent five-year stratum.
- `--query-lane <query>` (repeatable, at most three) — explicit semantic query
  formulations. If absent, form the lanes in Step 1 and pass them through the
  tool's `query_lanes` parameter. Literature sources search every lane; artifact
  sources (Context7, GitHub, and Hugging Face) search only the primary lane.
- `--exploratory` — cross-domain mode: the user is hunting structural
  analogies in *other* fields, so low topic overlap is expected (reported as
  `LOW-OVERLAP`, not a retry demand) and the digest adds a `### Field map`
  of which OpenAlex subfields the primary query lane spans.

Strip the flags from the topic text. If the topic (after removing flags) is
empty, ask the user what they want researched.

## Step 1 — Form the query lanes

Form two or three genuinely different semantic lanes; do not generate them by
merely deleting words from one long query:

1. Domain plus method or construct.
2. Failure mode, validity question, or limitation.
3. Standard, guideline, review, or canonical terminology when applicable.

Each lane must be **short and distinctive; short is not the same as generic**.
Anchor it in **two or more domain-specific terms** (the field plus the specific
method or concept). A bare common token — `data`, `model`, `network`, `signal`
— or a word that doubles as an author surname (`Sun`, `Li`) collides and returns
confident-looking noise. The tool refuses the most generic lanes outright.
Example for execution provenance:

- `computational experiment provenance reproducibility artifacts`
- `exploratory research raw data traceability`
- `minimum information experimental reporting provenance`

Mechanical shortening is only the first repair for one colliding lane. If that
shortening also collides, use a **semantic re-anchor** with different domain
terminology; repeatedly deleting words often makes the collision worse.

## Step 2 — Call the tool

Call `mcp__plugin_code-quorum_quorum__q_research` in Claude Code or
`mcp__quorum_codex__q_research` in Codex with:
- `topic`: the topic text
- `sources`: the list from `--source`, only if the user restricted it
- `limit`: the `--limit` value, only if given
- `purpose`: `"methods"` by default or `"currency"` when requested
- `query_lanes`: the two or three query strings from Step 1
- `mode`: `"exploratory"` only under `--exploratory` (default is grounded)

No-MCP fallback (headless, another agent, plugin not loaded): from a
code-quorum checkout,
`uv run quorum research "<topic>" [-s <source>]... [--purpose <purpose>]
[--query-lane <query>]... [--limit <n>]` — same engine, grounded mode only (the
CLI has no exploratory flag; an `--exploratory` pull needs the MCP tool).

## Step 3 — Act on the `Research status:` line before anything else

The digest opens with a combined deterministic quality verdict, followed by a
`### Source/lane status` table. The table is authoritative for diagnosis; one
good source must not hide a collision elsewhere, and one noisy source must not
sink a good lane. Each row is one of:

- `ON-TOPIC` — enough candidates share the lane vocabulary.
- `THIN` — fewer than three candidates; too little evidence to grade relevance.
- `QUERY-COLLISION` — enough candidates returned, but they do not match the
  lane. Try one labelled mechanical shortening; if it still collides, use a
  semantic re-anchor with different terminology.
- `SOURCE-MISMATCH` — this source returned nothing while a peer source found
  on-topic work for the same lane; do not force that source to fit the domain.
- `INFRASTRUCTURE` — the source failed after its bounded retry; retry the same
  lane because changing terms cannot repair an outage.
- `CONFIG` — a credential or access problem; changing the query cannot fix it.

The combined `Research status:` line summarizes those rows:

- `OK` — at least one source/lane is on-topic and no configuration problem
  needs priority; read every non-OK status row before proceeding. Infrastructure
  failures remain visible in the summary and table even when a peer paper source
  supplied usable evidence.
- `RETRY-RECOMMENDED` — the query whiffed or collided with unrelated work. It
  **usually** carries one explicitly labelled mechanical shortening (`· try: "…"`) —
  resubmit it once before concluding "no prior art". If that result still
  reports `QUERY-COLLISION`, perform a semantic re-anchor instead of shortening
  again. When the suggestion is **absent**,
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
hidden. A `SOURCE-MISMATCH` row is a real answer, not a whiff to rework.

## Step 4 — Present

- Open with the `Research status:` line quoted, then summarize every non-OK row
  from `### Source/lane status` so partial failures cannot be hidden.
- Then the findings, keeping the digest's links: papers (title, year, source,
  URL), then docs/repos/models if queried. Don't pad — the digest is already
  ranked and deduped.
- Europe PMC preprint hits labelled by server ("bioRxiv", "Research Square")
  are **not peer-reviewed** — say so wherever one carries weight. Published-lane
  hits expose full-text availability and may include MeSH metadata. Exact
  published matches have priority; `[strata: synonym-expanded]` identifies a
  candidate used only to backfill a thin exact result set.
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
