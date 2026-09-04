---
name: q-skystorm
description: Blue-sky, forward-looking ideation for Claude or Codex hosts — map the problem's cross-domain topology in the literature by default, dream wide, then ground the best threads and synthesize a path forward. Use for forward-research where the question is not just "what's a wild idea" but "how would I validate it". Pass the topic; add --no-research to dream from model priors alone, or --verbose to lift terse output caps.
---

# Q-Skystorm — map the territory, dream wide, then ground

## Completion gate

For every council start, `q_await` is the blocking completion notification. After completing the host's independent work, call it in the same turn as the start. Do not end the turn, tell the user you will check later, or leave a live `job_id` pending. If the council is still running, remain in the blocking call until it returns a result or error.

Skystorm is the forward-research companion to `q-brainstorm`. Brainstorm stays
grounded (assemble from solved problems); skystorm dreams past the frontier and
then asks how to validate what it found. Research-first is the **default**: the
anchor→harvest→pivot→map protocol surfaces the far-afield domains *from the
literature*, and the dream council is seeded with that evidence — the point is
cross-domain transfer driven by published structure, not by model priors alone.
Movements: **map → dream → ground → synthesize.**

## Arguments

`$ARGUMENTS` is the topic, optionally with flags:
- `--no-research` — skip the literature mapping (research is the **default**);
  the dream stage then runs on the agents' own priors, which is exactly what
  the mapping exists to improve on — prefer the default.
- `--verbose` — lift the terse output caps on **both** the dream and ground passes (fuller, unabridged ideation). Default is terse.

Strip the flags from the topic before passing `topic`. If the topic is empty, ask what to explore.

## Host routing

When this skill is running in Codex, read
`references/codex-host.md` completely and follow that workflow instead of the
steps below. It uses the `mcp__quorum_codex__*` tools and passes `host: "codex"`.

When running in Claude Code, continue below. Use the
`mcp__plugin_code-quorum_quorum__*` tools and pass `host: "claude"` on every
council start.

If either required MCP tool is absent from the current task's tool registry,
stop and report that the plugin MCP server did not load. Do not substitute the
synchronous CLI, direct Python imports, or another runtime path: only the
registered start/await pair enforces the structural anti-bias gate.

Every MCP council start requires an explicit absolute `cwd`. Never omit it: the
MCP server's own working directory is its installation or runtime location, not
the project selected by the user.

## Claude Code workflow

## Step 1 — Your own ideas FIRST

Before researching and before starting the council, list 3–5 of your own
blue-sky ideas for the topic, each with the non-obvious connection it exploits
and the smallest experiment that would test it. Keep them wild but each tied to
a way to get signal.

The order is the anti-bias gate in both directions: your ideas are committed
*before* the research digest can anchor them, and the council never sees them.

## Step 2 — Map the topology

Unless `--no-research` was passed (in which case skip this whole step), call
`mcp__plugin_code-quorum_quorum__q_research` and run the
**anchor → harvest → pivot → map** protocol rather than one query, so the
far-afield domains come back *from the literature* instead of your own priors.
**The anchor and the pivots use different `mode` values — this matters:**
- **Anchor (`mode="grounded"`).** One query in the topic's home domain (2+
  domain terms, the normal not-a-bare-word bar — the tool refuses a generic
  one outright). Use **grounded** so an off-topic home-domain result trips
  `RETRY-REQUIRED` and you fix the anchor *before* harvesting from it. A partial
  grounded collision degrades with filtered usable evidence; only no usable
  anchor evidence remains `RETRY-REQUIRED` —
  exploratory mode would accept a bad anchor as `LOW-OVERLAP` and its
  contaminated vocabulary would poison every pivot term you pull. The
  overlap check needs ≥3 papers to fire, so if the anchor returns only one
  or two, `OK` is not a clean bill — eyeball those titles yourself before
  harvesting.
  Form 2–3 home-domain semantic lanes and pass them as `query_lanes` with
  `purpose="methods"`; do not create variants by only deleting words.
- **Harvest.** From the anchor's returned abstracts, pick 1–2 *method*
  terms — a named algorithm, transform, or update rule that recurs but was
  *not* in your query (e.g. "Sinkhorn iteration", "low-rank matrix
  factorization" — never a bare word like "matmul"). The method layer is
  what transfers across fields, and harvesting it from retrieved text
  grounds the pivot in evidence, not guesswork. The pivot term must itself
  be distinctive — a pivot on "machine learning" connects everything,
  which is the same as connecting nothing.
- **Pivot (`mode="exploratory"`).** Re-query on the method term(s) alone,
  home anchor removed. Exploratory mode relaxes the low-overlap check (a
  deliberate cross-domain probe *should* read off-domain, so it reports
  `LOW-OVERLAP` instead of forcing a retry) and turns on the field map
  (built into exploratory mode). The scatter grounded mode would
  call noise is now the deliverable (verified: "Sinkhorn iteration optimal
  transport" with no field guess returns mostly ML papers plus a spatial-
  transcriptomics cell-biology paper using the identical machinery). If an
  idea *already* evokes a specific field (psychology, materials science,
  ecology…), you may also pivot straight into that field's terms.
- **Map.** The pivot digest's `### Field map` section (OpenAlex subfield
  distribution) shows which fields the method spans, ranked. Read it as a
  menu of adjacent domains that published work connects to your problem — a
  field present there is kin by evidence. **Absence is inconclusive, not
  proof of no connection:** the map shows only the top-ranked subfields, so
  a genuinely related field can sit just below the cut — probe it directly
  before ruling kinship out. Narrate *why* the domain you pull from is
  structurally kin, so a bad domain choice stays visible.

**Act on the `Research status:` line** (the digest's first line) before
anything else, and **quote it verbatim** in your Step 6 narration:
`RETRY-REQUIRED` may carry a suggested query for a query-shape or all-zero
result (shown as `· try: "…"`) — when present, resubmit it before concluding
"no prior art".
`Research needs action: true` and `### Required research actions` list every
exact operation: preserve the query for `RETRY-SAME`, or supply different
domain terms for `RE-ANCHOR`. `DEGRADED` means a source exhausted its bounded,
delayed retry but usable peer evidence remains; this includes collisions beside
usable on-topic rows, including an on-topic paper row when paper sources were
queried. Proceed on filtered anchor evidence and disclose each rejected
source/lane row without repeating the mechanical retry by hand. Exploratory pivots retain deliberate
`LOW-OVERLAP` candidates; do not filter them as collisions. `LOW-OVERLAP` is expected for a pivot
and needs no retry; `CONFIG` is a key/access problem no retry fixes.
Mechanically-fixable failures are retried inside the tool — a `↻` note means a
first-attempt error was handled, not hidden. A domain-legitimate 0 (GitHub/HuggingFace on a
non-software topic or another source marked `SOURCE-MISMATCH`) is a real
answer, not a whiff. Read every row in `### Source/lane status`; one combined
verdict must not hide `THIN`, `QUERY-COLLISION`, `SOURCE-MISMATCH`,
`INFRASTRUCTURE`, or `CONFIG`; `OK` itself means no required action remains.
`QUERY-COLLISION` is semantic failure: never mechanically shorten it; use a
semantic re-anchor with different terminology. If research still
comes back empty after the retry protocol, start the dream stage unseeded
rather than not at all. Keep the research-quality note in your own Step 6
narration, never in the dream pool itself in Step 5, which stays
agent-facing content only.

Read results for **portable structure, not precedent** — this is not
brainstorm's "has someone already solved this," it's "does this problem's
data or computation share a shape with something another domain already
grapples with, and what did that domain build to handle it." A hit from a
far-flung field is not noise to filter out — it's the find, so the
sanity-check here is different from brainstorm's: reject a hit only when
it shares no structural kinship with the problem at all (a pure keyword
collision, not a genuine analogy), never merely because it comes from
outside your literal subject area or because its field label doesn't
match the project's own. A project framed as "this is about diffusion
models" turning up a matmul-based update rule in a hydrology paper is not
disqualified by that category mismatch — judge it by whether the
underlying computation transfers, not whether the field labels match.
Attention and self-attention trace to psychology's "cocktail party
effect": nothing like a transformer on the surface, but the same
structural problem — selectively weighting competing signals — once
decomposed. Look for that kind of structural kinship with latitude: a
technique or data shape that could transfer with adaptation, not a
literal subject-matter match.

## Step 3 — Dream (start the divergent agents, seeded with the pivot digest)

Call `mcp__plugin_code-quorum_quorum__q_brainstorm_start` with:
- `topic`: the topic text
- `cwd`: the absolute project working directory
- `host`: `"claude"`
- `agents`: `["codex", "opencode", "gemini"]`
- `roles`: `["visionary:codex", "visionary:opencode", "pioneer:gemini"]`
- `research`: the **pivot digest** markdown from Step 2 (the cross-domain
  evidence + field map), passed **verbatim** — the raw `q_research` return
  value, unedited. Never your summary of it, and never your own Step-1 ideas
  (those stay behind the anti-bias gate). If several pivots ran, pass the one
  with the strongest cross-domain signal (or concatenate two short ones).
  Omit under `--no-research`, or when the final `Research status:` is still
  `RETRY-REQUIRED` after the retry protocol (known-bad evidence must not
  anchor the dream stage). Seed `DEGRADED` while preserving its outage
  disclosure. `LOW-OVERLAP` on a pivot is expected — seed it. A digest whose
  status is `OK` with zero hits from domain-legitimate sources also still gets
  seeded — the zero is itself evidence.
- `verbose`: `false` by default — ideas come back **terse** (tight, no padding). Pass `true` only if the user gave `--verbose`.

Pin `agents` explicitly (don't rely on the server default roster) so the dream
stage is always exactly two visionaries + one pioneer, even if the default
council changes.

Two visionaries (feasibility off, non-obvious connections) and one pioneer
(bold but buildable — smallest real first step). The seeded digest gives them
the same evidence you mapped — the published cross-domain structure to dream
from — without your interpretation of it. You get a `job_id` immediately; the
agents run in the background — you cannot see their output yet (the anti-bias
gate).

## Step 4 — Retrieve the dream pool

Call `mcp__plugin_code-quorum_quorum__q_await` with the Step 3 `job_id`.

The first line is the council liveness summary (`Council: …`). **Relay it** — say
in one line which members fired and worked, including any that returned nothing or
errored, so a quiet visionary is not mistaken for a silent failure.

If any expected seat is missing or failed, also report a line beginning
`Council degraded:` that names each unavailable seat and its status. Continue
with the available seats, but never describe the spread as complete or full
when the council is degraded.

## Step 5 — Ground (validation pass over the pool)

Assemble the dream pool: your ideas + the agents' ideas, as a short bulleted
list — **ideas only, no prior art**. Keep the research-quality note out of
this pool too: the whole pool reaches the analyst wrapped in "ideas already
on the table" framing, so anything placed inside it reads as an idea to
validate regardless of what it actually is — prior art routed through the
pool would be graded instead of leaned on. The note belongs only in your own
Step 6 narration to the user, not the analyst's prompt. The evidence travels on its
own channel instead: pass the digest via `research` below. Then call
`mcp__plugin_code-quorum_quorum__q_brainstorm_start` again with:
- `topic`: the same topic
- `cwd`: the same working directory
- `host`: `"claude"`
- `agents`: `["codex"]`
- `roles`: `["analyst:codex"]`
- `prior_ideas`: the assembled pool
- `research`: the same pivot digest seeded in Step 3, verbatim (omit under
  the same conditions as Step 3) — so the analyst grounds the threads
  against the actual evidence, not a paraphrase
- `grounding`: `true`
- `verbose`: same as Step 3 — terse by default; `true` only if `--verbose` was given.

This runs a single Codex analyst pass as a validation guide (not a refutation):
for each promising thread, the evidence/controls/experiment that would validate
it, and the gaps in current methods it must close to be validated and sustained.
`q_await` its `job_id` (relay its liveness line too).

## Step 6 — Synthesize (the orchestrator lands a path)

Unlike `q-brainstorm`, skystorm converges at the end. Unless `--no-research`
was passed, open with the one-line research-quality note from Step 2 so the
reader knows how much weight the grounding below can bear. Combine the dream
pool with the analyst grounding and present:
1. The source-attributed spread of ideas (multi-source ideas flagged as worth
   serious consideration; single-source ideas flagged as such; conflicts
   surfaced).
2. For the strongest 2–3 threads, the grounded path: the smallest experiment,
   the controls, and the falsifiability bar (junk-in→junk-out,
   structure-in→explainable-out) — constructive, never a teardown.
3. A short closing synthesis: **how to get this done while honoring the vision** —
   appealing to both the dreamers and the grounding.

As you land each thread, apply **Verify cited evidence** (below): the dream rounds
run on concepts, but a citation the *landed path* rests on is load-bearing and gets
a full read before you present it as grounded — then weight sourced evidence and
native ideation together.

## Verify cited evidence

The grounding pass cites prior art, and agents — including their own web-grounding —
can return plausible-but-wrong external citations: a future-dated or unresolvable
arXiv ID, a URL carrying tracking params (a `?utm_source=` tell that a claim came
from the model's own web search rather than the research digest), or "studies show…"
with no locator. Skystorm is deliberately loose at the front and strict at the back:

- **While dreaming** — concept level is the right tier. A citation seeding a wild
  idea needs only the **concept** confirmed as real and roughly as described;
  demanding full reads here smothers the divergence the mode exists for. Say
  that's the tier, and don't present it as grounding.
- **Once a thread lands** — anything the smallest experiment, a control, a
  falsifiability bar, a quoted number, or an "already solved / never been tried"
  claim rests on is **load-bearing**. The test: *if this source said the opposite,
  would the path change?* If yes: fetch the artifact itself — the arXiv PDF or
  HTML, the Europe PMC full text, the repo's actual files at a pinned commit, the
  model card — and read the parts that decide whether it transfers (method, the
  results table with its N and spread, limitations), then confirm it says what's
  claimed. Existence is **not** confirmation, and an abstract is not a read.

Anything that can't be confirmed at its own tier is down-weighted and marked
`[unverified]`. Never pass a flaky citation through as settled fact.

A failed fetch is a refusal until proven otherwise: registries refuse as well as
answer — arXiv signals its rate limit as HTTP 200 with a bare `Rate exceeded`
body, not an error status — so an identifier that won't resolve is `[unverified]`,
never evidence of fabrication on its own. Only a well-formed "no such record"
answer from the registry counts against it.

Then weight sourced evidence and **native ideation** together: the research informs
the path, it does not set the ceiling. An un-sourced but promising thread — yours or
a dreamer's — must not be crowded out just because no citation backs it yet.

## Notes

- `q_research` is not subject to the anti-bias gate — it is prior art, not peer
  output. But your OWN ideas are: write them before researching (Step 1), so
  they are not biased by reacting to the digest.
- The grounding pass is a single Codex analyst by design; you (the orchestrator)
  add the integrating layer in Step 6. If grounding feels thin, that is the place
  to add depth.
