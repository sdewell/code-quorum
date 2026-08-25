---
name: q-brainstorm
description: Generate divergent ideas from external council seats while the current Claude or Codex host writes independently, grounded in prior art by default (research-first). Use when you want a wide spread of testable options before committing to an approach. Pass the topic as the argument; add --no-research to skip the prior-art grounding, --extended for a second divergence round, or --verbose to lift terse output caps.
---

# Q-Brainstorm — divergent ideas from each registered agent

## Completion gate

For every council start, `q_await` is the blocking completion notification. After completing the host's independent work, call it in the same turn as the start. Do not end the turn, tell the user you will check later, or leave a live `job_id` pending. If the council is still running, remain in the blocking call until it returns a result or error.

Research-first: unless `--no-research` is passed, the run grounds itself in
prior art (arXiv + OpenAlex + Europe PMC published/preprints + Context7 +
GitHub + HuggingFace)
*before* the council generates, and the digest is seeded into the agents'
round-1 prompt as evidence. It is a research mode — treat the whole run as one.

## Arguments

`$ARGUMENTS` is the brainstorm topic, optionally with flags:
- `--no-research` — skip the prior-art grounding (research is the **default**).
- `--extended` — add a host-driven second round that diverges past round 1.
- `--verbose` — lift the terse output caps. Default is terse.
- `--gemini-model <id>` — run the gemini seat on a specific agy model.

Strip the flags from the topic text before passing `topic` to the tools. If the
topic (after removing flags) is empty, ask the user what they want to brainstorm.

## Host routing

Claude Code uses the `mcp__plugin_code-quorum_quorum__*` tools and passes
`host: "claude"`. Codex uses the `mcp__quorum_codex__*` tools and passes
`host: "codex"`. The workflows are otherwise identical.

If either required MCP tool is absent from the current task's tool registry,
stop and report that the plugin MCP server did not load. Do not substitute the
synchronous CLI, direct Python imports, or another runtime path: only the
registered start/await pair enforces the structural anti-bias gate.

Every MCP council start requires an explicit absolute `cwd`. Never omit it: the
MCP server's own working directory is its installation or runtime location, not
the project selected by the user.

## Step 1 — Your own ideas FIRST

Before researching and before starting the council, list 3–5 distinct ideas of
your own for the topic, each with a brief rationale, main trade-off, and the
cheapest test that would give signal. Keep them varied — don't converge
prematurely.

The order is the anti-bias gate in both directions: your ideas are committed
*before* the digest can anchor them (they must be genuinely yours, not
reactions to research), and the council never sees them at all.

## Step 2 — Research

Unless `--no-research` was passed (in which case skip this whole step), call
`mcp__plugin_code-quorum_quorum__q_research` (Claude Code) or
`mcp__quorum_codex__q_research` (Codex). It returns a markdown digest of
papers, library docs, GitHub repos, and HuggingFace models). This runs BEFORE
the council so round 1 can generate from the evidence.

**Form the query short *and distinctive*; short is not the same as generic.** Anchor it in **two or
more domain-specific terms** (the field *plus* the specific method/concept) so it can only match your
domain. A bare common token — `data`, `model`, `network`, `signal` — or a word that doubles as an
author surname (`Sun`, `Li`) collides and returns **confident-looking noise** (author names, generic
"A Survey of …" titles); the tool refuses the most generic ones outright. Example: for alt-data
finance, ❌ `data` / `alternative data` → ✓ `alternative data equity return prediction`. You don't
need to hand-tune per backend — the tool shapes each source's query and strips arXiv's boolean
operators for you.

Form 2–3 semantic query lanes and pass them as `query_lanes`: domain +
construct, failure/validity, and review/guideline terminology. Pass
`purpose="methods"` so OpenAlex balances all-time and recent candidates. Do
not manufacture lanes by repeatedly deleting words from one query.

**The digest opens with a `Research status:` line — act on it before anything else, and quote it in
Step 5.** It is the deterministic quality verdict:
- `RETRY-RECOMMENDED` — the query whiffed or collided with unrelated work (off-topic hits, or every
  paper source empty on a valid query). It **usually** carries a suggested shorter query (shown as
  `· try: "…"`) — when present, you must call `q_research` again with it before concluding "no
  prior art". When it is **absent**, do exactly what the detail text after the dash says — the two
  absent cases need opposite moves: on a **backend/infrastructure failure** (sources errored or
  timed out) the detail says *retry* — resubmit the **same** query, because changing terms cannot
  fix an outage; when the query simply **cannot be shortened** the detail says *re-anchor* —
  resubmit with DIFFERENT domain-specific terms of your own. Either way the escape
  hatch is gated behind an actual retry: shrugging the result off and falling back on your
  own knowledge — *"the research backend choked, I'll use what I know"* —
  is the **specific failure to avoid**.
- `CONFIG` — a key/anonymous-access problem no retry fixes; proceed on the other sources and say so.
- `OK` — results look on-topic; proceed.

Immediately inspect `### Source/lane status`. Its rows are `ON-TOPIC`, `THIN`,
`QUERY-COLLISION`, `SOURCE-MISMATCH`, `INFRASTRUCTURE`, or `CONFIG`. A combined
OK does not erase a weak row. On `QUERY-COLLISION`, use the suggested mechanical
shortening once; if that lane still collides, use a semantic re-anchor with
different terminology. Retry an `INFRASTRUCTURE` row with the same lane, fix or
disclose `CONFIG`, and accept `SOURCE-MISMATCH` rather than forcing a source to
fit the domain.

Mechanically-fixable failures (an arXiv 400, its 200-with-`Rate exceeded` rate refusal, a
transient flake) are **already retried inside the
tool** — a `↻` note marks a source that a first-attempt error you never saw was repaired on, not
hidden. A `SOURCE-MISMATCH` row is a real answer, not a whiff to rework. Only
after a **reworked* retry has **also** failed** may you note a
source unavailable and continue (never block on it) — start the council unseeded rather than not
at all.

## Step 3 — Start the agents (non-blocking, seeded with the digest)

Call `mcp__plugin_code-quorum_quorum__q_brainstorm_start` (Claude Code) or
`mcp__quorum_codex__q_brainstorm_start` (Codex) with:
- `topic`: the topic text
- `cwd`: the absolute path of the project working directory
- `host`: the current host selected above
- `research`: the digest markdown from Step 2, passed **verbatim** — the raw
  `q_research` return value, unedited. Never your summary of it, and never
  your own Step-1 ideas (those stay behind the anti-bias gate). The server
  frames it for the agents as evidence to build from. Omit under
  `--no-research`, or when the final `Research status:` is still
  `RETRY-RECOMMENDED` after the retry protocol (known-bad evidence must not
  anchor the council). A digest whose status is `OK` with zero hits from
  domain-legitimate sources still gets seeded — the zero is itself evidence.
- `verbose`: `false` by default — ideas come back **terse** (tight
  rationale/trade-offs, no padding). Set `true` only if the user gave
  `--verbose`.
- `gemini_model`: omit by default. Pass an agy model id (exactly as printed by
  `agy models`) when the user gave `--gemini-model <id>` or asked for a
  specific model in the gemini seat — e.g. `claude-opus-4-6-thinking` to have
  that seat answer with Claude on the same AI Pro plan.

You will receive a `job_id` immediately. The agents now generate ideas in
parallel in the background; you cannot see their output yet.

Expected wall-clock to completion: 30s–4min.

## Step 4 — Retrieve agent output

Call `mcp__plugin_code-quorum_quorum__q_await` (Claude Code) or
`mcp__quorum_codex__q_await` (Codex) with the `job_id` from Step 3.

When the council ran, the result begins with a council liveness summary line
(`Council: …`). Relay it in one line — which members fired and worked, and call
out any that returned nothing or errored — so a quiet or non-contentious member
is not mistaken for a silent failure. Throughout your report, attribute council
voices **stance-first** — "the visionary (gemini)", never a bare seat name —
matching the transcript's own labels: it shows the user the role assignment
that actually ran (deliberation anonymizes peers by stance) and reminds them
which stances they can reassign with `--role <stance>:<agent>`.

## Step 5 — Combine and present

The goal is breadth. Combine your ideas with the agents'. Surface:
- Ideas multiple sources raised independently — likely worth serious consideration.
- Ideas only one source raised — interesting but flag the singular origin.
- Conflicts between ideas — note them; don't paper over.
- Each idea's cheapest test — the collection should be comparable and
  deployable, not open-ended.

**Unless `--no-research` was passed**, open with the one-line research-quality
note from Step 2 (quote the `Research status:` line the way the `Council: …`
liveness line gets relayed, plus one line on per-source signal strength — so a
domain-legitimate whiff is never mistaken for missing prior art, and noise is
never mistaken for confirmed grounding). Then ground the spread in the prior
art: point out where existing methods or results could be *recombined* for
this topic ("don't reinvent geometry to make an app"), and flag any idea the
prior art shows is already solved or well-trodden. Keep the spread — research
informs, it does not collapse the options.

Don't synthesize into a single recommendation unless the user asks.

## Step 6 — Extended divergence round (only if `--extended`)

After Step 5, run one more round that pushes past everything on the table:

1. Assemble the **pre-existing ideas pool**: all round-1 ideas (yours + the
   agents'), as a short bulleted list. Keep the research-quality note out of
   this pool — `prior_ideas` reaches the agents wrapped in "ideas already on
   the table, do NOT repeat these" framing, so anything placed inside it
   reads the same way regardless of where you put it; the note belongs only
   in your own Step 5 narration to the user, not the agents' prompt. The
   digest itself does NOT go in this pool either — it rides the `research`
   param again (evidence framing), never the do-not-repeat wrap.
2. Write your own round-2 divergent/synthesis ideas in your visible response —
   BEFORE any sharper research call, same invariant as round 1 (own ideas
   precede research so they are not reactions to a new digest; the council
   never sees them).
3. Optionally — unless `--no-research` was passed — if round 1 revealed a
   sharper query, call `q_research` again with it now.
4. Call `mcp__plugin_code-quorum_quorum__q_brainstorm_start` (Claude Code) or
   `mcp__quorum_codex__q_brainstorm_start` (Codex) again with the same `topic`,
   `cwd`, and `host` as Step 3, plus `prior_ideas` set to that pool, `research` set to the same
   digest as Step 3 of the main flow (or the sharper one from the previous
   item), and `verbose` set to the same boolean as Step 3. This tells the
   agents not to repeat what's listed and to diverge past it, with the
   evidence still in view.
5. `q_await` the new `job_id`. Its result also begins with a `Council: …`
   liveness line — relay it in one line too (same as Step 4), so a quiet or
   failed member in this second round is not mistaken for a silent failure.
6. Present round 1 + round 2 together, foregrounding genuinely new directions,
   novel syntheses, and concrete ways to research, test, or establish
   feasibility for the strongest threads.

## Verify cited evidence

Agents — and their own web-grounding — can return plausible-but-wrong citations: a
future-dated or unresolvable arXiv ID, a URL carrying tracking params (a
`?utm_source=` tell that a claim came from the model's own web search rather than
the research digest), or "studies show…" with no locator. How hard to check depends
on what the citation is holding up — and brainstorm output is meant to be acted on,
so the bar here is higher than in blue-sky ideation:

- **Load-bearing** — the citation is doing evidentiary work: a quoted number or
  benchmark, a claimed result, a method presented as proven, or "this is (or
  isn't) already solved". The test: *if this source said the opposite, would the
  option change?* If yes: fetch the artifact itself — the arXiv PDF or HTML, the
  Europe PMC full text, the repo's actual files at a pinned commit, the model
  card — and read the parts that decide whether it transfers (method, the results
  table with its N and spread, limitations), then confirm it says what's claimed.
  Existence is **not** confirmation: a real paper is easy to cite for something it
  never said, and an abstract is not a read.
- **Idea-seeding** — the citation names a technique or gestures at a concept that
  shapes an option, and nothing rests on it yet. Confirming the **concept** is
  real and roughly as described is enough. Label it that tier rather than dressing
  it up as grounding.

Promote on the way out, not just on the way in: any citation still doing
evidentiary work in the spread you present has crossed into load-bearing, and gets
the full read before it goes in. Anything that can't be confirmed at its own tier
is down-weighted and marked `[unverified]`. Never pass a flaky citation through as
settled fact.

A failed fetch is a refusal until proven otherwise: registries refuse as well as
answer — arXiv signals its rate limit as HTTP 200 with a bare `Rate exceeded`
body, not an error status — so an identifier that won't resolve is `[unverified]`,
never evidence of fabrication on its own. Only a well-formed "no such record"
answer from the registry counts against it.

## Notes

- `q_research` is not subject to the anti-bias gate — it is external prior art,
  not peer output. But your OWN ideas are: write them before researching, so
  they are not biased by reacting to the digest.
- The bare `quorum q-brainstorm` CLI mirrors the research-first default
  (`--no-research` to skip) but not `--extended`, which is this
  host-orchestrated flow.
