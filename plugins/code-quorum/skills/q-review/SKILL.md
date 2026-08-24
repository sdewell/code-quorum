---
name: q-review
description: Have each agent independently review real code changes (a branch, an open PR, a commit range, or the whole codebase), then converge over multiple rounds into a single findings table. Use when you want critical pushback on committed-but-unmerged work before it lands. Pass the target spec as the argument; add --scope with a path to bound the review or --verbose to lift terse output caps.
---

# Q-Review — multi-round code review

## Completion gate

For every council start, `q_await` is the blocking completion notification. After completing the host's independent work, call it in the same turn as the start. Do not end the turn, tell the user you will check later, or leave a live `job_id` pending. If the council is still running, remain in the blocking call until it returns a result or error.

## Arguments

`$ARGUMENTS` is the **target spec** — what to review — optionally with flags. If
empty, it defaults to all work on this branch versus `main` (committed **and**
uncommitted). Recognized target forms:

- *(empty)* — the whole branch vs `main`: merge-base → working tree, including uncommitted tracked changes.
- `working` — uncommitted tracked changes only.
- `pr:123` or a PR URL — an open PR; its title/body/labels orient the review extent.
- `A..B` / `A...B` — an explicit commit range (two-dot literal / three-dot merge-base, PR-like).
- `all` — the **whole codebase**; agents read the repo at `cwd`. Pair this with `--scope`.

If the user supplied a `--scope <path>`, capture it for Step 1. When reviewing `all` with **no** scope doc, tell the user a scope doc is strongly recommended and point them at `skills/q-review/references/scope-template.md` (copy it into the repo as `code-quorum-review-scope.md` and pass `--scope code-quorum-review-scope.md`).

If the user supplied `--verbose` or `--gemini-model <id>`, capture them for
Step 1. Strip `--scope <path>`, `--verbose`, and `--gemini-model <id>` from the
target before passing `target`; otherwise the tool will try to resolve the flag
text as part of the review target.

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

## Step 1 — Start the agents (non-blocking)

Call `mcp__plugin_code-quorum_quorum__q_review_start` (Claude Code) or
`mcp__quorum_codex__q_review_start` (Codex) with:

- `target`: the stripped target spec (empty string for the default branch-vs-`main` review).
- `cwd`: the absolute path of the project working directory.
- `host`: the current host selected above.
- `scope_path`: the `--scope` path if the user supplied one. It may be absolute
  or relative, but after resolution it must remain inside `cwd`; otherwise the
  tool rejects it before reading. Otherwise omit it.
- `extended`: `false` by default (2-round review). Set `true` for a 4-round deliberation with a stance rotation at round 3 — use only when the change is large, contentious, or high-stakes.
- `mode`: `"revise"` (default) — agents soften or strengthen positions in light of peers, converging on agreed findings. Use `"critique"` when you want them to attack each other's points.
- `agents`: optional; omit it for every external seat in the current host profile (a seat the user has marked disabled in `quorum setup-models` is skipped from that default roster). Passing `agents` explicitly overrides the skip and runs a disabled seat anyway — a deliberate ask always runs.
- `roles`: optional per-agent stances, e.g. `["maintainer:opencode"]` or `["security:gemini"]`. The default skeptic is the other platform's seat; Gemini is architect and OpenCode is neutral.
- `verbose`: `false` by default. The council writes **terse** output — no prose padding, `path:line` citations instead of pasted code, and in the **final** round each still-held finding collapses to one `HELD …` line (the agreement count is preserved). Intermediate rounds of an `--extended` run re-list in full so later rounds deliberate from evidence, not stubs. Set `true` only if the user gave `--verbose` (every held finding is re-listed in full each round) — it produces a much larger matrix.
- `gemini_model`: omit by default. Pass an agy model id (exactly as printed by `agy models`) when the user gave `--gemini-model <id>` or asked for a specific model in the gemini seat — e.g. `claude-opus-4-6-thinking` to have that seat answer with Claude on the same AI Pro plan (useful when Gemini quota is tight or a Claude perspective is wanted).

You will receive a `job_id` immediately. The agents now review the change in parallel in the background; you cannot see their output yet. This is intentional — the structural anti-bias gate.

If the target has no changes (an empty diff, other than `all`), the job short-circuits and `q_await` returns a "nothing to review" message — surface it and stop; there is nothing to converge on.

Expected wall-clock to completion: 1–8min for the default 2-round flow; 4–15min for `extended: true` (4 rounds with rotation). Pick `extended` deliberately — the latency adds up, especially with 3+ agents.

## Step 2 — Form your own review while agents run

Before retrieving any agent output, read the change yourself and write your **own** independent code review of the diff (for `all`, read the relevant files at `cwd`). Produce a written review with:

- Concrete bugs, regressions, and security holes — `FILE: path:line`, severity (`critical` / `high` / `medium` / `low`), what is wrong and why, and the fix.
- Missing edge cases or risks.
- Structural / blast-radius concerns.

If a scope doc was supplied, apply your review only to the in-scope surface; note anything out-of-scope or an accepted risk once, but do not treat it as a finding.

Write it down before retrieving agent output.

## Step 3 — Retrieve agent output

Once your review is written, call `mcp__plugin_code-quorum_quorum__q_await`
(Claude Code) or `mcp__quorum_codex__q_await` (Codex) with the `job_id` from Step 1.
Returns the full rounds matrix as markdown.

When the council ran, the result begins with a council liveness summary line (`Council: …`). Relay it in one line — which members fired and worked, and call out any that returned nothing or errored — so a quiet or non-contentious member is not mistaken for a silent failure. Throughout your report, attribute council voices **stance-first** — "the skeptic (codex)", never a bare seat name — matching the transcript's own labels: it shows the user the role assignment that actually ran (deliberation anonymizes peers by stance) and reminds them which stances they can reassign with `--role <stance>:<agent>`.

## Step 4 — Converge into a findings table

Merge your own review with the agents' rounds matrix into **one** converged result. Cluster findings by **file + claim** (the line number is a *hint*, not the merge key — line positions shift across hunks and rounds). For each cluster, union the agents that flagged it and derive an agreement count (N of M).

In the default terse output, a later-round `HELD <path:line> | <severity> | <claim>` line means that agent **still holds** a finding it stated in full in an earlier round — resolve it back to that earlier finding and count it toward agreement, exactly as a full re-statement would. A finding that stops appearing (no full entry and no `HELD` line) was dropped; a `CHANGING`/`RETRACTING` note updates or removes it. (Under `verbose`, findings are re-listed in full each round and there are no `HELD` lines.)

Emit, in this order:

1. **A verdict line on top** — exactly one of `approve` / `approve-with-fixes` / `request-changes`. This answers "is this branch ready to merge?"
2. **A findings table** with these columns:

   | FILE:line | SEVERITY | FINDING | flagged-by | SCOPE | RECOMMENDATION |
   |---|---|---|---|---|---|

   - **SEVERITY** is one of `critical` / `high` / `medium` / `low`.
   - **flagged-by** is the convergence count, `N of M` agents (count your own review as a separate signal, not one of the M agents).
   - **SCOPE** is `in` or `out`. If a scope doc declares a finding out of bounds (out-of-scope, a known edge case, or an accepted risk — agents tag these `[OUT-OF-SCOPE]` on the FINDING line), **strike the row and exclude it from the converged set**: render it struck through and marked `[OUT-OF-SCOPE]`, do not let it influence the verdict. You make the final keep/exclude call — a mis-scoped-but-real critical bug is still worth surfacing, just outside the converged table.

If agreement is low (most findings flagged by only 1 of M, or agents disagree on whether something is a finding at all), **recommend an `--extended` re-run** so the council spends more rounds converging.

## High-signal cases to watch for

- **Factual disagreement between agents about the codebase.** If one agent says "X is broken in `foo.py:42`" and another says "there is no such code" — at least one is hallucinating. Open the file and verify before acting on either. Agents disagreeing on opinion is normal; disagreeing on facts almost always points at hallucination.
- **Unanimous concern from all agents on a point you missed.** Worth treating as a near-certain real issue rather than a coincidence — a 3/3 finding is the strongest signal in the table.

Treat agent feedback as input from a junior reviewer. You are the lead and have final say on the verdict.

## Verify cited evidence

Beyond codebase facts, agents — and their own web-grounding — can return plausible-but-wrong external citations: a future-dated or unresolvable arXiv ID or CVE, a URL carrying tracking params (a `?utm_source=` tell that the link was copied off a search-results page rather than opened), or "the docs say…" with no locator. This mode fetches no research digest, so every external citation came from an agent's own web grounding — there is no vetted alternative behind it, and how hard you check is the only thing standing between a plausible-looking citation and a finding someone acts on.

- **Load-bearing** — a finding rests on it: a CVE or advisory, a quoted affected-version range, "the docs say this API behaves like X", a benchmark number, or "this pattern is known-broken upstream". The test: *if this source said the opposite, would the finding change?* If yes: fetch the artifact itself — the advisory record, the doc page **at the version this code actually uses**, the upstream repo's code at a pinned commit, the paper's PDF — and read the part that decides whether it applies to *this* code, then confirm it says what's claimed. Existence is **not** confirmation, and that is the specific trap here: a real CVE cited against a version range this code isn't in, or a real doc page for the wrong major version, is a false finding wearing a working link.
- **Idea-seeding** — the citation names a technique or points at prior art for context, and no finding rests on it. Confirming the **concept** is real and roughly as described is enough. Label it that tier — it is not allowed to become the thing that carries a finding.

Anything that can't be confirmed at its own tier is down-weighted and marked `[unverified]`: it may ride along in the FINDING column as an unverified note, but on its own it must not justify a `critical`/`high` severity or a `request-changes` verdict. Never pass a flaky citation through as settled fact.

"Unresolvable" is itself a claim to verify: registries refuse as well as answer — arXiv signals its rate limit as HTTP 200 with a bare `Rate exceeded` body, not an error status — so a fetch that fails or returns a malformed body makes the identifier `[unverified]`, never fabricated on its own. Only a well-formed "no such record" answer from the registry counts against it.
