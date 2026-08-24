---
name: q-validate
description: Have each agent independently review a plan file, then deliberate over additional rounds. Use when you have a draft plan and want critical pushback before committing. Pass the path to the plan file as the argument; add --verbose to lift terse output caps.
---

# Q-Validate — multi-round plan review

## Completion gate

For every council start, `q_await` is the blocking completion notification. After completing the host's independent work, call it in the same turn as the start. Do not end the turn, tell the user you will check later, or leave a live `job_id` pending. If the council is still running, remain in the blocking call until it returns a result or error.

## Arguments

`$ARGUMENTS` is the path to the plan file, optionally with `--verbose` to lift
the terse output caps and/or `--gemini-model <id>` to run the gemini seat on a
specific agy model. Strip `--verbose` and `--gemini-model <id>` from the path
before passing `plan_path`. If the remaining path is empty, ask the user which
plan file to review.

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

Call `mcp__plugin_code-quorum_quorum__q_validate_start` (Claude Code) or
`mcp__quorum_codex__q_validate_start` (Codex) with:

- `plan_path`: a **bare file path only** (absolute, or relative to `cwd`) that
  resolves inside `cwd`; paths outside the working directory are rejected before reading. `q_validate_start`
  `open()`s this path directly, so it CANNOT carry an appended context/boundaries block — passing
  `<path>` + prose errors with `File name too long`. If `$ARGUMENTS` includes reviewer context beyond
  the path, do NOT forward it here; instead put that context, boundaries, or specific questions
  **inside the plan file** (e.g. an `## Open questions for review` section) — the agents read the file.
- `cwd`: the absolute path of the project working directory
- `host`: the current host selected above
- `extended`: `false` by default (2-round review). Set `true` for a 4-round deliberation with a stance rotation at round 3 — use only when the plan is contentious or high-stakes.
- `mode`: `"revise"` (default) — agents soften or strengthen positions in light of peers. Use `"critique"` when you want them to attack each other's points.
- `agents`: optional; omit it for every external seat in the current host profile (a seat the user has marked disabled in `quorum setup-models` is skipped from that default roster). Passing `agents` explicitly overrides the skip and runs a disabled seat anyway — a deliberate ask always runs.
- `verbose`: `false` by default. The council writes **terse** output — no prose padding, `path:line` citations instead of pasted code, and later rounds omit restating what is unchanged. Set `true` only if the user gave `--verbose`; it produces a much larger matrix.
- `gemini_model`: omit by default. Pass an agy model id (exactly as printed by `agy models`) when the user gave `--gemini-model <id>` or asked for a specific model in the gemini seat — e.g. `claude-opus-4-6-thinking` to have that seat answer with Claude on the same AI Pro plan.

You will receive a `job_id` immediately. The agents now review the plan in parallel in the background; you cannot see their output yet. This is intentional — the structural anti-bias gate.

Expected wall-clock to completion: 1–8min for the default 2-round flow; 4–15min for `extended: true` (4 rounds with rotation). Pick `extended` deliberately — the latency adds up, especially with 3+ agents.

## Step 2 — Form your own review while agents run

Read the plan at the stripped `plan_path` and write your own staff-engineer-style review. Produce a written review with:

- Critical flaws or unsound steps.
- Missing edge cases or risks.
- Opportunities for simplification.
- A fundamentally better approach, if any.

Write it down before retrieving agent output.

## Step 3 — Retrieve agent output

Once your review is written, call `mcp__plugin_code-quorum_quorum__q_await`
(Claude Code) or `mcp__quorum_codex__q_await` (Codex) with the `job_id` from Step 1.
Returns the full rounds matrix as markdown.

When the council ran, the result begins with a council liveness summary line (`Council: …`). Relay it in one line — which members fired and worked, and call out any that returned nothing or errored — so a quiet or non-contentious member is not mistaken for a silent failure. Throughout your report, attribute council voices **stance-first** — "the skeptic (codex)", never a bare seat name — matching the transcript's own labels: it shows the user the role assignment that actually ran (deliberation anonymizes peers by stance) and reminds them which stances they can reassign with `--role <stance>:<agent>`.

## Step 4 — Compare and decide

Unlike `q-brainstorm` (which preserves spread), the goal here is convergence on a final go/no-go and a revised plan. For each issue raised across your own review and the agents' rounds:

- Accept it and update the plan, or
- Override with a brief explanation of why the current approach is better.

If an agent flips position between rounds, note both the original and revised stance — divergence often points at a real ambiguity in the plan.

Treat agent feedback as input from a junior engineer. You are the lead and have final say.

## High-signal cases to watch for

- **Factual disagreement between agents about the codebase.** If one agent says "X is implemented in `foo.py`" and another says "there is no `foo.py`" — at least one is hallucinating. Open the file and verify before acting on either suggestion. Agents disagreeing on opinion is normal; disagreeing on facts almost always points at hallucination.
- **Unanimous concern from all agents on a point you missed.** Worth treating as a near-certain real issue rather than a coincidence.

## Verify cited evidence

Beyond codebase facts, agents — and their own web-grounding — can return plausible-but-wrong external citations: a future-dated or unresolvable arXiv ID, a URL carrying tracking params (a `?utm_source=` tell that the link was copied off a search-results page rather than opened), or "studies show…" with no locator. This mode fetches no research digest, so every external citation came from an agent's own web grounding. For any citation load-bearing to a conclusion, confirm it cheaply — the paper/repo exists and says what's claimed — or down-weight it and mark it `[unverified]`. Never pass a flaky citation through as settled fact.
