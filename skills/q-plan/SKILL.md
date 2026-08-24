---
name: q-plan
description: Generate alternative implementation plans from external council seats while the current Claude or Codex host writes its own independent plan. Use when you want second opinions before committing to a plan. Pass the task description as the argument; add --verbose to lift terse output caps.
---

# Q-Plan — parallel plans from each registered agent

## Completion gate

For every council start, `q_await` is the blocking completion notification. After completing the host's independent work, call it in the same turn as the start. Do not end the turn, tell the user you will check later, or leave a live `job_id` pending. If the council is still running, remain in the blocking call until it returns a result or error.

## Arguments

`$ARGUMENTS` is the task description, optionally with `--verbose` to lift the
terse output caps and/or `--gemini-model <id>` to run the gemini seat on a
specific agy model. Strip `--verbose` and `--gemini-model <id>` from the task
text before passing `task`. If the remaining task is empty, ask the user what
they want a plan for.

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

Call `mcp__plugin_code-quorum_quorum__q_plan_start` (Claude Code) or
`mcp__quorum_codex__q_plan_start` (Codex) with:

- `task`: the value of `$ARGUMENTS`
- `cwd`: the absolute path of the project working directory
- `host`: the current host selected above
- `verbose`: `false` by default — plans come back **terse** (no padding, `path:line` over pasted code). Set `true` only if the user gave `--verbose`.
- `gemini_model`: omit by default. Pass an agy model id (exactly as printed by `agy models`) when the user gave `--gemini-model <id>` or asked for a specific model in the gemini seat — e.g. `claude-opus-4-6-thinking` to have that seat answer with Claude on the same AI Pro plan (useful when Gemini quota is tight or a Claude perspective is wanted).

You will receive a `job_id` immediately. The agents now run in parallel in the background; you cannot see their output yet. This is intentional — the structural anti-bias gate.

Expected wall-clock to completion: 30s–4min depending on the agents and codebase size.

## Step 2 — Draft your own plan while agents run

In your response, draft your own implementation plan for the stripped task text. Cover architecture, ordered steps, edge cases, and trade-offs. Write it down before retrieving agent output. The parallel execution means this costs you no wall-clock time.

## Step 3 — Retrieve agent output

Once your plan is written, call `mcp__plugin_code-quorum_quorum__q_await`
(Claude Code) or `mcp__quorum_codex__q_await` (Codex) with the `job_id` from Step 1.
Returns the agents' rounds matrix as markdown.

When the council ran, the result begins with a council liveness summary line (`Council: …`). Relay it in one line — which members fired and worked, and call out any that returned nothing or errored — so a quiet or non-contentious member is not mistaken for a silent failure. Throughout your report, attribute council voices **stance-first** — "the skeptic (codex)", never a bare seat name — matching the transcript's own labels: it shows the user the role assignment that actually ran (deliberation anonymizes peers by stance) and reminds them which stances they can reassign with `--role <stance>:<agent>`.

## Step 4 — Compare and integrate

Unlike `q-brainstorm` (which preserves spread), the goal here is convergence on a single plan. Compare each agent's plan against your own. For each non-trivial point either agent raised:

- Accept it and update your plan, or
- Override it with a brief explanation of why the current approach is better.

Treat agent suggestions as input from a junior engineer. You are the lead and have final say.

## High-signal cases to watch for

- **Factual disagreement between agents about the codebase.** If one agent says "X is implemented in `foo.py`" and another says "there is no `foo.py`" — at least one is hallucinating. Open the file and verify before acting on either suggestion. Agents disagreeing on opinion is normal; disagreeing on facts almost always points at hallucination.
- **A point neither you nor the other agent caught.** Worth scrutinizing — it's either a real insight or a confident fabrication.

## How to treat responses

- Never assume an agent's suggestion is correct. Validate against the codebase.
- Use suggestions as a starting point, not authoritative answers.
- If an agent points at a file or pattern, open it and verify before acting on the suggestion.

## Verify cited evidence

Beyond codebase facts, agents — and their own web-grounding — can return plausible-but-wrong external citations: a future-dated or unresolvable arXiv ID, a URL carrying tracking params (a `?utm_source=` tell that the link was copied off a search-results page rather than opened), or "studies show…" with no locator. This mode fetches no research digest, so every external citation came from an agent's own web grounding. For any citation load-bearing to a conclusion, confirm it cheaply — the paper/repo exists and says what's claimed — or down-weight it and mark it `[unverified]`. Never pass a flaky citation through as settled fact.
