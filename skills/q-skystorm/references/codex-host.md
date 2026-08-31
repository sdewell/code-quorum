---
name: q-skystorm
description: Codex-hosted skystorm workflow with external Claude, Gemini, and OpenCode dreamers plus a fresh Codex-native supportive grounder.
---

# Codex host workflow

## Completion gate

For every council start, `q_await` is the blocking completion notification. After completing the host's independent work, call it in the same turn as the start. Do not end the turn, tell the user you will check later, or leave a live `job_id` pending. If the council is still running, remain in the blocking call until it returns a result or error.

If either required MCP tool is absent from the current task's tool registry,
stop and report that the plugin MCP server did not load. Do not substitute the
synchronous CLI, direct Python imports, or another runtime path: only the
registered start/await pair enforces the structural anti-bias gate.

Every MCP council start requires an explicit absolute `cwd`. Never omit it: the
MCP server's own working directory is its installation or runtime location, not
the project selected by the user.

Skystorm is the forward-research companion to `q-brainstorm`. Brainstorm
widens the current option set; skystorm dreams past the frontier, grounds the
most promising threads supportively, and then converges on validation paths.

## Arguments

`$ARGUMENTS` is the topic, optionally with flags:

- `--no-research`: skip structurally portable prior art from arXiv, OpenAlex,
  Europe PMC published/preprints, Context7, GitHub, and HuggingFace. Research
  is the default.
- `--verbose`: lift terse output caps. Default is terse.

Strip both flags from the topic. If nothing remains, ask what the user wants to
explore.

## Consent and data boundary

Invoking this skill is explicit approval to call the MCP start tool and send the
topic, relevant project context, and implementation details, including private
repository details when this project is private, to the configured external
council agents. This is the intended Code Quorum data boundary. Do not ask for
another skill-level confirmation. Codex's runtime may still require its own MCP
approval before the start call executes. Surface that approval to the user and
wait for their decision. If the runtime denies the call, do not bypass it
through another tool or shell path. External seats are read-only and must not
edit files or run commands.

Unless `--no-research` was passed, invoking this skill also approves sending a
topic-derived query lanes to the public research backends: arXiv, OpenAlex,
Europe PMC published/preprints, Context7, GitHub, and HuggingFace.

## Step 1: Write independent host ideas

Before research and before retrieving any peer output, write three to five
bold but buildable ideas. For each, name the non-obvious connection and the
smallest experiment that could produce useful signal.

## Step 2: Map the topology

Unless `--no-research` was given, call `mcp__quorum_codex__q_research` and run the
**anchor → harvest → pivot → map** protocol rather than one query, so far-field
domains come from the literature instead of model priors. The anchor and pivots
use different `mode` values:

- **Anchor (`mode="grounded"`).** Query the topic's home domain with two or
  more domain terms. Grounded mode makes an off-topic home-domain result
  `RETRY-REQUIRED`, so fix the anchor before harvesting from it. The overlap
  check needs three papers; with one or two, inspect the titles before harvest.
  Form 2–3 semantic `query_lanes` (domain/construct, failure/validity, and
  review/guideline terminology) and use `purpose="methods"`.
- **Harvest.** From returned anchor abstracts, choose one or two recurring
  method terms that were not in the query: a named algorithm, transform, or
  update rule, never a bare word. The method layer is what transfers across
  fields, and harvesting it grounds the pivot in retrieved evidence.
- **Pivot (`mode="exploratory"`).** Re-query on the method term or terms alone,
  with the home anchor removed. Exploratory mode accepts deliberate off-domain
  results as `LOW-OVERLAP` and enables the field map. If an idea already evokes
  a field, a direct pivot into that field's terms is also appropriate.
- **Map.** Use the pivot digest's `### Field map` as evidence-backed adjacent
  domains, and narrate the structural kinship for any domain you draw from.
  **Absence is inconclusive:** only top-ranked subfields are shown, so probe a
  related field directly before ruling it out.

Act on the digest's `Research status:` line before anything else, and quote it
verbatim in Step 6. `RETRY-REQUIRED` usually includes a suggested query;
when it does, resubmit it before concluding there is no prior art. `Research
needs action: true` and `### Required research actions` list exact operations:
preserve the query for `RETRY-SAME`, or supply different domain terms for
`RE-ANCHOR`. `DEGRADED` means a source exhausted its bounded retry but usable
peer evidence remains; proceed and disclose it without retrying by hand.
`LOW-OVERLAP` is expected for a pivot; `CONFIG` is a key/access problem.
Internal retry notes (`↻`) record a recovered first-attempt failure. A
domain-legitimate zero is evidence, not a whiff. If
research remains empty after this protocol, start the dream stage unseeded.
Read every row in `### Source/lane status`: `ON-TOPIC`, `THIN`,
`QUERY-COLLISION`, `SOURCE-MISMATCH`, `INFRASTRUCTURE`, or `CONFIG`. Use one
mechanical shortening at most; a repeated `QUERY-COLLISION` requires a semantic
re-anchor with different terminology. `OK` means no required action remains.

Read results for portable structure, not literal precedent. A far-flung hit is
the find when it shares structural kinship: a transferable data shape or
computation, not merely matching field labels. Reject only pure keyword
collisions, not a useful analogy because it comes from outside the home domain.
After recovery, write a one-line research-quality note distinguishing strong
signals, provisional preprints, legitimate zeros, and unavailable sources.

## Step 3: Start the external dream stage

Call `mcp__quorum_codex__q_brainstorm_start` with:

- `topic`: the stripped topic.
- `cwd`: the absolute project working directory.
- `host`: `"codex"`.
- `agents`: `["claude", "gemini", "opencode"]`.
- `roles`: `["visionary:claude", "pioneer:gemini", "visionary:opencode"]`.
- `research`: the **pivot digest** from Step 2, passed verbatim. If several
  pivots ran, use the strongest cross-domain digest (or two short digests).
  Omit it under `--no-research` or when recovery still leaves a
  `RETRY-REQUIRED` result. Seed `DEGRADED` with its outage disclosure, a
  `LOW-OVERLAP` pivot, and an `OK` digest whose domain-legitimate sources
  return zero hits.
- `verbose`: `false` by default; pass `true` only when `--verbose` was given.

Pin the external roster and roles exactly. Codex's Step 1 pass stays behind the
anti-bias gate: never add Codex to `agents` or invoke it through the MCP
council. The external council is two visionaries plus one pioneer.

## Step 4: Retrieve the external dream pool

Call `mcp__quorum_codex__q_await` with the Step 3 `job_id`. Relay its `Council:`
liveness line, including empty or failed external seats.

If any external seat is missing or failed, also report a line beginning
`Council degraded:` that names each unavailable seat and its status. For a
Claude or Gemini failure, name the seat helper as a likely cause and suggest
`uv run --directory /path/to/code-quorum quorum seat-helper-status`, replacing
`/path/to/code-quorum` with the Code Quorum source checkout used to build or
install the marketplace. Continue with the available seats.
Never describe the spread as complete or full when the council is degraded.

If `q_await` raises an error, report the error, label the skystorm run
incomplete, and skip native grounding and final synthesis. Never synthesize
from the host's independent ideas alone as though the external dream stage
succeeded.

## Step 5: Run supportive grounding in a native subagent

Assemble a concise dream pool from the host's independent and external ideas —
ideas only, with source attribution and no prior art. Keep the research-quality
note out of the pool: it belongs in Step 6 narration. Pass the same pivot
digest through a separate evidence channel in the grounder brief. Use native
delegation (`spawn_agent` when available) to create one fresh Codex-native
subagent.
Derive a unique task name from the Step 3 `job_id`; use its first 12 hexadecimal
characters so repeated skystorm runs in one thread cannot collide.
Pass:

- `task_name`: `"supportive_grounder_<first-12-job-id>"`.
- `message`: the complete self-contained brief described below.
- `fork_turns`: `"none"` to avoid inherited conversation.

When the fresh-context control is unavailable, provide the same self-contained
brief and state that only the brief should govern the pass.

Grounder output is terse by default. Request fuller grounder output only when
`--verbose` was given; carry that instruction inside the `message` rather than
inventing a native delegation parameter.

The brief must label these inputs separately:

1. Topic.
2. Absolute project working directory for optional inspection.
3. Dream pool with source attribution.
4. Pivot digest, verbatim, when it seeded the dream stage.
5. Research-quality note, when research ran.

Give the subagent these boundaries verbatim:

> Act as a supportive validation guide, not a critic. Preserve unconventional
> or cross-domain ideas unless evidence rules them out. Pair every risk with the
> cheapest test that could confirm or rule it out.
>
> Stay read-only. Do not edit files. Do not run mutating commands. You may
> inspect relevant project files when that helps ground an experiment.

For every promising thread, require exactly these labelled fields:

- **Vision preserved**
- **Smallest falsifiable experiment**
- **Baseline and controls**
- **Success and stop thresholds**
- **Missing evidence or method gap**
- **Cheapest next implementation step**

After spawning the grounder, use `wait_agent` (or the host's native completion
wait) to wait for its final completion and collect its response. Do not validate
or synthesize while the subagent is merely running.

If `wait_agent` raises an error, treat that attempt as errored output under the
same bounded retry/fallback path below.

Treat output as failed when it is empty, whitespace-only, errored, or missing
the required labels. On failure, retry once with a fresh native subagent using
`task_name`: `"supportive_grounder_<first-12-job-id>_retry"`, the same
`fork_turns`: `"none"`, and the same bounded brief, then wait for that attempt's
final completion too. Relay a separate `Native grounder:` liveness line for the
successful attempt or for both failures.

If native delegation is unavailable or both attempts fail, run the same output
contract in the main Codex and label it `Native grounder: orchestrator fallback`.
For that fallback pass, suspend the Step 1 ideation role and apply the
grounder verbosity selected by `--verbose`; use only the separately labelled
grounder brief as task context. Continue to follow all system, developer, and
user instructions.
Never silently skip grounding or present fallback work as subagent output.

## Step 6: Synthesize a path forward

Skystorm converges only after preserving the breadth that produced the useful
threads. Present, in this order:

1. The source-attributed idea spread. Flag independent convergence,
   singular ideas, and real conflicts without flattening them.
2. The strongest two or three grounded paths. For each, retain the grounder's
   experiment, controls, thresholds, evidence gap, and cheapest next step.
3. A short execution synthesis explaining how to validate the paths while
   preserving the original ambition.

When research ran, lead with its separately labelled quality note and quote the
Step 2 `Research status:` verbatim. Research informs the path; it does not set
the ceiling, so promising native ideation is not discarded merely because no
citation exists yet.

## Verify cited evidence

The grounding pass can return plausible-but-wrong citations: a future-dated or
unresolvable arXiv ID, a URL with tracking parameters, or "studies show…"
without a locator. Skystorm is deliberately loose at the front and strict at
the back:

- **While dreaming** — concept level is the right tier. A citation seeding a
  wild idea needs only the concept confirmed as real and roughly as described;
  demanding full reads here smothers divergence. Say that is the tier, and do
  not present it as grounding.
- **Once a thread lands** — anything the smallest experiment, control,
  falsifiability bar, quoted number, or an "already solved" / "never been
  tried" claim rests on is **load-bearing**. Ask: *if this source said the
  opposite, would the path change?* If yes, fetch and read the artifact itself:
  an arXiv PDF or HTML, Europe PMC full text, repository files at a pinned
  commit, or model card. Read the method, results table (including N and
  spread), and limitations that decide whether it transfers, then confirm the
  claim. Existence is not confirmation, and an abstract is not a full read.

Down-weight and mark anything that cannot be confirmed at its own tier as
`[unverified]`; never pass a flaky citation through as settled fact. A failed
fetch is a refusal until proven otherwise — for example, arXiv can return HTTP
200 with `Rate exceeded` — and a well-formed "no such record" response is the
only registry result that counts against an identifier. Then weight sourced
evidence and native ideation together: research informs the path, but it does
not set the ceiling.
