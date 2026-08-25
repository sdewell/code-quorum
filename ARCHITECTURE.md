# Architecture — code-quorum

This document describes the directory and module structure, important data and
control flows, and the design tradeoffs behind them.

code-quorum is a multi-agent council for Claude Code and Codex. It fans a
prompt out to external seats in parallel, optionally across deliberation
rounds, then hands the result back to the active host to synthesize. There are
two plugin hosts and two execution surfaces over one shared spine: a Typer CLI
and an MCP server. Every MCP council start carries an explicit `host` and
absolute project `cwd`, so roster selection, containment, and grounding do not
depend on ambient process inference or the server's installation directory.

## Layout

| Path | Responsibility |
|---|---|
| `quorum/cli.py` | Typer CLI — modes, setup, diagnostics, and helper management. |
| `quorum/hosts.py` | Claude/Codex host profiles: external default roster and stances. |
| `quorum/orchestration.py` | The `run_mode` funnel: resolve context, build the mode prompt, select agents + roles, resolve a review target. Holds the mode prompts and `AGENT_REGISTRY`. |
| `quorum/council.py` | `run_council` — the round-by-round fan-out that returns a `rounds × agents` result matrix; per-agent containment; output/liveness formatting. |
| `quorum/roles.py` | Cognitive stances (skeptic, architect, …) as per-phase prompt prefixes. |
| `quorum/context.py` | Project-context assembly (LEARNINGS/CLAUDE.md + git + gh) injected into every prompt. |
| `quorum/research.py` | Prior-art research over arXiv/OpenAlex/Europe PMC published+preprints/Context7/GitHub/HuggingFace. Standalone — no council coupling. |
| `quorum/agents/` | Backend adapters plus `seat_helper.py`, the Codex-host bridge for Claude and agy. |
| `quorum_mcp/server.py` | Shared FastMCP server. `q_*_start` / `q_await` / `q_research` tools. |
| `quorum_mcp/jobs.py` | The non-blocking background-job model (start returns a `job_id`; `q_await` blocks). |
| `scripts/` | Version/release helpers, including `build_codex_marketplace.py` for a self-contained Codex artifact. |
| `.claude-plugin/`, `.codex-plugin/`, `.mcp.json`, `hooks/`, `skills/` | Two manifests over shared skills, hooks, and runtime; the Codex builder stages them with its host-specific MCP adapter. |

The skills, MCP tools, and CLI subcommands use a `q-`/`q_` prefix (`q-plan`,
`q_await`, …), not `co-`.

## Key components

### The spine

Every council run funnels through **`run_mode` → `run_council`**. `run_mode` (orchestration.py) resolves project context, prepends it to the mode prompt in a fixed order — **role → context → mode task** — selects the agents, assigns their stances, and delegates to `run_council`. `run_council` (council.py) drives the rounds: each round runs the selected agents concurrently (`asyncio.gather`), collects an `AgentResult` per agent, and returns the full `rounds × agents` matrix. The two surfaces differ only in how they call this spine — the CLI prints the matrix; the MCP server runs it as a background job (see *MCP server*).

### Modes

Four deliberation modes share the mode-agnostic spine (`run_mode` → `run_council` → `apply_role`):

- `q-plan` — parallel implementation plans (1 round).
- `q-brainstorm` — divergent ideas, no synthesis (1 round). Research-first by default; `--no-research` opts out.
- `q-validate` — multi-round critique of a plan file (2 rounds; 4 with `--extended`).
- `q-review` — multi-round review of **real code changes** (a branch, an open PR, a commit range, or the whole codebase). It reuses the spine unchanged and differs in three places: target ingestion (`resolve_review_target` turns a target spec into byte-capped diff text), an optional `--scope` doc embedded verbatim (`read_scope_file`), and a `review` phase added to the stances. Convergence is inherent (`mode='revise'` + ≥2 rounds), not a flag; Claude synthesizes the verdict + findings table at the SKILL step.

Round count comes from `resolve_rounds(extended)` → 2 by default, 4 (`MAX_ROUNDS`) extended; q-plan/q-brainstorm hardcode 1. Round 1 is independent (no peer output); rounds 2+ inject each agent's own prior plus peers' priors with a `revise`/`critique` instruction; the extended round 3 rotates stances and presents all priors flat.

### Host profiles and agents

All seats implement one `Agent` interface. A Claude host defaults to Codex +
Gemini + OpenCode; a Codex host defaults to Claude + Gemini + OpenCode. The
active host is rejected if explicitly requested as a subprocess seat.

- **claude** — `claude -p` after a strict `claude.ai` subscription-auth check.
  API/cloud-provider routing variables are removed. The child is non-persistent,
  uses `dontAsk`, and receives only `Read,Glob,Grep`.
- **codex** — `codex exec --ephemeral --ignore-user-config --ignore-rules -s
  read-only`; prompt on stdin and output through a temporary file. Ignoring user
  config/rules keeps a council seat from inheriting mutating or behavioral
  customizations from the developer's interactive Codex setup.
- **gemini** (architect) — **one seat, two engines.** `GeminiCliAgent` (the default, `CODE_QUORUM_GEMINI_BACKEND=cli`) shells out to `agy` on the Google AI Pro quota; `GeminiAgent` (`=sdk`) runs the Antigravity SDK in-process on the metered API. The `make_gemini_agent` registry factory selects between them and **raises on an unrecognized backend value** rather than silently falling back to a different quota source. The cli engine's model resolves per-invocation ask first (`gemini_model` on the council-start tools, e.g. a Claude slug for one run) → `CODE_QUORUM_GEMINI_MODEL` → seat default; combining the per-run ask with the sdk backend raises (agy slugs are not google-genai ids). At runtime the cli seat also carries a **quota reflex**: a failure with `RESOURCE_EXHAUSTED`/429 evidence in the attempt's own agy log re-runs once on `claude-opus-4-6-thinking`, labeling the swap in the transcript. The seat pins its default model in **two forms** because two different things need proving and neither check covers the other: the display name (`Gemini 3.1 Pro (High)`) is what gets invoked and is verified by a live **routing canary** that reads the engine agy actually used out of its diagnostic log, while the slug (`gemini-3.1-pro-high`) is what `agy models` lists and is verified by a zero-quota **membership** test. Membership alone is not enough — agy 1.1.7 accepted that slug and ran Flash. Because `agy models` prints the slug, and the seat's own default used to be it, the seat **rewrites that one known-bad slug to the display name wherever it arrives from** (env var, per-run ask, direct construction) rather than only fixing its own default; every other id passes through untouched so an unknown model is still rejected by agy, by name. Above both pins sits a **runtime routing check that needs no table**: after any successful run the seat reads the engine label out of the log it already writes and compares it to what that attempt asked for, reducing both of agy's naming conventions to letters and digits so `gemini-3.1-pro-high` and `Gemini 3.1 Pro (High)` compare equal while `Gemini 3.6 Flash (High)` does not. That costs nothing (the log is already parsed for quota evidence) and — unlike a pinned expectation — catches *the next* substitution rather than only the known one. A mismatch warns and prefixes the transcript instead of failing the seat: an answer from the wrong engine is still worth something, being unable to tell is not. No labels in the log means missing evidence, not a mis-route, so a change to agy's log format degrades to silence rather than to a false alarm on every run.
The agy adapter separates authentication from transport failure. A failed
attempt whose private diagnostic log proves the OAuth token endpoint had a
network error retries once on the same model; a repeated transport failure is
reported as `network`, while definitive login failures remain `authentication`.
`quorum auth-check --seat gemini --host <host>` runs `agy models` inside the
sandbox and requires at least one valid model row. It verifies login readiness
without sending a prompt or switching silently to the metered SDK backend.

- **opencode** (neutral) — `opencode run` subprocess (DeepSeek via OpenRouter), run in a sandboxed `HOME` with a generated read-only `council.md`. Two unbounded-growth/stall guards sit on this seat, both traced to upstream behaviour rather than guessed at. (1) The generated config **arms opencode's own stream watchdog** (`provider.openrouter.options.chunkTimeout`, in milliseconds): on the openai-compatible path that watchdog is unset by default, so a silently dropped SSE stream hangs forever — and because opencode's retries are error-driven, a stall that never errors never retries. Set below the seat's own idle timeout so opencode's internal retry gets its chance before our watchdog gives up. (2) `scripts/prune_opencode_db.py` caps the sandbox database, which nothing else bounds (measured at 516 MB after a couple of months, mostly message `part` rows and the `event` log). It deletes **only `agent = 'council'` sessions**, oldest first, keeping the newest few because the seat recovers a dropped answer by reading it back out of that database.

On a Codex host, Claude and Gemini cannot start their required containment from
inside Codex's outer sandbox: nested Seatbelt application fails. A LaunchAgent
helper receives only those two seat types through a mode-0700 file spool,
restricts request working directories to configured roots, then runs Claude
with its tool allowlist and agy under the same Seatbelt profile used by a Claude
host. Protocol v2 adds a Gemini-only `auth-check` operation; older helpers are
rejected before a request is queued and must be reinstalled. Orphaned result
files expire after five minutes. Missing helper state fails both seats closed.

### Roles

A role is a **cognitive stance = a prompt prefix** that changes *how* an agent answers, not *what* (modes own the deliverable shape). `ROLES` carries one prompt **per phase** (plan/brainstorm/validate/review) for each stance, because the seed stances were written as evaluators and would contradict the generative modes if reused verbatim. `apply_role` prepends the phase-appropriate prefix (`neutral` is a no-op); `rotate_roles` cyclically shifts assignments for the extended round-3 rotation.

### Context

`context.resolve` assembles a `<project_context>` block prepended to every prompt: keyword-scored `LEARNINGS*.md` / `CLAUDE.md` filenames (CLAUDE.md always included; it tells agents what to read rather than inlining contents), git state (branch, recent commits touching prompt-mentioned files), and open `gh` PRs matching prompt keywords. It is best-effort — a missing tool, timeout, or nonzero exit collapses to empty rather than failing the run. `--no-context` / `--skip-gh` short-circuit the respective sources.

### Prior-art research

`research_topic` (research.py) accepts one to three semantic query lanes and runs the seven source adapters concurrently while keeping each adapter's own lanes sequential; arXiv lanes observe its polite request gap. Each attempt is wrapped in `_contained` so a dead source lands in `digest.errors` instead of sinking its peers, and rate-retry queueing has a bounded wait horizon. Four adapters are paper sources (`_PAPER_SOURCES`: arXiv, OpenAlex, Europe PMC published, Europe PMC preprints) whose results merge into `digest.papers` and dedup by identifier/normalized-title. The published Europe PMC adapter searches the non-preprint corpus with `core` metadata (abstracts, MeSH, and full-text availability); the preprint adapter remains pinned to `SRC:PPR NOT PUBLISHER:"arXiv"`. Methods-purpose OpenAlex searches balance all-time relevance candidates with a recent five-year stratum, while currency purpose keeps only the recent stratum. `SourceLaneStatus` classifies every source/lane attempt before `compute_status` summarizes it, preventing an aggregate verdict from hiding a collision or source mismatch. The keyword backends (GitHub, HuggingFace) retain their distinctive-term union and per-term failure isolation. The module is standalone and is **not** subject to the anti-bias gate — it returns external prior art, not peer output, so it can be called during the orchestrator's own-work window.

### MCP server & job model

The MCP surface exposes `q_plan_start` / `q_brainstorm_start` /
`q_validate_start` / `q_review_start` (each returns a `job_id` immediately),
`q_await(job_id)`, and synchronous `q_research`. Every start requires an
explicit absolute project `cwd`, accepts `host="claude"|"codex"`, and passes
both through the shared orchestration path. Missing or blank `cwd` values fail
before a job is created. `jobs.py` runs councils as background tasks with a TTL
reaper. The CLI stays synchronous.

### Plugin packaging

Claude installs the source tree directly through `.claude-plugin/plugin.json`;
its `.mcp.json` uses `${CLAUDE_PLUGIN_ROOT}`. Codex requires conventional
`skills/` and `.mcp.json` paths, so `.codex-plugin/plugin.json` points at the
same shared tree and `build_codex_marketplace.py` stages a self-contained local
marketplace, including the shared SessionStart hooks and their three referenced
housekeeping scripts, while rewriting `.mcp.json` to use the staged plugin
directory. The generated Codex adapter deliberately names its server
`quorum_codex`, producing the `mcp__quorum_codex__*` tool prefix, while Claude
keeps its existing `quorum` identity and `mcp__plugin_code-quorum_quorum__*`
prefix. Distinct server names prevent Claude's source-tree `.mcp.json` from
shadowing the installed Codex adapter. Codex discovers the default
`hooks/hooks.json`, but each hook definition's current hash must be reviewed and
trusted before those commands run. Host-neutral skills select the correct MCP
namespace and pass `host` explicitly.
`q-skystorm` shares the research-mapping and citation-verification contract,
but routes Codex to its native-grounder reference because the active host cannot
run itself as an MCP subprocess seat. The Claude host uses an external Codex
analyst; the Codex host uses a fresh native Codex subagent and includes external
Claude in the dream council.

## Design decisions

- **The anti-bias gate is two distinct mechanisms.** (a) Inside `run_council`, round 1 uses the original prompt with **no peer block** — round-1 spread comes only from *different agents × different stances*; peer priors enter only at round 2+. (b) On the MCP surface, the **non-blocking `*_start` → `q_await` split** forces Claude to commit its own work between the two calls — it literally cannot see peer output before then. The CLI surface has no gate (b) because a human, not an agent under evaluation, reads the result. A related guard: peer priors in rounds 2+ are tagged **by stance only, never by agent name** (same convention as the round-3 rotation), so a seat weighs peer output on content rather than brand — tag metadata only; a peer's own text may still self-identify, and the orchestrator still sees the fully attributed matrix at synthesis.

- **Stateless by design.** There is no session resumption and no shared memory between rounds beyond the prompt text the orchestrator rebuilds each round. Simpler to reason about and reproduce; the cost is re-sending priors in each round's prompt.

- **Read-only is backend-specific, with one shared policy: fail closed when the
  boundary cannot start.** Codex uses its native read-only sandbox plus
  ephemeral/config isolation; Claude exposes only read tools; Gemini SDK has an
  SDK allowlist; agy uses macOS Seatbelt because its own config failed on 1.1.2;
  OpenCode uses an owner-only isolated HOME and read-only permissions. Its CLI
  session database persists there; failure captures are owner-only, omit prompt
  metadata, and retain a fixed newest-20 window. Prompt text and agy config are
  defence in depth, not the guarantee.

- **Input paths are constrained before prompt construction.** Every MCP council
  start requires an explicit working directory under the configured project
  roots; it never falls back to the server's plugin-cache or runtime directory.
  Plan/scope documents must resolve inside that working directory. Whole-codebase
  reviews carry an explicit repository-inspection instruction rather than an
  ambiguous blank diff.

- **Failures are surfaced, never swallowed.** Each agent's `run` is wrapped in `asyncio.wait_for` + a full exception catch so a timeout or crash becomes a failed `AgentResult` (with a distinct return code) instead of hanging the council or aborting succeeding peers. A clean-exit-but-blank turn maps to its own non-zero code (125) so an empty agent turn is visible rather than vanishing. A `format_liveness` line reports per-member status so a quiet member is not mistaken for a silent failure.

- **Subprocess groups are contained aggressively.** Every subprocess agent spawns with `start_new_session=True` and is torn down by process-group kill, with a `lstart`-matched escapee sweep (pid-reuse defense), a global pgid registry, and atexit/SIGTERM/SIGHUP handlers that SIGKILL any live agent groups on parent shutdown. `/bin/ps` is pinned by absolute path so a PATH-injected shim cannot forge the process snapshot.

- **Byte caps with graceful degradation.** Plan files, scope docs, and review diffs cap at 100 KB; an over-cap diff degrades to a `git diff --stat` / gh-metadata summary that points agents at the working tree rather than failing the run.

- **One version source.** `pyproject.toml` `[project].version` is canonical;
  `sync_version.py` propagates it to the package, both plugin manifests, and the
  shared cheat-sheet badge, and CI fails on drift.
