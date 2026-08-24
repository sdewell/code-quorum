---
name: q-help
description: Cheat sheet for the code-quorum multi-agent council. Use when the user asks about quorum modes, default council, stance assignments, or which workflow fits the task. Does not run the council itself — point to q-plan, q-brainstorm, q-skystorm, q-validate, q-review, or q-research.
---

# Instructions

Print everything below the `---OUTPUT---` line verbatim as your response. Do not paraphrase, summarize, omit sections, or add commentary before or after. Stop after the last line. This skill is a static cheat sheet — your job is to surface it to the user, not to interpret it.

---OUTPUT---

## code-quorum — quick reference

`code-quorum v0.0.63`

### Modes

| Workflow | When |
|---|---|
| `q-plan <task>` | Second-opinion plans before committing. Convergence. |
| `q-brainstorm <topic>` | Wide spread of testable ideas, grounded in prior art by default (research-first; `--no-research` opts out). Preserve divergence. |
| `q-skystorm <topic>` | Forward-research blue-sky. Map the cross-domain topology in the literature → dream → ground → synthesize. |
| `q-validate <plan-path>` | Critical pushback on a draft plan. 2-round default; `--extended` = 4 rounds with stance rotation at R3. |
| `q-review [target]` | Critical pushback on real code changes (branch/PR/range/whole codebase). Convergence into a findings table. `--scope <path>` bounds it; 2-round default, `--extended` = 4. |
| `q-research <topic>` | Standalone prior-art digest (arXiv + OpenAlex + Europe PMC + Context7 + GitHub + HuggingFace) with links. No council — one call. `-s <source>` restricts; `--exploratory` for cross-domain analogy hunting. |

### Invocation

The host can select a workflow from a matching plain-language request. To select one explicitly:

| Surface | Example |
|---|---|
| Claude Code | `/q-plan <task>` |
| Codex | `$code-quorum:q-plan <task>` |
| Shell | `uv run quorum q-plan <task>` |

Use the same `q-*` workflow name in Claude Code and Codex. `q-skystorm` and `q-help` are host-only skills; they have no shell command. The standalone shell command is `uv run quorum research <topic>`; `--exploratory` is host-only, and the shell `research` command is the grounded digest.

### Default council

The host never runs as a subprocess seat.

| Host | External council |
|---|---|
| Claude Code | Codex + Gemini + OpenCode |
| Codex | Claude subscription + Gemini + OpenCode |

### Default stances

The other platform's seat is skeptic, Gemini is architect, and OpenCode is neutral.

Override per-agent with `roles=["security:gemini","maintainer:opencode"]`. Stances: `skeptic`, `architect`, `security`, `maintainer`, `analyst`, `neutral`, `visionary`, `pioneer`.

These defaults apply to `q-plan`, `q-brainstorm`, `q-validate`, and `q-review`. `q-skystorm` keeps a host-specific workflow: Claude Code uses external Codex in both dream and analyst grounding; Codex uses external Claude, Gemini, and OpenCode dreamers plus a fresh native Codex supportive grounder. `--extended` (q-validate/q-review) rotates stances at round 3.

### More

Full reference: `README.md`.
