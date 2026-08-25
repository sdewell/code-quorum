# code-quorum

Independent reviews have become an important part of how I use agentic tools.
Inspired by similar work, I built Code Quorum for my own use and am sharing it
in case others find it useful.

Code Quorum is a macOS multi-agent council for Claude Code and Codex. The active
host writes its own review, assessment, or plan while external seats work in
parallel. A structural anti-bias gate keeps every perspective independent until
the final synthesis.

It can use existing Claude Code, ChatGPT/Codex, and Gemini/Antigravity
subscriptions. The OpenCode seat uses OpenRouter, with DeepSeek V4 Flash as its
default model. Both CLI hosts are supported, along with Codex in the ChatGPT
desktop app and the Code surface in the Claude desktop app.

| Host | Default external council |
|---|---|
| Claude Code | Codex + Gemini + OpenCode |
| Codex | Claude subscription + Gemini + OpenCode |

The host is never also a subprocess seat. The start/await split requires the
host to form its own answer before `q_await` exposes peer output, and every
council skill calls the blocking completion notification in the same turn as
its start.

External seats are read-only, but read-only does not mean data-local. Review
[Security and data boundaries](SECURITY.md) before using Code Quorum on private
material.

Every repository-reading MCP council start requires an explicit absolute project
`cwd`. The MCP tools reject an omitted or blank value rather than falling back
to the server's plugin-cache or runtime directory. The host skills supply this
value during normal `/q-*` and `$code-quorum:q-*` use. Plan and scope files must
also resolve inside that directory; absolute paths, `..` traversal, and symlink
escapes are rejected before their contents are read.

## Workflows

The host can select a workflow from a matching plain-language request. Use the
forms below to select one explicitly:

| Workflow | Claude Code | Codex | Shell |
|---|---|---|---|
| Plan | `/q-plan <task>` | `$code-quorum:q-plan <task>` | `uv run quorum q-plan <task>` |
| Brainstorm | `/q-brainstorm <topic>` | `$code-quorum:q-brainstorm <topic>` | `uv run quorum q-brainstorm <topic>` |
| Skystorm | `/q-skystorm <topic>` | `$code-quorum:q-skystorm <topic>` | host-only skill |
| Validate | `/q-validate <plan-path>` | `$code-quorum:q-validate <plan-path>` | `uv run quorum q-validate <plan-path>` |
| Review | `/q-review [target]` | `$code-quorum:q-review [target]` | `uv run quorum q-review [target]` |
| Research | `/q-research <topic>` | `$code-quorum:q-research <topic>` | `uv run quorum research <topic>` |
| Help | `/q-help` | `$code-quorum:q-help` | host-only skill |

`uv run quorum --help` lists the exact shell surface. The main modifiers are:

| Option | Effect |
|---|---|
| `--extended` | Adds another divergence round for host brainstorming, or expands validation/review to 4 rounds with a stance rotation. |
| `--mode critique` | Makes later validation/review rounds attack peer positions instead of revising toward agreement. |
| `--scope <doc>` | Declares in-bounds, out-of-bounds, and accepted-risk areas for a whole-codebase review. |
| `--exploratory` | Makes host research hunt for cross-domain analogies instead of direct prior art. |
| `--no-research` | Runs brainstorming or skystorm from model priors alone. |

### Research sources and credentials

`q-research` queries all seven sources by default. Repeat `--source <name>` to
restrict a run. It accepts up to three `--query-lane` formulations and reports a
`Source/lane status` table so a strong source cannot hide a collision elsewhere.
Use `--purpose methods` (default) for balanced all-time and recent OpenAlex
strata, or `--purpose currency` for the recent five-year stratum only. Literature
sources search every lane; artifact sources (Context7, GitHub, and Hugging Face)
search only the primary lane to avoid redundant results and API traffic. The
per-source result limit stays fixed across lanes, so additional lanes broaden
coverage without growing the digest without bound.

| Source | Target | Credential policy |
|---|---|---|
| arXiv (`arxiv`) | Scholarly papers and preprints from arXiv search. | None. |
| OpenAlex (`openalex`) | Methods searches balance all-time relevance/canonical candidates with recent five-year candidates and retain the stratum labels; currency searches use only the recent stratum. Exploratory mode also produces a subfield map. | `OPENALEX_API_KEY` or `QUORUM_OPENALEX_API_KEY` is required for normal OpenAlex use. `QUORUM_OPENALEX_EMAIL` identifies the client but does not replace the key. |
| Europe PMC published (`europepmc-published`) | Published life-sciences literature, including PubMed/MEDLINE records, reviews, MeSH metadata, and full-text availability. Exact matches have priority; MeSH synonym expansion only backfills a thin exact result set. | None. |
| Europe PMC preprints (`europepmc-preprints`) | Life-sciences preprints from bioRxiv, medRxiv, Research Square, and similar sources; arXiv records are excluded. | None. |
| Context7 (`context7`) | High-trust library matches and documentation snippets. | `CONTEXT7_API_KEY` is recommended because anonymous requests can be rate-limited. |
| GitHub (`github`) | Public repositories matched by name, description, and topics, then ranked by stars. | `GH_TOKEN` or `GITHUB_TOKEN` is recommended for higher limits. Private repositories are excluded. |
| Hugging Face (`huggingface`) | Public model IDs and metadata, ranked by downloads. Term-fallback results carry `[broadened]`. | `HF_TOKEN` or `QUORUM_HF_TOKEN` is recommended for account-level Hub limits. Private models are filtered out. |

Code Quorum reads credentials from the process environment and sends tokens
only in authorization headers. The generated Codex adapter forwards the named
variables but does not store their values in the plugin artifact. OpenAlex is
the only source that requires a key for normal use; the others improve
reliability or rate limits.

Verify Hugging Face search from a checkout with:

```bash
uv run quorum research "sentence embedding" --source huggingface --limit 5
uv run pytest tests/test_research_live.py -m live -k huggingface -q
```

The CLI check must return model links and a nonzero `HuggingFace` source count.
The live tests cover direct search, configured-token authentication,
distinctive-term union, and the full `research_topic` path. Without a token,
the authentication test skips while anonymous checks still run.

## Requirements

Code Quorum currently supports macOS and requires Python 3.13+, the
[`uv`](https://docs.astral.sh/uv/) package manager, and the binaries for the
seats you intend to use. Each seat relies on its own login or key; Code Quorum
does not write credential values into plugin artifacts. Seat CLIs retain their
own authentication and runtime state as described in [SECURITY.md](SECURITY.md).

| Seat | Binary | Auth | Cost |
|---|---|---|---|
| codex | [`codex`](https://developers.openai.com/codex/cli) | the CLI's own login (`codex login`; `--with-api-key` for metered use) | ChatGPT subscription or metered API key |
| gemini | [`agy`](https://antigravity.google/docs/cli/install/) | Google OAuth via `agy` | Google AI subscription; metered `GEMINI_API_KEY` is an explicit SDK opt-in |
| opencode | [`opencode`](https://opencode.ai/docs/) | `OPENROUTER_API_KEY` in the environment | metered through OpenRouter |
| claude (Codex host only) | [`claude`](https://code.claude.com/docs/en/setup) | the CLI's own claude.ai login | Claude subscription only; API-key routing is stripped |

The Gemini seat depends on the macOS Seatbelt sandbox; Claude uses a read-tool
allowlist instead. A Codex host also needs a narrowly scoped LaunchAgent helper
for its Claude and Gemini seats. Codex and OpenCode have no Seatbelt dependency
but are untested on other platforms.

## Install

Clone the stable checkout and configure the seats:

```bash
git clone https://github.com/sdewell/code-quorum.git
cd code-quorum
uv sync
agy  # complete Google OAuth login, then exit
uv run quorum setup-agy                    # one-time Gemini seat config
uv run quorum setup-models --host claude   # use --host codex for Codex
uv run quorum doctor --host claude         # or codex / both
uv run quorum auth-check --seat gemini --host claude
```

`doctor` checks binaries, configuration shape, and sandbox readiness. It does
not test live credentials. `auth-check` runs `agy models` inside the same
sandbox used by the seat and requires at least one valid model row without
sending a model prompt. A missing or revoked login directs the user back to
interactive `agy`; Code Quorum never silently changes to a metered API route.

On a Codex host, install and verify the helper from a real terminal before the
Codex authentication check:

```bash
uv run quorum install-seat-helper-launchagent   # --allowed-root <dir> to widen
uv run quorum seat-helper-status
uv run quorum auth-check --seat gemini --host codex
```

Every MCP workflow permits `cwd` under `~/Code` and `~/src` by default; the
Codex helper applies the same roots before accepting Claude or Gemini requests.
Set `CODE_QUORUM_HELPER_ALLOWED_ROOTS` or install the helper with repeated
`--allowed-root` options to use other project roots. Reinstall it from the
updated stable checkout after every Code Quorum upgrade. Incompatible helper
protocols fail closed, and installation from Codex's replaceable plugin cache
is rejected.

## Read-only boundaries

Every external seat is read-only and refuses to run if its boundary cannot be
applied. Enforcement differs by seat: Codex uses its native read-only sandbox,
Claude exposes only read tools, Gemini uses macOS Seatbelt, and OpenCode uses an
isolated HOME with restricted permissions.

Gemini's Seatbelt profile denies other home-directory reads, with explicit
exceptions for agy authentication and runtime state. It does not deny readable
paths outside `$HOME`. Code Quorum provides no universal path fence for Claude,
Codex, or OpenCode. Council material can leave the machine under the user's
configured provider accounts. The full boundary table, data-egress map,
strict-isolation guidance, and credential handling are in
[SECURITY.md](SECURITY.md).

## Data and approvals on Codex

Codex treats tool approval, filesystem containment, and authorization to send
material off-machine as separate decisions. A target such as `main...HEAD`
bounds the prepared review diff; it does not restrict an external seat's
read-only access to the working directory.

When `approvals_reviewer = "auto_review"` is enabled, a council start may need
explicit authorization naming the payload and recipients. Users who want
unattended access can opt in per tool, but Code Quorum never writes those
approval entries itself.

[SECURITY.md](SECURITY.md) contains the one-off authorization example, all six
Codex approval blocks (including the shared `q_await` tool), project
`AGENTS.md` guidance, path-confinement limits, and the approval-preserving
update procedure. Review it before enabling unattended workflows.

## Configuration

Each seat resolves its model through one ladder, first hit wins: per-run flag
-> environment variable -> the choice recorded by `quorum setup-models` -> the
shipped pin. Recorded choices never change silently; a seat that cannot honor
one fails loudly while the rest of the council continues.

```bash
uv run quorum setup-models --host claude
uv run quorum setup-models --seat codex --model gpt-5.6-terra --effort medium
```

| Variable | Effect |
|---|---|
| `CODE_QUORUM_HOST` | default host profile (`claude` or `codex`) |
| `CODE_QUORUM_GEMINI_MODEL` | Gemini seat model (an id from `agy models`) |
| `CODE_QUORUM_GEMINI_BACKEND` | `cli` (subscription) or `sdk` (metered `GEMINI_API_KEY`) |
| `CODE_QUORUM_OPENCODE_MODEL` | OpenCode seat model |
| `CODE_QUORUM_OPENCODE_DEBUG` | `0` disables failed-run diagnostic capture |
| `CODE_QUORUM_CLAUDE_MODEL` / `_EFFORT` | Claude seat model and effort |
| `CODE_QUORUM_CODEX_MODEL` / `_EFFORT` | Codex seat model and reasoning effort |

### OpenCode configuration

The OpenCode seat requires `OPENROUTER_API_KEY` and does not load your personal
OpenCode configuration. It uses an isolated HOME and rebuilds this generated
configuration before every run. The shipped model is
`openrouter/deepseek/deepseek-v4-flash`; its OpenRouter `chunkTimeout` is
`90000` milliseconds.

Code Quorum sets `OPENCODE_DISABLE_PROJECT_CONFIG=1` and `OPENCODE_PURE=1`.
The generated council agent permits only `Read`, `glob`, and `list`, denies
shell and mutation tools, blocks `.env` and `.env.*`, and permits
`.env.example`. Failed or empty runs write raw stdout/stderr captures to
`~/.cache/code-quorum/opencode-debug` unless
`CODE_QUORUM_OPENCODE_DEBUG=0` is set. The directory is `0700`, capture files
are `0600`, prompt text is omitted from command metadata, and only the newest 20
captures are retained. Raw streams can still contain reviewed material. See
[ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY.md](SECURITY.md) for the full
boundary design.

### Disabling a seat

If a live probe fails because a seat is absent or logged out,
`quorum setup-models` can mark it `disabled` in `models.toml`. Disabled seats
are skipped by the default roster and reported by `doctor`, but an explicit
`--agent <seat>` request still runs them.

## Install as a plugin

**Claude Code:**

```text
/plugin marketplace add sdewell/code-quorum
/plugin install code-quorum@code-quorum
```

For the Claude Code CLI, load `OPENROUTER_API_KEY` and optional research keys
before starting the host. For example:

```bash
source ~/.zshrc.local
claude
```

For the Claude desktop app, make the keys available to the current macOS login
session before opening it:

```bash
source ~/.zshrc.local
launchctl setenv OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
```

The launchctl value is inherited by every subsequently launched application
until it is unset, logout occurs, or the machine reboots. Start Claude Code,
then remove the login-session copy; the already-running app retains its copy for
plugin subprocesses:

```bash
launchctl unsetenv OPENROUTER_API_KEY
```

After installing or upgrading the plugin, or after changing a key, quit Claude
Code completely and start a new Claude Code session. A plugin reload can pick up
code changes but cannot change the environment inherited by the running host.

The Claude plugin starts its MCP server with
`uv run --directory ${CLAUDE_PLUGIN_ROOT} quorum-mcp`, so `uv` and Python 3.13+
must be on `PATH`.

**Codex:**

Register the public marketplace and install the plugin:

```bash
codex plugin marketplace add sdewell/code-quorum --ref main
codex plugin add code-quorum@code-quorum
codex plugin list
```

For the ChatGPT desktop app, fully quit and reopen the app after registering
the marketplace. Open **Plugins**, choose **Personal**, select **Code Quorum**,
and click **Install**.

After either Codex surface installs the plugin, prepare the stable checkout
from a real terminal:

```bash
uv sync
uv run quorum install-seat-helper-launchagent
uv run quorum seat-helper-status
```

The Codex launcher starts that checkout's prepared `.venv` directly. MCP
startup therefore does not depend on a writable `uv` cache, network downloads,
or an environment inside the replaceable plugin directory. For later upgrades,
one command refreshes the checkout, plugin, environment, approvals, and helper:

```bash
uv run quorum update-codex
```

Fully restart Codex and start a new thread afterward. If an older checkout does
not yet have `update-codex`, use the one-time legacy sequence in `SECURITY.md`.

In Codex CLI, open `/hooks` to review and trust each code-quorum command hook.
Codex skips plugin hooks until each current definition hash is trusted. A
changed definition requires re-review and trust for that changed definition.

The generated launcher recovers standard user and Homebrew binary directories
(`~/.local/bin`, `~/.opencode/bin`, `/opt/homebrew/bin`, and `/usr/local/bin`)
for a desktop app with a minimal `PATH`.

For Codex CLI, source the key environment before launching Codex. For the
ChatGPT desktop app, set keys in the current macOS login session:

```bash
source ~/.zshrc.local
launchctl setenv OPENROUTER_API_KEY "$OPENROUTER_API_KEY"
```

The value is visible to every subsequently launched application until it is
removed. Start Codex, then remove the login-session copy; the running app keeps
the value it already inherited:

```bash
launchctl unsetenv OPENROUTER_API_KEY
```

After a plugin upgrade or key change, fully quit and reopen Codex and start a
brand-new Codex thread. Do not resume a thread created before the restart; its
tool registry may still refer to the prior plugin process.

For approval-preserving updates, never use `codex plugin remove` as the normal
path. Follow the verified sequence in [SECURITY.md](SECURITY.md).

## Design

[ARCHITECTURE.md](ARCHITECTURE.md) describes the shared CLI/MCP spine, seat
adapters, round model, anti-bias mechanisms, packaging, and failure boundaries.
The short version: round 1 contains no peer output, later rounds anonymize peers
by stance, and the host does not receive the council matrix until `q_await`.

## Attribution

These projects inspired the workflow shape; no code, prompts, or documentation
were copied:

- [SnakeO/claude-co-commands](https://github.com/SnakeO/claude-co-commands) - independent host work before peer output.
- [agentic-box/owlex](https://github.com/agentic-box/owlex) - multi-agent council architecture and cognitive roles.
- [karpathy/llm-council](https://github.com/karpathy/llm-council) - anonymized peer review by content rather than model identity.

## License

MIT.
