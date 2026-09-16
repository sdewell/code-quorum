# Security and data boundaries

Code Quorum is read-only by design, but read-only does not mean data-local.
Council modes send the configured external seats enough material to do their
work and let those seats inspect the selected working directory. Use Code
Quorum only where that off-machine access is acceptable.

## Credential non-interference

Treat a credential as sensitive according to what it can authorize: data
access, mutation, billing, quota use, or future permission expansion. A key is
not less sensitive because its current account tier is free.

Code Quorum's guarantee is limited to credential routing it controls. The host
and MCP process can receive credential values needed by enabled workflows; they
are not credential-free. For subprocess-backed seat execution and authentication,
Code Quorum builds a positive environment allowlist from basic runtime names and
explicit adapter additions instead of passing the ambient host environment. The
Claude adapter additionally permits `CLAUDE_CONFIG_DIR` and its selected
`CLAUDE_CODE_OAUTH_TOKEN`, Codex permits `CODEX_HOME`, and OpenCode receives its
resolved `OPENROUTER_API_KEY` after its other environment values have been
filtered.

Research authentication is isolated at the request-header level, not the
process level. OpenAlex, Context7, GitHub, and Hugging Face requests receive only
their selected provider's Bearer token. arXiv and Europe PMC requests receive no
authorization header.

| Responsibility | Operator | Code Quorum |
|---|---|---|
| Provisioning | Supply the intended credential with the narrowest practical capability, availability, and lifetime. | Bundle no developer credential values and keep generated plugin artifacts free of them. |
| Routing | Select the intended seat or research source and keep secrets out of transmitted or seat-readable material. | Apply positive subprocess-environment filtering and provider-specific request-header routing where described above. |
| Storage and response | Choose an appropriate platform storage/delivery mechanism; set service limits; rotate or revoke after suspected exposure. | Avoid intentional credential persistence or disclosure and document the boundaries that remain outside this guarantee. |

Authentication failures follow each adapter's actual behavior rather than one
uniform promise. A missing `OPENROUTER_API_KEY` prevents OpenCode use, and a
missing `GEMINI_API_KEY` prevents the opt-in Gemini SDK backend. Claude and agy
subscription failures make those seats unavailable. Codex delegates login
handling to its CLI and reports a failed run. Research sources follow the
provider-specific anonymous behavior documented in the README. Code Quorum does
not borrow another provider's environment credential for a failed route.

This guarantee does not cover credential values deliberately placed in task
text, diffs, plans, scopes, arguments, `.env`, `.envrc`, embedded-secret
configuration, or other files a selected seat can read. It also excludes
provider-owned login/session stores, Keychain access, the opt-in in-process
Gemini SDK backend, host-side context gathering such as `gh`, setup and version
probes, and auxiliary processes. Standard proxy and CA variables remain
available as network plumbing; proxy URLs can contain credentials. Raw seat
diagnostics are permission-protected and retention-bounded as described below,
but are not guaranteed to be credential-sanitized.

## Seat containment

Every seat is prevented from modifying the selected project, and a seat that
cannot apply its boundary refuses to run rather than degrading:

| Host | Seat | Boundary |
|---|---|---|
| Claude Code | Codex | `codex exec --ephemeral --ignore-user-config --ignore-rules -s read-only` |
| Claude Code | Gemini/agy | macOS Seatbelt profile; refuses if the profile smoke test fails |
| Claude Code | OpenCode | isolated HOME plus read-only OpenCode permissions |
| Codex | Claude | shared helper; `claude -p` with only `Read,Glob,Grep`, no session persistence |
| Codex | Gemini/agy | shared helper, then the same macOS Seatbelt profile |
| Codex | OpenCode | isolated HOME and permissions, inside the outer Codex sandbox |

For the Gemini seat the sandbox - not agy's config - is the guarantee. It denies
project writes, permits writes only to observed agy runtime-state paths, and
denies other home-directory reads except the selected `cwd`, agy credential and
runtime state, the login keychain file, and its Playwright cache. The process can
still read otherwise-readable paths outside `$HOME`. The sandbox covers only the
council seat, not interactive `agy` use.

Each agy attempt writes a diagnostic log containing routing/error evidence and
potentially reviewed prompt material. Code Quorum pre-creates the log directory
as `0700`, files as `0600`, deletes ordinary successful-attempt logs, and retains
at most the newest 20 diagnostic logs needed to explain failures or model
substitutions.

On a Codex host, Claude and Gemini run through a narrowly scoped LaunchAgent
helper because their containment cannot start inside Codex's outer sandbox.
The helper accepts only those two seat types, checks that the requested working
directory is under an allowed root, and rejects incompatible protocol versions
before queueing a request.

### OpenCode isolation

The OpenCode seat requires `OPENROUTER_API_KEY` but does not load your personal
OpenCode configuration. Each run uses an isolated HOME at
`~/.cache/code-quorum/opencode-sandbox`, and Code Quorum rebuilds this generated
configuration before each run.

The shipped model is `openrouter/deepseek/deepseek-v4.1-flash`, routed to the
Novita and Parasail backends. DeepSeek's own endpoint is not in that order:
it fails the OpenRouter account's zero-data-retention and no-training policy,
and both listed backends pass it. Its generated configuration sets the
OpenRouter `chunkTimeout` to `90000` milliseconds, `reasoning.effort` to `medium`, and
sets `OPENCODE_DISABLE_PROJECT_CONFIG=1` and `OPENCODE_PURE=1`. The generated
council agent permits only `Read`, `glob`, and `list`; it denies shell, write,
and edit tools. Reads of `.env` and `.env.*` are denied while `.env.example`
remains available.

OpenCode persists its session database under the isolated HOME. Failed or empty
runs also retain raw stdout/stderr under
`~/.cache/code-quorum/opencode-debug`. Both roots are owner-only (`0700`), debug
files are `0600`, prompt text is omitted from command metadata, and no more than
20 debug captures are retained. Raw streams and the session database can still
contain reviewed material; set `CODE_QUORUM_OPENCODE_DEBUG=0` to disable the
extra failure captures.
The SessionStart database pruner activates above 500 MB and removes oldest
council sessions toward 400 MB while preserving the newest 10; non-council
OpenCode sessions are not deleted.

To record which OpenRouter backend served each request, the seat routes
OpenCode through a loopback relay (`quorum/agents/openrouter_relay.py`)
bound to `127.0.0.1` on an ephemeral port for the life of one run. The
relay sees the full request, including the `Authorization` header and the
reviewed material, and forwards it unchanged to `https://openrouter.ai`. It
does not log or store request or response bodies or headers. It appends one
JSON line per request to `~/.cache/code-quorum/opencode-served.jsonl`
(directory `0700`, file `0600`): backend name, generation id, model, token
counts, cost, HTTP status, finish reason, and timings. The listener accepts
connections from any local process during the run; such a process runs as
the same user and already holds `OPENROUTER_API_KEY`, so the relay does not
widen that boundary. `CODE_QUORUM_OPENCODE_RELAY=0` disables the relay and
the ledger.

## Material available outside the host

Council seats cannot modify the project, but they receive read-only access to
the selected `cwd` under the user's configured provider accounts.

| Workflow | Material available outside the host |
|---|---|
| `q-plan`, `q-brainstorm`, `q-skystorm` | Task or topic, project-context summary, and read-only access to `cwd` |
| `q-validate` | Full plan, project-context summary, and read-only access to `cwd` |
| `q-review` | Selected diff, optional scope document, project-context summary, and read-only access to `cwd` |
| `q-research` | Query sent to the selected research APIs; no council or project `cwd` access |

Every repository-reading MCP council start requires an explicit absolute `cwd` and
rejects an omitted or blank value before a job is created. The MCP server's own
plugin-cache or runtime directory is never used as project context. The selected
directory must be under `CODE_QUORUM_HELPER_ALLOWED_ROOTS` (`~/Code`, `~/src`,
and `~/.codex/agent-worktrees` — a common agent-worktree location — by
default).
Plan and scope documents must resolve inside that `cwd`; absolute
paths, `..` escapes, and symlink escapes are rejected before their contents are
read. Because a `cwd` under `~/.codex/agent-worktrees` is hidden, the Gemini
SDK backend would otherwise widen its workspace to the home directory with no
sandbox to fence that read scope; it refuses to run for such a `cwd` instead.

A bare `q-review` compares the branch with the default-branch merge-base and
includes committed work, uncommitted tracked changes, and a list of untracked
filenames. An explicit range such as `main...HEAD` or a target such as `pr:123`
bounds the prepared diff only; it does not restrict the external seats'
read-only access to `cwd`.

Code Quorum does not enforce universal `cwd` read confinement. The Claude,
Codex, and OpenCode seats have no Code Quorum path fence. Gemini denies other
reads inside `$HOME` except its explicit agy/keychain/cache requirements, but its
Seatbelt profile leaves otherwise-readable paths outside `$HOME` available.
`CODE_QUORUM_HELPER_ALLOWED_ROOTS` controls which working directories every MCP
workflow accepts, not which paths a running seat may read. A separate clean clone
reduces accidental exposure from normal relative reads, but it is not a security
boundary. When strict filesystem isolation matters, use a dedicated macOS VM or
machine with its own helper installation and provider logins, exposing only the
material those seats may inspect. A separate macOS user account is a weaker
operational boundary because some paths remain shared.

## Desktop credential environment

`launchctl setenv` is an optional delivery mechanism, not a secure credential
store. It puts a credential into the macOS login-session environment, where
every subsequently launched application can inherit it. For a desktop host,
start the app after setting the value, then remove the login-session copy with
`launchctl unsetenv OPENROUTER_API_KEY`. Unsetting limits exposure to later
applications; it does not remove the already-running host's inherited copy, and
an enabled downstream route still receives the credential it requires. Logout
or reboot also clears the login-session value. Code Quorum's portable public
interface remains environment variables and does not require Apple Keychain.

## Codex approvals and authorization

Codex treats tool approval separately from filesystem sandboxing and data
authorization. By default, eligible approval requests are routed to the user.
When `approvals_reviewer = "auto_review"` is configured, Codex's automatic
reviewer can deny a council start unless the request explicitly authorizes the
payload and recipients.

For a one-off review, make the authorization part of the request:

> Run `$code-quorum:q-review` on `main...HEAD`. I authorize Code Quorum to send
> that committed diff to the configured external Claude, Gemini, and OpenCode
> seats and authorize their read-only access to this repository directory for
> the review. I understand Code Quorum does not enforce universal confinement
> to that directory.

Users who deliberately want unattended access to every Code Quorum workflow
can opt in through `~/.codex/config.toml`:

```toml
[plugins."code-quorum@code-quorum".mcp_servers.quorum_codex.tools.q_plan_start]
approval_mode = "approve"

[plugins."code-quorum@code-quorum".mcp_servers.quorum_codex.tools.q_research]
approval_mode = "approve"

[plugins."code-quorum@code-quorum".mcp_servers.quorum_codex.tools.q_brainstorm_start]
approval_mode = "approve"

[plugins."code-quorum@code-quorum".mcp_servers.quorum_codex.tools.q_validate_start]
approval_mode = "approve"

[plugins."code-quorum@code-quorum".mcp_servers.quorum_codex.tools.q_review_start]
approval_mode = "approve"

[plugins."code-quorum@code-quorum".mcp_servers.quorum_codex.tools.q_await]
approval_mode = "approve"
```

`q_await` is shared by every council mode. The other entries approve each
workflow's start or research call. This is a per-user choice; Code Quorum does
not write these entries during installation.

The Codex adapter uses the distinct `quorum_codex` server identity so Claude's
source-tree `quorum` descriptor cannot shadow it. Upgrading from a release that
used `quorum` is a one-time namespace migration: explicitly approve the six
tools again in a new Codex thread. Old `quorum` approval blocks do not authorize
`quorum_codex`, and must not be copied into the new table as a substitute for
consent. One workflow invokes only some of the tools, so review and configure
every entry shown above before expecting the six-entry verifier to pass. Claude
keeps its existing server identity and tool prefix.

Tool approval and data-transfer authorization remain separate. For recurring
use on a private repository, put standing authorization in the project `AGENTS.md`,
scoped to Code Quorum's documented behavior. A global `AGENTS.md`
rule grants the same off-machine transmission authority for every repository
it covers, so use global scope only when that broader authorization is
intended. Start with the specific workflow being authorized. For example:

> A request to run `code-quorum:q-review` authorizes its registered MCP tools,
> configured external council seats, documented review payload, normal provider
> usage, and read-only access to the selected working directory. This excludes
> `.env`, credential paths, and other secret-bearing files; do not run until any
> such material is removed from the selected directory. Do not ask again or
> substitute a sanitized-copy run solely because the repository is private.

That exclusion is not enforced by Code Quorum; the user maintains it. Use this
standing authorization only for repositories and working directories whose
off-machine exposure has already been accepted.

Keep the scope read-only and limited to the requested Code Quorum workflow.
The standing rule supersedes per-run confirmation for the authorized material:
with the approval entries in place, a bare `q-review` can transmit working-tree
material off-machine and let external seats inspect `cwd` without another
prompt. Replace `q-review` with `q-*` only when every workflow is intended;
that broader form also covers `q-research` and its configured providers.

See Codex's
[approval security](https://learn.chatgpt.com/docs/agent-approvals-security#automatic-approval-reviews)
and
[plugin-scoped MCP policy](https://developers.openai.com/plugins/build/plugins#bundled-mcp-servers-and-lifecycle-hooks)
documentation.

## Approval-preserving Codex updates

`codex plugin remove` deletes the plugin-scoped approval entries above. Do not
use removal as a normal update step. From the stable Code Quorum checkout, run:

```bash
uv run quorum update-codex
```

The command snapshots your chosen approval settings, fast-forwards the checkout,
prepares its environment, refreshes and non-destructively reinstalls the plugin,
restarts the seat helper, and then verifies the plugin and helper versions match
and your approval settings did not change. This keeps the helper protocol
consistent with the installed plugin. It never runs `codex plugin remove`.

If an older checkout does not yet provide `update-codex`, use this legacy
sequence once; later upgrades use the single command above:

```bash
git pull --ff-only
uv sync
codex plugin marketplace upgrade code-quorum
codex plugin add code-quorum@code-quorum
uv run quorum install-seat-helper-launchagent
uv run quorum seat-helper-status
```

`codex plugin add code-quorum@code-quorum` performs a non-destructive reinstall
and preserved the approval entries in testing. That behavior was verified with
Codex CLI 0.148.0 and again with 0.149.0. `seat-helper-status` must report the
target Code Quorum version before a new Codex thread is started.

If removal is unavoidable, restore all six approval blocks after reinstalling.
Then:

1. Run `codex --strict-config --version` to validate the schema.
2. Run `codex plugin list` and confirm the target version.
3. Inspect `~/.codex/config.toml` and confirm that all six named tool blocks use
   the `approve` approval mode.

After those checks pass, fully restart Codex and start a brand-new thread.
