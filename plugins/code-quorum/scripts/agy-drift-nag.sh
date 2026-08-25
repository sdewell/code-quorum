#!/usr/bin/env bash
# Session-start drift nag for the agy seat -- free, automatic, at most once/day.
#
# The gemini seat's read-only guarantee is a macOS seatbelt whose fit against
# agy is an empirical fact about one specific agy version, recorded as
# SEAT_VERIFIED_AGY_VERSION in quorum/agents/gemini_cli.py. agy updates often
# and a sandbox-fit regression fails SILENTLY, so drift has to surface without
# anyone remembering to look. This script is the free half of that deal: it
# compares two version strings and prints one paragraph. It NEVER runs the
# live canaries -- those spend real Gemini quota, so verification only happens
# when a person chooses to run it:
#
#   bash scripts/verify-agy-seat.sh
#
# Wired as a plugin SessionStart hook (hooks/hooks.json). Every exit path that
# is not a nag exits 0 silently: a missing agy, an unparseable version, or an
# unreadable pin all mean there is nothing true to say, and a hook that errors
# on a user machine is worse than a missed nudge.
set -euo pipefail

command -v agy >/dev/null 2>&1 || exit 0

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pin_file="${root}/quorum/agents/gemini_cli.py"
if [[ ! -r "$pin_file" ]]; then
  exit 0
fi

pinned="$(sed -n 's/^SEAT_VERIFIED_AGY_VERSION = "\(.*\)"$/\1/p' "$pin_file")"
if [[ -z "$pinned" ]]; then
  exit 0
fi

# Bounded probe: a wedged agy must cost ~5s at most, never the hook's whole
# budget. Deliberately a temp FILE, not $(...): agy spawns helper processes,
# and a hung grandchild holding the pipe's write end would block a command
# substitution past the hook budget even after the direct child is killed.
probe_out="$(mktemp "${TMPDIR:-/tmp}/agy-drift-nag.XXXXXX" 2>/dev/null || true)"
if [[ -z "$probe_out" ]]; then
  exit 0
fi
agy --version >"$probe_out" 2>&1 </dev/null &
probe_pid=$!
for _ in {1..50}; do
  kill -0 "$probe_pid" 2>/dev/null || break
  sleep 0.1
done
kill "$probe_pid" 2>/dev/null || true
wait "$probe_pid" 2>/dev/null || true
# Same lenient parse as verify-agy-seat.sh: first semver anywhere in the
# output, so a prefixed or multi-line --version does not kill the nag.
installed="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "$probe_out" 2>/dev/null | sed -n 1p || true)"
rm -f "$probe_out" 2>/dev/null || true
if [[ -z "$installed" || "$installed" == "$pinned" ]]; then
  exit 0
fi

# At most one nag per day per version pair, or it becomes wallpaper. A changed
# pair (either side) nags immediately regardless of the date. Every cache
# operation is best-effort: an unwritable or malformed cache must degrade to
# an extra nag, never to a failing hook. Two sessions starting in the same
# instant can each print one nag (no lock) -- accepted, self-correcting race.
stamp_dir="${XDG_CACHE_HOME:-$HOME/.cache}/code-quorum"
stamp_file="${stamp_dir}/agy-drift-nag"
stamp="${installed}|${pinned}|$(date +%F)"
if [[ "$(cat "$stamp_file" 2>/dev/null || true)" == "$stamp" ]]; then
  exit 0
fi

if [[ -e "${root}/.git" && -f "${root}/scripts/verify-agy-seat.sh" ]]; then
  remedy="  bash \"${root}/scripts/verify-agy-seat.sh\""
elif [[ -f "${root}/.claude-plugin/plugin.json" && -f "${root}/scripts/verify-agy-seat.sh" ]]; then
  remedy="  From a matching code-quorum source checkout, run:
  bash scripts/verify-agy-seat.sh
  After it succeeds, follow the normal commit/release flow, then update or
  reinstall the Claude plugin so it receives the updated verified pin."
else
  remedy="  From a matching code-quorum source checkout, run:
  bash scripts/verify-agy-seat.sh
  After it succeeds, rebuild into a fresh --output directory and reinstall the
  Codex marketplace artifact so this hook receives the updated verified pin."
fi

notice=""
IFS= read -r -d '' notice <<EOF || true
[code-quorum] agy is v${installed} but the gemini seat's read-only sandbox was
last live-verified against v${pinned}. The seatbelt still applies and fails
closed; whether it still contains everything this agy version can attempt is
unproven. Re-verify when ready -- it runs live canaries that spend real Gemini
quota (~2 min), so it is opt-in and never runs automatically:
${remedy}
(This notice is free and repeats at most once a day until the versions match.)
EOF

# Codex hook stdout is strict JSON-or-silent. Claude accepts the same
# hookSpecificOutput envelope. Encode in Bash so this safety notice does not
# add a cold uv/Python startup inside the hook's 10-second budget.
json_escape() {
  local value="$1"
  local result=""
  local char code escaped i
  local LC_ALL=C
  for ((i = 0; i < ${#value}; i++)); do
    char="${value:i:1}"
    case "$char" in
      '"') result+='\"' ;;
      \\) result+='\\' ;;
      $'\b') result+='\b' ;;
      $'\f') result+='\f' ;;
      $'\n') result+='\n' ;;
      $'\r') result+='\r' ;;
      $'\t') result+='\t' ;;
      *)
        printf -v code '%d' "'$char"
        if ((code < 32)); then
          printf -v escaped '\\u%04x' "$code"
          result+="$escaped"
        else
          result+="$char"
        fi
        ;;
    esac
  done
  printf '%s' "$result"
}
escaped_notice="$(json_escape "$notice")"
payload="{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"${escaped_notice}\"}}"

# Claim delivery only after the complete JSON response was written to stdout.
# If stdout fails, a later session gets another chance to warn.
printf '%s\n' "$payload"
mkdir -p "$stamp_dir" 2>/dev/null || true
printf '%s' "$stamp" 2>/dev/null > "$stamp_file" || true
