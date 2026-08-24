#!/usr/bin/env bash
# Re-verify the agy seat against the installed agy, then pin it.
#
# WHY THIS EXISTS: SEAT_VERIFIED_AGY_VERSION records what has been live-verified,
# and agy changes often. When the remedy is "remember to run a suite, then
# hand-edit a constant", it does not get done -- the pin sat three minor versions
# stale once, and the warning became wallpaper. This makes the remedy one command.
#
#   bash scripts/verify-agy-seat.sh
#
# It bumps the constant ONLY when every live canary passes. On any failure it
# leaves the pin alone, which is the whole point: the warning is supposed to keep
# firing until the seat is actually proven on the installed version.
#
# NOTE: routing no longer depends on this. The seat checks on every run that agy
# served the model it was asked for (routing_complaint), reading the log it
# already writes. This script covers what a single run cannot: the SEATBELT --
# whether agy has found a new way to write files, run shells, or fetch URLs.
set -euo pipefail

cd "$(dirname "$0")/.."
PIN_FILE="${CODE_QUORUM_AGY_PIN_FILE:-quorum/agents/gemini_cli.py}"

if ! command -v agy >/dev/null 2>&1; then
  echo "agy is not on PATH -- nothing to verify against." >&2
  exit 1
fi

# Extract the first semantic version from anywhere in agy's output, matching what
# _agy_version() does in the seat. A stricter "the whole first line must be X.Y.Z"
# would refuse to pin the moment agy prefixes it ("agy version 1.1.9") or writes a
# telemetry line first -- and refusing after the canaries all passed is the one
# outcome that would send someone back to hand-editing the pin. Captured to a
# variable first so nothing runs in a pipeline under `pipefail`.
if ! version_output="$(agy --version 2>&1)"; then
  echo "agy --version failed:" >&2
  printf '%s\n' "$version_output" >&2
  echo "refusing to pin from failed version output." >&2
  exit 1
fi
installed="$(printf '%s' "$version_output" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | sed -n 1p)"
if [[ -z "$installed" ]]; then
  echo "no version number in \`agy --version\` output:" >&2
  printf '%s\n' "$version_output" >&2
  echo "refusing to pin to something unparseable." >&2
  exit 1
fi

current="$(sed -n 's/^SEAT_VERIFIED_AGY_VERSION = "\(.*\)"$/\1/p' "$PIN_FILE")"
if [[ -z "$current" ]]; then
  echo "could not find SEAT_VERIFIED_AGY_VERSION in ${PIN_FILE}." >&2
  exit 1
fi

echo "installed agy: ${installed}"
echo "seat pinned to: ${current}"
echo
echo "running the live containment canaries (spends real AI Pro quota; ~2 min)..."
if ! canary_output="$(
  CODE_QUORUM_AGY_E2E=1 UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
    uv run pytest -q -m live tests/test_gemini_cli.py 2>&1
)"; then
  printf '%s\n' "$canary_output"
  echo
  echo "FAILED -- pin left at ${current}." >&2
  echo "Read the containment failure before touching the pin." >&2
  exit 1
fi
printf '%s\n' "$canary_output"
summary="$(
  printf '%s\n' "$canary_output" |
    grep -E '^[0-9]+ (failed|passed|skipped|deselected|xfailed|xpassed)' |
    tail -n 1 || true
)"
if printf '%s\n' "$summary" | grep -Eq '[0-9]+ skipped'; then
  echo
  echo "FAILED -- one or more live canaries were skipped; pin left at ${current}." >&2
  echo "Satisfy every live prerequisite before re-running the verifier." >&2
  exit 1
fi
if ! printf '%s\n' "$summary" | grep -Eq '^[1-9][0-9]* passed'; then
  echo
  echo "FAILED -- no live canary passed; pin left at ${current}." >&2
  echo "Check the pytest selection before re-running the verifier." >&2
  exit 1
fi

if [[ "$installed" == "$current" ]]; then
  echo
  echo "all canaries pass and the pin already matches ${installed} -- nothing to do."
  exit 0
fi

# Anchored to line start and the exact assignment, so no other occurrence of the
# version string (comments, history notes) can be rewritten by accident.
# temp-then-rename rather than `sed -i`: the in-place flag is spelled differently
# on BSD and GNU sed ('' vs nothing), and the wrong one for the host either fails
# or silently creates a backup file.
tmp="$(mktemp "${PIN_FILE}.tmp.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
cp -p "$PIN_FILE" "$tmp"
sed "s/^SEAT_VERIFIED_AGY_VERSION = \".*\"\$/SEAT_VERIFIED_AGY_VERSION = \"${installed}\"/" \
  "$PIN_FILE" > "$tmp"
candidate="$(sed -n 's/^SEAT_VERIFIED_AGY_VERSION = "\(.*\)"$/\1/p' "$tmp")"
if [[ "$candidate" != "$installed" ]]; then
  echo "generated pin is invalid; refusing to replace ${PIN_FILE}." >&2
  exit 1
fi
mv "$tmp" "$PIN_FILE"
trap - EXIT

now="$(sed -n 's/^SEAT_VERIFIED_AGY_VERSION = "\(.*\)"$/\1/p' "$PIN_FILE")"
if [[ "$now" != "$installed" ]]; then
  echo "the pin did not take (still ${now}) -- edit ${PIN_FILE} by hand." >&2
  exit 1
fi

echo
echo "all canaries pass. pin bumped ${current} -> ${installed}."
echo "Review and commit it through the normal PR route:"
echo "  git diff ${PIN_FILE}"
