# code-quorum review scope

> Passed to `q-review` via `--scope <path>`. code-quorum reads this verbatim and tells
> every council agent what is in bounds. Out-of-scope / accepted-risk items are noted but
> never treated as findings to converge on. Delete these guidance comments before use.

## Threat model
<!-- Copy the repo's threat model verbatim from its CLAUDE.md (trusted / semi-trusted /
     out-of-scope). Seats treat hardening against an out-of-scope threat as
     [OUT-OF-SCOPE], not as a finding. A repo with no threat model gets one written
     into CLAUDE.md before the review. -->
-

## In scope
<!-- What this review SHOULD focus on: the changed surface, specific files/dirs, the
     behaviors or risk classes that matter for this change. Be concrete. -->
-

## Out of scope
<!-- What the council must NOT flag as a finding: generated/vendored code, a legacy module
     you are not touching this pass, formatting the formatter owns, etc. -->
-

## Known edge cases
<!-- Edge cases already considered and consciously handled or deferred, so agents do not
     re-raise them as new findings. -->
-

## Accepted risks
<!-- Risks you know about and accept for this change, with a brief why. Agents may note them
     tagged [OUT-OF-SCOPE] but should not push on them. -->
-

<!-- Optional severity bar — the lowest severity worth reporting this pass, e.g.
     "report high and above; skip low/nits." -->
