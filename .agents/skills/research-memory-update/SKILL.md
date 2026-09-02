---
name: research-memory-update
description: "Fold one closed-out numereng experiment into research memory: write its branch under experiments/<id>/, append to the topic ledgers, and rewrite CURRENT.md."
user-invocable: true
argument-hint: "<experiment-id>"
---

# Research Memory Update

Fold one closed-out experiment into `.numereng/notes/__RESEARCH_MEMORY__/`. Read, under
`.numereng/experiments/<id>/`, the finalized
`agentic_research/closeout/EXPERIMENT.closeout.md`, then
`agentic_research/closeout/evidence_summary.json`, then `EXPERIMENT.md` for the model's own working notes.
Use only their numbers. Never edit anything under `.numereng/experiments/`.

## 1. Decide The Disposition

Choose one and write it, with a one-paragraph rationale, at the top of the branch README:

- `master`: the experiment changes a champion, candidate, or frontier; confirms or contradicts a key
  result; closes a route with evidence; adds ensemble, package, or live-gate evidence; exposes a
  material confound; or yields a reusable design rule.
- `branch_only`: informative but incomplete, a reproduction with no new decision, or operational
  evidence worth keeping without changing master understanding.
- `exclude`: smoke, harness, or infra work, an empty failed search, or a rerun already folded into
  its parent. Write nothing further.

Label the comparison class and carry it into everything you write. A `broad screening surface`
covers many targets, feature sets, or variants and moves priors. A `narrow candidate-quality packet`
tests one family or a small set and moves candidate confidence, not broad claims. `champion /
production evidence` is validated handoff evidence with scoring, ensemble, exposure, and operating
gates. A narrow packet never overwrites a broad prior unless it is directly comparable or repeated.

## 2. Write The Branch

Create `experiments/<id>/` with exactly `README.md` and six topic files: `features.md`,
`targets.md`, `models.md`, `hyperparameters.md`, `ensembling.md`, `neutralization-exposure.md`. The
README names the experiment id, links each topic file, and holds the baseline context, comparison
anchors, and selection reasoning. Topic files hold design-lever evidence only.

Each topic file carries these level-2 sections in order: Experiment-Specific Takeaway; Evidence
Snapshot, with numbers and metric names; Evidence Level, exactly one of `verified artifact`,
`computed metric`, `supported inference`, `hypothesis / next-step`; Design-Space Role, exactly one of
`varied`, `controlled`, `inherited`, `observed`, `not_tested`, `confounded`; Confounds; What Not To
Infer; Not Established; Scope And Caveats; Future Implication; Master Ledger Update. A topic the
experiment did not exercise is still written honestly as `hypothesis / next-step` and `not_tested`.

## 3. Update The Ledgers And CURRENT.md

Skip this step for `branch_only` and `exclude`.

For each topic the experiment materially evidenced, append one entry under
`## Append-Only Experiment Learnings` in `topics/<topic>.md`, headed `### <experiment-id>`, linking
`../experiments/<id>/<topic>.md` and carrying the evidence level and comparison class. Never rewrite
prior entries. Replace the body of `## Current Overview` or `## Current Best Understanding` only
when this experiment changes the standing synthesis for that topic; otherwise leave both untouched.

Rewrite `CURRENT.md` as a compression, not an accumulation: fold this experiment in, drop what it
supersedes, keep the sections `## Compressed Frontier`, `## Comparison Anchors`, and
`## Current Constraints`, name the experiment id, and add a `Full record:` line pointing at
`experiments/<id>/README.md`. State the scope boundary, the comparison class, the active candidate
set, what changed in the frontier belief, the champion state (usually `none`), the blocking gates,
and the confounds that changed together.

## Rules

- Carry evidence levels and design-space roles into every promoted claim. Never upgrade a
  hypothesis into a settled result.
- Never promote a candidate to champion without explicit production-ready evidence.
- Do not create decision notes, flat experiment files, or extra topic files.
- `External Signals (Unverified)` sections stay append-only and never become comparison anchors.
