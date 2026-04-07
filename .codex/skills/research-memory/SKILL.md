---
name: research-memory
description: "Analyze completed numereng experiment evidence or resynthesize existing research memory under .numereng/notes/__RESEARCH_MEMORY__."
user-invocable: true
argument-hint: "<experiment-id | resynthesize>"
---

# Research Memory

Use this skill when the user wants to update or recompute the rolling research-memory system.

This skill updates the canonical notes root:

- `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md`
- `.numereng/notes/__RESEARCH_MEMORY__/experiments/<experiment-id>.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/*.md`
- optional `.numereng/notes/__RESEARCH_MEMORY__/decisions/*.md`

Do not use this skill for:

- creating or training experiments
- store repair or cleanup
- submission operations
- broad Numerai literature review without concrete experiment evidence

Use:

- `experiment-design` for round planning before results exist
- `numereng-experiment-ops` for experiment contract/layout questions
- `store-ops` for drift, reset, reindex, or destructive cleanup

## Workflow Modes

Support exactly two workflow modes:

- `experiment-ingest`
  - anchored to one completed experiment id
- `resynthesis`
  - no new experiment id required
  - rereads current memory plus high-signal experiment reviews and recomputes `CURRENT.md` and touched ledgers

Do not create a new artifact type for resynthesis.

## Hard Gates

For `experiment-ingest`, before writing anything, confirm all of the following:

- the experiment exists
- at least one completed / scored run exists
- there are no active experiment run jobs
- required artifacts exist for analyzed runs:
  - `run.json`
  - `resolved.json`
  - `metrics.json`
  - `results.json`
  - `score_provenance.json`
- the experiment is not obviously draft-only or zero-result

If any gate fails:

- stop
- report the exact blocker(s)
- do not update research memory

For `resynthesis`:

- read current memory plus the high-signal linked experiment reviews first
- do not invent new experiment evidence
- do not write a decision note unless one of the explicit materiality triggers fires

## Canonical Truth Hierarchy

Research-memory has one present-tense truth layer:

- `CURRENT.md`
  - the only file allowed to state:
    - current promoted defaults
    - active frontier
    - top-ranked next experiments
    - blocked paths in present tense

Other files are subordinate:

- `topics/*.md`
  - scoped durable beliefs that support `CURRENT.md`
- `experiments/*.md`
  - original result interpretation plus current interpretation of one experiment
- `decisions/*.md`
  - historical rationale for changes to `CURRENT.md`
- `legacy-progression/`
  - historical input only

If `CURRENT.md` and another file disagree, `CURRENT.md` wins and the subordinate file must be updated on the next valid ingest or resynthesis.

## Canonical Metric Basis

Use the current numereng metric contract:

- primary: `bmc_last_200_eras.mean`
- tie-break: `bmc.mean`
- supporting: `corr.mean`, `mmc.mean`, `cwmm.mean`

Do not use payout-estimate framing as the canonical basis in new research-memory notes.

Keep these outputs separate:

- best score evidence
- best next experiment

Do not collapse them into one decision rule.

## Source Priority

Read in this order for `experiment-ingest`:

1. `experiment.json`
2. run artifacts:
   - `run.json`
   - `metrics.json`
   - `resolved.json`
   - `results.json`
   - `score_provenance.json`
3. `EXPERIMENT.pack.md` if present
4. `EXPERIMENT.md`
5. `uv run numereng experiment details --id <id> --format json`
6. `uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format json`

Secondary priors are allowed only after executed experiment evidence is understood:

- `.numereng/notes/NUMERAI_KEY_DYNAMICS/*`
- `.numereng/notes/NUMERAI_RESEARCH_STRATEGY/*`
- relevant files under `.numereng/notes/research/research-briefs/*`
- `.numereng/notes/__RESEARCH_MEMORY__/legacy-progression/*`

Executed experiment artifacts remain the default evidence base, but only after a lightweight evidence-quality check.

## Evidence Posture And Claim Quality

Do not assign exactly one evidence class to the whole experiment.

Instead:

- assign one experiment-level `overall evidence posture`:
  - `frontier-grade`
  - `mixed`
  - `supporting`
- also record claim-level quality notes inside the review:
  - `primary comparable`
  - `mixed`
  - `supporting`

Use overall posture as a summary, not a substitute for claim-level interpretation.

## Comparison Policy

Use two comparison passes.

### 1. Primary relevant subset

Select in this order:

1. evaluation surface / validation profile
2. target family / horizon
3. feature scope
4. model family
5. stated hypothesis / branch intent

Rules:

- this pass drives frontier interpretation
- if surface comparability fails, evidence cannot become direct frontier proof
- if multiple candidates tie, cite all tied candidates instead of silently choosing one

### 2. Base-rate and contradiction sweep

Read broader history for:

- contradiction checks
- repeated dead-end detection
- adjacent supporting evidence
- base-rate reconciliation

Every frontier recommendation must explicitly state whether broader history:

- supports
- weakens
- contradicts

Do not implement weighted scoring.

## Evidence-Quality Gate

Before promoting any belief into a topic ledger or `CURRENT.md`, explicitly check:

- are artifacts complete?
- is the run surface trustworthy for the claim being made?
- is the comparison actually comparable?
- are there obvious confounds or missing contextual caveats?
- does broader history support, weaken, or contradict the claim?

If the gate is weak:

- the claim may still be recorded
- but it must be marked provisional via posture / confidence / freshness
- and it must not be promoted as a strong default unless the evidence is strong enough

## Update Rules

One successful run of this skill must:

- create or refresh the experiment review when relevant
- update touched topic ledgers
- refresh `CURRENT.md`
- create a decision note only if the frontier materially changes

V1 topic ledgers:

- `topics/baselines.md`
- `topics/targets.md`
- `topics/feature-scope.md`
- `topics/model-families.md`
- `topics/evaluation-surfaces.md`
- `topics/postprocessing.md`

## Review Lifecycle

Experiment reviews are dual-layered:

- `Original Result Summary`
  - what the experiment established when first reviewed
- `Current Interpretation`
  - current meaning in frontier terms

Review rendering rules:

- default to aggregate-primary rendering for matrix-like experiments such as:
  - target sweeps
  - model-family sweeps
  - feature-scope comparisons
  - similar structured experiment matrices
- use run-primary rendering only when one or a few runs are the actual inferential unit
- `Current Interpretation` should be a short verdict, not a mini-essay
- include `Interpretation context` metadata:
  - `chronological rebuild as of <date>` for rebuild flows
  - `current-state ingest` for normal ingests
- include `Evidence risk` metadata:
  - `low` for clean comparable evidence
  - `medium` for partial limitations
  - `high` for mixed or degraded evidence
- accepted reviews should use the canonical filename:
  - `experiments/<experiment-id>.md`
- `.parallel.md` is preview-only and should not remain once a review is promoted

Lifecycle rules:

- preserve historical result interpretation
- allow `Current Interpretation` to be updated
- keep dated addenda for material reinterpretations or artifact changes
- do not rely on strict append-only semantics for the whole review body

## Materiality Triggers

A change is material if it changes any of:

- a promoted default in `CURRENT.md`
- the top-ranked next experiment
- a blocked path or its reopen conditions
- the confidence or freshness of a major current belief
- the interpretation of a previously cited key evidence anchor

Only then:

- write a decision note
- append an addendum for reinterpretation

If none of those triggers fire, prefer a no-op or a light non-material refresh.

## Freshness And Reopen Conditions

Track freshness lightly as:

- `fresh`
- `aging`
- `stale`

Apply freshness to:

- promoted defaults in `CURRENT.md`
- scoped beliefs in topic ledgers
- major current interpretations when relevant

Anti-patterns and blocked paths must include reopen conditions.

## Required Output Shapes

Read the templates in `assets/` before writing:

- `assets/CURRENT.template.md`
- `assets/experiment-review.template.md`
- `assets/topic-ledger.template.md`
- `assets/decision-note.template.md`
- `assets/AGENTS.template.md`

When the task needs details on sourcing, comparison, or writing policy, load:

- `references/source-priority.md`
- `references/comparison-policy.md`
- `references/write-contract.md`

## Writing Rules

- Keep notes decision-oriented.
- Prefer links over repeated prose.
- Use note-relative links inside `__RESEARCH_MEMORY__`.
- Use explicit app routes for experiment navigation:
  - `/experiments/<id>`
  - `/experiments/<id>/runs/<run_id>`
- Be explicit about uncertainty, confidence, scope conditions, and freshness.
- Optimize for “what should we do next?” while preserving enough provenance to explain why.

## Minimal Workflow

### Experiment-ingest

1. Validate gates.
2. Read canonical structured artifacts.
3. Read experiment narrative artifacts.
4. Assign overall evidence posture.
5. Build the primary comparison subset in ordered rubric order.
6. Run the broader base-rate / contradiction check.
7. Run the evidence-quality gate.
8. Write or refresh the experiment review.
9. Update touched topic ledgers.
10. Refresh `CURRENT.md`.
11. If a materiality trigger fired, write a decision note or addendum.

### Resynthesis

1. Read `CURRENT.md`, touched topic ledgers, and high-signal linked reviews.
2. Reconcile drift between current defaults and scoped ledger beliefs.
3. Refresh frontier, defaults, blocked paths, and freshness markers in `CURRENT.md`.
4. Refresh touched ledgers so they support the recomputed current state.
5. Write a decision note only if a materiality trigger fired.
