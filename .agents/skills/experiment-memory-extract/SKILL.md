---
name: experiment-memory-extract
description: "Extract finalized numereng experiment conclusions into per-experiment design-space topic branches under .numereng/notes/__RESEARCH_MEMORY__/experiments."
user-invocable: true
argument-hint: "<experiment-id or experiment path>"
---

# Experiment Memory Extract

## Role / Purpose

Extract one finalized numereng experiment into a per-experiment research-memory branch under:

```text
.numereng/notes/__RESEARCH_MEMORY__/experiments/<experiment-id>/
```

This skill is extraction-only. It does not update master topic ledgers or `CURRENT.md`.

## Personality / Collaboration Style

Write compact research notes. Preserve exact evidence, but keep every claim scoped to what the finalized experiment actually tested.

## Goal

Create or refresh one experiment branch with `README.md` plus the six canonical training-design topic files.

## Success Criteria

- Source `EXPERIMENT.md` is finalized and has an explicit verdict/final decision.
- The branch contains exactly `README.md` plus the six canonical topic files.
- Every topic file cites the finalized report and distinguishes tested evidence from inference.
- Broad comparison and selection context stays in `README.md`, not topic ledgers.
- Master topic ledgers and `CURRENT.md` are unchanged.

## Constraints

- Treat finalized `.numereng/experiments/<id>/EXPERIMENT.md` as the source of truth.
- Use `EXPERIMENT.pack.md` only as supporting evidence after the report is understood.
- Do not create extra topic files.
- Do not update `.numereng/notes/__RESEARCH_MEMORY__/topics/*.md`.
- Do not update `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md`.
- Do not promote global defaults from one experiment.
- Do not turn candidate evidence into champion evidence.
- Do not turn a multi-axis result into a single-axis rule.

## Canonical Topic Files

Every branch must contain exactly:

```text
README.md
features.md
targets.md
models.md
hyperparameters.md
ensembling.md
neutralization-exposure.md
```

Topics are training/model design levers only. Baseline context, evaluation, validation, leakage, and submission operations belong in `README.md` or `CURRENT.md`, not topic ledgers.

## Evidence And Scope Rules

Use these labels when extracting:

- `verified artifact`: file, run, status, manifest, or count checked by the final report.
- `computed metric`: exact value from the final report or pack.
- `supported inference`: scoped interpretation supported by the report.
- `hypothesis / next-step`: plausible but untested follow-up.

Each topic with real evidence should state:

- the scoped design-space takeaway
- evidence level
- design-space role
- confounds changed together
- what is not established
- future implication
- master-ledger update suggestion

Design-space roles:

- `varied`
- `controlled`
- `inherited`
- `observed`
- `not_tested`
- `confounded`

If a topic was not tested, say so directly and keep the file short.

## Workflow

1. Resolve the experiment id from an id or `.numereng/experiments/<id>/` path.
2. Read, in order:
   - `.numereng/experiments/<id>/EXPERIMENT.md`
   - `.numereng/experiments/<id>/experiment.json`
   - `.numereng/experiments/<id>/EXPERIMENT.pack.md` when present
   - `.numereng/notes/NUMERAI_KEY_DYNAMICS/NUMERAI_MASTER_MODEL_DESIGN_SPACE.md`
   - existing `.numereng/notes/__RESEARCH_MEMORY__/experiments/<id>/README.md` when refreshing a migrated branch
   - `references/source-priority.md`
   - `references/write-contract.md`
   - `references/comparison-policy.md`
3. Confirm the report has a verdict and is not draft-only.
4. Extract broad experiment context into `README.md`:
   - experiment id
   - source `EXPERIMENT.md` path
   - extraction date
   - final verdict
   - evidence/completeness status
   - compact decision table
   - topic index
   - comparison and selection context
   - preserved prior flat review content when a migrated branch had one
5. Extract each canonical topic into its own file.
6. For each topic, include:
   - `Experiment-Specific Takeaway`
   - `Evidence Snapshot`
   - `Evidence Level`
   - `Design-Space Role`
   - `Confounds`
   - `What Not To Infer`
   - `Not Established`
   - `Scope And Caveats`
   - `Future Implication`
   - `Master Ledger Update`
7. Confirm the branch shape and that master files were not changed.

## Output

Topic files should usually be 300-650 words when the topic has real evidence, and shorter when the topic was not tested. Use at most one compact table per topic unless the source evidence truly needs another.

`README.md` should be 500-900 words and may carry broad comparison, evaluation, and selection context. Explicitly distinguish:

- best single run
- candidate family
- ensemble candidate
- stabilizer candidate
- champion
- no champion

## Verification

After extraction, confirm:

- `experiments/<id>/` exists
- all canonical topic files exist
- no flat `experiments/<id>.md` file exists for that experiment
- topic files cite the finalized experiment report, not dashboard impressions
- topic files include evidence level and confounds when the topic has real evidence
- master ledgers and `CURRENT.md` were not changed by this skill

## Stop Rules

Stop without writing when:

- `EXPERIMENT.md` is missing
- `experiment.json` is missing or malformed
- the experiment is obviously draft-only or zero-result
- the finalized report lacks an explicit verdict or final decision
- source evidence is insufficient to make scoped design-space claims
- the requested extraction would require updating master ledgers or `CURRENT.md`
