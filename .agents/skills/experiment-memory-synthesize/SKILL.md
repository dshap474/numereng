---
name: experiment-memory-synthesize
description: "Integrate per-experiment research-memory branch files into master topic ledgers and a compressed CURRENT.md frontier."
user-invocable: true
argument-hint: "<experiment-id | experiment folder | all>"
---

# Research Memory Synthesize

## Role / Purpose

Use this skill after `experiment-memory-extract` has extracted one or more experiment branches under:

```text
.numereng/notes/__RESEARCH_MEMORY__/experiments/<experiment-id>/
```

This is the only research-memory workflow that updates:

- `.numereng/notes/__RESEARCH_MEMORY__/topics/*.md`
- `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md`

It must not edit finalized experiment reports under `.numereng/experiments/`.

## Personality / Collaboration Style

Be conservative, concise, and scope-aware. Integrate the learning that is actually supported, preserve useful uncertainty, and avoid turning a single experiment into global truth.

## Goal

Synthesize one or more experiment branch folders into the six master training-design ledgers and a compressed frontier state in `CURRENT.md`.

## Success Criteria

- Every relevant experiment branch learning is linked from the matching master topic ledger.
- Master topic overviews are updated only when accumulated evidence changes the current best synthesis.
- `CURRENT.md` stays compact and explains the experiment history chain, active candidate set, frontier belief, champion state, comparison anchors, and next experiments.
- Scope boundaries, comparison class, blocking gates, and confounds remain visible.
- No extra rationale notes, decision notes, flat experiment reviews, or noncanonical topic files are created.

## Constraints

- Preserve the six-topic taxonomy exactly:

```text
features.md
targets.md
models.md
hyperparameters.md
ensembling.md
neutralization-exposure.md
```

- Do not update `.numereng/experiments/**`.
- Do not create or revive `decisions/`, flat `experiments/<id>.md` files, or obsolete topic ledgers.
- Do not promote a candidate to champion unless production-ready evidence is explicitly present.
- Keep evaluation and next-step selection in `CURRENT.md`; keep detailed training-design synthesis in `topics/*.md`.

## Comparison Classes

Label each integrated experiment so later synthesis does not mix unlike evidence:

- `broad screening surface`: many targets, feature sets, or model variants; useful for priors and search direction.
- `narrow candidate-quality packet`: focused test of one family or small candidate set; useful for candidate confidence but not broad replacement claims.
- `champion / production evidence`: validated handoff evidence with sufficient scoring, ensemble, exposure, and operating gates.

When an experiment is narrow, do not let it overwrite broad-screening priors unless it is directly comparable or repeated.

## Frontier Update Rules

Before changing `CURRENT.md`, check and state:

- scope boundary: what exactly the experiment tested
- comparison class: broad screen, narrow packet, or champion evidence
- active candidate set: which families/runs remain worth testing
- frontier belief: what changed in the research direction
- champion state: usually `none`
- blocking gates: missing ensemble, full scoring, exposure, live, or direct-comparison evidence
- confounds changed together: features, targets, depth/model class, horizons, seed count, or neutralization

Do not compress `medium standard-large Ender worked` into `medium is better`; preserve the actual tested surface.

## Workflow

1. Resolve the input:
   - experiment id
   - experiment branch folder
   - `all` experiment branches
2. Read each experiment branch:
   - `README.md`
   - `features.md`
   - `targets.md`
   - `models.md`
   - `hyperparameters.md`
   - `ensembling.md`
   - `neutralization-exposure.md`
3. Read the current master ledgers under `.numereng/notes/__RESEARCH_MEMORY__/topics/`.
4. Read `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md`.
5. Read `.numereng/notes/NUMERAI_KEY_DYNAMICS/NUMERAI_MASTER_MODEL_DESIGN_SPACE.md` for the design-space frame.
6. For each topic ledger:
   - append one experiment learning entry with a link to `../experiments/<id>/<topic>.md`
   - preserve append-only historical entries
   - update the mutable overview only when the new evidence changes the current best understanding
7. Update `CURRENT.md` as a compressed frontier/history file:
   - experiment history chain
   - current candidate set
   - frontier belief
   - champion state
   - comparison anchors
   - evaluation and next-step rationale
   - current constraints and blocking gates

## Output

Update only:

- `.numereng/notes/__RESEARCH_MEMORY__/topics/features.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/targets.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/models.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/hyperparameters.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/ensembling.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/neutralization-exposure.md`
- `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md`

Each master topic file keeps two layers:

- mutable top overview: current best understanding
- append-only experiment learnings: one section per integrated experiment

## Verification

After integration, confirm:

- every master topic file exists
- every master topic file has `Current Overview` and `Append-Only Experiment Learnings`
- integrated experiment links point to `../experiments/<id>/<topic>.md`
- `CURRENT.md` is concise and does not duplicate topic ledgers
- `CURRENT.md` distinguishes candidate set, frontier belief, and champion state
- broad screening and narrow candidate packet evidence are not blurred
- no obsolete rationale-note references remain
- no flat `experiments/<id>.md` references remain

## Stop Rules

Stop and report the blocker if:

- the experiment branch is missing or incomplete
- a required topic file is absent
- branch evidence contradicts the finalized experiment report in a way that changes the conclusion
- the requested synthesis would require editing finalized experiment artifacts
- the evidence is too weak to update a master overview; append the scoped learning instead
