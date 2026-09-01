# Write Contract

Research memory has two write phases. `experiment-memory-extract` performs only the first phase.

## Experiment Branches

`experiment-memory-extract` writes only:

```text
.numereng/notes/__RESEARCH_MEMORY__/experiments/<experiment-id>/
```

Each branch has `README.md` plus one topic file per canonical training-design topic.

Topic files must include:

- experiment-specific takeaway
- compact evidence snapshot from finalized `EXPERIMENT.md`
- evidence level
- design-space role
- confounds
- what not to infer
- not established
- scope and caveats
- future implication
- master ledger update suggestion

Evidence levels:

- `verified artifact`
- `computed metric`
- `supported inference`
- `hypothesis / next-step`

Evidence snapshots should use one short table or bullet list when it materially improves the branch. Topic files should usually be 300-650 words when the topic has real evidence, and shorter when the topic was not tested.

The branch README must carry the broad experiment decision context:

- final verdict
- evidence/completeness status
- compact decision table
- comparison and selection context

When relevant, distinguish best single run, candidate family, ensemble candidate, stabilizer candidate, champion, and no champion. Do not promote any category into another unless the finalized report explicitly supports it.

Design-space roles:

- `varied`
- `controlled`
- `inherited`
- `observed`
- `not_tested`
- `confounded`

## Scope Discipline

Every topic takeaway should state what changed together when it affects interpretation:

- feature set
- target family / horizon
- model family
- model recipe / capacity
- hyperparameters
- target preselection
- scoring stage availability
- post-selection comparison against prior experiments

Do not compress a multi-axis result into a single-axis design rule.

## Master Topic Ledgers

`experiment-memory-synthesize` writes `topics/*.md`.

Each topic ledger has:

- `Current Overview`
- `Current Best Understanding`
- `Append-Only Experiment Learnings`

The overview may be edited in place. The learning log should append experiment sections and link to experiment branch topic files.

## CURRENT.md

`CURRENT.md` is compressed.

It should include:

- compressed frontier
- experiment history chain
- comparison anchors
- evaluation and next-step selection rationale
- current research direction
- current next experiments
- current constraints

It should not include extensive topic knowledge or full topic ledgers. Evaluation, selection, validation, leakage, and submission operations are guardrails or decision context, not topic-ledger branches.

## Links

- use note-relative links under `__RESEARCH_MEMORY__`
- link experiment branches as `experiments/<id>/README.md`
- link topic evidence as `experiments/<id>/<topic>.md`
- use explicit app routes only when linking to the dashboard or run pages
