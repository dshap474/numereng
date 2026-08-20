<!--
stage-3_extract.md — closeout phase 3 prompt (tracked, generic; no machine-specific paths).
The runner substitutes {{CONTEXT_JSON}} with the bounded EXTRACT context (decision memo, then the
deterministic evidence summary, then a compact per-round table) and calls the LLM with the shared
files envelope schema. Output is ONE JSON object whose files are exactly the seven canonical
research-memory branch files for this experiment: README.md plus one file per topic ledger. This
phase writes the per-experiment branch only; it never touches the master ledgers or CURRENT.md.
This prompt is unchanged apart from this header comment: extraction had no doctrine to port.
-->

# Closeout Phase 3 — EXTRACT

You are extracting one completed agentic research experiment into its permanent research-memory
branch: a small directory of exactly seven files that later synthesis (and future experiments) read.
You are given the finalized decision memo plus the bounded, deterministic evidence bundle.

## Inputs

Everything you may use is in the JSON context below. `decision_memo` is the human-facing verdict;
`evidence_summary` is the authoritative deterministic record; `rounds_table` is a compact per-round
view. Use ONLY these numbers — never invent metrics, and never claim a metric the record marks as
unavailable.

```json
{{CONTEXT_JSON}}
```

## Standing doctrine (apply throughout)

- Within-lane BMC200 is a **candidate-ranker**, not a deploy signal.
- Scout-tier outputs are **candidates, not results**.
- Every topic file must classify its evidence and the design-space role of its axis (see below).
- This phase writes the experiment branch only. Do NOT propose or reference edits to `CURRENT.md`
  or to any master ledger here — that is the next phase's job.

## Required output

Return exactly one JSON object with exactly these seven files:

```json
{"files": [
  {"path": "README.md", "content": "<branch overview>"},
  {"path": "ensembling.md", "content": "<topic file>"},
  {"path": "features.md", "content": "<topic file>"},
  {"path": "hyperparameters.md", "content": "<topic file>"},
  {"path": "models.md", "content": "<topic file>"},
  {"path": "neutralization-exposure.md", "content": "<topic file>"},
  {"path": "targets.md", "content": "<topic file>"}
 ],
 "notes": "<one-line summary of what this branch records>"}
```

`README.md` MUST name the experiment id explicitly and link every one of the six topic files by its
filename (e.g. `[Features](features.md)`, `[Targets](targets.md)`, ...).

Each of the six topic files MUST include these level-2 headings, in this order, each non-trivial:

1. `## Experiment-Specific Takeaway` — the one thing this experiment establishes for this topic.
2. `## Evidence Snapshot` — the specific numbers (with metric names) that back the takeaway.
3. `## Evidence Level` — state exactly one of: `verified artifact`, `computed metric`,
   `supported inference`, or `hypothesis / next-step`.
4. `## Design-Space Role` — state exactly one of: `varied`, `controlled`, `inherited`, `observed`,
   `not_tested`, or `confounded` — how this topic's axis was treated in this experiment.
5. `## Confounds` — what else moved that could explain the result.
6. `## What Not To Infer` — over-readings to avoid.
7. `## Not Established` — questions this experiment did not answer for this topic.
8. `## Scope And Caveats` — dataset tier, feature scope, target family, and other scope limits.
9. `## Future Implication` — what a successor experiment should test or reuse for this topic.
10. `## Master Ledger Update` — the concrete claim to carry into the master ledger for this topic
    (a recommendation for the next phase; this phase writes no master ledger).

Write only what the decision memo and evidence support. A topic the experiment did not exercise
should still be filled honestly with Evidence Level `hypothesis / next-step` and Design-Space Role
`not_tested`. Output only the JSON object, nothing else.
