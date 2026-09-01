# Comparison Policy

Research-memory updates use two comparison passes.

## 1. Primary Relevant Subset

This is the main context for deciding what the experiment means.

Select in this order:

1. target family / horizon
2. feature set
3. model
4. hyperparameter region
5. stated hypothesis / branch intent

Rules:

- this pass drives frontier interpretation in `CURRENT.md`
- if evaluation comparability fails, record that in `CURRENT.md`, not as a topic branch
- if multiple candidates tie, cite all tied candidates rather than silently choosing one

## 2. Base-Rate And Contradiction Sweep

This is a global consistency check across the whole experiment history.

Use it for:

- contradiction checks
- repeated dead-end detection
- adjacent supporting evidence
- base-rate reconciliation

Every frontier recommendation must explicitly state whether broader history:

- supports
- weakens
- contradicts

Do not let weakly related global evidence silently override a strong comparable subset, but do let it reduce confidence.

## Evidence Posture

- `frontier-grade`: strong enough to shape the frontier directly
- `mixed`: useful but heterogeneous or only partly comparable
- `supporting`: incomplete, degraded, or too weak for direct frontier movement

## Comparison Class

Classify the experiment before writing memory:

- `broad screening surface`: many targets/configs; useful for candidate discovery and base rates.
- `narrow candidate-quality packet`: pruned target/config set; useful for candidate development.
- `champion / production evidence`: scored ensemble or production-ready workflow with required handoff checks.

Do not compare these classes as if they answer the same question. A narrow candidate packet can be cleaner than a broad screen without replacing the broad screen as a search surface.

## Surface Discipline

- comparable strong surfaces can directly change defaults and the top-ranked next move
- smoke / `simple` / staged results can narrow menus and identify challengers
- staged wins must be labeled as promotion candidates, not final frontier conclusions
- mixed-surface experiments should usually produce mixed posture plus claim-level quality notes
