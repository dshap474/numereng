# Comparison Policy

Research-memory updates use two comparison passes.

## 1. Primary relevant subset

This is the main context for deciding what the experiment means.

Select in this order:

1. evaluation surface / validation profile
2. target family / horizon
3. feature scope
4. model family
5. stated hypothesis / branch intent

Rules:

- this pass drives frontier interpretation
- if surface comparability fails, evidence cannot be treated as direct frontier proof
- if multiple candidates tie, cite all tied candidates rather than silently choosing one

## 2. Base-rate and contradiction sweep

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

- `frontier-grade`
  - strong enough to shape the frontier directly
- `mixed`
  - useful but heterogeneous or only partly comparable
- `supporting`
  - incomplete, degraded, or too weak for direct frontier movement

## Surface Discipline

- comparable strong surfaces can directly change defaults and the top-ranked next move
- smoke / `simple` / staged results can narrow menus and identify challengers
- staged wins must be labeled as promotion candidates, not final frontier conclusions
- mixed-surface experiments should usually produce mixed posture plus claim-level quality notes
