<!--
Hidden local rules file for __RESEARCH_MEMORY__.
-->

# Research Memory Contract

## Purpose

- compress evidence into decisions
- keep one rolling frontier
- keep one durable review per experiment
- preserve history without turning the system into a diary

## Canonical Files

- `CURRENT.md`
- `experiments/<experiment-id>.md`
- `topics/*.md`
- `decisions/*.md`

## Canonical Truth Hierarchy

- `CURRENT.md` is the sole canonical present-tense decision layer
- `topics/*.md` support `CURRENT.md` with scoped durable beliefs
- `experiments/*.md` preserve original result plus current interpretation for one experiment
- `decisions/*.md` explain why `CURRENT.md` changed

## Workflow Modes

- `experiment-ingest`
- `resynthesis`

## Update Rules

- reviews preserve provenance but must surface `Current Interpretation`
- ledgers store scoped beliefs, not universal claims
- freshness must be tracked lightly as `fresh | aging | stale`
- anti-patterns require reopen conditions
- decision notes only on material frontier shifts
- executed experiment artifacts remain the default evidence base only after a lightweight evidence-quality check
- broader history must always be consulted as a base-rate check before promotion

## Metric Basis

- `bmc_last_200_eras.mean`
- `bmc.mean`
- `corr.mean`
- `mmc.mean`
- `cwmm.mean`
