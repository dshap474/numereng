<!--
Hidden local rules file for __RESEARCH_MEMORY__.
-->

# Research Memory Contract

## Purpose

- compress evidence into decisions
- keep one rolling frontier
- keep one durable review per experiment
- preserve history without turning the system into a diary

## Canonical files

- `CURRENT.md`
- `experiments/<experiment-id>.md`
- `topics/*.md`
- `decisions/*.md`

## Update rules

- reviews are mostly immutable
- later material changes append addenda
- decision notes only on material frontier shifts

## Metric basis

- `bmc_last_200_eras.mean`
- `bmc.mean`
- `corr.mean`
- `mmc.mean`
- `cwmm.mean`
