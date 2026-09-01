# <experiment-id>

**Extracted**: <YYYY-MM-DD>
**Source report**: `.numereng/experiments/<experiment-id>/EXPERIMENT.md`
**Purpose**: per-experiment design-space topic branch

## Final Verdict

- <one-paragraph or short-bullet verdict from finalized EXPERIMENT.md>

## Evidence Status

- <artifact completeness and source quality summary>

## Decision Snapshot

| Decision | Result | Evidence level | Why |
|---|---|---|---|
| Best single run | `<run_id or n/a>` | `<computed metric|supported inference>` | <metric-backed reason> |
| Candidate family | `<family or n/a>` | `<computed metric|supported inference>` | <replication/tradeoff reason> |
| Stabilizer candidate | `<candidate or n/a>` | `<supported inference|hypothesis / next-step>` | <why it is or is not established> |
| Ensemble candidate | `<candidate or n/a>` | `<supported inference|hypothesis / next-step>` | <only use as established if an ensemble artifact exists> |
| Champion | `<none or run/model>` | `<verified artifact|supported inference>` | <promotion decision and caveat> |

## Topic Branches

- [Features](features.md)
- [Targets](targets.md)
- [Models](models.md)
- [Hyperparameters](hyperparameters.md)
- [Ensembling](ensembling.md)
- [Neutralization And Exposure](neutralization-exposure.md)

## Comparison And Selection Context

- Comparison class: `<broad screening surface|narrow candidate-quality packet|champion / production evidence>`.
- Scope boundary: <what this experiment can and cannot be compared against>.
- Confounds: <design choices that changed together>.
- <baseline/comparison/evaluation context that should not be copied into topic ledgers>

## Preserved Prior Review

- None.
