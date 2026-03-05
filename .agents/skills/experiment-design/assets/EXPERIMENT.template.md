# <experiment-name>

**ID**: `<YYYY-MM-DD_slug>`
**Created**: <YYYY-MM-DD>
**Updated**: <ISO timestamp>
**Status**: <draft|active|complete|archived>
**Champion run**: <run_id|none>
**Tags**: <tag1,tag2>

## Summary

- **Hypothesis**: <short hypothesis>
- **Primary metric**: `bmc_last_200_eras.mean`
- **Tie-break metric**: `bmc.mean`

## Ambiguity Resolution (If Applicable)

| interpretation_id | interpretation | scout configs tested | winner run_id | rationale |
|---|---|---|---|---|
| `int-a` | `<interpretation>` | `<config_a,config_b>` | `<run_id>` | `<why selected>` |

## Scout -> Scale Tracker

- Current stage: `<scout|scale>`
- Scout compute profile used (for example dataset variant/model capacity):
- Scale gate met: `<yes|no>`
- Confirmatory scaled round run: `<yes|no>`

## Plateau Gate Settings

- Primary stop metric: `bmc_last_200_eras.mean`
- Meaningful gain threshold: `<default 1e-4 to 3e-4>`
- Consecutive non-improving rounds required: `<default 2>`

## Round Log

### Round <N>

#### Intent
- Question this round answers:
- Single variable changed:
- Why this change now:

#### Configs Executed
| config_path | command | status |
|---|---|---|
| `.numereng/experiments/<id>/configs/<config>.json` | `uv run numereng experiment train --id <id> --config <config>.json` | <planned|running|done|failed> |

#### Results
| run_id | status (`run.json`) | created_at (`run.json`) | bmc_last_200_eras.mean | bmc.mean | corr.mean | mmc.mean | cwmm.mean | notes |
|---|---|---|---:|---:|---:|---:|---:|---|

#### Decision
- Winner:
- Round-best delta vs prior-best (`bmc_last_200_eras.mean`):
- Why winner:
- Risks observed:
- Plateau gate status: `<continue|pivot|stop>`
- Next round action:

## Ensemble Log (Optional)

### Ensemble <N>
- Command used:
- Method: `rank_avg`
- Optimization metric:
- Neutralization mode:
- Include heavy artifacts: `<yes|no>`
- Artifacts path:

| ensemble_id | run_ids | method | metric | weights source | heavy artifacts | artifacts_path | notes |
|---|---|---|---|---|---|---|---|
| `<ensemble_id>` | `<run_a,run_b,...>` | `rank_avg` | `<corr20v2_sharpe|corr20v2_mean|max_drawdown>` | `<explicit|optimized|equal>` | `<yes|no>` | `<path>` | `<selection rationale>` |

Artifact checklist:
- `weights.csv`
- `component_metrics.csv`
- `era_metrics.csv`
- `regime_metrics.csv`
- `lineage.json`
- optional heavy: `component_predictions.parquet`, `bootstrap_metrics.json`
- optional final-neutralization: `predictions_pre_neutralization.parquet`

## Remaining Knobs Audit

| knob/dimension | tried ranges | remaining options | expected value | overfit risk | decision |
|---|---|---|---|---|---|
| `model.params.learning_rate` | `<...>` | `<...>` | `<high|medium|low>` | `<high|medium|low>` | `<continue|defer|drop>` |

## Final Decision

- Selected champion run:
- Promotion command used:
- Promotion metric/value:

## Findings

- What worked:
- What did not:
- Unexpected observations:

## Anti-Patterns Observed

- <anti-pattern>

## Next Experiments

1. 
2. 
3. 

## Repro Commands

```bash
uv run numereng experiment details --id <id> --format json
uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format table
uv run numereng ensemble details --ensemble-id <ensemble_id> --format json
uv run numereng experiment promote --id <id> --metric bmc_last_200_eras.mean
```
