# Research Round Log: <experiment-id>

## Experiment Setup
- Hypothesis:
- Baseline run ref:
- Feature set:
- Primary metric: `bmc_last_200_eras.mean`
- Tie-break metric: `bmc.mean`

## Round <N>

### Intent
- What question this round answers:
- Interpretation tag (if ambiguity was resolved):
- Sweep dimension:
- Why now:
- Stage: `<scout|scale>`

### Configs
| config | single change | interpretation tag | command | status |
|---|---|---|---|---|
| `<config>.json` | `<what changed>` | `<int-a|n/a>` | `uv run numereng experiment train --id <id> --config <config>.json` | `<planned|running|done|failed>` |

### Results
| run_id | config | bmc_last_200_eras.mean | delta vs prior-best | bmc.mean | corr.mean | mmc.mean | cwmm.mean | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|

### Decision
- Winner:
- Why winner:
- Risks observed:
- Plateau gate status: `<continue|pivot|stop>`
- Next round action:

### Ensemble Follow-Up (Optional)
- Candidate run IDs:
- Planned build command: `uv run numereng ensemble build --experiment-id <id> --run-ids <run_a,run_b,...> --method rank_avg --metric corr20v2_sharpe`
- Weight mode: `<explicit|optimized|equal>`
- Ensemble ID:
- Artifacts path:
- Artifact check (`weights.csv`, `component_metrics.csv`, `era_metrics.csv`, `regime_metrics.csv`, `lineage.json`):
