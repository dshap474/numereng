# Medium LGBM GPU Ender20 HPO

**ID**: `2026-04-09_medium-lgbm-gpu-ender20-hpo`
**Created**: 2026-04-10
**Updated**: 2026-04-10T03:53:34.446587+00:00
**Status**: draft
**Champion run**: none
**Tags**: medium,lgbm,gpu,ender20,hpo,v5.2

## Summary

- **Hypothesis**: Optuna can find a stronger medium-feature GPU LightGBM parameter set for target_ender_20 on the 3090 than the current fixed anchors.
- **Primary metric**: `bmc_last_200_eras.mean`
- **Tie-break metric**: `bmc.mean`
- **Outcome**: <what the experiment established so far>

## Abstract

- What was tested:
- Headline result:
- Best run / leading candidate:
- Why this result matters:

## Method

- Data split / CV setup: `purged_walk_forward`
- Feature set or dataset scope: `medium`, `train_plus_validation`, `non_downsampled`
- Model family and key hyperparameters: `LGBMRegressor` on GPU via `device_type="gpu"` with an Optuna TPE search around the medium standard-large anchor
- Target(s) evaluated: `target_ender_20`
- Transforms / neutralization / special scoring notes: study objective is `post_fold_champion_objective`, so the search uses fold-time corr/BMC evidence without requiring deferred round scoring

## Execution Inventory

- Planned configs:
- Executed configs:
- Failed / interrupted configs:
- Skipped / superseded configs:
- Launcher path (if used): `experiments/2026-04-09_medium-lgbm-gpu-ender20-hpo/run_scripts/launch_all.sh`
- Verification source of truth: `experiment.json`, `run_plan.csv`, `run.json`, `metrics.json`, `results.json`, `score_provenance.json`
- Notes on what was actually run versus what remains only planned:

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
| `experiments/2026-04-09_medium-lgbm-gpu-ender20-hpo/configs/<config>.json` | `uv run numereng experiment train --id 2026-04-09_medium-lgbm-gpu-ender20-hpo --config <config>.json --post-training-scoring none` | <planned|running|done|failed> |

Execution notes:
- State explicitly if a config was skipped, superseded, deduplicated, or replaced by a recovered canonical run.
- If a round used a launcher script, still record the canonical per-config command family.

#### Results
| run_id | status (`run.json`) | created_at (`run.json`) | bmc_last_200_eras.mean | bmc.mean | corr.mean | mmc.mean | cwmm.mean | notes |
|---|---|---|---:|---:|---:|---:|---:|---|

Artifact checks:
- Confirm whether each listed run has `run.json`, `resolved.json`, `metrics.json`, `results.json`, `score_provenance.json`, and persisted predictions.
- If a required artifact is missing, state it here instead of treating the run as fully complete.

#### Decision
- Winner:
- Round-best delta vs prior-best (`bmc_last_200_eras.mean`):
- Why winner:
- Risks observed:
- Plateau gate status: `<continue|pivot|stop>`
- Next round action:

## Results

- Best run on primary metric:
- Best run on tie-break metric:
- Main trade-offs across leading runs:
- Detailed evidence source: `Round Log` and `EXPERIMENT.pack.md`

Compact summary table (use one row per executed run or per leading candidate set):

| run_id | config_path | round | status | bmc_last_200_eras.mean | bmc.mean | corr.mean | mmc.mean | avg_corr_with_benchmark | notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `<run_id>` | `experiments/2026-04-09_medium-lgbm-gpu-ender20-hpo/configs/<config>.json` | `<N>` | `<FINISHED>` | `<...>` | `<...>` | `<...>` | `<...|n/a>` | `<...|n/a>` | `<trade-off or selection note>` |

Result interpretation:
- What pattern actually changed the leaderboard:
- What looked good on one metric but failed on another:
- Whether the best result is robust enough to promote:

## Standard Plots / Visual Checks

- Plot objective: `<benchmark comparison|candidate comparison|seed stability|other>`
- Plot generation command:
- Artifact path(s):
- Included in report: `<yes|no>`
- If no plot was generated, explain why and what scalar evidence substituted for it:

| plot_id | purpose | command | artifact_path | notes |
|---|---|---|---|---|
| `<plot-1>` | `<what this plot shows>` | `<command>` | `<relative/path.png>` | `<interpretation or caveat>` |

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

## Stopping Rationale

- Why iteration stopped:
- Plateau or diminishing-returns evidence:
- Confirmatory run or scale-check evidence:
- Remaining uncertainty accepted:

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

## Final Checks

- `EXPERIMENT.md` clearly separates executed configs from planned-only configs.
- Metrics reported here match the underlying run artifacts.
- Linked artifact paths and plot paths resolve.
- Champion run is either recorded or `none` is explained.
- If the experiment is complete, `EXPERIMENT.pack.md` has been regenerated.

## Repro Commands

```bash
uv run numereng experiment details --id 2026-04-09_medium-lgbm-gpu-ender20-hpo --format json
uv run numereng experiment report --id 2026-04-09_medium-lgbm-gpu-ender20-hpo --metric bmc_last_200_eras.mean --format table
uv run numereng hpo create --study-config experiments/2026-04-09_medium-lgbm-gpu-ender20-hpo/configs/hpo_study_v1.json
bash experiments/2026-04-09_medium-lgbm-gpu-ender20-hpo/run_scripts/launch_all.sh
uv run numereng experiment pack --id 2026-04-09_medium-lgbm-gpu-ender20-hpo
uv run numereng ensemble details --ensemble-id <ensemble_id> --format json
uv run numereng experiment promote --id 2026-04-09_medium-lgbm-gpu-ender20-hpo --metric bmc_last_200_eras.mean
# optional plot / viz command:
# <plot command here>
```
