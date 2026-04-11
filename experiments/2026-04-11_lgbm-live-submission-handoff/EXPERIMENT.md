# LGBM Live Submission Handoff

**ID**: `2026-04-11_lgbm-live-submission-handoff`
**Created**: 2026-04-11
**Updated**: 2026-04-11T17:31:59.640077+00:00
**Status**: draft
**Champion run**: none
**Tags**: submission,handoff,lgbm,live

## Summary

- **Hypothesis**: Final-fit the April 8 winning LGBM blend components and prepare a live submission handoff artifact set.
- **Primary metric**: `bmc_last_200_eras.mean`
- **Tie-break metric**: `bmc.mean`
- **Outcome**: handoff prepared; four `full_history_refit` configs are frozen and the live blend weights are locked in `submission_handoff.json`

## Abstract

- What was tested: no new comparative training; this experiment packages the selected winner from `2026-04-08_lgbm-cross-scope-blend-selection` into final-fit components that can be turned into a live submission
- Headline result: winning live handoff remains the unneutralized weighted cross-scope blend
- Best run / leading candidate: weighted `top2_medium_top2_small`
- Why this result matters: numereng can submit live predictions, but it cannot yet generate them end-to-end from the selected blend; this experiment freezes the exact refit and blend contract

## Method

- Data split / CV setup: `full_history_refit` on `train_plus_validation`
- Feature set or dataset scope: two `medium` targets and two `small` targets
- Model family and key hyperparameters: `LGBMRegressor`, `learning_rate=0.01`, `num_leaves=64`, `max_depth=6`, `n_estimators=3000`, `colsample_bytree=0.1`
- Target(s) evaluated: `target_alpha_20`, `target_charlie_20`, `target_jasper_60`, `target_alpha_20`
- Transforms / neutralization / special scoring notes: no neutralization promotion; live blend rule is per-era rank of each component followed by weighted sum with no final rerank

## Execution Inventory

- Planned configs: four final-fit configs listed in `run_plan.csv`
- Executed configs: none yet
- Failed / interrupted configs:
- Skipped / superseded configs:
- Launcher path (if used): `experiments/2026-04-11_lgbm-live-submission-handoff/run_scripts/launch_all.sh`
- Verification source of truth: `experiment.json`, `run_plan.csv`, `run.json`, `metrics.json`, `results.json`, `score_provenance.json`
- Notes on what was actually run versus what remains only planned: this experiment is prepared but not executed; the blend winner came from `2026-04-08_lgbm-cross-scope-blend-selection/analysis/blends/final_selection.json`

## Ambiguity Resolution (If Applicable)

| interpretation_id | interpretation | scout configs tested | winner run_id | rationale |
|---|---|---|---|---|
| `int-a` | `<interpretation>` | `<config_a,config_b>` | `<run_id>` | `<why selected>` |

## Scout -> Scale Tracker

- Current stage: `scale`
- Scout compute profile used (for example dataset variant/model capacity): completed on March baseline sweeps plus April blend-selection workflow
- Scale gate met: `yes`
- Confirmatory scaled round run: `pending final-fit execution`

## Plateau Gate Settings

- Primary stop metric: `bmc_last_200_eras.mean`
- Meaningful gain threshold: `n/a`
- Consecutive non-improving rounds required: `n/a`

## Round Log

### Round 1

#### Intent
- Question this round answers: can the selected April 8 winner be refit cleanly into four final-fit components for live use
- Single variable changed: training profile only, from `purged_walk_forward` to `full_history_refit`
- Why this change now: the blend is selected and post-processing did not improve it; the next value is live data, not more local tuning

#### Configs Executed
| config_path | command | status |
|---|---|---|
| `experiments/2026-04-11_lgbm-live-submission-handoff/configs/r1_medium_target_alpha_20_seed42_full_history_refit.json` | `uv run numereng experiment train --id 2026-04-11_lgbm-live-submission-handoff --config configs/r1_medium_target_alpha_20_seed42_full_history_refit.json --profile full_history_refit --post-training-scoring none` | planned |
| `experiments/2026-04-11_lgbm-live-submission-handoff/configs/r1_medium_target_charlie_20_seed43_full_history_refit.json` | `uv run numereng experiment train --id 2026-04-11_lgbm-live-submission-handoff --config configs/r1_medium_target_charlie_20_seed43_full_history_refit.json --profile full_history_refit --post-training-scoring none` | planned |
| `experiments/2026-04-11_lgbm-live-submission-handoff/configs/r1_small_target_jasper_60_seed42_full_history_refit.json` | `uv run numereng experiment train --id 2026-04-11_lgbm-live-submission-handoff --config configs/r1_small_target_jasper_60_seed42_full_history_refit.json --profile full_history_refit --post-training-scoring none` | planned |
| `experiments/2026-04-11_lgbm-live-submission-handoff/configs/r1_small_target_alpha_20_seed42_full_history_refit.json` | `uv run numereng experiment train --id 2026-04-11_lgbm-live-submission-handoff --config configs/r1_small_target_alpha_20_seed42_full_history_refit.json --profile full_history_refit --post-training-scoring none` | planned |

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
- Winner: pending execution; frozen blend contract is in `submission_handoff.json`
- Round-best delta vs prior-best (`bmc_last_200_eras.mean`): `n/a`
- Why winner: selected upstream in April 8 blend-selection workflow
- Risks observed: numereng still lacks first-class live inference generation from final-fit runs
- Plateau gate status: `continue`
- Next round action: run the four refits, generate one live predictions parquet per component, blend them with `run_scripts/blend_live_predictions.py`, then submit

## Results

- Best run on primary metric: weighted `top2_medium_top2_small` from `2026-04-08_lgbm-cross-scope-blend-selection`
- Best run on tie-break metric: same winner for this handoff
- Main trade-offs across leading runs: the winner is a thin but real gain over `small_target_jasper_60`; medium branches mainly diversify the small-side core
- Detailed evidence source: `Round Log` and `EXPERIMENT.pack.md`

Compact summary table (use one row per executed run or per leading candidate set):

| run_id | config_path | round | status | bmc_last_200_eras.mean | bmc.mean | corr.mean | mmc.mean | avg_corr_with_benchmark | notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `blend_winner` | `experiments/2026-04-11_lgbm-live-submission-handoff/submission_handoff.json` | `0` | `selected` | `0.0034606039` | `0.0076615204` | `0.0176040021` | `n/a` | `0.3398025430` | `weighted components: 0.05 medium_alpha_20, 0.00 medium_charlie_20, 0.65 small_jasper_60, 0.30 small_alpha_20` |

Result interpretation:
- What pattern actually changed the leaderboard: weighted cross-scope blending slightly improved the strong `small_only` branch
- What looked good on one metric but failed on another: neutralized variants lowered benchmark correlation but hurt both recent and full BMC
- Whether the best result is robust enough to promote: yes for live collection, but it should be treated as a thin-edge candidate rather than a locked champion

## Standard Plots / Visual Checks

- Plot objective: `not generated in handoff experiment`
- Plot generation command:
- Artifact path(s):
- Included in report: `no`
- If no plot was generated, explain why and what scalar evidence substituted for it: the handoff reuses scalar evidence already frozen in the April 8 selection artifacts

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

- Selected champion run: weighted `top2_medium_top2_small`
- Promotion command used: `n/a`
- Promotion metric/value: `bmc_last_200_eras.mean = 0.0034606039`

## Stopping Rationale

- Why iteration stopped: local selection is complete and the next information source is live tournament behavior
- Plateau or diminishing-returns evidence: post-processing and neutralization sweeps did not beat the raw winner
- Confirmatory run or scale-check evidence: blend beat its components on the primary metric, but only narrowly
- Remaining uncertainty accepted: the live edge may be smaller than local validation suggests

## Findings

- What worked: cross-scope weighted blending of LGBM targets
- What did not: feature neutralization at `0.1` through `0.5`
- Unexpected observations: `medium_target_charlie_20` survived selection but ended with zero final weight

## Anti-Patterns Observed

- treating CV prediction files as if they were directly submit-ready live artifacts

## Next Experiments

1. Run the four `full_history_refit` configs in this experiment.
2. Generate one live predictions parquet per refit component for the current Numerai round.
3. Blend those live predictions with `run_scripts/blend_live_predictions.py` and submit the resulting parquet.

## Final Checks

- `EXPERIMENT.md` clearly separates executed configs from planned-only configs.
- Metrics reported here match the underlying run artifacts.
- Linked artifact paths and plot paths resolve.
- Champion run is either recorded or `none` is explained.
- If the experiment is complete, `EXPERIMENT.pack.md` has been regenerated.

## Repro Commands

```bash
uv run numereng experiment details --id 2026-04-11_lgbm-live-submission-handoff --format json
uv run numereng experiment report --id 2026-04-11_lgbm-live-submission-handoff --metric bmc_last_200_eras.mean --format table
bash experiments/2026-04-11_lgbm-live-submission-handoff/run_scripts/launch_all.sh
uv run numereng experiment pack --id 2026-04-11_lgbm-live-submission-handoff
uv run numereng experiment promote --id 2026-04-11_lgbm-live-submission-handoff --metric bmc_last_200_eras.mean
uv run python experiments/2026-04-11_lgbm-live-submission-handoff/run_scripts/blend_live_predictions.py \
  --input medium_target_alpha_20=/abs/path/medium_alpha_live.parquet \
  --input medium_target_charlie_20=/abs/path/medium_charlie_live.parquet \
  --input small_target_jasper_60=/abs/path/small_jasper_live.parquet \
  --input small_target_alpha_20=/abs/path/small_alpha_live.parquet \
  --output experiments/2026-04-11_lgbm-live-submission-handoff/predictions/live_weighted_blend.parquet
uv run numereng run submit \
  --model-name <model_name> \
  --predictions experiments/2026-04-11_lgbm-live-submission-handoff/predictions/live_weighted_blend.parquet
```
