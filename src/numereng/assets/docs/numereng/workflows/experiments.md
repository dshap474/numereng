# Experiments

Experiments group related runs, track champions, and keep experiment-local artifacts together.

## Create

```bash
uv run numereng experiment create \
  --id 2026-03-12_lgbm-baseline \
  --name "LGBM baseline" \
  --hypothesis "small feature-set baseline before wider sweeps" \
  --tags baseline,lgbm
```

This creates `.numereng/experiments/<experiment_id>/` with:

- `experiment.json`
- `EXPERIMENT.md` seeded to the current report-section scaffold
- `configs/`
- `run_plan.csv` with the canonical header stub
- `run_scripts/launch_all.py`, `launch_all.sh`, and `launch_all.ps1`

Fill in `run_plan.csv` only when the experiment is using an ordered sweep.

## Train Within An Experiment

```bash
uv run numereng experiment train \
  --id 2026-03-12_lgbm-baseline \
  --config configs/run.json
```

Optional overrides:

- `--output-dir <path>`
- `--profile <simple|purged_walk_forward|full_history_refit>`
- `--post-training-scoring <none|core|full|round_core|round_full>`
- `--workspace <path>`

`experiment train` uses the same training engine profiles as `run train`, but it links the resulting run to the experiment manifest and report surfaces.

Recommended scoring policy by round:

- scripted sweeps using the generated `run_scripts/launch_all.*`: keep every config at `training.post_training_scoring = "none"` and let the launcher call `experiment score-round`
- manual non-scripted workflows: early configs in a round use `none`, and the last `rN_*` config may use `round_core` or `round_full`

Those patterns both train the round first, then trigger one deferred `experiment score-round`
batch pass after the last config links into the manifest.

## Batch Score One Round

```bash
uv run numereng experiment score-round \
  --id 2026-03-12_lgbm-baseline \
  --round r1 \
  --stage post_training_full
```

Use this to materialize deferred scoring for all eligible `FINISHED` runs in one
round. It is also the manual recovery path if automatic `round_core` or
`round_full` scoring fails.

## Inspect

```bash
uv run numereng experiment list
uv run numereng experiment details --id 2026-03-12_lgbm-baseline
uv run numereng experiment report --id 2026-03-12_lgbm-baseline --format table
uv run numereng experiment pack --id 2026-03-12_lgbm-baseline
```

Useful artifacts under `.numereng/experiments/<experiment_id>/`:

- `experiment.json`
- `EXPERIMENT.md`
- `EXPERIMENT.pack.md`
- `configs/*.json`
- `hpo/`
- `ensembles/`

`experiment pack` overwrites `EXPERIMENT.pack.md` in the experiment folder. The packed file
contains the current `EXPERIMENT.md` narrative plus one run-summary table with dashboard-aligned
scalar metrics only, not per-era or other time-series artifacts.

## Promote A Champion

```bash
# default metric-based promotion
uv run numereng experiment promote --id 2026-03-12_lgbm-baseline

# explicit run promotion
uv run numereng experiment promote --id 2026-03-12_lgbm-baseline --run <run_id>
```

If `--run` is omitted, numereng promotes the best candidate by the selected metric.

## High-Risk Gotchas

- `experiment train` requires both `--id` and `--config`
- config files must end in `.json`
- if you override `--workspace`, keep experiment paths and output paths aligned to the same store
- `full_history_refit` should be reserved for final refits because it emits no validation metrics
- `round_core` and `round_full` require `experiment train` plus an `rN_*` config filename
