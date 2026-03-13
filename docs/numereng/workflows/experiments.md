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

This creates `.numereng/experiments/<experiment_id>/` with an `experiment.json` manifest.

## Train Within An Experiment

```bash
uv run numereng experiment train \
  --id 2026-03-12_lgbm-baseline \
  --config configs/run.json
```

Optional overrides:

- `--output-dir <path>`
- `--profile <simple|purged_walk_forward|full_history_refit>`
- `--store-root <path>`

`experiment train` uses the same training engine profiles as `run train`, but it links the resulting run to the experiment manifest and report surfaces.

## Inspect

```bash
uv run numereng experiment list
uv run numereng experiment details --id 2026-03-12_lgbm-baseline
uv run numereng experiment report --id 2026-03-12_lgbm-baseline --format table
```

Useful artifacts under `.numereng/experiments/<experiment_id>/`:

- `experiment.json`
- `EXPERIMENT.md`
- `configs/*.json`
- `hpo/`
- `ensembles/`

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
- if you override `--store-root`, keep experiment paths and output paths aligned to the same store
- `full_history_refit` should be reserved for final refits because it emits no validation metrics
