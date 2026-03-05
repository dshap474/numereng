# Experiments

Use experiments to group related runs, track champions, and generate reports.

## Create

```bash
uv run numereng experiment create \
  --id 2026-02-22_baseline \
  --name "Baseline" \
  --hypothesis "Initial LGBM benchmark" \
  --tags baseline,lgbm
```

## Train Within an Experiment

```bash
uv run numereng experiment train \
  --id 2026-02-22_baseline \
  --config configs/run.json
```

Optional training overrides are the same as `run train`:

- `--output-dir`
- `--profile <simple|purged_walk_forward|submission>`
- `--store-root`

## Inspect

```bash
uv run numereng experiment list
uv run numereng experiment details --id 2026-02-22_baseline
uv run numereng experiment report --id 2026-02-22_baseline --format table
```

## Promote Champion

```bash
# Promote best run by metric (default metric: bmc_last_200_eras.mean)
uv run numereng experiment promote --id 2026-02-22_baseline

# Or promote an explicit run
uv run numereng experiment promote --id 2026-02-22_baseline --run <run_id>
```

## High-Risk Gotchas

- `experiment train` requires `--id` and `--config`.
- Config file must be `.json`.
- If you pass `--store-root`, keep `--output-dir` aligned with the same root.
