# Ensembles

Use `numereng ensemble` when you want numereng to blend scored runs into one persisted ensemble artifact.

## Build

```bash
uv run numereng ensemble build \
  --run-ids run_a,run_b,run_c \
  --method rank_avg \
  --metric corr_sharpe \
  --target target_ender_20 \
  --selection-note "diversity first"
```

Optional controls:

- `--weights <w1,w2,...>`
- `--optimize-weights`
- `--regime-buckets <n>`
- `--neutralize-members`
- `--neutralize-final`
- `--neutralizer-path <path>`
- `--include-heavy-artifacts`

## Inspect

```bash
uv run numereng ensemble list
uv run numereng ensemble details --ensemble-id <ensemble_id>
```

## Select From Historical Source Experiments

Create one source-rules JSON file whose `experiment_id` order matches `--source-experiment-ids`:

```json
[
  {
    "experiment_id": "2026-04-10_seed-sweep",
    "selection_mode": "top_n",
    "top_n": 3
  },
  {
    "experiment_id": "2026-04-12_diversifiers",
    "selection_mode": "explicit_targets",
    "explicit_targets": ["target", "target_ender_20"]
  }
]
```

```bash
uv run numereng ensemble select \
  --experiment-id 2026-04-18_lgbm-baseline \
  --source-experiment-ids 2026-04-10_seed-sweep,2026-04-12_diversifiers \
  --source-rules configs/ensembles/source_rules.json
```

Use this when you want numereng to freeze candidate bundles from one or more source experiments, then persist the winning selection under one target experiment.

## Artifact Locations

Ensembles are stored either:

- globally under `.numereng/ensembles/<ensemble_id>/`
- under an experiment at `.numereng/experiments/<experiment_id>/ensembles/<ensemble_id>/`

Typical files:

- `predictions.parquet`
- `metrics.json`
- `weights.parquet`
- `correlation_matrix.parquet`
- `component_metrics.parquet`
- `lineage.json`

Optional heavier artifacts include component predictions and regime or bootstrap diagnostics.

## High-Risk Gotchas

- `--run-ids` must contain at least two runs
- `rank_avg` is the current public blend method
- `--weights` length must match the number of component runs
- `ensemble select` requires both `--source-experiment-ids` and `--source-rules`
- for `selection_mode = "explicit_targets"`, `explicit_targets` refers to source `target_col` values, not run IDs
- `source_rules` experiment order must match `--source-experiment-ids` exactly
- if you neutralize, treat the neutralizer inputs like any other production-side dependency

## Read Next

- [Serving & Model Uploads](serving.md)
- [Submissions](submission.md)
