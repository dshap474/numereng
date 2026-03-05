# Ensembles

Build and inspect rank-average ensembles from multiple run predictions.

## Build

```bash
uv run numereng ensemble build \
  --run-ids run_a,run_b,run_c \
  --method rank_avg \
  --metric corr20v2_sharpe \
  --target target_ender_20 \
  --selection-note "diversity first" \
  --regime-buckets 4
```

Optional weighting controls:

- `--weights 0.5,0.3,0.2` (must match component count)
- `--optimize-weights`

Optional neutralization stages:

- `--neutralize-members`
- `--neutralize-final`
- `--neutralizer-path <path>`
- `--neutralization-proportion <0..1>`
- `--neutralization-mode <era|global>`
- `--neutralizer-cols <csv>`
- `--no-neutralization-rank`

Optional diagnostics:

- `--include-heavy-artifacts`

## Inspect

```bash
uv run numereng ensemble list
uv run numereng ensemble details --ensemble-id <ensemble_id>
```

## Artifacts

Ensemble artifacts are persisted under either:

- `.numereng/ensembles/<ensemble_id>/`
- `.numereng/experiments/<experiment_id>/ensembles/<ensemble_id>/`

Canonical files include:

- `predictions.parquet`
- `correlation_matrix.csv`
- `metrics.json`
- `weights.csv`
- `component_metrics.csv`
- `era_metrics.csv`
- `regime_metrics.csv`
- `lineage.json`

Optional files:

- `component_predictions.parquet` (heavy)
- `bootstrap_metrics.json` (heavy)
- `predictions_pre_neutralization.parquet` (when `--neutralize-final`)

## High-Risk Gotchas

- `--run-ids` must contain at least two run IDs.
- `--method` currently only supports `rank_avg`.
- `--weights` length must match number of components.
