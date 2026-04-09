# Ensembles

Numereng builds rank-average ensembles from multiple scored runs.

## Build

```bash
uv run numereng ensemble build \
  --run-ids run_a,run_b,run_c \
  --method rank_avg \
  --metric corr_sharpe \
  --target target_ender_20 \
  --selection-note "diversity first" \
  --regime-buckets 4
```

Optional weighting controls:

- `--weights <w1,w2,...>`
- `--optimize-weights`

Optional neutralization stages:

- `--neutralize-members`
- `--neutralize-final`
- `--neutralizer-path <path>`
- `--neutralization-proportion <0..1>`
- `--neutralization-mode <era|global>`
- `--neutralizer-cols <csv>`
- `--no-neutralization-rank`

Optional heavy diagnostics:

- `--include-heavy-artifacts`

## Inspect

```bash
uv run numereng ensemble list
uv run numereng ensemble details --ensemble-id <ensemble_id>
```

## Artifact Locations

Ensembles are stored either:

- globally under `.numereng/ensembles/<ensemble_id>/`
- under an experiment at `experiments/<experiment_id>/ensembles/<ensemble_id>/`

Canonical files include:

- `predictions.parquet`
- `metrics.json`
- `weights.parquet`
- `correlation_matrix.parquet`
- `component_metrics.parquet`
- `era_metrics.parquet`
- `regime_metrics.parquet`
- `lineage.json`

Optional files:

- `component_predictions.parquet`
- `bootstrap_metrics.json`
- `predictions_pre_neutralization.parquet`

Numereng-managed parquet artifacts are written with `ZSTD` compression level `3`.

## High-Risk Gotchas

- `--run-ids` must contain at least two run IDs
- `--method` currently supports `rank_avg`
- `--weights` length must match the component count
- use `--optimize-weights` or explicit `--weights`, not conflicting weighting assumptions
