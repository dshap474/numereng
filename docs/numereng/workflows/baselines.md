# Baselines & Active Benchmark

Use this workflow when you want benchmark-relative scoring to work reliably for
local training runs.

## Two Different Benchmark Inputs

Numereng uses two different benchmark-related inputs that are easy to confuse:

- official Numerai benchmark-model dataset files under
  `.numereng/datasets/<data_version>/*benchmark_models.parquet`
- a separate local scoring artifact under
  `.numereng/datasets/baselines/active_benchmark/`

The official dataset files are source data. They are not the same thing as the
active benchmark artifact used by Numereng's default benchmark-relative scoring.

## Baseline Directory Contract

Named baselines live under:

- `.numereng/datasets/baselines/<baseline_name>/`

Typical files inside one baseline directory:

- `baseline.json`
- one predictions parquet
- optional shared per-era CORR artifacts

Example local shape:

```text
.numereng/datasets/baselines/
  medium_ender20_ender60_6run_blend/
    baseline.json
    pred_medium_ender20_ender60_6run_blend.parquet
    val_per_era_corr20v2_target_ender_20.parquet
    val_per_era_corr20v2_target_ender_60.parquet
  active_benchmark/
    benchmark.json
    predictions.parquet
```

`baseline.json` is the durable metadata for one named baseline. In the current
local store it records things like:

- `name`
- `kind`
- `default_target`
- `available_targets`
- `source_experiment_id`
- `source_run_ids`
- `artifacts.predictions`
- optional shared per-era CORR artifact paths

## Active Benchmark Contract

The canonical active benchmark lives under:

- `.numereng/datasets/baselines/active_benchmark/predictions.parquet`
- `.numereng/datasets/baselines/active_benchmark/benchmark.json`

This is the artifact Numereng uses when a training config leaves
`data.benchmark_source` at its default.

The active metadata file is a promoted copy of one named baseline's metadata.
The active predictions parquet is a promoted copy of that baseline's prediction
artifact.

## Two Supported Scoring Modes

### `benchmark_source.source = "active"`

This is the default.

Use it when the repository already has a seeded active benchmark artifact.

Pros:

- shortest config
- shared benchmark provenance across runs
- supports shared active-benchmark CORR delta artifacts when present

Risk:

- runs can complete training but leave benchmark-relative scoring pending if
  `active_benchmark/predictions.parquet` has not been seeded yet

### `benchmark_source.source = "path"`

Use an explicit parquet path when there is no seeded active benchmark yet, or
when you want one run to score against a specific baseline without changing the
repo-wide active benchmark.

Example:

```json
{
  "data": {
    "benchmark_source": {
      "source": "path",
      "predictions_path": ".numereng/datasets/baselines/medium_ender20_ender60_6run_blend/pred_medium_ender20_ender60_6run_blend.parquet",
      "pred_col": "prediction",
      "name": "medium_ender20_ender60_6run_blend"
    }
  }
}
```

This is the safer explicit path for smoke runs, experiments, and environments
that have not yet been initialized with an active benchmark.

## Current Repo Gap

There is currently no public `numereng baselines ...` CLI family.

What does exist today:

- `data.benchmark_source` in the public training config contract
- internal helper `seed_active_benchmark(...)` in
  `src/numereng/features/training/repo.py`

So the workflow is supported by the repo, but the seeding step is not yet
exposed as a public CLI command.

## Operating Recipe

1. Train or build a baseline-producing run or ensemble.
2. Persist a named baseline directory under
   `.numereng/datasets/baselines/<baseline_name>/`.
3. Write `baseline.json` in that directory and keep the predictions parquet
   there as the durable source artifact.
4. Seed the canonical active benchmark by copying that baseline into
   `.numereng/datasets/baselines/active_benchmark/`.
5. Confirm later runs using default `benchmark_source.source = "active"` can
   score BMC, MMC, and `corr_delta_vs_baseline`.

## Choosing Between `active` And `path`

Use `active` when:

- the repo already has a trusted shared baseline
- you want a stable default benchmark across many runs

Use `path` when:

- bootstrapping a new machine or clone
- evaluating a one-off candidate baseline
- scoring against a specific baseline without promoting it to shared default

## High-Risk Gotchas

- the official `*benchmark_models.parquet` files are not a drop-in replacement
  for `active_benchmark/predictions.parquet`
- leaving `benchmark_source` implicit still resolves to `source = "active"`
- if `active_benchmark` is missing, native training can still succeed while
  benchmark-relative scoring stays incomplete or pending
- if you use `source = "path"`, `predictions_path` is required
