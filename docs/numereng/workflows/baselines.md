# Baselines & Active Benchmark

Use the `baseline` workflow when you want a shared benchmark artifact for benchmark-relative scoring and BMC-style comparisons.

## Why This Exists

Numereng distinguishes between:

- official Numerai benchmark-model dataset files under `.numereng/datasets/<data_version>/...`
- numereng-managed baseline artifacts under `.numereng/datasets/baselines/...`

The second one is what numereng uses for the shared active benchmark and benchmark-relative scoring.

## Build A Named Baseline

```bash
uv run numereng baseline build \
  --run-ids run_a,run_b,run_c \
  --name medium_blend \
  --default-target target_ender_20 \
  --description "medium blend for active benchmark seeding"
```

To promote the result immediately to the shared active benchmark:

```bash
uv run numereng baseline build \
  --run-ids run_a,run_b,run_c \
  --name medium_blend \
  --promote-active
```

## Baseline Artifact Layout

Named baselines live under:

- `.numereng/datasets/baselines/<baseline_name>/`

Typical files:

- `baseline.json`
- one predictions parquet
- optional shared per-era CORR artifacts

The promoted shared active benchmark lives under:

- `.numereng/datasets/baselines/active_benchmark/predictions.parquet`
- `.numereng/datasets/baselines/active_benchmark/benchmark.json`

## Config Interaction

Training configs can score against:

- the shared active benchmark with `benchmark_source.source = "active"`
- one explicit parquet with `benchmark_source.source = "path"`

Use `active` when the repo already has a trusted shared benchmark. Use `path` when bootstrapping a machine, testing a one-off baseline, or avoiding a repo-wide promotion.

## High-Risk Gotchas

- official `*benchmark_models.parquet` dataset files are not the same thing as numereng’s active benchmark artifact
- `benchmark_source.source = "active"` expects the promoted active artifact to exist locally
- if the active benchmark is missing, training can still succeed while benchmark-relative scoring stays incomplete or pending

## Read Next

- [Training Models](training.md)
- [Metrics](../reference/metrics.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
