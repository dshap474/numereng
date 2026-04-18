# Dataset Tools

Use `numereng dataset-tools` when you want numereng to materialize derived dataset artifacts under the local store.

## Current Public Surface

The public dataset-tools command family currently exposes one workflow:

```bash
uv run numereng dataset-tools build-downsampled-full --data-version v5.2
```

Optional overrides:

- `--data-dir <path>`
- `--downsample-eras-step <n>`
- `--downsample-eras-offset <n>`
- `--rebuild`

## What It Builds

For one `data_version`, numereng writes the derived downsampled artifacts under:

- `.numereng/datasets/<data_version>/downsampled_full.parquet`
- `.numereng/datasets/<data_version>/downsampled_full_benchmark_models.parquet`

These are derived from the canonical non-downsampled dataset files under the same data-version root.

## When To Use It

- you want a smaller derived dataset for faster iteration
- you want deterministic repo-local downsampled artifacts instead of ad hoc notebooks
- your training config uses `dataset_variant = "downsampled"`

## High-Risk Gotchas

- this command does not download Numerai datasets for you; the required source parquet files must already exist locally
- the public quantization commands are removed; `build-downsampled-full` is the supported path
- if you want a clean rebuild of the derived artifact, use `--rebuild`

## Read Next

- [Numerai Operations](numerai-ops.md)
- [Training Models](training.md)
- [Project Layout](../getting-started/project-layout.md)
