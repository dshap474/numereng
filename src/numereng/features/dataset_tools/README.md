# dataset-tools

Official-style dataset downsampling utilities.

Run with `uv run python`:

- `uv run -m numereng.features.dataset_tools.build_full_datasets --data-version v5.2`
- `uv run numereng dataset-tools build-full-datasets --data-version v5.2`

This builds:
- `.numereng/datasets/v5.2/full.parquet`
- `.numereng/datasets/v5.2/full_benchmark_models.parquet`
- `.numereng/datasets/v5.2/downsampled_full.parquet`
- `.numereng/datasets/v5.2/downsampled_full_benchmark_models.parquet`

Downsampling keeps every 4th era (`offset=0`) to match official example-scripts.
