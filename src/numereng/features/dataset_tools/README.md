# dataset-tools

Official-style dataset downsampling utilities.

Run with `uv run python`:

- `uv run -m numereng.features.dataset_tools.build_downsampled_full --data-version v5.2`
- `uv run numereng dataset-tools build-downsampled-full --data-version v5.2`

This builds:
- `.numereng/datasets/v5.2/downsampled_full.parquet`
- `.numereng/datasets/v5.2/downsampled_full_benchmark_models.parquet`

Canonical non-downsampled storage remains split-source only:

- `.numereng/datasets/v5.2/train.parquet`
- `.numereng/datasets/v5.2/validation.parquet`

Downsampling keeps every 4th era (`offset=0`) to match official example-scripts.
