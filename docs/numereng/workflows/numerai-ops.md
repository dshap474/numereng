# Numerai Operations

The `numerai` and `docs` command families cover local Numerai docs sync, dataset/model/round queries, and deterministic forum scraping.

## Official Numerai Docs Mirror

```bash
uv run numereng docs sync numerai
```

This syncs the official Numerai docs into `docs/numerai/` in the current checkout.

## Datasets

List available datasets:

```bash
uv run numereng numerai datasets list
```

Download one dataset file:

```bash
uv run numereng numerai datasets download \
  --filename v5.2/train.parquet \
  --dest-path .numereng/datasets
```

Use `--round` or `--tournament <classic|signals|crypto>` when needed.

## Models And Current Round

```bash
uv run numereng numerai models
uv run numereng numerai round current
```

These are useful before submissions and hosted model uploads.

## Forum Scraping

Incremental scrape:

```bash
uv run numereng numerai forum scrape
```

Full refresh:

```bash
uv run numereng numerai forum scrape --full
```

By default, numereng writes forum artifacts under `docs/numerai/forum/`.

## Read Next

- [Installation](../getting-started/installation.md)
- [Dataset Tools](dataset-tools.md)
- [Submissions](submission.md)
