# Numerai Operations

The `numerai` and `docs` command families cover local Numerai docs sync, dataset/model/round queries, and deterministic forum scraping.

## Official Numerai Docs Mirror

```bash
uv run numereng docs sync numerai
```

This mirrors the official Numerai docs into `docs/numerai/` in the current checkout.

- Source: [`github.com/numerai/docs`](https://github.com/numerai/docs) (shallow clone of `main`)
- Pulls **everything** from upstream, not just markdown — including images under `docs/numerai/.gitbook/assets/` (PNGs, JPGs, GIFs) used by the dashboard reader
- Re-running refreshes the mirror to the latest upstream commit. Added files appear, deleted files are removed, edits overwrite local copies. Do not hand-edit files under `docs/numerai/` — they will be wiped on the next sync
- Local-only files are preserved across syncs: `SYNC_POLICY.md`, `.sync-meta.json`, `.gitignore`, and the `forum/` scrape target
- Expect a few hundred files on first sync. `.sync-meta.json` records the upstream commit and timestamp

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
