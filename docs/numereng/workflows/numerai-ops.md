# Numerai Operations

The `numerai` command family covers account-scoped dataset/model queries and deterministic forum scraping. The separate `docs` command downloads the official Numerai docs into the current repo checkout on demand.

## Official Docs Mirror

Download or refresh the official Numerai docs locally:

```bash
uv run numereng docs sync numerai
```

Optional:

- `--workspace <path>`

The mirror lands in `docs/numerai/`.

Notes:

- Numerai docs are not preinstalled into the repo checkout
- the sync preserves local `SYNC_POLICY.md`, `.sync-meta.json`, and `forum/` content inside `docs/numerai/`

## Datasets

List available datasets:

```bash
uv run numereng numerai datasets list
```

Optional filters:

- `--round <num>`
- `--tournament <classic|signals|crypto>`

Download one dataset file:

```bash
uv run numereng numerai datasets download \
  --filename v5.2/train.parquet \
  --dest-path .numereng/datasets
```

If `--dest-path` is omitted, numereng chooses the default dataset destination for that filename.

## Models

List account model name to model-id mappings:

```bash
uv run numereng numerai models
```

Optional:

- `--tournament <classic|signals|crypto>`

## Current Round

```bash
uv run numereng numerai round current
```

Use this before uploads when you want to confirm the current tournament round.

## Forum Scraping

Incremental scrape:

```bash
uv run numereng numerai forum scrape
```

Full refresh:

```bash
uv run numereng numerai forum scrape --full
```

Optional paths:

- `--output-dir <path>`
- `--state-path <path>`

Current defaults:

- output directory: `docs/numerai/forum`
- default state path: `<output_dir>/.forum_scraper_state.json`

The scraper writes:

- `INDEX.md`
- `posts/YYYY/MM/*.md`
- `.forum_scraper_manifest.json`
- `.forum_scraper_state.json`

## Tournament Flag

Dataset, model, and round commands accept:

- `classic`
- `signals`
- `crypto`

Forum scraping is not tournament-specific.
