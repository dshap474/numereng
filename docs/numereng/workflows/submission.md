# Submissions

Use `run submit` when you already have a submit-ready predictions source: either a run-backed artifact or an explicit predictions file.

If you still need to freeze a package, rebuild live predictions, or prepare a hosted upload, use [Serving & Model Uploads](serving.md) first.

## Submit By Run ID

```bash
uv run numereng run submit \
  --model-name MY_MODEL \
  --run-id <run_id>
```

## Submit By Predictions File

```bash
uv run numereng run submit \
  --model-name MY_MODEL \
  --predictions predictions/live.parquet
```

## Optional Pre-Submit Neutralization

```bash
uv run numereng run submit \
  --model-name MY_MODEL \
  --run-id <run_id> \
  --neutralize \
  --neutralizer-path data/neutralizer.parquet
```

Useful controls:

- `--neutralization-proportion <0..1>`
- `--neutralization-mode <era|global>`
- `--neutralizer-cols <csv>`
- `--no-neutralization-rank`

## Resolution Rules

When submitting by run ID, numereng resolves predictions from:

1. the artifact path recorded in `run.json`
2. canonical files under `artifacts/predictions/`
3. single-file fallback inside that predictions directory

## High-Risk Gotchas

- exactly one of `--run-id` or `--predictions` is required
- `--model-name` is required
- `--tournament` defaults to `classic`
- if `--neutralize` is set, `--neutralizer-path` is required
- `--allow-non-live-artifact` bypasses the normal classic live-artifact eligibility check

## Read Next

- [Serving & Model Uploads](serving.md)
- [Numerai Operations](numerai-ops.md)
