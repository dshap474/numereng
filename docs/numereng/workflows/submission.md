# Submissions

Submit predictions to Numerai from a run ID or an explicit predictions file.

If you need numereng to rebuild a live blend or create a Numerai model-upload pickle first, use [Serving And Model Uploads](serving.md).

## Submit By Run ID

```bash
uv run numereng run submit \
  --model-name MY_MODEL \
  --run-id <run_id>
```

## Submit By File

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
  --neutralizer-path data/neutralizer.parquet \
  --neutralization-proportion 0.5 \
  --neutralization-mode era
```

Additional controls:

- `--neutralizer-cols <csv>`
- `--no-neutralization-rank`

## Submission Contract

- exactly one of `--run-id` or `--predictions` is required
- `--model-name` is required
- `--tournament` defaults to `classic`
- if `--neutralize` is set, `--neutralizer-path` is required
- `--allow-non-live-artifact` bypasses the default classic live-artifact eligibility check

## Run Artifact Resolution

When submitting by run ID, numereng resolves predictions in this order:

1. the artifact path recorded in `run.json`
2. canonical files under `artifacts/predictions/`
3. single-file fallback in that predictions directory

Keep run outputs under `.numereng/runs/<run_id>/artifacts/predictions/` for deterministic behavior.

## Neutralized Outputs

When submission neutralization runs without an explicit `--output-path`, numereng writes sidecar artifacts such as:

- `<original>.neutralized.parquet`

The response metadata records the source path, neutralizer path, mode, proportion, and row-match counts.

## Round Awareness

```bash
uv run numereng numerai round current
```

Use this before upload when you want to confirm the active round.
