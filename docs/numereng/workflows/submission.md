# Submissions

Submit predictions to Numerai from a run ID or an explicit predictions file.

## Step 1: Choose Submission Source

### Submit by Run ID

```bash
uv run numereng run submit \
  --model-name MY_MODEL \
  --run-id <run_id>
```

### Submit by File

```bash
uv run numereng run submit \
  --model-name MY_MODEL \
  --predictions predictions/round_XXX.csv
```

## Step 2: Optional Pre-Submit Neutralization

```bash
uv run numereng run submit \
  --model-name MY_MODEL \
  --run-id <run_id> \
  --neutralize \
  --neutralizer-path data/neutralizer.parquet \
  --neutralization-proportion 0.5 \
  --neutralization-mode era
```

## Submission Contract

- Exactly one of `--run-id` or `--predictions` is required.
- `--model-name` is required.
- If `--neutralize` is set, `--neutralizer-path` is required.
- `--neutralization-proportion` must be `0.0..1.0`.
- Tournament options: `classic`, `signals`, `crypto`.

## Run Artifact Discovery

When you submit by run ID, Numereng resolves predictions in this order:

1. `run.json` artifact reference
2. `artifacts/predictions/` canonical names (`live_predictions.*`, `predictions.*`, `val_predictions.*`)
3. Single `.csv`/`.parquet` fallback in that directory

Keep run artifacts under `.numereng/runs/<run_id>/artifacts/predictions/` for deterministic submission behavior.

## Neutralized Outputs

Neutralization writes a sidecar file by default:

- `<original>.neutralized.parquet`
- `<original>.neutralized.csv`

The output metadata includes source path, neutralizer path, mode, proportion, and matched row counts.

## Round Awareness

```bash
uv run numereng numerai round current
```

Use this to check the active round before upload.
