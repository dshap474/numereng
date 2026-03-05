---
name: numerai-model-upload
description: Submit Numerai predictions through numereng CLI/API (no MCP) using run artifacts or explicit prediction files, with optional pre-submit neutralization.
user-invocable: true
---

# Numerai Model Upload

Submit predictions to Numerai using numereng's API boundary and CLI.

Run from:
- `<repo>`

## Scope

This skill covers Numerai **prediction submission** via:
- `uv run numereng run submit ...`
- `import numereng.api as api_module` with `api_module.submit_predictions(...)`

This skill does **not** use MCP workflows.

Out of scope in current numereng contract:
- creating Numerai model slots
- compute pickle upload/assign/trigger flows

If the user asks for those, state that numereng does not expose them and continue with prediction submission flow.

## Contract Guardrails

- Submission source is XOR:
  - exactly one of `run_id` or `predictions_path`
- Supported tournaments:
  - `classic | signals | crypto`
- Default local store root:
  - `.numereng`
- API boundary behavior:
  - return/raise `PackageError` at public API boundary, not raw internals

## Preflight Checklist

1. Verify Numerai connectivity and credentials with the package surface:
```bash
uv run numereng numerai models list --tournament classic
uv run numereng numerai round current --tournament classic
```

2. Verify target model name exists in model mapping output.

3. Verify submission source:
- run-based: ensure run exists under `.numereng/runs/<run_id>/`
- file-based: ensure predictions file exists (`.csv` or `.parquet`)

4. Ask for explicit user confirmation before a real live submission.

## CLI Workflow (Preferred)

### A) Submit from a run artifact

```bash
uv run numereng run submit \
  --run-id <run_id> \
  --model-name <model_name> \
  --tournament classic
```

Notes:
- The service resolves predictions from run manifest first, then known fallback filenames in `artifacts/predictions/`.
- Use `--store-root <path>` when not using default `.numereng`.

### B) Submit from an explicit predictions file

```bash
uv run numereng run submit \
  --predictions path/to/predictions.parquet \
  --model-name <model_name> \
  --tournament classic
```

### C) Optional pre-submit neutralization

```bash
uv run numereng run submit \
  --run-id <run_id> \
  --model-name <model_name> \
  --neutralize \
  --neutralizer-path path/to/neutralizer.parquet \
  --neutralization-proportion 0.5 \
  --neutralization-mode era
```

Optional flags:
- `--neutralizer-cols col_a,col_b`
- `--no-neutralization-rank`

## Python API Workflow

```python
import numereng.api as api_module

response = api_module.submit_predictions(
    api_module.SubmissionRequest(
        model_name="main",
        tournament="classic",
        run_id="abc123def456",  # OR predictions_path="path/to/preds.csv"
    )
)

print(response.model_dump())
```

Neutralized submit via API:

```python
import numereng.api as api_module

response = api_module.submit_predictions(
    api_module.SubmissionRequest(
        model_name="main",
        tournament="classic",
        predictions_path="path/to/preds.parquet",
        neutralize=True,
        neutralizer_path="path/to/neutralizer.parquet",
        neutralization_proportion=0.5,
        neutralization_mode="era",
    )
)
```

## Common Errors (Surface-Level)

- `submission_request_invalid`
  - invalid XOR source or malformed request.
- `submission_model_not_found`
  - model name not in account model mapping.
- `submission_predictions_file_not_found`
  - explicit predictions path missing.
- `submission_run_not_found`
  - run ID not found under store root.
- `submission_run_predictions_not_found`
  - run exists but predictions artifact not discoverable.
- `submission_neutralizer_path_required`
  - `neutralize=true` without neutralizer path.
- `numerai_upload_predictions_failed`
  - upstream Numerai upload failure.

## Done Criteria

- Preflight checks completed (`models list`, `round current`).
- User explicitly confirmed live upload.
- Submission returned payload with:
  - `submission_id`
  - `model_name`
  - `model_id`
  - resolved `predictions_path`
