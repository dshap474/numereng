# Serving And Model Uploads

Use `numereng serve` when you want to freeze a production winner, rebuild live predictions locally, or package a Numerai-hosted model upload.

## When To Use `serve`

- use `numereng run submit` when you already have a live-eligible predictions parquet
- use `numereng serve` when you need numereng to:
  - freeze explicit weighted components into one submission package
  - rebuild live predictions from persisted run artifacts or, as a dev fallback, from configs
  - inspect whether a package is safe for local live builds or Numerai model uploads
  - build a Numerai-compatible `cloudpickle` model upload

## Package Lifecycle

Create a package from explicit components and weights:

```bash
uv run numereng serve package create \
  --experiment-id 2026-04-11_lgbm-live-submission-handoff \
  --package-id april8_winner_v1 \
  --components components.json
```

Inspect compatibility before building:

```bash
uv run numereng serve package inspect \
  --experiment-id 2026-04-11_lgbm-live-submission-handoff \
  --package-id april8_winner_v1
```

Inspection writes a stable report under:

- `experiments/<experiment_id>/submission_packages/<package_id>/artifacts/preflight/report.json`

The report classifies the package separately for:

- local live builds
- artifact-backed live readiness
- Numerai-hosted model uploads

The deployment classification is one of:

- `local_live_only`
- `artifact_backed_live_ready`
- `pickle_upload_ready`

## Local Live Build

```bash
uv run numereng serve live build \
  --experiment-id 2026-04-11_lgbm-live-submission-handoff \
  --package-id april8_winner_v1
```

This workflow:

1. downloads the current Classic `live.parquet`
2. prefers persisted `full_history_refit` model artifacts from run-backed components
3. falls back to local retraining only for config-backed/dev packages
4. predicts on the live frame
5. rank-blends the component predictions
6. writes a submit-ready parquet

Release-grade packages should be built from run IDs whose `full_history_refit` runs already wrote:

- `.numereng/runs/<run_id>/artifacts/model/model.pkl`
- `.numereng/runs/<run_id>/artifacts/model/manifest.json`

Use `serve live submit` if you want numereng to upload that parquet immediately.

## Model Upload Pickles

```bash
uv run numereng serve pickle build \
  --experiment-id 2026-04-11_lgbm-live-submission-handoff \
  --package-id jasper60_single_test
```

`serve pickle build` only succeeds when the package passes hosted-inference preflight.
It is now artifact-backed only and does not retrain components.

Current v1 rules are intentionally conservative:

- Classic only
- every component must be run-backed and have a loadable persisted model artifact
- external baseline side-input files are rejected
- custom module/plugin components are rejected for hosted inference
- the package must fit Numerai model-upload dependency/runtime expectations

If the package is compatible, numereng writes:

- `artifacts/pickle/model.pkl`

Upload it with:

```bash
uv run numereng serve pickle upload \
  --experiment-id 2026-04-11_lgbm-live-submission-handoff \
  --package-id jasper60_single_test \
  --model-name MY_SPARE_MODEL
```

## Auth And Environment

Uploads require Numerai credentials in the active shell environment.

Typical checks:

```bash
uv run numereng numerai models list
uv run numereng numerai round current
```

Numerai model uploads are hosted under strict limits:

- no internet access
- self-contained pickle only
- limited CPU, RAM, and runtime

If a package is local-live compatible but not model-upload compatible, keep using `serve live build` plus `run submit`.
