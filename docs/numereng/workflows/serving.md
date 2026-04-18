# Serving & Model Uploads

Use `numereng serve` when you want to freeze a winning set of components into one production package, rebuild live predictions, or prepare a Numerai-hosted model upload.

## Package Lifecycle

Create a package:

```bash
uv run numereng serve package create \
  --experiment-id 2026-04-18_live-handoff \
  --package-id april_winner_v1 \
  --components components.json
```

Inspect compatibility:

```bash
uv run numereng serve package inspect \
  --experiment-id 2026-04-18_live-handoff \
  --package-id april_winner_v1
```

List packages:

```bash
uv run numereng serve package list --experiment-id 2026-04-18_live-handoff
```

Score one package on validation data:

```bash
uv run numereng serve package score \
  --experiment-id 2026-04-18_live-handoff \
  --package-id april_winner_v1 \
  --runtime auto
```

Sync hosted diagnostics:

```bash
uv run numereng serve package sync-diagnostics \
  --experiment-id 2026-04-18_live-handoff \
  --package-id april_winner_v1
```

## Local Live Build

```bash
uv run numereng serve live build \
  --experiment-id 2026-04-18_live-handoff \
  --package-id april_winner_v1
```

Submit immediately after building:

```bash
uv run numereng serve live submit \
  --experiment-id 2026-04-18_live-handoff \
  --package-id april_winner_v1 \
  --model-name MY_MODEL
```

## Model Upload Pickles

Build a hosted-compatible pickle package:

```bash
uv run numereng serve pickle build \
  --experiment-id 2026-04-18_live-handoff \
  --package-id hosted_candidate \
  --docker-image "Python 3.12"
```

Upload it:

```bash
uv run numereng serve pickle upload \
  --experiment-id 2026-04-18_live-handoff \
  --package-id hosted_candidate \
  --model-name MY_SPARE_MODEL
```

## Artifact Locations

Serving packages live under:

- `.numereng/experiments/<experiment_id>/submission_packages/<package_id>/`

Important artifacts include:

- `package.json`
- `artifacts/preflight/report.json`
- `artifacts/eval/validation/<runtime>/...`
- `artifacts/live/*`
- `artifacts/pickle/model.pkl`
- `artifacts/diagnostics/<upload_id>/...`

## High-Risk Gotchas

- hosted model uploads are stricter than local live builds
- local live builds may use artifact-backed local runtime paths that hosted uploads reject
- pickle uploads must be self-contained and cannot depend on importing `numereng` at Numerai runtime
- `serve pickle build` only succeeds when the package passes hosted-inference preflight and isolated smoke

## Read Next

- [Submissions](submission.md)
- [Experiments](experiments.md)
