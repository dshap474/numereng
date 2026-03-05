# Hyperparameter Optimization

Run Optuna-backed HPO studies with the `hpo` command family.

## Create a Study

You can create a study from a study-config JSON file (recommended) or from inline flags.

### Config-Driven

```bash
uv run numereng hpo create \
  --study-config configs/hpo/study.json
```

### Inline

```bash
uv run numereng hpo create \
  --study-name lgbm-sweep \
  --config configs/run.json \
  --metric bmc_last_200_eras.mean \
  --direction maximize \
  --n-trials 50 \
  --sampler tpe
```

## Inspect Studies

```bash
uv run numereng hpo list
uv run numereng hpo details --study-id <study_id>
uv run numereng hpo trials --study-id <study_id>
```

## Optional Neutralized Objective

```bash
uv run numereng hpo create \
  --study-config configs/hpo/study.json \
  --neutralize \
  --neutralizer-path data/neutralizer.parquet
```

Optional neutralization controls:

- `--neutralization-proportion <0..1>`
- `--neutralization-mode <era|global>`
- `--neutralizer-cols <csv>`
- `--no-neutralization-rank`

## High-Risk Gotchas

- Study config and training config paths are `.json` only.
- If `--neutralize` is set, `--neutralizer-path` is required.
- Trial indexing is mandatory; indexing failures mark a trial as failed.
