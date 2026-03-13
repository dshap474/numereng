# Hyperparameter Optimization

Numereng uses Optuna-backed studies through the `hpo` command family.

## Create A Study

Recommended: study-config JSON.

```bash
uv run numereng hpo create --study-config configs/hpo/study.json
```

Inline alternative:

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

## Neutralized Objective

Trials can be scored with prediction-stage neutralization:

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

## Artifacts

Study artifacts are stored either:

- globally under `.numereng/hpo/<study_id>/`
- under an experiment at `.numereng/experiments/<experiment_id>/hpo/<study_id>/`

Trial runs still produce normal run artifacts and are indexed like any other run.

## High-Risk Gotchas

- study config paths and training config paths are JSON-only
- if `--neutralize` is set, `--neutralizer-path` is required
- trial indexing is mandatory; indexing failures mark the trial failed
- the objective metric must exist in the scored run outputs or study evaluation fails
