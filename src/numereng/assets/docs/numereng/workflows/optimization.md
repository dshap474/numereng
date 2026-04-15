# Hyperparameter Optimization

Numereng uses Optuna-backed studies through the `hpo` command family.

## Smoke Check

Run the dedicated HPO v2 smoke script to validate the repaired end-to-end path against a throwaway workspace:

```bash
./scripts/hpo_v2_smoke.sh
```

Convenience entrypoints:

```bash
just hpo-smoke
```

Optional:

- pass an empty workspace path as the first argument if you want to keep the artifacts
- set `NUMERENG_HPO_SMOKE_DATASETS_DIR=/path/to/.numereng/datasets` if you want to source datasets from another workspace instead of the script's built-in tiny synthetic fixture

The script validates:

- random-sampler JSON stays canonical
- resume uses total-study caps (`1` then `2` attempted trials, not `3`)
- duplicate deterministic params reuse the finished run inside and across studies

## Create A Study

Recommended: study-config JSON.

```bash
uv run numereng hpo create --study-config configs/hpo/study.json
```

Inline alternative:

```bash
uv run numereng hpo create \
  --study-id ender20_lgbm_gpu_v1 \
  --study-name lgbm-sweep \
  --config configs/run.json \
  --search-space '{"model.params.learning_rate":{"type":"float","low":0.001,"high":0.05,"log":true}}' \
  --metric post_fold_champion_objective \
  --direction maximize \
  --n-trials 50 \
  --timeout-seconds 21600 \
  --max-completed-trials 40 \
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
- under an experiment at `experiments/<experiment_id>/hpo/<study_id>/`

Each study root now includes:

- `study_spec.json` for the immutable study identity and base-config hash
- `study_summary.json` for the current mutable status snapshot
- `optuna_journal.log` for resumable Optuna execution
- `configs/trial_0000.json` and later materialized trial configs
- `trials_live.parquet`

Trial runs still produce normal run artifacts and are indexed like any other run.

## High-Risk Gotchas

- study config paths and training config paths are JSON-only
- HPO v2 study configs require `study_id`, `objective`, `search_space`, `sampler`, and `stopping`
- rerunning `numereng hpo create` with the same `study_id` resumes the existing study when the immutable spec matches
- `--timeout-seconds` is per invocation; rerunning the same study does not consume prior wall-clock budget
- `sampler.kind=random` only accepts `kind` and `seed`
- if `--neutralize` is set, `--neutralizer-path` is required
- trial indexing is mandatory; indexing failures mark the trial failed
- `post_fold_champion_objective` reads `post_fold_snapshots.parquet` first and falls back to `results.json` if the snapshot artifacts are missing or unusable
- identical parameter draws reuse a completed trial value or a finished deterministic run; stuck or partial run dirs fail loudly and require operator cleanup
- the objective metric must exist in the scored run outputs or study evaluation fails
