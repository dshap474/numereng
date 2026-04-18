# Runtime Artifacts & Paths

This page describes where numereng writes durable runtime state and which files matter when you are debugging, comparing runs, or operating agents against the workspace.

## Workspace Model

The repo checkout is the workspace.

Numereng-managed runtime state lives under:

- `.numereng/`

Repo-local extension roots live under:

- `src/numereng/features/models/custom_models/`
- `src/numereng/features/agentic_research/programs/`
- `.agents/skills/` for local custom skills

## Store Roots

Important top-level paths under `.numereng/`:

- `numereng.db`
- `runs/`
- `experiments/`
- `notes/`
- `datasets/`
- `cache/`
- `tmp/`
- `remote_ops/`
- optional `hpo/`
- optional `ensembles/`

## Run Artifacts

Canonical files under `.numereng/runs/<run_id>/` include:

- `run.json`
- `runtime.json`
- `run.log`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json`
- `artifacts/predictions/*.parquet`
- `artifacts/scoring/*`
- `artifacts/model/model.pkl`
- `artifacts/model/manifest.json`

## Experiment Artifacts

Canonical files under `.numereng/experiments/<experiment_id>/` include:

- `experiment.json`
- `EXPERIMENT.md`
- `EXPERIMENT.pack.md`
- `configs/*.json`
- `run_plan.csv`
- `run_scripts/*`
- `submission_packages/<package_id>/*`
- `agentic_research/*`
- `hpo/<study_id>/*`
- `ensembles/<ensemble_id>/*`

Archived experiments move to:

- `.numereng/experiments/_archive/<experiment_id>/`

## Dataset And Baseline Artifacts

Key dataset locations:

- `.numereng/datasets/<data_version>/train.parquet`
- `.numereng/datasets/<data_version>/validation.parquet`
- `.numereng/datasets/<data_version>/downsampled_full.parquet`
- `.numereng/datasets/<data_version>/downsampled_full_benchmark_models.parquet`

Named baselines live under:

- `.numereng/datasets/baselines/<baseline_name>/`

The promoted shared active benchmark lives under:

- `.numereng/datasets/baselines/active_benchmark/predictions.parquet`
- `.numereng/datasets/baselines/active_benchmark/benchmark.json`

## Remote And Cloud State

Numereng stores orchestration state under:

- `.numereng/remote_ops/`
- `.numereng/cache/remote_ops/`
- `.numereng/cache/cloud/<provider>/...`

Cloud pull archives are staging artifacts. Durable run provenance still lands under `.numereng/runs/<run_id>/`.

## Notes And Research Memory

Repo-local notes live under:

- `.numereng/notes/`

Canonical rolling research-memory paths include:

- `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md`
- `.numereng/notes/__RESEARCH_MEMORY__/experiments/*.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/*.md`

## Path Rules

- Treat filesystem artifacts as canonical; SQLite is an index over them.
- Keep experiment work inside one experiment root instead of spreading configs and reports around the repo.
- Use `--workspace` only when you intentionally want to target another checkout’s `.numereng/` store.

## Read Next

- [Project Layout](../getting-started/project-layout.md)
- [Store Operations](../workflows/store-ops.md)
- [Experiments](../workflows/experiments.md)
