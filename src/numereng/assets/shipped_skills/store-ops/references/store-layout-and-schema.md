# Store Layout and Schema

## Canonical Local Store Paths

Default store root:
- `.numereng`

Common paths:
- `.numereng/numereng.db`
- `.numereng/runs/<run_id>/`
- `experiments/<experiment_id>/experiment.json`
- `experiments/<experiment_id>/EXPERIMENT.md`
- `experiments/<experiment_id>/configs/*.json`

## Run Artifact Expectations

Run directories typically contain:
- `run.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json`
- `artifacts/predictions/*.parquet`

## Key SQLite Tables

Experiment-linked tables:
- `experiments` (`experiment_id`)
- `runs` (`experiment_id`, `run_id`)
- `run_jobs` (`experiment_id`, `canonical_run_id`, `job_id`)
- `logical_runs` (`experiment_id`, `logical_run_id`)
- `hpo_studies` (`experiment_id`, `study_id`)
- `ensembles` (`experiment_id`, `ensemble_id`)

Run-linked tables:
- `metrics` (`run_id`)
- `run_artifacts` (`run_id`)
- `run_attempts` (`canonical_run_id`, `logical_run_id`, `job_id`)
- `run_job_logs` (`job_id`)
- `run_job_events` (`job_id`)
- `run_job_samples` (`job_id`)
- `ensemble_components` (`run_id`, `ensemble_id`)
- `hpo_trials` (`run_id`, `study_id`)

Ensemble-linked tables:
- `ensemble_components` (`ensemble_id`)
- `ensemble_metrics` (`ensemble_id`)

HPO-linked tables:
- `hpo_trials` (`study_id`)

## Important Relationships (Operational)

There are no strict FK constraints enforcing delete order, so cleanup should delete dependent rows first:
1. `run_job_logs`, `run_job_events`, `run_job_samples`
2. `run_attempts`
3. `metrics`, `run_artifacts`
4. `ensemble_components`, `ensemble_metrics` (as relevant)
5. `hpo_trials`
6. `run_jobs`, `logical_runs`, `hpo_studies`, `ensembles`
7. `runs`

Use transaction boundaries for destructive cleanup.

## Manifest Reset Contract

For a clean experiment reset (preserve design files):
- Keep `configs/`
- Keep `EXPERIMENT.md`
- Update `experiment.json`:
  - `status = "draft"`
  - `runs = []`
  - `champion_run_id = null`
  - `updated_at = <current ISO timestamp>`
