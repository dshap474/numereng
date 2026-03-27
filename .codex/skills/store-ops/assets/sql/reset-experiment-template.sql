-- Reset experiment-linked run state in SQL.
-- REVIEW and edit target IDs before execution.
-- Intended to run inside sqlite3 on .numereng/numereng.db.

BEGIN;

-- 1) Define scope
WITH target_experiments(experiment_id) AS (
  VALUES
    ('2026-02-22_medium-baseline-stability')
),
target_runs(run_id) AS (
  SELECT run_id
  FROM runs
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
  UNION
  SELECT canonical_run_id
  FROM run_jobs
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
    AND canonical_run_id IS NOT NULL
    AND canonical_run_id != ''
)
SELECT COUNT(*) AS target_run_count FROM target_runs;

-- 2) Delete job-linked dependent rows
DELETE FROM run_job_logs
WHERE job_id IN (
  SELECT DISTINCT job_id
  FROM run_jobs
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
);

DELETE FROM run_job_events
WHERE job_id IN (
  SELECT DISTINCT job_id
  FROM run_jobs
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
);

DELETE FROM run_job_samples
WHERE job_id IN (
  SELECT DISTINCT job_id
  FROM run_jobs
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
);

-- 3) Delete HPO and ensemble dependents
DELETE FROM hpo_trials
WHERE study_id IN (
  SELECT study_id
  FROM hpo_studies
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
)
OR run_id IN (SELECT run_id FROM target_runs);

DELETE FROM ensemble_metrics
WHERE ensemble_id IN (
  SELECT ensemble_id
  FROM ensembles
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
);

DELETE FROM ensemble_components
WHERE ensemble_id IN (
  SELECT ensemble_id
  FROM ensembles
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
)
OR run_id IN (SELECT run_id FROM target_runs);

-- 4) Delete run-linked rows
DELETE FROM metrics
WHERE run_id IN (SELECT run_id FROM target_runs);

DELETE FROM run_artifacts
WHERE run_id IN (SELECT run_id FROM target_runs);

DELETE FROM run_attempts
WHERE canonical_run_id IN (SELECT run_id FROM target_runs)
   OR logical_run_id IN (
     SELECT logical_run_id
     FROM logical_runs
     WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
   );

-- 5) Delete run orchestration rows
DELETE FROM run_jobs
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
   OR canonical_run_id IN (SELECT run_id FROM target_runs);

DELETE FROM logical_runs
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments);

-- 6) Delete top-level rows
DELETE FROM hpo_studies
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments);

DELETE FROM ensembles
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments);

DELETE FROM runs
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
   OR run_id IN (SELECT run_id FROM target_runs);

COMMIT;
