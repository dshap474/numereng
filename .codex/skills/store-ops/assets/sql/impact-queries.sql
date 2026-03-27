-- Impact queries for store operations.
-- Edit VALUES lists before running.

-- Example:
-- sqlite3 .numereng/numereng.db < .agents/skills/store-ops/assets/sql/impact-queries.sql

WITH
  target_experiments(experiment_id) AS (
    VALUES
      ('2026-02-22_medium-baseline-stability')
  ),
  target_runs(run_id) AS (
    VALUES
      ('example_run_id')
  )
SELECT 'target_experiments' AS section, experiment_id AS value
FROM target_experiments
UNION ALL
SELECT 'target_runs', run_id
FROM target_runs;

-- Experiment-level row counts
WITH target_experiments(experiment_id) AS (
  VALUES ('2026-02-22_medium-baseline-stability')
)
SELECT 'experiments' AS table_name, COUNT(*) AS row_count
FROM experiments
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
UNION ALL
SELECT 'runs', COUNT(*)
FROM runs
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
UNION ALL
SELECT 'run_jobs', COUNT(*)
FROM run_jobs
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
UNION ALL
SELECT 'logical_runs', COUNT(*)
FROM logical_runs
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
UNION ALL
SELECT 'hpo_studies', COUNT(*)
FROM hpo_studies
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
UNION ALL
SELECT 'ensembles', COUNT(*)
FROM ensembles
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments);

-- Resolve run IDs by experiment from DB
WITH target_experiments(experiment_id) AS (
  VALUES ('2026-02-22_medium-baseline-stability')
)
SELECT run_id, experiment_id
FROM runs
WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
ORDER BY experiment_id, created_at DESC;

-- Run-linked row counts for target runs
WITH target_runs(run_id) AS (
  VALUES ('example_run_id')
)
SELECT 'metrics' AS table_name, COUNT(*) AS row_count
FROM metrics
WHERE run_id IN (SELECT run_id FROM target_runs)
UNION ALL
SELECT 'run_artifacts', COUNT(*)
FROM run_artifacts
WHERE run_id IN (SELECT run_id FROM target_runs)
UNION ALL
SELECT 'run_attempts(canonical)', COUNT(*)
FROM run_attempts
WHERE canonical_run_id IN (SELECT run_id FROM target_runs)
UNION ALL
SELECT 'run_jobs(canonical)', COUNT(*)
FROM run_jobs
WHERE canonical_run_id IN (SELECT run_id FROM target_runs)
UNION ALL
SELECT 'hpo_trials(run_id)', COUNT(*)
FROM hpo_trials
WHERE run_id IN (SELECT run_id FROM target_runs)
UNION ALL
SELECT 'ensemble_components(run_id)', COUNT(*)
FROM ensemble_components
WHERE run_id IN (SELECT run_id FROM target_runs);

-- Job-linked row counts by experiment (for cleanup planning)
WITH target_experiments(experiment_id) AS (
  VALUES ('2026-02-22_medium-baseline-stability')
),
job_ids(job_id) AS (
  SELECT DISTINCT job_id
  FROM run_jobs
  WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
)
SELECT 'run_job_logs' AS table_name, COUNT(*) AS row_count
FROM run_job_logs
WHERE job_id IN (SELECT job_id FROM job_ids)
UNION ALL
SELECT 'run_job_events', COUNT(*)
FROM run_job_events
WHERE job_id IN (SELECT job_id FROM job_ids)
UNION ALL
SELECT 'run_job_samples', COUNT(*)
FROM run_job_samples
WHERE job_id IN (SELECT job_id FROM job_ids);

-- Shared-run overlap check across experiments
WITH
  target_experiments(experiment_id) AS (
    VALUES ('2026-02-22_medium-baseline-stability')
  ),
  target_runs(run_id) AS (
    SELECT run_id
    FROM runs
    WHERE experiment_id IN (SELECT experiment_id FROM target_experiments)
  )
SELECT run_id, experiment_id
FROM runs
WHERE run_id IN (SELECT run_id FROM target_runs)
  AND COALESCE(experiment_id, '') NOT IN (SELECT experiment_id FROM target_experiments)
ORDER BY run_id, experiment_id;
