# Baseline Workflow

Use this reference when the task is about benchmark-relative scoring
prerequisites for numereng experiments.

## Core Distinction

- official Numerai benchmark-model dataset files live under
  `.numereng/datasets/<data_version>/*benchmark_models.parquet`
- the default diagnostics benchmark for numereng scoring is a separate shared
  artifact under `.numereng/datasets/baselines/active_benchmark/`

Do not confuse the dataset files with the active benchmark artifact.

## Canonical Paths

- named baselines:
  - `.numereng/datasets/baselines/<baseline_name>/baseline.json`
  - `.numereng/datasets/baselines/<baseline_name>/<predictions>.parquet`
- shared active benchmark:
  - `.numereng/datasets/baselines/active_benchmark/predictions.parquet`
  - `.numereng/datasets/baselines/active_benchmark/benchmark.json`

## Config Choice

- default:
  - `data.benchmark_source.source = "active"`
  - use only when the shared `active_benchmark` artifact already exists
- explicit override:
  - `data.benchmark_source = { "source": "path", "predictions_path": "...", "pred_col": "prediction", "name": "<label>" }`
  - use this for bootstrap environments, one-off comparisons, and smoke runs

## Current Repo Gap

- there is no public `numereng baselines ...` CLI family
- internal helper exists:
  - `src/numereng/features/training/repo.py::seed_active_benchmark(...)`

## Agent Guidance

- if benchmark-relative metrics are required and `active_benchmark` is absent,
  prefer `benchmark_source.source = "path"` over leaving the config implicit
- if the user asks how to establish the shared default, explain the named
  baseline directory contract and the active-benchmark seeding step
- if the task becomes cleanup or drift repair around `.numereng/datasets` or
  `.numereng/numereng.db`, hand off to `store-ops`
