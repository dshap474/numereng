# Experiment Contract

Use this reference when the task is about numereng's concrete experiment layout and execution
surface.

## Canonical Paths

- `.numereng/experiments/<experiment_id>/experiment.json`
- `.numereng/experiments/<experiment_id>/EXPERIMENT.md`
- `.numereng/experiments/<experiment_id>/EXPERIMENT.pack.md`
- `.numereng/experiments/<experiment_id>/configs/*.json`
- `.numereng/experiments/<experiment_id>/run_plan.csv`
- `.numereng/runs/<run_id>/`
- `.numereng/numereng.db`

## Manifest Expectations

- `experiment.json` is the source of truth for experiment status, run list, and champion state.
- `run_plan.csv` is optional and only applies when a sweep order has been defined.
- `EXPERIMENT.md` is the durable narrative for findings, decisions, anti-patterns, and next steps.
- `EXPERIMENT.pack.md` is a generated snapshot that embeds `EXPERIMENT.md` plus one dashboard-aligned scalar run-metrics table; it excludes per-era/time-series metrics.

## Valid Command Families

- `uv run numereng experiment create|list|details|train|promote|report|pack ...`
- `uv run numereng run train ...`
- `uv run numereng ensemble build|list|details ...`
- `uv run numereng hpo create ...`
- `uv run numereng store init|index|rebuild|doctor ...`

## Run Output Expectations

For completed scored runs, expect:

- `run.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json`

At minimum:

- `run.json.metrics_summary` should match `metrics.json`
- artifact paths declared in `run.json` should exist
- the manifest and store should agree on the run set unless drift is being investigated
