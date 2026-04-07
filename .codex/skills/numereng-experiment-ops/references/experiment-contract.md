# Experiment Contract

Use this reference when the task is about numereng's concrete experiment layout and execution
surface.

## Canonical Paths

- `.numereng/experiments/<experiment_id>/experiment.json`
- `.numereng/experiments/<experiment_id>/EXPERIMENT.md`
- `.numereng/experiments/<experiment_id>/EXPERIMENT.pack.md`
- `.numereng/experiments/<experiment_id>/configs/*.json`
- `.numereng/experiments/<experiment_id>/run_plan.csv`
- `.numereng/experiments/<experiment_id>/run_scripts/*`
- `.numereng/runs/<run_id>/`
- `.numereng/numereng.db`

## Manifest Expectations

- `experiment.json` is the source of truth for experiment status, run list, and champion state.
- `experiment create` scaffolds `EXPERIMENT.md`, `configs/`, `run_plan.csv`, and `run_scripts/launch_all.*`.
- `run_plan.csv` is created as a header-only stub and only becomes meaningful once a sweep order has been defined.
- `run_scripts/` is the canonical home for experiment-local launchers and recovery helpers.
- `EXPERIMENT.md` is the durable narrative for findings, decisions, anti-patterns, and next steps.
- `EXPERIMENT.pack.md` is a generated snapshot that embeds `EXPERIMENT.md` plus one dashboard-aligned scalar run-metrics table; it excludes per-era/time-series metrics.

## Valid Command Families

- `uv run numereng experiment create|list|details|train|promote|report|pack ...`
- `uv run numereng run train ...`
- `uv run numereng remote experiment pull --target <target_id> --experiment-id <id>`
- `uv run numereng ensemble build|list|details ...`
- `uv run numereng hpo create ...`
- `uv run numereng store init|index|rebuild|doctor ...`

## Scripted Sweep Contract

- scripted sweeps must keep `run_plan.csv` at the experiment root
- scripted sweeps should call `uv run numereng experiment train --post-training-scoring none`
- scripted sweeps own round scoring and should call `uv run numereng experiment score-round` after the last planned config for each `rN`
- default scripted batch stage is `post_training_core`

## Run Output Expectations

For completed runs, expect:

- `run.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json` after post-training scoring has been materialized

At minimum:

- `run.json.metrics_summary` should match `metrics.json`
- `run.json.training.scoring` should record the scoring `policy`, `status`,
  `requested_stage`, and `refreshed_stages`
- artifact paths declared in `run.json` should exist
- the manifest and store should agree on the run set unless drift is being investigated

## Remote Pullback Closeout

When an experiment was trained on a remote target, finished runs are not available in the local
canonical run store until they are pulled back explicitly.

Use:

- `uv run numereng remote experiment pull --target <target_id> --experiment-id <id>`

Contract:

- pullback is manual
- only `FINISHED` remote runs materialize locally
- successful pullback writes canonical local run dirs under `.numereng/runs/<run_id>/`
- rerunning the same pull is idempotent and should no-op for already materialized runs
