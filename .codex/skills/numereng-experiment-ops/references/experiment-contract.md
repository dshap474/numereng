# Experiment Contract

Use this reference when the task is about numereng's concrete experiment layout and execution
surface.

## Canonical Paths

- `experiments/<experiment_id>/experiment.json`
- `experiments/<experiment_id>/EXPERIMENT.md`
- `experiments/<experiment_id>/EXPERIMENT.pack.md`
- `experiments/<experiment_id>/configs/*.json`
- `experiments/<experiment_id>/run_plan.csv`
- `experiments/<experiment_id>/run_scripts/*`
- optional `experiments/<experiment_id>/analysis/`
- optional `experiments/<experiment_id>/deployment/<deployment_id>/`
- `.numereng/runs/<run_id>/`
- `.numereng/numereng.db`

## Manifest Expectations

- `experiment.json` is the source of truth for experiment status, run list, and champion state.
- `experiment create` scaffolds `EXPERIMENT.md`, `configs/`, `run_plan.csv`, and `run_scripts/launch_all.*`.
- `run_plan.csv` is created as a header-only stub and only becomes meaningful once a sweep order has been defined.
- `run_scripts/` is the canonical home for experiment-local launchers and recovery helpers.
- `EXPERIMENT.md` is the durable narrative for findings, decisions, anti-patterns, and next steps.
- `EXPERIMENT.pack.md` is a generated snapshot that embeds `EXPERIMENT.md` plus one dashboard-aligned scalar run-metrics table; it excludes per-era/time-series metrics.
- Optional `analysis/` and `deployment/<deployment_id>/` areas hold experiment-local decision evidence and handoff artifacts. Load `references/experiment-local-artifacts.md` for that contract.
- Use `experiment-finalize` for completed-report rewrite and pack rendering once scoring artifacts are complete.

## Valid Command Families

- `numereng experiment create|list|details|train|promote|report|pack ...`
- `numereng run train ...`
- `numereng remote experiment pull --target <target_id> --experiment-id <id> --mode <scoring|full>`
- `numereng ensemble build|list|details ...`
- `numereng hpo create ...`
- `numereng store init|index|rebuild|doctor ...`

## Scripted Sweep Contract

- scripted sweeps must keep `run_plan.csv` at the experiment root
- scripted sweeps should call `numereng experiment train --post-training-scoring none`
- scripted sweeps own round scoring and should call `numereng experiment score-round` after the last planned config for each `rN`
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

- `numereng remote experiment pull --target <target_id> --experiment-id <id> --mode <scoring|full>`

Contract:

- pullback is manual
- only `FINISHED` remote runs materialize locally
- successful pullback writes canonical local run dirs under `.numereng/runs/<run_id>/`
- rerunning the same pull is idempotent and should no-op for already materialized runs
- use `--mode scoring` for metrics/reporting artifacts and `--mode full` only when prediction parquets are needed
