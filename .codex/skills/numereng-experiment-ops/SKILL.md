---
name: numereng-experiment-ops
description: "Use for numereng-specific experiment execution contracts: experiment layout, config locations, schema sources, template files, run artifacts, and current CLI/API entrypoints."
user-invocable: true
argument-hint: "<experiment operation intent> (e.g., create experiment, where do configs go, training config schema, run artifact layout)"
---

# Numereng Experiment Ops

Use this skill for the concrete numereng experiment contract.

This is the source of truth for:
- where experiments live
- how experiment IDs and config files are organized
- what `EXPERIMENT.md` should contain
- where training schema and templates live
- how baseline directories and the active benchmark fit into experiment-time scoring
- which numereng commands are valid for experiment execution
- what run outputs and metrics files should exist

Do not use this skill for experiment strategy or plateau decisions. Use `experiment-design`
for that. Do not use this skill for drift repair, reindex, or destructive cleanup. Use
`store-ops` for that.

Run from:
- `<workspace>`

## Scope

In scope:
- experiment directory layout
- experiment creation and resume conventions
- config file naming and placement
- `EXPERIMENT.md` and round-log formatting
- training config schema entrypoints
- baseline-directory and active-benchmark setup guidance for benchmark-relative scoring
- current CLI/API command contract for experiment execution
- run output and metric artifact expectations
- remote finished-run pullback into local canonical storage
- champion handoff for final refit and official Numerai submission
- deciding when to hand off to `store-ops`

Out of scope:
- experiment methodology
- sweep design
- plateau decisions
- store repair or cleanup

## Hard Rules

- Prefer package CLI entrypoints and explicit commands:
  - `numereng experiment create|list|details|archive|unarchive|train|promote|report|pack ...`
  - `numereng run train ...`
  - `numereng remote experiment pull --target <target_id> --experiment-id <id>`
  - `numereng ensemble build|list|details ...`
  - `numereng hpo create ...`
  - `numereng store init|index|rebuild|doctor ...`
- Do not use removed or unsupported families:
  - `orchestrator ...`
  - `optimize ...`
  - `baselines ...`
  - `neutralize-sweep ...`
  - `experiment summarize|show|compare|set-status|conclude|build-ensemble|rebuild-registry`
  - `db validate|db rebuild`
- Keep experiment learnings inside the experiment-local `EXPERIMENT.md`.
- Do not create or rely on a separate `.numereng/knowledge/` workflow from this skill.

## Dataset Variant Policy

Use this default unless the user explicitly overrides it:

- scout and smoke validation:
  - `data.dataset_variant = "non_downsampled"`
- explicit low-cost scout override:
  - `data.dataset_variant = "downsampled"`

If `downsampled` is used, record the reason in `EXPERIMENT.md`.

## Experiment Layout

Experiments live under:

- `experiments/<experiment_id>/`
- `experiments/_archive/<experiment_id>/` when archived

Canonical experiment files:

- `experiments/<experiment_id>/experiment.json`
- `experiments/<experiment_id>/EXPERIMENT.md`
- `experiments/<experiment_id>/EXPERIMENT.pack.md`
- `experiments/<experiment_id>/configs/*.json`
- `experiments/<experiment_id>/run_scripts/*` for launcher and recovery helpers
- `experiments/<experiment_id>/run_plan.csv` when the experiment is using a planned sweep
- `experiments/_archive/<experiment_id>/...` for archived experiment-local files

Archive semantics:

- archive moves the full experiment directory from the live root into `_archive/`
- archive sets `experiment.json.status = "archived"` and preserves the prior status for unarchive
- archived experiments are hidden from normal experiment/config catalogs
- archived experiments remain readable by direct experiment ID
- archived experiments are read-only; do not train or promote them until unarchived

Use one experiment directory for one line of inquiry.

Recommended experiment ID format:

- `YYYY-MM-DD_slug`

Example:

- `2026-03-06_small-lgbm-all-targets-3seed-baseline`

## Experiment Creation And Logging

Create experiments with:

```bash
numereng experiment create \
  --id <YYYY-MM-DD_slug> \
  --name "<name>" \
  --hypothesis "<hypothesis>" \
  --tags "tag1,tag2"
```

After creation:

- `experiment create` now scaffolds the deterministic experiment skeleton for you:
  - `EXPERIMENT.md` with the current report-section contract
  - `configs/`
  - `run_plan.csv` with the canonical header stub
  - `run_scripts/launch_all.py|.sh|.ps1`
- keep the canonical narrative in `EXPERIMENT.md`
- keep configs in `configs/`
- keep launcher and recovery helpers in `run_scripts/`
- fill in `run_plan.csv` only when a sweep ordering is intentionally defined
- update the experiment log after each completed round
- when refreshing a report, bring `EXPERIMENT.md` up to the current template contract rather than only appending ad hoc notes

Use these assets:

- `assets/EXPERIMENT.template.md`
- `assets/research-round-template.md`

## EXPERIMENT.md Contract

Treat `assets/EXPERIMENT.template.md` as the narrative schema for experiment reporting.
For new experiments, prefer the CLI-generated scaffold first and use the asset as the refresh/reference contract.

Required sections for current reports:

- summary with `hypothesis`, primary metric, tie-break metric, and outcome
- abstract
- method
- execution inventory
- ambiguity resolution when relevant
- scout -> scale tracker
- plateau gate settings
- round log
- results
- standard plots / visual checks
- ensemble log when relevant
- remaining knobs audit
- final decision
- stopping rationale
- findings
- anti-patterns observed
- next experiments
- final checks
- repro commands

Section expectations:

- `Execution Inventory` must separate planned, executed, failed/interrupted, and skipped/superseded configs.
- `Round Log` must state what changed in the round, what actually ran, and the round decision.
- `Results` must include a compact summary table and should include `avg_corr_with_benchmark` when available from run artifacts.
- `Standard Plots / Visual Checks` should record the plot command and artifact path when plots exist; if no plot was generated, say why and what scalar evidence substituted for it.
- `Final Checks` should be used as the completion gate for the markdown itself.

Completion bar:

- Do not call an `EXPERIMENT.md` complete just because it exists.
- A completed report should clearly distinguish executed work from planned-only work, match the underlying run artifacts, explain champion status, and avoid stale placeholders.
- If the experiment is complete or the user asks for a final write-up, regenerate `EXPERIMENT.pack.md` after the narrative is updated.

## Config File Conventions

Keep training configs under:

- `experiments/<experiment_id>/configs/`

Config naming should make the changed variable obvious.

Examples:

- `base.json`
- `r1_target_rowan_60_seed42.json`
- `r2_lr_0p003_numleaves_128.json`

For manual sweeps:

- start from `assets/training-config-template.json`
- change one variable at a time per config inside a round
- keep the baseline config in the same experiment directory
- use explicit `data.benchmark_source={source:\"path\", ...}` when the repo has
  not yet seeded `.numereng/datasets/baselines/active_benchmark/`
- use default `benchmark_source.source=active` only when the shared
  `active_benchmark` artifact is already present

Post-training scoring policy:

- `training.post_training_scoring` defaults to `none`
- use `core` or `full` when one config should score itself immediately after training
- use `round_core` or `round_full` only in experiment workflows and only on the
  last `rN_*` config in a round when you want one deferred batch scoring pass at
  the end of that round
- earlier configs in the same round should usually keep `post_training_scoring = "none"`
- `run train` rejects `round_core` and `round_full`; those policies require
  `experiment train`

Scripted sweep policy:

- experiment-local launchers live under `run_scripts/`
- scripted sweeps must call `numereng experiment train ... --post-training-scoring none`
- scripted sweeps own round scoring and must call `numereng experiment score-round --round <rN> --stage <post_training_core|post_training_full>` after the last planned config for each round
- default scripted batch stage is `post_training_core`; use `post_training_full` only when the round needs the heavy feature diagnostics
- do not combine scripted batch scoring with config-level `round_core` or `round_full`

For HPO study definitions:

- use `assets/hpo-study-template.json`

## Schema Source Of Truth

Installed users should treat the template asset as the primary starting point and the live CLI/API validation as the runtime contract.

Repo-internal schema sources, only if you are editing numereng itself:

- `src/numereng/config/training/CLAUDE.md`
- `src/numereng/config/training/schema/training_config.schema.json`

Template alignment rule:

- `assets/training-config-template.json` must stay aligned with the live schema and current repo contract

If the schema and template disagree:

1. treat the schema and loader contract as authoritative
2. update the template before using it for new configs
3. record any notable fallback in `EXPERIMENT.md` if it affected a live experiment

## Run Execution Commands

Use these current command families:

```bash
numereng experiment archive --id <id>
numereng experiment unarchive --id <id>
numereng experiment train --id <id> --config <config.json> [--post-training-scoring <none|core|full|round_core|round_full>]
numereng experiment score-round --id <id> --round <rN> --stage <post_training_core|post_training_full>
numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format table
numereng experiment pack --id <id>
numereng experiment details --id <id> --format json
numereng experiment promote --id <id> --metric bmc_last_200_eras.mean
numereng remote experiment pull --target <target_id> --experiment-id <id>
numereng ensemble build --experiment-id <id> --run-ids <run_a,run_b,...> --method rank_avg
numereng hpo create --study-config <path.json>
```

Use `numereng run train ...` only when the task is intentionally outside the experiment
manifest workflow.

Guardrail:
- if an experiment is archived, unarchive it before running `experiment train` or `experiment promote`

Recommended scripted sweep layout:

```text
experiments/<experiment_id>/
  experiment.json
  EXPERIMENT.md
  configs/
  run_plan.csv
  run_scripts/
    launch_all.py
    launch_all.sh
    launch_all.ps1
    run_with_recovery.py   # optional per-experiment helper
```

Recommended scripted sweep commands:

- default launcher:
  - `bash experiments/<id>/run_scripts/launch_all.sh`
- explicit heavy round scoring:
  - `bash experiments/<id>/run_scripts/launch_all.sh --score-stage post_training_full`
- manual recovery path:
  - `numereng experiment score-round --id <id> --round <rN> --stage <post_training_core|post_training_full>`

## Remote Experiment Pullback

If training ran on a remote target such as the GPU PC, treat finished-run pullback as a
standard post-experiment closeout step.

Use:

```bash
numereng remote experiment pull --target <target_id> --experiment-id <id>
```

Current contract:

- pullback is manual; it does not happen automatically when remote training finishes
- only `FINISHED` remote runs are materialized locally
- pulled runs become canonical local runs under `.numereng/runs/<run_id>/`
- the local experiment manifest is reconciled so viz and local reporting see the run history
- rerunning the same pull is idempotent: already-materialized runs are treated as no-ops
- active or incomplete remote runs remain remote-only until they finish and a later pull is run

Default workflow rule:

1. finish the remote experiment or finish the current batch of remote runs
2. run `numereng remote experiment pull --target <target_id> --experiment-id <id>`
3. only then treat the experiment as fully available for local viz, reporting, packing, and research-memory ingestion

If the user is working on the remote PC workflow and asks what to do at the end of an
experiment, explicitly remind them to run the pull command.

## Pack Completed Experiment

When the user asks to package or pack a completed experiment, run:

```bash
numereng experiment pack --id <id>
```

This writes `experiments/<id>/EXPERIMENT.pack.md` in the experiment folder.

The packed markdown includes:
- the current `EXPERIMENT.md` body
- one run-summary table using dashboard-aligned scalar metrics for every manifest-listed run:
  - `bmc_last_200_eras_mean`
  - `bmc_mean`
  - `corr_sharpe`
  - `corr_mean`
  - `mmc_mean`
  - `cwmm_mean`
  - `fnc_mean`
  - `feature_exposure_mean`
  - `max_feature_exposure`
  - `max_drawdown`
  - `mmc_coverage_ratio_rows`

Do not treat the pack file as a replacement for `EXPERIMENT.md`.
It is a generated snapshot and should not include per-era or other time-series metrics.
Run pack after the markdown narrative is current and the report-level final checks have been satisfied.

## Champion Handoff And Submission

When an experiment has a clear winner and the user wants an official Numerai submission:

1. pick the winning run or config from the experiment based on the recorded experiment metric
2. treat `purged_walk_forward` runs as evaluation artifacts only; do not directly submit their CV prediction parquet
3. rerun the winning config with `training.engine.profile = "full_history_refit"` to produce the final-fit model artifact
4. generate a live predictions parquet for the current Numerai round from that final-fit model
5. submit the live predictions through numereng

Current numereng contract:

- numereng supports official submission of a live predictions parquet through:
  - `numereng run submit --model-name <model_name> --predictions <live_predictions.parquet>`
- numereng also supports submission by run ID only when the referenced run already contains a live-eligible predictions artifact:
  - `numereng run submit --model-name <model_name> --run-id <run_id>`
- numereng does not currently expose a first-class command that downloads `live.parquet`, runs live inference, and writes the submission parquet for you
- numereng submission is predictions-based, not Numerai Compute pickle upload

Practical handoff rule:

- if the task is "how do I choose the winner and get it ready for submission?", keep the task in this skill
- if the task is "how do I inspect models, rounds, tournaments, diagnostics, or direct Numerai API operations?", use `numerai-api-ops`

## Run Output Contract

Each completed run should have a run directory under:

- `.numereng/runs/<run_id>/`

Core files expected for completed runs:

- `run.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json` after post-training scoring has been materialized

Remote pullback note:

- when runs were trained on a remote target, they do not become part of the local canonical run
  store until `numereng remote experiment pull --target <target_id> --experiment-id <id>`
  succeeds
- after pullback, the canonical expectation is the same: `.numereng/runs/<run_id>/` should exist locally

Expect `run.json` to carry:

- run identity
- config provenance
- metrics summary
- declared artifact paths
- `training.scoring` metadata with `policy`, `status`, `requested_stage`, and
  `refreshed_stages`; failed or deferred scoring may also include `reason` and
  `error`

At minimum, artifact paths declared in `run.json` should exist on disk for successful runs.
Prediction parquet outputs should match the recorded artifact paths.
Archiving does not move or delete `.numereng/runs/<run_id>/`; only experiment-local files move.

High-signal submission caveat:
- a run-local predictions artifact is only submittable when it is a live-eligible predictions file for the active Numerai round
- historical validation/CV prediction artifacts from experiment training are not the same thing as live submission files

For experiment progress, the manifest in `experiment.json` is the source of truth for run order
and completion count.

## Store Sync And Handoff To Store Ops

The store DB at `.numereng/numereng.db` should stay aligned with run directories and experiment
manifests.

Use `store-ops` when the task stops being normal experiment execution and becomes maintenance or
repair work.

Typical handoff triggers:
- interrupted or partial runs that need cleanup/reset
- experiment manifest vs run-directory vs DB drift
- missing indexed runs or stale experiment state in sqlite
- destructive cleanup of experiment-linked run data
- post-repair verification after store mutation

Primary integrity flow:

```bash
numereng store doctor
```

If `doctor` reports issues:

- one known run missing from the index:
  - `numereng store index --run-id <run_id>`
- broad drift or suspected corruption:
  - `numereng store rebuild`

Hand off to `store-ops` when the task involves:

- deleting runs
- resetting experiment-linked run state
- diagnosing filesystem/DB drift in detail
- any mutation that should follow a dry-run impact summary

## Reference Loading Guide

| Request Type | Load |
|---|---|
| Experiment directories, manifests, command families, artifact expectations | `references/experiment-contract.md` |
| Training schema source paths, template alignment, config authoring rules | `references/config-schema-and-template.md` |
| Baseline directories, `active_benchmark`, benchmark-source choice | `references/baseline-workflow.md` |

If a task spans multiple domains, load each relevant reference and avoid unrelated files.

## Asset Usage Guide

| Task | Use |
|---|---|
| Refresh experiment narrative or backfill an older stub | `assets/EXPERIMENT.template.md` |
| Log one research round | `assets/research-round-template.md` |
| Start a new training config | `assets/training-config-template.json` |
| Backfill launcher files for an older experiment that predates CLI scaffolding | `assets/launch_all.py`, `assets/launch_all.sh`, `assets/launch_all.ps1` |
| Start an HPO study config | `assets/hpo-study-template.json` |

Explicit path gates:
- If the task is about experiment narrative or round logging, use the markdown templates.
- If the task is about refreshing `EXPERIMENT.md`, align the file to the current `assets/EXPERIMENT.template.md` section contract instead of preserving stale structure.
- If the task is about training config shape, use `assets/training-config-template.json` plus the schema source paths.
- If the task is about HPO study authoring, use `assets/hpo-study-template.json`.
