---
name: numereng-experiment-ops
description: "Use for the concrete numereng experiment contract: layout, configs, templates, scoring artifacts, remote pullback, and current CLI/API entrypoints."
user-invocable: true
argument-hint: "<experiment operation intent> (e.g., create experiment, training config schema, pull scoring artifacts)"
---

# Numereng Experiment Ops

Use this skill for numereng's concrete experiment operating contract. It answers where experiment files live, which commands are valid, and what artifacts should exist.

Do not use this skill for research strategy, plateau decisions, or sweep methodology; use `experiment-design`. Do not use it for drift repair, reindex, destructive cleanup, or reset workflows; use `store-ops`.

Use `experiment-finalize` when scoring artifacts are complete and the task is finalizing `EXPERIMENT.md` or rendering `EXPERIMENT.pack.md`.

Run commands from `<workspace>`.

## Operating Contract

- Keep one experiment directory per line of inquiry under `experiments/<experiment_id>/`.
- Keep configs under `experiments/<experiment_id>/configs/`.
- Keep experiment-local launchers and recovery helpers under `experiments/<experiment_id>/run_scripts/`.
- Keep the durable narrative in `experiments/<experiment_id>/EXPERIMENT.md`.
- Treat `EXPERIMENT.pack.md` as a generated snapshot, not as the canonical narrative.
- Keep experiment learnings inside `EXPERIMENT.md`; do not create a separate `.numereng/knowledge/` workflow from this skill.
- Prefer package CLI entrypoints and explicit commands.
- Use `numereng run train ...` only when the task is intentionally outside the experiment manifest workflow.

Valid command families:

```bash
numereng experiment create|list|details|archive|unarchive|train|promote|report|pack ...
numereng experiment score-round --id <id> --round <rN> --stage <post_training_core|post_training_full>
numereng run train ...
numereng remote experiment pull --target <target_id> --experiment-id <id> --mode <scoring|full>
numereng ensemble build|list|details ...
numereng hpo create ...
numereng store init|index|rebuild|doctor ...
```

Do not use removed or unsupported families:

- `orchestrator ...`
- `optimize ...`
- `baselines ...`
- `neutralize-sweep ...`
- `experiment summarize|show|compare|set-status|conclude|build-ensemble|rebuild-registry`
- `db validate|db rebuild`

Archive guardrail:

- Archived experiments are read-only. Unarchive before `experiment train` or `experiment promote`.
- Archiving moves experiment-local files only; it does not move or delete `.numereng/runs/<run_id>/`.

## Experiment Layout

Canonical files:

- `experiments/<experiment_id>/experiment.json`
- `experiments/<experiment_id>/EXPERIMENT.md`
- `experiments/<experiment_id>/EXPERIMENT.pack.md`
- `experiments/<experiment_id>/configs/*.json`
- `experiments/<experiment_id>/run_plan.csv` when a sweep order is intentionally defined
- `experiments/<experiment_id>/run_scripts/*`
- `experiments/_archive/<experiment_id>/...` for archived experiment-local files

Optional experiment-local artifact areas such as `analysis/` and `deployment/<deployment_id>/` are defined in `references/experiment-local-artifacts.md`.

Create experiments with:

```bash
numereng experiment create \
  --id <YYYY-MM-DD_slug> \
  --name "<name>" \
  --hypothesis "<hypothesis>" \
  --tags "tag1,tag2"
```

`experiment create` scaffolds `EXPERIMENT.md`, `configs/`, `run_plan.csv`, and `run_scripts/launch_all.py|.sh|.ps1`. Fill `run_plan.csv` only when the sweep ordering is intentional.

## Config And Scoring Policy

Default dataset policy unless the user explicitly overrides it:

- scout and smoke validation: `data.dataset_variant = "non_downsampled"`
- explicit low-cost scout override: `data.dataset_variant = "downsampled"`

Record any `downsampled` override in `EXPERIMENT.md`.

Config rules:

- Start from `assets/training-config-template.json`.
- Name config files for the changed variable, for example `r1_target_rowan_60_seed42.json`.
- Change one variable at a time per round unless testing an intentional interaction.
- Use explicit `data.benchmark_source={source:"path", ...}` when `.numereng/datasets/baselines/active_benchmark/` is absent.
- Use `benchmark_source.source=active` only when the shared active benchmark exists.
- Use `assets/hpo-study-template.json` for HPO study definitions.

Post-training scoring policy:

- `training.post_training_scoring` defaults to `none`.
- Use `core` or `full` when one config scores itself immediately after training.
- Use `round_core` or `round_full` only with `experiment train`, and only on the last `rN_*` config in a round.
- Scripted sweeps should call `numereng experiment train ... --post-training-scoring none`, then call `numereng experiment score-round ...` after the last planned config for each round.
- Default scripted batch scoring is `post_training_core`; use `post_training_full` only when the round needs heavy feature diagnostics.
- Do not combine scripted batch scoring with config-level `round_core` or `round_full`.

Schema source of truth:

- Installed users should treat the template asset as the starting point and live CLI/API validation as the runtime contract.
- Repo-internal sources, only when editing numereng itself:
  - `src/numereng/config/training/CLAUDE.md`
  - `src/numereng/config/training/schema/training_config.schema.json`
- If the template disagrees with the schema or loader, the schema and loader win; update the template before using it for new configs.

## Report And Pack Contract

`EXPERIMENT.md` is the canonical experiment narrative. `EXPERIMENT.pack.md` is generated output.

For completed scored experiments, hand off to `experiment-finalize`; it verifies artifacts, updates the final narrative, and renders the pack table.

## Remote Pullback

If training ran on a remote target such as the GPU PC, finished-run pullback is a standard closeout step. Pullback is manual; it does not happen automatically.

Use scoring mode for metrics/reporting artifacts only:

```bash
numereng remote experiment pull --target <target_id> --experiment-id <id> --mode scoring
```

Use full mode only when prediction parquets are needed for submit, ensemble, package, or local rescore work:

```bash
numereng remote experiment pull --target <target_id> --experiment-id <id> --mode full
```

Remote pullback contract:

- only `FINISHED` remote runs are materialized locally
- pulled runs become canonical local runs under `.numereng/runs/<run_id>/`
- the local experiment manifest is reconciled so viz and local reporting see the run history
- rerunning the same pull is idempotent; already-materialized runs are treated as no-ops
- active or incomplete remote runs remain remote-only until they finish and a later pull runs

Treat an experiment as fully available for local viz, reporting, packing, and research-memory ingestion only after the needed pullback has succeeded.

## Run Output Contract

Each completed run should have `.numereng/runs/<run_id>/` with:

- `run.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json` after post-training scoring has been materialized

Expect `run.json` to carry identity, config provenance, metrics summary, declared artifact paths, and `training.scoring` metadata with `policy`, `status`, `requested_stage`, and `refreshed_stages`. Failed or deferred scoring may also include `reason` and `error`.

At minimum, artifact paths declared in `run.json` should exist on disk for successful runs. Prediction parquet outputs should match the recorded artifact paths when predictions were intentionally materialized.

For experiment progress, `experiment.json` is the source of truth for run order and completion count.

## Verification Gates

Before treating experiment work as complete:

- remote-trained runs have been pulled locally with the correct `--mode`
- manifest-listed runs exist under `.numereng/runs/<run_id>/`
- required run metadata and scoring artifacts exist for the claims being made
- store drift has been handed off to `store-ops` when normal experiment commands are no longer enough

## Store Handoff

The store DB at `.numereng/numereng.db` should stay aligned with run directories and experiment manifests.

Use `store-ops` when the task becomes maintenance or repair, especially:

- interrupted or partial runs that need cleanup/reset
- experiment manifest vs run-directory vs DB drift
- missing indexed runs or stale experiment state in sqlite
- destructive cleanup of experiment-linked run data
- post-repair verification after store mutation
- any mutation that should follow a dry-run impact summary

Primary integrity command:

```bash
numereng store doctor
```

If `doctor` reports one known run missing from the index, use `numereng store index --run-id <run_id>`. For broad drift or suspected corruption, use `numereng store rebuild` or hand off to `store-ops`.

## Champion Handoff And Submission

When an experiment has a clear winner and the user wants an official Numerai submission:

1. pick the winning run or config from the recorded experiment metric
2. treat `purged_walk_forward` runs as evaluation artifacts only; do not directly submit their CV prediction parquet
3. rerun the winning config with `training.engine.profile = "full_history_refit"`
4. generate a live predictions parquet for the current Numerai round
5. submit the live predictions through numereng

Current submission caveats:

- `numereng run submit --model-name <model_name> --predictions <live_predictions.parquet>` submits explicit live predictions.
- `numereng run submit --model-name <model_name> --run-id <run_id>` works only when that run contains a live-eligible predictions artifact.
- Historical validation/CV prediction artifacts from experiment training are not live submission files.
- Numereng submission is predictions-based, not Numerai Compute pickle upload.

Use `numerai-api-ops` for direct Numerai API operations, model inspection, rounds, tournaments, and diagnostics.

## Reference Loading Guide

| Request Type | Load |
|---|---|
| Experiment directories, manifests, command families, artifact expectations | `references/experiment-contract.md` |
| Optional experiment-local analysis and deployment artifact areas | `references/experiment-local-artifacts.md` |
| Training schema source paths, template alignment, config authoring rules | `references/config-schema-and-template.md` |
| Baseline directories, `active_benchmark`, benchmark-source choice | `references/baseline-workflow.md` |

If a task spans multiple domains, load each relevant reference and avoid unrelated files.

## Asset Usage Guide

Assets are source templates and backfill material. Copy or use them to update experiment-local files; do not treat asset launchers as the live experiment launcher path.

| Task | Use |
|---|---|
| Refresh experiment narrative or backfill an older stub | `assets/EXPERIMENT.template.md` |
| Log one research round | `assets/research-round-template.md` |
| Start a new training config | `assets/training-config-template.json` |
| Backfill launcher files for an older experiment that predates CLI scaffolding | `assets/launch_all.py`, `assets/launch_all.sh`, `assets/launch_all.ps1` |
| Start an HPO study config | `assets/hpo-study-template.json` |
