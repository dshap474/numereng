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
- which numereng commands are valid for experiment execution
- what run outputs and metrics files should exist

Do not use this skill for experiment strategy or plateau decisions. Use `experiment-design`
for that. Do not use this skill for drift repair, reindex, or destructive cleanup. Use
`store-ops` for that.

Run from:
- `<repo>`

## Scope

In scope:
- experiment directory layout
- experiment creation and resume conventions
- config file naming and placement
- `EXPERIMENT.md` and round-log formatting
- training config schema entrypoints
- current CLI/API command contract for experiment execution
- run output and metric artifact expectations
- deciding when to hand off to `store-ops`

Out of scope:
- experiment methodology
- sweep design
- plateau decisions
- store repair or cleanup

## Hard Rules

- Prefer package CLI entrypoints and explicit commands:
  - `uv run numereng experiment create|list|details|train|promote|report ...`
  - `uv run numereng run train ...`
  - `uv run numereng ensemble build|list|details ...`
  - `uv run numereng hpo create ...`
  - `uv run numereng store init|index|rebuild|doctor ...`
- Do not use removed or unsupported families:
  - `orchestrator ...`
  - `optimize ...`
  - `baselines ...`
  - `neutralize-sweep ...`
  - `experiment summarize|show|compare|set-status|conclude|archive|build-ensemble|rebuild-registry`
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

- `.numereng/experiments/<experiment_id>/`

Canonical experiment files:

- `.numereng/experiments/<experiment_id>/experiment.json`
- `.numereng/experiments/<experiment_id>/EXPERIMENT.md`
- `.numereng/experiments/<experiment_id>/configs/*.json`
- `.numereng/experiments/<experiment_id>/run_plan.csv` when the experiment is using a planned sweep

Use one experiment directory for one line of inquiry.

Recommended experiment ID format:

- `YYYY-MM-DD_slug`

Example:

- `2026-03-06_small-lgbm-all-targets-3seed-baseline`

## Experiment Creation And Logging

Create experiments with:

```bash
uv run numereng experiment create \
  --id <YYYY-MM-DD_slug> \
  --name "<name>" \
  --hypothesis "<hypothesis>" \
  --tags "tag1,tag2"
```

After creation:

- keep the canonical narrative in `EXPERIMENT.md`
- keep configs in `configs/`
- use `run_plan.csv` only when a sweep ordering is intentionally defined
- update the experiment log after each completed round

Use these assets:

- `assets/EXPERIMENT.template.md`
- `assets/research-round-template.md`

## Config File Conventions

Keep training configs under:

- `.numereng/experiments/<experiment_id>/configs/`

Config naming should make the changed variable obvious.

Examples:

- `base.json`
- `r1_target_rowan_60_seed42.json`
- `r2_lr_0p003_numleaves_128.json`

For manual sweeps:

- start from `assets/training-config-template.json`
- change one variable at a time per config inside a round
- keep the baseline config in the same experiment directory

For HPO study definitions:

- use `assets/hpo-study-template.json`

## Schema Source Of Truth

Training config schema lives under:

- `src/numereng/config/training/`

Primary sources:

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
uv run numereng experiment train --id <id> --config <config.json>
uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format table
uv run numereng experiment details --id <id> --format json
uv run numereng experiment promote --id <id> --metric bmc_last_200_eras.mean
uv run numereng ensemble build --experiment-id <id> --run-ids <run_a,run_b,...> --method rank_avg
uv run numereng hpo create --study-config <path.json>
```

Use `uv run numereng run train ...` only when the task is intentionally outside the experiment
manifest workflow.

## Run Output Contract

Each completed run should have a run directory under:

- `.numereng/runs/<run_id>/`

Core files expected for completed scored runs:

- `run.json`
- `resolved.json`
- `results.json`
- `metrics.json`
- `score_provenance.json`

Expect `run.json` to carry:

- run identity
- config provenance
- metrics summary
- declared artifact paths

At minimum, artifact paths declared in `run.json` should exist on disk for successful runs.
Prediction parquet outputs should match the recorded artifact paths.

For experiment progress, the manifest in `experiment.json` is the source of truth for run order
and completion count.

## Store Sync And Handoff To Store Ops

The store DB at `.numereng/numereng.db` should stay aligned with run directories and experiment
manifests.

Primary integrity flow:

```bash
uv run numereng store doctor
```

If `doctor` reports issues:

- one known run missing from the index:
  - `uv run numereng store index --run-id <run_id>`
- broad drift or suspected corruption:
  - `uv run numereng store rebuild`

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

If a task spans multiple domains, load each relevant reference and avoid unrelated files.

## Asset Usage Guide

| Task | Use |
|---|---|
| Create or refresh experiment narrative | `assets/EXPERIMENT.template.md` |
| Log one research round | `assets/research-round-template.md` |
| Start a new training config | `assets/training-config-template.json` |
| Start an HPO study config | `assets/hpo-study-template.json` |

Explicit path gates:
- If the task is about experiment narrative or round logging, use the markdown templates.
- If the task is about training config shape, use `assets/training-config-template.json` plus the schema source paths.
- If the task is about HPO study authoring, use `assets/hpo-study-template.json`.
