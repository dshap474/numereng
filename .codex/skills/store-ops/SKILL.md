---
name: store-ops
description: "Use for numereng store maintenance, including drift diagnosis, experiment-linked run cleanup/reset, store reindex, and postcondition verification with strict safety gates."
user-invocable: true
argument-hint: "<store operation intent> (e.g., reset experiment runs, diagnose store drift, reindex one run)"
---

# Store Ops

Canonical skill for safe maintenance of numereng local store state.

Run from:
- `<repo>`

Use this only after the experiment/runtime contract is clear. For experiment layout, config
schema, template files, and run artifact expectations, use `numereng-experiment-ops`.

Use this skill when requests involve:
- store drift diagnosis
- run/index mismatch repair
- experiment-linked run cleanup/reset
- store reindex/rebuild with verification

## Scope

In scope:
- store inspection and health checks
- run/store drift diagnosis
- targeted run deletion/reset for one or more experiments
- run reindex (`store index`) and full rebuild (`store rebuild`)
- post-mutation verification

Out of scope:
- experiment strategy
- config schema or template authoring
- experiment directory/bootstrap conventions
- experiment archive/unarchive lifecycle changes
- schema migration authoring
- cloud archival/backfill strategy
- product feature implementation in `src/numereng/*`

## Hard Rules

- Prefer package CLI entrypoints and explicit commands:
  - `uv run numereng store init|index|rebuild|doctor ...`
  - `uv run numereng experiment details|report ...`
- Archive/unarchive is an experiment lifecycle workflow, not a store-reset workflow.
- Use `uv run numereng experiment archive|unarchive ...` for archive state changes.
- Use SQLite inspection only when needed to explain or repair index drift.
- Destructive operations must follow two-step confirmation:
  1. run dry-run impact summary
  2. obtain explicit user confirmation before mutation
- Default reset behavior must preserve experiment design assets:
  - keep `configs/`
  - keep `EXPERIMENT.md`
  - clear linked run data and runtime index state
- Abort destructive mutations when active training writers are detected unless user explicitly overrides.
- Do not load unrelated references/assets.

## Reference Loading Guide

| Request Type | Load |
|---|---|
| Store shape, canonical paths, DB tables | `references/store-layout-and-schema.md` |
| End-to-end operation procedures | `references/core-workflows.md` |
| Confirmation flow, guardrails, abort conditions | `references/safety-and-guardrails.md` |

If a task spans multiple domains, load each relevant reference and avoid unrelated files.

Explicit path gates:
- If task involves store shape/table dependencies, load `references/store-layout-and-schema.md`.
- If task involves execution sequencing, load `references/core-workflows.md`.
- If task involves safety decisions or abort logic, load `references/safety-and-guardrails.md`.

## Asset Usage Guide

| Task | Use |
|---|---|
| Impact report before destructive action | `scripts/collect_store_impact.py` |
| Reset/delete one or more specific runs | `scripts/reset_runs.py` |
| Reset/delete run data for experiments | `scripts/reset_experiment_runs.py` |
| SQL row-impact checks and validation | `assets/sql/impact-queries.sql` |
| SQL delete-order reference | `assets/sql/reset-experiment-template.sql` |
| Process preflight check | `assets/bash/preflight-checks.sh` |
| Post-operation validation | `assets/bash/verification-checks.sh` |

Explicit path gates:
- If task needs impact preview before mutation, use `scripts/collect_store_impact.py`.
- If task resets one or more explicit run IDs, use `scripts/reset_runs.py`.
- If task resets experiment-linked run state, use `scripts/reset_experiment_runs.py`.
- If task needs shell-level preflight/verification bundles, use `assets/bash/preflight-checks.sh` and `assets/bash/verification-checks.sh`.

## Core Workflow

### 1) Read-Only Diagnosis

Run this first for any drift/cleanup request:

```bash
uv run numereng store doctor
uv run numereng experiment details --id <experiment_id> --format json
uv run numereng experiment report --id <experiment_id> --format json
```

Then compute impact without mutation:

```bash
uv run python .agents/skills/store-ops/scripts/collect_store_impact.py \
  --experiment-id <experiment_id>
```

### 2) Single-Run Repair

When one known run is missing from the DB index:

```bash
uv run numereng store index --run-id <run_id>
uv run numereng store doctor
```

### 3) Single-Run Reset (Delete Run Folder + Indexed State)

First, impact preview:

```bash
uv run python .agents/skills/store-ops/scripts/reset_runs.py \
  --run-id <run_id>
```

Then execute after explicit confirmation:

```bash
uv run python .agents/skills/store-ops/scripts/reset_runs.py \
  --run-id <run_id> \
  --execute
```

Multi-run reset:

```bash
uv run python .agents/skills/store-ops/scripts/reset_runs.py \
  --run-id <run_id_a> \
  --run-id <run_id_b> \
  --execute
```

### 4) Broad Reindex

When index drift is broad or uncertain:

```bash
uv run numereng store rebuild
uv run numereng store doctor
```

### 5) Experiment Run Reset (Default Preserve Policy)

First, impact preview:

```bash
uv run python .agents/skills/store-ops/scripts/reset_experiment_runs.py \
  --experiment-id <experiment_id>
```

Then execute after explicit confirmation:

```bash
uv run python .agents/skills/store-ops/scripts/reset_experiment_runs.py \
  --experiment-id <experiment_id> \
  --execute
```

Optional destructive policies:
- `--preserve-policy preserve_only_configs`
- `--preserve-policy hard_wipe_experiment_dirs`

### 6) Verification

```bash
uv run numereng experiment details --id <experiment_id> --format json
uv run numereng experiment report --id <experiment_id> --format json
uv run numereng store doctor
```

Use `assets/bash/verification-checks.sh` for a bundled check.

## Confirmation Protocol (Required)

For destructive store changes:
1. Provide dry-run impact summary with affected run IDs, run directories, and DB rows by table.
2. Confirm exact preserve policy and scope with user.
3. Execute mutation only after explicit confirmation.
4. Report postconditions with objective checks.

## Error Handling

- `active training writer detected`:
  - stop and request explicit override
- `run IDs referenced by non-target experiments`:
  - stop by default; surface overlaps
- `manifest/run-dir mismatch`:
  - include DB + filesystem evidence in dry-run report
- `partial SQL failure`:
  - rollback transaction and report exact failing table/statement

## Done Criteria

Operation is complete only when all apply:
- expected manifests updated to requested state
- target run directories removed (or unchanged for read-only actions)
- DB row counts for target scope reflect intended result
- `uv run numereng store doctor` returns `ok: true`
- verification output is shared with user
