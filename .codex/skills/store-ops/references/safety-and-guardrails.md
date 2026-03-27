# Safety and Guardrails

## Default Safety Mode

All destructive operations use two-step confirmation:

1. Dry-run impact summary.
2. Explicit user confirmation for the exact scope and preserve policy.

Do not execute destructive actions without both steps.

## Preflight Rules

Before mutation:
- detect active training writer processes
- detect run ID overlap with non-target experiments
- detect missing manifest paths for target experiments
- detect store DB accessibility

If any preflight check fails, stop and report evidence.

## Active Writer Detection

Block by default when any process indicates local training write activity, for example:
- `numereng run train`
- `numereng experiment train`
- `numereng cloud aws train submit`
- `numereng cloud modal train submit`

Allow override only when user explicitly requests proceeding despite active writers.

## Overlap Protection

Before deleting a run directory or run DB row:
- verify the run is not referenced by a non-target experiment manifest
- if overlap exists, abort and surface overlap details

## Preserve Policies

Supported policies:
- `preserve_design_files` (default):
  - keep `configs/`
  - keep `EXPERIMENT.md`
  - reset `experiment.json` runtime fields
- `preserve_only_configs`:
  - keep `configs/`
  - rewrite `EXPERIMENT.md` to minimal summary template
- `hard_wipe_experiment_dirs`:
  - delete entire experiment directories and experiment DB rows

## Transaction Safety

Use one SQL transaction for destructive DB work:
- `BEGIN`
- ordered deletes
- `COMMIT`

On any failure:
- rollback
- report failing statement/table
- do not perform additional mutation steps

## Required Postconditions

After mutation:
- target run directories removed as expected
- target experiment manifests reflect expected status
- target DB row counts are zero (or expected value)
- `uv run numereng store doctor` reports `ok: true`

If postconditions fail, stop and report residual artifacts/rows.
