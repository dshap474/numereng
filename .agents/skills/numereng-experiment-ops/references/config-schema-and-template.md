# Config Schema And Template

Use this reference when the task is about the numereng training config contract.

## Source Of Truth

Primary schema and loader paths:

- `src/numereng/config/training/CLAUDE.md`
- `src/numereng/config/training/contracts.py`
- `src/numereng/config/training/loader.py`
- `src/numereng/config/training/schema/training_config.schema.json`

## Template Alignment Rule

- `assets/training-config-template.json` is a starter template, not a second schema.
- If the template disagrees with the live schema or loader behavior, the schema and loader win.
- Update the template before using it for new experiment configs.

## Authoring Rules

- keep configs under `.numereng/experiments/<experiment_id>/configs/`
- make file names reflect the changed variable
- change one variable at a time within a research round unless the round is intentionally testing an interaction
- keep dataset variant, feature scope, and benchmark reference explicit in the config
- prefer `data.benchmark_source.source = "path"` when a machine has not yet seeded
  `.numereng/datasets/baselines/active_benchmark/`
- use default `benchmark_source.source = "active"` only when the shared active
  benchmark artifact is known to exist
