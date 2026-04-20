"""Agent-first usage guidance for numereng users working from a cloned checkout."""

# numereng User Guide For Agents

This file is for agents operating `numereng` after a user clones the repo to do Numerai model development.

Default assumption:
- the user wants to use numereng to train, score, compare, package, and submit models
- the user does not want routine edits to `src/` unless they explicitly ask for package-development work

## Working Model
- The repo checkout is the workspace.
- Run everything from the repo root.
- Use the repo-managed environment: `uv sync --extra dev`.
- Prefer `uv run numereng ...` for numereng commands.
- Runtime state lives under `.numereng/`.

## First Commands
```bash
uv sync --extra dev
uv run numereng store init
uv run numereng --help
just viz
```

If Numerai docs are needed locally:
```bash
uv run numereng docs sync numerai
```

## Canonical Paths
- `.numereng/runs/`: run artifacts and scored outputs
- `.numereng/experiments/`: experiment manifests, configs, reports, round-scored workflows
- `.numereng/notes/`: repo-local notes and research memory
- `.numereng/datasets/`: local Numerai datasets, downsampled artifacts, baselines
- `.numereng/cache/`: runtime caches, pulled cloud archives, remote/cache state
- `.numereng/tmp/`: managed scratch paths
- `.numereng/remote_ops/`: remote orchestration state
- `src/numereng/features/models/custom_models/`: default custom model discovery root
- `src/numereng/features/agentic_research/programs/`: default research-program catalog
- `.agents/skills/`: local custom skills; gitignored

## Choose The Right Workflow
- Use `run train` for one standalone local run.
- Use `experiment ...` when the user is comparing related configs, tracking champions, or packaging a project.
- Use `research ...` when the user wants numereng to mutate configs and run an autonomous research loop.
- Use `hpo ...` for Optuna-backed search over one config/search space.
- Use `ensemble ...` when combining scored runs into one ranked blend.
- Use `serve ...` when freezing a production package, rebuilding live predictions, or preparing a Numerai model upload.
- Use `run submit` when a submit-ready parquet or run artifact already exists.
- Use `remote ...` for SSH-driven remote repo sync, experiment launch, pullback, and maintenance.
- Use `cloud ...` for EC2, managed AWS, or Modal workflows.
- Use `store ...` when the filesystem artifacts and SQLite index need repair or reconciliation.
- Use `monitor snapshot` and `just viz` / `numereng viz` for read-only monitoring.

## Default Agent Loop
1. Confirm the user’s current experiment or create one with `numereng experiment create`.
2. Keep configs under `.numereng/experiments/<id>/configs/` when the work belongs to an experiment.
3. Train with `experiment train` for tracked experiment work, or `run train` for a one-off run.
4. Materialize deferred scoring with `run score` or `experiment score-round` when needed.
5. Review results with `experiment report`, `monitor snapshot`, or the dashboard.
6. Use `serve` or `run submit` only after validating the winning run/package.

## Safety Rules
- Do not treat the dashboard as a control plane. It is read-only.
- Do not delete or rewrite `.numereng/` state casually.
- Prefer `store doctor` and `store rebuild` over manual SQLite edits.
- Keep experiment-local work under one experiment root instead of scattering configs around the repo.
- Submission source is XOR: exactly one of `run_id` or `predictions_path`.
- Neutralization source is XOR: exactly one of `run_id` or `predictions_path`.
- Training and HPO configs are JSON-only and reject unknown keys.
- `experiment train` is the correct path for experiment-linked round scoring policies.
- `research init` requires `--program` and persists the selected program snapshot into the experiment.
- `serve pickle build` and `serve pickle upload` are stricter than local live builds; do not assume local success implies hosted-upload success.
- Remote and cloud commands can create or pull runtime state; confirm the target experiment or run before launching them.

## High-Signal Commands
```bash
uv run numereng experiment list
uv run numereng experiment details --id <experiment_id>
uv run numereng experiment report --id <experiment_id>
uv run numereng run score --run-id <run_id>
uv run numereng monitor snapshot --json
just viz
```

## When To Touch `src/`
Only edit `src/` when the user explicitly wants package development, bug fixes, new features, or documentation changes to numereng itself.

If the user just wants to develop Numerai models with numereng, stay in:
- `.numereng/experiments/`
- `.numereng/notes/`
- `.numereng/datasets/`
- `docs/numereng/`
