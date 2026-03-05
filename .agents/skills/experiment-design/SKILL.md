---
name: experiment-design
description: "Complete experiment lifecycle on current numereng contracts: hypothesis, config rounds, training, reporting, champion promotion, and submission handoff."
user-invocable: true
---

# Experiment Design

Canonical execution skill for the full experiment lifecycle in the current numereng contract.

Run from:
- `<repo>`

## Contract Guardrails

Use only commands that exist in the current CLI contract:

- `uv run numereng experiment create|list|details|train|promote|report ...`
- `uv run numereng run train ...`
- `uv run numereng run submit ...`
- `uv run numereng ensemble build|list|details ...`
- `uv run numereng store init|index|rebuild|doctor ...`

Do not use these removed/unsupported families:

- `orchestrator ...`
- `optimize ...`
- `baselines ...`
- `neutralize-sweep ...`
- `experiment summarize|show|compare|set-status|conclude|archive|build-ensemble|rebuild-registry`
- `db validate|db rebuild`

If the user asks for removed command families, translate intent to supported commands and record the fallback in `EXPERIMENT.md`.

## Dataset Variant Policy

Use this dataset-variant default unless the user explicitly overrides it:

- Default for scout rounds and smoke validation: `data.dataset_variant = "downsampled"`.
- Use `data.dataset_variant = "non_downsampled"` for scale/champion/final validation runs.
- If `non_downsampled` is used before scale/final validation, record the reason in `EXPERIMENT.md`.

## Experiment-Local Learning Policy

Do **not** use a separate knowledge-base output system in this skill.

- Do not create or update `.numereng/knowledge/`.
- Keep all learnings in the experiment-local `EXPERIMENT.md`.
- For completed experiments, record findings, anti-patterns, decisions, and next questions directly in that same `EXPERIMENT.md`.

## Store Sync Requirement

The store DB (`.numereng/numereng.db`) should stay aligned with on-disk runs and experiment manifests.

Primary integrity flow:

```bash
uv run numereng store doctor
```

If `doctor` reports issues:

```bash
# single-run drift
uv run numereng store index --run-id <run_id>

# broad drift or suspected corruption
uv run numereng store rebuild

# verify repair
uv run numereng store doctor
```

Bootstrap (new or reset store):

```bash
uv run numereng store init
```

Notes:
- Experiment metadata indexing is triggered by `experiment create`, `experiment train`, and `experiment promote`.
- There is no standalone `experiment rebuild-registry` command in this package.

## Deterministic Routing

Read only the files required for the active task domain.

### Reference Loading Guide

Load the matching reference file before acting in that domain.

| When the task involves... | Load this reference |
|---------------------------|---------------------|
| Hypothesis framing, round planning, stop criteria, reporting | `references/research-strategy.md` |
| Manual parameter sweeps and config-variant comparisons | `references/tuning-and-optimization.md` |
| Feature neutralization timing, objective, and placement in HPO/ensemble flows | `references/feature-neutralization.md` |
| Blend strategy, candidate selection, and weight decisions | `references/ensemble-building.md` |
| Seed variance reduction strategy | `references/seed-ensembling.md` |

If a task spans multiple domains, load each relevant reference and avoid unrelated files.

### Asset Usage Guide

Use assets by explicit task mapping.

| Task | Use this asset |
|------|----------------|
| Create/update experiment narrative | `assets/EXPERIMENT.template.md` |
| Record per-round decisions and outcomes | `assets/research-round-template.md` |
| Start a new training config variant | `assets/training-config-template.json` |
| Define HPO study schema for `hpo create` | `assets/hpo-study-template.json` |
| Draft/track blend weight plans | `assets/weights-template.csv` |

## Methodology Requirements

### Persistence Expectation (Required)

Do not conclude after a single promising run.

- Work in rounds.
- Default round size: 4-5 configs (one base plus single-variable variants).
- Only run fewer variants when the user explicitly requests a minimal pass.
- Finalize only after explicit plateau checks are satisfied.

### Planning Checklist (Before Round 1)

- State the model idea and novelty.
- Choose baseline config and keep baseline alignment explicit.
- Set primary and tie-break metrics:
  - primary: `bmc_last_200_eras.mean`
  - tie-break: `bmc.mean`
- Choose sweep dimension(s) tied to the hypothesis.
- Define risk checks (`corr.mean`, `mmc.mean`, `cwmm.mean`).
- Define plateau gate settings for this experiment.

### Handling Ambiguity (Fast Disambiguation)

If the request is underspecified:
1. List 2-4 materially different interpretations.
2. Create quick scout variants for each interpretation.
3. Use lower-cost settings for scout runs (for example, downsampled dataset variant and smaller model capacity).
4. Compare interpretations on primary/tie-break metrics plus stability checks.
5. Pick one direction and record why in `EXPERIMENT.md`.

### Scout -> Scale Policy

Use a staged approach:
1. Scout phase:
  - controlled, lower-cost runs
  - single-variable comparisons
2. Scale phase:
  - only after scout rounds show repeatable improvements
  - expand compute or feature scope for top candidates only

Scale gate:
- Run at least one confirmatory scaled round before concluding the idea is maxed out.

### Numeric Plateau Stop Gate (Default)

Stop iterating only when all conditions hold:
1. At least two consecutive rounds do not beat the best `bmc_last_200_eras.mean` by a meaningful margin.
2. Default meaningful margin threshold is `1e-4` to `3e-4`.
3. Remaining untried knobs are documented as likely redundant or high overfit risk.
4. A scaled confirmatory round (for top candidates) has been run.

## Pipeline Phases

### Phase 1: Create and Initialize

Create the experiment and initialize baseline artifacts.

```bash
uv run numereng experiment create \
  --id <YYYY-MM-DD_slug> \
  --name "<name>" \
  --hypothesis "<hypothesis>" \
  --tags "tag1,tag2"
```

Required ID format: `YYYY-MM-DD_slug` (example: `2026-02-22_ender20-lgbm-sweep`).

After creation:
- Use `assets/EXPERIMENT.template.md` for experiment logging.
- Use `assets/research-round-template.md` for round-by-round notes.
- Use `assets/training-config-template.json` as the starter training config schema.
- The starter config template is scout-oriented (`val_predictions_scout` naming); update naming and compute profile for scaled rounds.
- Keep configs under `.numereng/experiments/<experiment_id>/configs/`.
- Canonical contract source:
  - `src/numereng/config/training/CLAUDE.md`
  - `src/numereng/config/training/schema/training_config.schema.json`

### Phase 2: Research Rounds (Train + Evaluate)

Run 4-5 config variants per round by default, changing one variable at a time.

```bash
uv run numereng experiment train --id <id> --config <config.json>
uv run numereng experiment train --id <id> --config <config.json> --profile purged_walk_forward
uv run numereng experiment train --id <id> --config <config.json> --profile simple
uv run numereng experiment train --id <id> --config <config.json> --profile submission
```

After each round:

```bash
uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format table
uv run numereng experiment details --id <id> --format table
```

Update `EXPERIMENT.md` with:
- what changed
- ranked run table
- round-best delta versus prior best
- decision and next round

Per-round synthesis requirements:
- Choose current winner by `bmc_last_200_eras.mean` (primary) and `bmc.mean` (tie-break).
- Sanity-check `corr.mean`, `mmc.mean`, and `cwmm.mean`.
- Decide continue, pivot, or stop using the numeric plateau gate.

### Phase 3: Manual Tuning and Optimization

There is no built-in `optimize` command family in the current CLI.

Use manual config sweeps:
1. Define a base config.
2. Create variant configs in `.numereng/experiments/<id>/configs/`.
3. Execute each variant with `experiment train`.
4. Compare with `experiment report` using a fixed metric.

Schema helper:
- `assets/hpo-study-template.json` provides the canonical `hpo create --study-config` shape.

Optional HPO execution path:
- `uv run numereng hpo create --study-config <path.json>`

Sweep selection should match research type:
- target/label idea -> sweep target or preprocessing variants first
- model architecture idea -> sweep model hyperparameters
- ensemble/blend idea -> sweep candidate composition and weights
- training procedure idea -> sweep procedure controls (profile, loading/scoring mode, neutralization settings)
- data scope idea -> sweep feature scope and dataset variant (`downsampled` scout vs `non_downsampled` scale)

Avoid broad unfocused sweeps; each round should answer one concrete question.

Feature neutralization policy in this phase:
- Neutralization is a **prediction-stage transform**, not a model retraining method.
- If you plan to use neutralization in production/submission, include it in trial scoring.
- Evaluate trials as: `train -> predict -> neutralize -> score`.
- If compute is constrained, run neutralization re-ranking on top-K base trials, then choose final winner from those neutralized scores.

### Phase 4: Ensemble and Seed Strategy

Use this phase for:
- candidate run selection strategy
- weight planning
- blend construction with `ensemble build`
- external blend generation fallback (if needed)

Build and inspect blends with the current CLI:

```bash
uv run numereng ensemble build \
  --experiment-id <id> \
  --run-ids <run_a,run_b,run_c> \
  --method rank_avg \
  --metric corr20v2_sharpe \
  --target target_ender_20 \
  --name "<blend name>" \
  --weights 0.50,0.30,0.20 \
  --selection-note "diversity-first blend" \
  --regime-buckets 4

uv run numereng ensemble list --experiment-id <id> --format table
uv run numereng ensemble details --ensemble-id <ensemble_id> --format json
```

Optional heavy diagnostics and final-blend neutralization:

```bash
uv run numereng ensemble build \
  --experiment-id <id> \
  --run-ids <run_a,run_b,run_c> \
  --method rank_avg \
  --metric corr20v2_sharpe \
  --target target_ender_20 \
  --name "<blend name>" \
  --optimize-weights \
  --include-heavy-artifacts \
  --neutralize-final \
  --neutralizer-path <neutralizer.parquet> \
  --neutralization-proportion 0.50 \
  --neutralization-mode era
```

Notes:
- CLI currently supports `--method rank_avg` only.
- If `--weights` is omitted, equal weights are used. Explicit weights are validated, then normalized to sum to 1.
- Use `--optimize-weights` when target labels are available for weight tuning.
- Optimization objective is suffix-driven; prefer `corr20v2_mean`, `corr20v2_sharpe`, or `max_drawdown`.
- Artifacts are persisted under `.numereng/experiments/<id>/ensembles/<ensemble_id>/` (or `.numereng/ensembles/<ensemble_id>/` without `--experiment-id`):
  - Always: `predictions.parquet`, `correlation_matrix.csv`, `metrics.json`, `weights.csv`, `component_metrics.csv`, `era_metrics.csv`, `regime_metrics.csv`, `lineage.json`
  - Optional (`--include-heavy-artifacts`): `component_predictions.parquet`, `bootstrap_metrics.json`
  - Conditional (`--neutralize-final`): `predictions_pre_neutralization.parquet`
- Neutralization with ensembles should usually start as: `member predictions -> blend -> neutralize once`.
- Add member-level neutralization only if diagnostics show unresolved exposure risk after final-blend neutralization.
- Avoid heavy neutralization at both member and final levels unless explicitly validated.

If blended predictions are produced externally, submit via:

```bash
uv run numereng run submit --model-name <numerai_model> --predictions <predictions.csv>
```

Keep a promoted fallback run in the manifest:

```bash
uv run numereng experiment promote --id <id> --run <run_id>
```

Track planned blend weights in `assets/weights-template.csv` and record rationale in `EXPERIMENT.md`.

### Phase 5: Analyze and Report

Primary reporting loop:

```bash
uv run numereng experiment list --status active --format table
uv run numereng experiment details --id <id> --format json
uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --limit 20 --format json
```

Use report metrics to identify stable winners and regression risk.

Reporting requirements (each round and final summary):
- clear statement of what changed and why
- table of run metrics with primary, tie-break, and sanity metrics
- explicit winner rationale
- explicit risk discussion
- explicit next action (continue, pivot, or stop)
- final story from hypothesis -> rounds -> final decision

### Phase 6: Promote and Submit

Promote champion by explicit run or automatic best metric:

```bash
uv run numereng experiment promote --id <id> --metric bmc_last_200_eras.mean
uv run numereng experiment promote --id <id> --run <run_id>
```

Submit champion predictions:

```bash
# source from a run (preferred)
uv run numereng run submit --model-name <numerai_model> --run-id <run_id>

# source from an explicit predictions file
uv run numereng run submit --model-name <numerai_model> --predictions <predictions.csv>
```

Submission source is XOR: provide exactly one of `--run-id` or `--predictions`.

## Training Config Contract (Current)

Training config is JSON-only and defined under:
- `src/numereng/config/training/contracts.py` (typed source of truth)
- `src/numereng/config/training/schema/training_config.schema.json` (machine-readable schema)
- `src/numereng/config/training/CLAUDE.md` (operational contract guide)

Required keys:
- top-level: `data`, `model`, `training`
- nested: `model.type`, `model.params`

Unknown keys are forbidden by contract.

Use this as a valid training config shape:

```json
{
  "data": {
    "data_version": "v5.2",
    "dataset_variant": "downsampled",
    "feature_set": "small",
    "target_col": "target_ender_20",
    "target_horizon": "20d",
    "era_col": "era",
    "id_col": "id"
  },
  "preprocessing": {
    "nan_missing_all_twos": false,
    "missing_value": 2.0
  },
  "model": {
    "type": "LGBMRegressor",
    "params": {
      "n_estimators": 200,
      "learning_rate": 0.03,
      "num_leaves": 64
    },
    "x_groups": [
      "features",
      "era"
    ]
  },
  "training": {
    "engine": {
      "profile": "purged_walk_forward"
    }
  }
}
```

Training profile constraints:
- `simple`: train on train eras and validate on validation eras (split sources only).
- `purged_walk_forward`: 156-era walk-forward CV over train+validation with embargo by horizon (`20d -> 8`, `60d -> 16`).
- `submission`: train on full history and skip validation metrics.

Legacy config fields like `training.method`, `training.strategy`, or `training.cv` are hard-fail.

## Run Output Contract (Current)

Training artifacts are written under `.numereng/runs/<run_id>/`:

- `run.json`
- `metrics.json`
- `results.json`
- `resolved.json`
- optional `score_provenance.json`
- `artifacts/` (predictions/model assets)

Experiment artifacts are written under `.numereng/experiments/<experiment_id>/`:

- `experiment.json`
- `EXPERIMENT.md`
- `configs/`
- `ensembles/<ensemble_id>/` when `ensemble build --experiment-id <experiment_id>` is used

Global ensemble artifacts are written under `.numereng/ensembles/<ensemble_id>/` when no experiment ID is provided.

Canonical report/promotion metric keys:
- `bmc_last_200_eras.mean`
- `bmc.mean`
- `corr.mean`
- `mmc.mean`
- `cwmm.mean`

## Quick Command Reference

| Task | Command |
|------|---------|
| Create experiment | `uv run numereng experiment create --id <YYYY-MM-DD_slug> --hypothesis "..."` |
| List experiments | `uv run numereng experiment list --status active --format table` |
| Experiment details | `uv run numereng experiment details --id <id> --format json` |
| Train (experiment-linked) | `uv run numereng experiment train --id <id> --config <json>` |
| Train (standalone) | `uv run numereng run train --config <json>` |
| Rank runs | `uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean` |
| Build ensemble | `uv run numereng ensemble build --experiment-id <id> --run-ids <run_a,run_b> --method rank_avg` |
| List ensembles | `uv run numereng ensemble list --experiment-id <id> --format table` |
| Ensemble details | `uv run numereng ensemble details --ensemble-id <ensemble_id> --format json` |
| Promote champion | `uv run numereng experiment promote --id <id> --metric bmc_last_200_eras.mean` |
| Submit run | `uv run numereng run submit --model-name <name> --run-id <run_id>` |
| Store diagnostics | `uv run numereng store doctor` |
| Store rebuild | `uv run numereng store rebuild` |

## Verification

- `experiment list --status active` includes the experiment.
- `experiment details --id <id>` shows expected run count and champion state.
- `experiment report --id <id>` returns ranked rows for the selected metric.
- `ensemble details --ensemble-id <ensemble_id>` returns expected components, metrics, and `artifacts_path`.
- `store doctor` reports `ok: true` after indexing/rebuild repairs.
- `EXPERIMENT.md` includes ambiguity resolution notes (if applicable), plateau gate tracking, and remaining-knobs audit.

## Assets

- Experiment log template: `assets/EXPERIMENT.template.md`
- Research round log template: `assets/research-round-template.md`
- Starter training config template: `assets/training-config-template.json`
- HPO study config template: `assets/hpo-study-template.json`
- Ensemble weight worksheet: `assets/weights-template.csv`

## References

- `references/research-strategy.md`
- `references/tuning-and-optimization.md`
- `references/feature-neutralization.md`
- `references/ensemble-building.md`
- `references/seed-ensembling.md`
