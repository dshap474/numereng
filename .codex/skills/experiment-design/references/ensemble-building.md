## Ensemble Building (Strategy and CLI Execution)

Use this reference when combining multiple runs into a blended prediction strategy.

### Current Contract

The current numereng CLI provides:
- `uv run numereng ensemble build ...`
- `uv run numereng ensemble list ...`
- `uv run numereng ensemble details ...`

`ensemble build` currently supports `--method rank_avg` only.

Key behavior:
- Provide at least two run IDs via `--run-ids`.
- If `--weights` is omitted, equal weights are used.
- Explicit weights are normalized to sum to 1.
- `--optimize-weights` uses a suffix-based objective; prefer `corr20v2_mean`, `corr20v2_sharpe`, or `max_drawdown`.

Artifacts are written under:
- `.numereng/experiments/<experiment_id>/ensembles/<ensemble_id>/` when `--experiment-id` is set
- `.numereng/ensembles/<ensemble_id>/` when no experiment ID is set

Artifact contract:
- Always-on: `predictions.parquet`, `correlation_matrix.csv`, `metrics.json`, `weights.csv`, `component_metrics.csv`, `era_metrics.csv`, `regime_metrics.csv`, `lineage.json`
- Optional (`--include-heavy-artifacts`): `component_predictions.parquet`, `bootstrap_metrics.json`
- Conditional (`--neutralize-final`): `predictions_pre_neutralization.parquet`

### Candidate Selection Heuristics

- Prefer complementary runs, not only highest single-run score.
- Keep model and target diversity where possible.
- Avoid strongly redundant candidates unless they improve stability.

### Blend Research Pattern (Round-Based)

Use incremental complexity across rounds:
1. Round A: simple equal-weight blend of top complementary runs.
2. Round B: explicit-weight blend if Round A is promising.
3. Round C: optional `--optimize-weights` when labels are available.
4. Round D: optional final-blend neutralization only if exposure diagnostics require it.

Only move to the next complexity tier if the previous tier improves primary experiment objectives.
If a tier fails to improve results, revert to the strongest simpler blend.

### Recommended Process

1. Select candidates from experiment report:

```bash
uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --limit 20 --format table
```

2. Build a candidate worksheet using:
- `../numereng-experiment-ops/assets/weights-template.csv`

3. Build a blend with the CLI:

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
```

Optional heavy diagnostics and final neutralization:

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

4. Inspect blend records:

```bash
uv run numereng ensemble list --experiment-id <id> --format table
uv run numereng ensemble details --ensemble-id <ensemble_id> --format json
```

5. If blended predictions are produced externally, submit them:

```bash
uv run numereng run submit --model-name <numerai_model> --predictions <predictions.csv>
```

### Champion Tracking

Even for blend workflows, keep a champion run in the manifest:

```bash
uv run numereng experiment promote --id <id> --run <run_id>
```

Use the run closest to the production blend inputs or the strongest standalone fallback.

### Verification

- `ensemble list` / `ensemble details` show the saved blend and artifacts path.
- Expected artifact files exist under the reported `artifacts_path`.
- `../numereng-experiment-ops/assets/weights-template.csv` is filled with rank, weight, and rationale.
- `EXPERIMENT.md` documents candidate selection and final blend logic.
- Submission source (`--predictions`) is explicit when using external blends.
- `EXPERIMENT.md` includes why added complexity (optimized weights/heavy diagnostics/neutralization) was justified.
