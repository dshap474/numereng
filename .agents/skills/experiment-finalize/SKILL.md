---
name: experiment-finalize
description: "Finalize a completed numereng experiment after scoring artifacts exist: verify evidence, rewrite EXPERIMENT.md, and render canonical EXPERIMENT.pack.md with one Run Ops-style row per run."
user-invocable: true
argument-hint: "<experiment_id or experiment path>"
---

# Experiment Finalize

## Role / Purpose

Finalize one completed numereng experiment after training and scoring artifacts exist. The skill verifies artifact evidence, rewrites `EXPERIMENT.md` as the durable decision memo, and renders the generated `EXPERIMENT.pack.md`.

Use `experiment-ops` for the general experiment contract and pullback, `experiment-design` for new experiment strategy, and `research-memory-update` only after the final report is done.

## Personality / Collaboration Style

Write like a cautious research operator. Prefer exact run ids, target names, metric names, artifact counts, and concrete caveats over broad claims.

## Goal

Produce a completed closeout package:

- finalized `.numereng/experiments/<id>/EXPERIMENT.md`
- generated `.numereng/experiments/<id>/EXPERIMENT.pack.md`
- generated `.numereng/tmp/experiment-finalize/<id>.context.md`

## Success Criteria

- Every manifest-listed run exists locally and is `FINISHED`.
- Required run metadata and core scoring artifacts exist for every claim.
- Full-summary metrics are used only when `post_training_full_summary.parquet` exists.
- `EXPERIMENT.md` states verdict, evidence status, candidate hierarchy, metric conflicts, risks, and next experiment pass criteria.
- `EXPERIMENT.pack.md` starts with the base prompt, then experiment-specific context, then `# Experiment Pack`.
- The pack table has one row per manifest run, or one row per ensemble artifact for ensemble-only experiments, and uses Run Ops metric columns.

## Constraints

- Run from the repo root.
- Predictions are not required for report finalization. Require prediction parquets only for submission, ensemble, package, or local rescore work.
- Ensemble-only closeout is valid when `experiment.json` has no manifest runs and `ensembles/` contains completed ensemble artifacts.
- Treat `EXPERIMENT.md` as the source narrative and `EXPERIMENT.pack.md` as generated output.
- Use canonical filename `EXPERIMENT.pack.md`.
- Do not create or use `references/prompt.md`.
- Do not invent metrics or fill gaps with guesses.
- Render missing metric values as `n/a` and call out meaningful gaps in `EXPERIMENT.md`.
- Do not leave the strongest interpretation only in `EXPERIMENT.pack.md`, generated context, or a later Pro response.
- Preserve run ids, config names, target names, and metric names exactly.
- Put the full per-run table in `EXPERIMENT.pack.md`, not in `EXPERIMENT.md`.
- Put the full per-run or per-ensemble table in `EXPERIMENT.pack.md`, not in `EXPERIMENT.md`.
- Do not update research memory from this skill.

## Evidence Language

Use these labels when writing or checking conclusions:

- `verified artifact`: file, run, status, manifest, or count directly checked on disk.
- `computed metric`: value directly computed or read from metrics/scoring tables.
- `supported inference`: interpretation supported by artifact evidence but not directly stored as a file.
- `hypothesis / next-step`: plausible claim that still requires another scored experiment.

Candidate wording must stay evidence-level accurate:

- Use `best single run` for the top row.
- Use `candidate family` when seed/family evidence supports follow-up.
- Use `stabilizer candidate` until a scored blend proves stabilization.
- Use `ensemble candidate` until an ensemble artifact is built and scored.
- Reserve `champion` for production-ready evidence with appropriate handoff checks.
- Use `no champion` whenever evidence is validation-only, single-row selected, missing ensemble/correlation checks, or missing production-readiness checks.

## Metric Conflict Severity

Describe metric conflicts with severity, not binary language.

- BMC: `strong`, `moderate`, `weak`, or `mixed`.
- MMC: `strong`, `positive but marginal`, `mixed`, `weak`, or `missing`.
- FNC: `positive`, `mixed`, `negative`, or `missing`.
- Drawdown: `clean`, `target-dependent`, or `warning`.
- Exposure: `measured`, `missing`, or `promotion gate`.
- Coverage/comparability: call out limited coverage, missing full summaries, target preselection, feature-set differences, model-recipe changes, and post-selection effects.

Avoid unqualified phrases like `no major conflict` when a supporting metric is small, mixed, seed-sensitive, or coverage-limited. If relying on `mmc_coverage_ratio_rows`, explain what it means or mark MMC interpretation as coverage-limited.

## Workflow

1. Resolve the experiment id from an id or `.numereng/experiments/<id>/` path.
2. Read:
   - `.numereng/experiments/<id>/experiment.json`
   - `.numereng/experiments/<id>/EXPERIMENT.md`
   - `.numereng/experiments/<id>/configs/`
   - `.numereng/experiments/<id>/run_plan.csv` when present
   - `.numereng/runs/<run_id>/run.json`
   - `.numereng/runs/<run_id>/metrics.json`
   - `.numereng/runs/<run_id>/results.json`
   - `.numereng/runs/<run_id>/resolved.json`
   - `.numereng/runs/<run_id>/score_provenance.json`
   - scoring parquet summaries under `.numereng/runs/<run_id>/artifacts/scoring/`
   - `references/report-contract.md`
   - `references/base-prompt.md`
   - `references/exp-specific-prompt.md`
3. Confirm every manifest-listed run exists locally and is `FINISHED`. If the manifest has no runs, confirm this is an ensemble-only experiment and `ensembles/` contains completed artifacts.
4. Confirm required metadata and scoring evidence exists:
   - `run.json`
   - `metrics.json`
   - `results.json`
   - `resolved.json`
   - `score_provenance.json`
   - `artifacts/scoring/manifest.json`
   - `artifacts/scoring/run_metric_series.parquet`
   - `artifacts/scoring/post_training_core_summary.parquet`
5. Use `post_training_full_summary.parquet` for FNC and feature-exposure claims only when present.
   For ensemble-only experiments, use ensemble `lineage.json`, `metrics.json`, `weights.parquet`, `era_metrics.parquet`, and `predictions.parquet`; compute benchmark-relative BMC by joining ensemble predictions to source-run target columns and the active benchmark.
6. Build the evidence table:

```bash
uv run python .agents/skills/experiment-finalize/scripts/render_experiment_pack.py --experiment-id <id> --dry-run
```

7. Build the evidence brief:

```bash
uv run python .agents/skills/experiment-finalize/scripts/summarize_experiment_evidence.py --experiment-id <id>
```

8. Rewrite `EXPERIMENT.md` from the artifact evidence and evidence brief. Include:
   - explicit hypothesis verdict: supported, partially supported, or rejected
   - evidence status and operational caveats
   - matrix-level result
   - best single run or ensemble versus best candidate family
   - target-family, seed, and horizon analysis
   - metric conflicts with severity
   - candidate assessment and champion/no-champion decision
   - special-case candidates separated from production-ready candidates
   - risks and weak claims
   - next experiment with pass criteria
   - pre-production questions, including hidden selection pressure from target, recipe, metric, and prior experiment choices
   - final checks and repro commands
9. Generate the experiment-specific ChatGPT Pro context from the completed report and scoring evidence. Save it to:

```bash
.numereng/tmp/experiment-finalize/<experiment_id>.context.md
```

10. Render the final pack:

```bash
uv run python .agents/skills/experiment-finalize/scripts/render_experiment_pack.py \
  --experiment-id <id> \
  --experiment-context-path .numereng/tmp/experiment-finalize/<experiment_id>.context.md
```

## Output

Final response should summarize:

- artifact completeness
- final verdict
- best single run, candidate family, ensemble candidate, and champion/no-champion decision
- key metric conflicts and caveats
- next experiment
- verification commands run

## Pack Table Contract

When `--experiment-context-path` is provided, the pack renderer writes:

1. static base prompt from `references/base-prompt.md`
2. generated experiment-specific context from `--experiment-context-path`
3. the existing experiment pack body

The pack renderer writes one row per manifest-listed run, or one row per ensemble artifact when the experiment has no runs and has `ensembles/`. Rows are sorted by `bmc_last_200_eras_mean` descending and then id.

Base columns:

- `run_id`
- `config`
- `model`
- `target`
- `feature_set`
- `status`

Metric columns follow `RUNOPS_ALL_SCORING_METRICS` from `viz/web/src/lib/metrics/canonical.ts`. If that source cannot be read, the script uses its fallback order.

## Verification

After finalization:

```bash
uv run python .agents/skills/experiment-finalize/scripts/render_experiment_pack.py --experiment-id <id> --dry-run
uv run python .agents/skills/experiment-finalize/scripts/summarize_experiment_evidence.py --experiment-id <id>
uv run python .agents/skills/experiment-finalize/scripts/render_experiment_pack.py \
  --experiment-id <id> \
  --experiment-context-path .numereng/tmp/experiment-finalize/<experiment_id>.context.md
```

Confirm:

- dry-run reports one table row per manifest run or ensemble artifact
- `references/prompt.md` does not exist
- evidence brief reports one row per manifest run or ensemble artifact
- `EXPERIMENT.md` has no stale placeholders or obsolete launcher paths
- `EXPERIMENT.md` metrics match run artifacts
- `EXPERIMENT.pack.md` exists and starts with `references/base-prompt.md`
- generated experiment-specific context appears before `# Experiment Pack`
- pack table contains Run Ops scoring columns
- missing metrics render as `n/a`

## Stop Rules

Stop and report exact blockers when:

- `experiment.json` is missing or malformed
- the manifest has no runs and no completed ensemble artifacts are available
- any manifest run directory is missing
- any manifest run is not `FINISHED`
- any required metadata file is missing
- required scoring artifacts are missing for the claims being made
- `EXPERIMENT.md` is missing
- the evidence brief cannot be generated
- the pack renderer fails validation
- the generated experiment-specific context file is missing or empty when prompt injection is expected
- the report would need prediction artifacts, full summaries, or external evidence that are not available

Do not update research memory or mark the experiment complete when blockers remain.
