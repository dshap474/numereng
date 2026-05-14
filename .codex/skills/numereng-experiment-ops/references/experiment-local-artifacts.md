# Experiment-Local Artifacts

Use this reference when an experiment contains derived evidence or deployment handoff artifacts beyond the root manifest, configs, run plan, and launch scripts.

## Core Boundaries

- `experiment.json` is the active experiment manifest.
- `EXPERIMENT.md` is the canonical narrative and decision record.
- `.numereng/runs/<run_id>/` is the canonical storage for run-owned artifacts.
- `EXPERIMENT.pack.md` is generated output, not source narrative.
- Experiment-local artifact areas support the narrative; they do not replace it.

## `analysis/`

Use `experiments/<experiment_id>/analysis/` for durable derived decision evidence created inside the experiment.

Good fits:

- candidate freezes and candidate rankings
- target, seed, feature-scope, or model-family comparison tables
- correlation matrices and pruning recommendations
- blend or ensemble sweeps
- dashboard/package comparison payloads used as evidence
- slimmed evidence notes explaining removed or regenerable heavy artifacts

Guardrails:

- keep `analysis/` reproducible from recorded inputs when practical
- if heavy derived files are removed locally, leave a short note explaining what was removed, why, and where the source or regeneration path lives
- do not store canonical run metadata here when it belongs under `.numereng/runs/<run_id>/`
- summarize the decision impact in `EXPERIMENT.md`; do not make readers inspect `analysis/` to know the outcome

## `deployment/<deployment_id>/`

Use `experiments/<experiment_id>/deployment/<deployment_id>/` when an experiment winner becomes a package, live submission, hosted pickle, or handoff artifact.

Good fits:

- package manifests and package metadata
- live prediction or submission parquet artifacts
- pickle or live-build artifacts
- handoff manifests and handoff notes
- dashboard diagnostics and package-validation comparisons
- deployment-specific helper scripts
- deployment-specific configs retained for provenance

Guardrails:

- use one stable `<deployment_id>` for each deployment lineage, for example `lgbm_cross_scope_v1`
- keep deployment records inside the source experiment when they are part of the same line of inquiry
- create a separate experiment only when the deployment work is itself a new research or training question
- preserve upload ids, model ids, labels, and package ids in machine-readable metadata when available
- distinguish historical upload metadata from the current live assignment

## What Not To Do

- Do not create a second experiment solely to hold handoff files when the handoff belongs to the source experiment winner.
- Do not treat `analysis/` or `deployment/` as required folders for every experiment.
- Do not define global subfolder schemas under these areas; use names that match the local workflow and document them in `EXPERIMENT.md`.
- Do not let supporting files drift ahead of `EXPERIMENT.md`; the narrative should state what each artifact area proves.
