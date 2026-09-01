# Agent Instructions: Generate Experiment-Specific Prompt Context

This file is not injected directly into `EXPERIMENT.pack.md`. Use it as instructions for writing the generated experiment-specific context section that will be passed to `render_experiment_pack.py --experiment-context-path`.

Write the generated context for ChatGPT Pro before rendering the pack. Save it to:

```bash
.numereng/tmp/experiment-finalize/<experiment_id>.context.md
```

## Read First

Use the current experiment evidence, not memory or dashboard appearance alone:

- `.numereng/experiments/<id>/EXPERIMENT.md`
- `.numereng/experiments/<id>/experiment.json`
- `.numereng/experiments/<id>/run_plan.csv`
- `.numereng/runs/<run_id>/run.json`
- `.numereng/runs/<run_id>/metrics.json`
- `.numereng/runs/<run_id>/score_provenance.json`
- `.numereng/runs/<run_id>/artifacts/scoring/post_training_core_summary.parquet`
- `.numereng/runs/<run_id>/artifacts/scoring/post_training_full_summary.parquet` when present

## Write This Generated Section

Start with this heading:

```markdown
# Experiment-Specific Context For ChatGPT Pro
```

Then write concise but high-signal context that helps an external model analyze the pack correctly:

- what question the experiment was trying to answer
- why this experiment was run now
- the hypothesis and what would count as support or rejection
- the experiment design: model family, feature set, targets, seeds, dataset variant, scoring stages
- the primary and tie-break metrics, and why those metrics matter here
- how to interpret target families, horizons, and seed replication for this specific experiment
- known execution caveats, such as reruns, remote pullback, rescoring, missing metrics, or no promoted champion
- the strongest apparent signals and the weakest apparent signals from `EXPERIMENT.md`
- what claims ChatGPT Pro should verify against the run table
- what kinds of recommendations would be premature

Do not copy the full run table. Do not invent metrics. Do not make new conclusions beyond what `EXPERIMENT.md` and the scoring artifacts support. The goal is orientation for a stronger second-pass analysis, not a second final report.
