# Source Priority

Use this order for experiment-topic extraction.

## Canonical Narrative Source

1. `.numereng/experiments/<id>/EXPERIMENT.md`

The finalized experiment report is the source of truth for extraction. If it is missing or still reads like a progress log, stop and ask for experiment finalization first.

## Supporting Structured Sources

Use these to verify identity, run membership, and artifact completeness:

1. `.numereng/experiments/<id>/experiment.json`
2. `.numereng/runs/<run_id>/run.json`
3. `.numereng/runs/<run_id>/metrics.json`
4. `.numereng/runs/<run_id>/resolved.json`
5. `.numereng/runs/<run_id>/results.json`
6. `.numereng/runs/<run_id>/score_provenance.json`

## Supporting Narrative Sources

Use these only after `EXPERIMENT.md` is understood:

- `.numereng/experiments/<id>/EXPERIMENT.pack.md`
- `uv run numereng experiment details --id <id> --format json`
- `uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format json`

## Design-Space Source

Use this to map takeaways to branch topics:

- `.numereng/notes/NUMERAI_KEY_DYNAMICS/NUMERAI_MASTER_MODEL_DESIGN_SPACE.md`

## Evidence-Quality Gate

Before writing a topic takeaway, check:

- are artifacts complete enough for the claim?
- is the claim a verified artifact, computed metric, supported inference, or hypothesis / next-step?
- is the surface trustworthy for the claim?
- is the topic actually tested, or only observed?
- are there confounds or missing contextual caveats?
- what is not established by this experiment?
- should the topic be marked `not_tested` or `confounded` instead of `varied`?
