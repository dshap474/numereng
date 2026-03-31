# Source Priority

Use this order when building a research-memory update for one experiment.

## Canonical structured sources

1. `experiment.json`
2. `run.json`
3. `metrics.json`
4. `resolved.json`
5. `results.json`
6. `score_provenance.json`

These are the source of truth for:

- experiment identity
- run membership
- feature scope
- target
- model family
- surface/profile
- metrics

## Narrative sources

Use these only after structured artifacts are understood:

1. `EXPERIMENT.pack.md`
2. `EXPERIMENT.md`
3. `uv run numereng experiment details --id <id> --format json`
4. `uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format json`

`EXPERIMENT.pack.md` is preferred over `EXPERIMENT.md` when the latter is still a progress log.

## Secondary priors

Use broader notes only as supporting context:

- `.numereng/notes/NUMERAI_KEY_DYNAMICS/*`
- `.numereng/notes/NUMERAI_RESEARCH_STRATEGY/*`
- relevant `.numereng/notes/research/research-briefs/*`
- `.numereng/notes/__RESEARCH_MEMORY__/legacy-progression/*`

Executed experiment evidence always dominates broader priors.
