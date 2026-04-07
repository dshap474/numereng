# Source Priority

Use this order when building a research-memory update.

## Canonical structured sources

1. `experiment.json`
2. `run.json`
3. `metrics.json`
4. `resolved.json`
5. `results.json`
6. `score_provenance.json`

These are the default source of truth for:

- experiment identity
- run membership
- feature scope
- target
- model family
- surface / profile
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

## Evidence-Quality Gate

Executed experiment artifacts dominate only after a lightweight evidence-quality check:

- are artifacts complete?
- is the surface trustworthy for the claim?
- is the comparison actually comparable?
- are there obvious confounds or missing contextual caveats?
- does broader history support, weaken, or contradict the claim?

If the gate is weak, record the claim as provisional rather than promoting it as a strong default.
