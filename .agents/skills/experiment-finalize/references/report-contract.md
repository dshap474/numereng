# Comprehensive EXPERIMENT.md Contract

Use this reference before rewriting `EXPERIMENT.md` for a completed scored experiment.

The final report is the primary decision memo. It should stand on its own without `EXPERIMENT.pro.md` or the generated pack prompt.

## Required Decision Sections

- Executive conclusion: state whether the hypothesis is supported, partially supported, or rejected.
- Evidence status: confirm manifest rows, finished runs, scoring artifacts, and operational caveats; for ensemble-only experiments, confirm completed ensemble artifacts and source prediction availability.
- Matrix-level readout: summarize whether the recipe worked broadly or only in specific slices.
- Best row versus best candidate: distinguish top single run or ensemble from target family, seed family, ensemble candidate, champion, and no champion.
- Target-family analysis: identify strongest and weakest target families with exact metrics.
- Seed and horizon analysis: explain whether seed, target, or horizon drove the result.
- Metric-conflict analysis: call out BMC/CORR, BMC/MMC, recent-BMC/full-BMC, FNC, CWMM, drawdown, exposure, and coverage/comparability.
- Candidate assessment: name primary and secondary candidates, plus why each is or is not production-ready.
- Risks and weak claims: explicitly say what the evidence does not support.
- Next experiment: give the smallest useful follow-up plus concrete pass criteria.
- Pre-production questions: list checks required before champion promotion or live submission, including hidden selection pressure from prior target, recipe, metric, or feature choices.

## Evidence Labels

Use these labels mentally when writing claims:

- `verified artifact`: file, run, status, manifest, or count directly checked.
- `computed metric`: value read from or computed from run metrics.
- `supported inference`: interpretation supported by artifact evidence.
- `hypothesis / next-step`: plausible but untested claim requiring another scored experiment.

Do not word a supported inference as if it were a verified artifact. Do not word a next-step hypothesis as if it were already tested.

## Metric Severity

Use severity language instead of binary conflict language:

- BMC: `strong`, `moderate`, `weak`, or `mixed`.
- MMC: `strong`, `positive but marginal`, `mixed`, `weak`, or `missing`.
- FNC: `positive`, `mixed`, `negative`, or `missing`.
- Drawdown: `clean`, `target-dependent`, or `warning`.
- Exposure: `measured`, `missing`, or `promotion gate`.

Avoid unqualified `no major conflict` when a supporting metric is small, seed-sensitive, mixed, or coverage-limited. If `mmc_coverage_ratio_rows` is used to support an MMC claim, define it or explicitly caveat coverage.

## Candidate Wording

- `best single run`: top row by the selected metric.
- `candidate family`: target/config family supported by replicated evidence.
- `stabilizer candidate`: lower-risk sleeve that might improve an ensemble; do not call it a stabilizer until a scored ensemble proves it.
- `ensemble candidate`: candidate for future ensemble construction; do not call it an ensemble result until an ensemble artifact is scored.
- `champion`: production-ready candidate with all required handoff checks.
- `no champion`: required when evidence is validation-only, single-row selected, missing ensemble/correlation checks, or missing production-readiness checks.
- `ensemble-only closeout`: valid when the experiment evaluates built ensemble artifacts rather than manifest-listed training runs; still treat validation-only ensemble winners as candidates, not champions.

## Confound Tracking

Call out changed axes that affect interpretation:

- feature set
- target family or horizon
- model family
- model recipe / capacity
- ensemble recipe / component weights
- hyperparameters
- target preselection
- scoring stage availability
- post-selection comparison against prior experiments

Do not compress a multi-axis win into a single-axis conclusion.

## Writing Rules

- Do not promote a champion just because one row ranks first.
- Do not describe a recipe as broadly reliable when matrix-level BMC or MMC is weak.
- Do not let high CORR override weak BMC under a BMC-first objective.
- Treat validation scoring artifacts as evidence, not live tournament results.
- Keep `EXPERIMENT.md` concise enough to read, but comprehensive enough to be the durable decision source.
- Prefer tight Pro-style prose over long inventory when both carry the same evidence.
- Preserve special-case candidates precisely: a target can be useful for complementarity, regime testing, risk control, or ensemble diversity without being production-ready.
- Put the full per-run table in `EXPERIMENT.pack.md`, not in `EXPERIMENT.md`.

## Acceptance Gate

Before finalizing, confirm the report answers:

- What did we learn?
- Was the hypothesis supported, partially supported, or rejected?
- What is verified, computed, inferred, or still hypothetical?
- What is the best single run?
- What is the best-supported candidate type?
- Why is there a champion or no champion?
- Which metric conflicts matter, and how severe are they?
- Which confounds limit causal interpretation?
- What should happen next, and what would count as success?
