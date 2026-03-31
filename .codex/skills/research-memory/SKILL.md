---
name: research-memory
description: "Analyze one completed numereng experiment, update rolling research memory under .numereng/notes/__RESEARCH_MEMORY__, contextualize it against relevant prior evidence, and recommend what to do next."
user-invocable: true
argument-hint: "<experiment-id>"
---

# Research Memory

Use this skill when the user wants to analyze a completed experiment, store its learnings in the rolling research-memory system, and decide what experiment should come next.

This skill updates the canonical notes root:

- `.numereng/notes/__RESEARCH_MEMORY__/CURRENT.md`
- `.numereng/notes/__RESEARCH_MEMORY__/experiments/<experiment-id>.md`
- `.numereng/notes/__RESEARCH_MEMORY__/topics/*.md`
- optional `.numereng/notes/__RESEARCH_MEMORY__/decisions/*.md`

Do not use this skill for:

- creating or training experiments
- store repair or cleanup
- submission operations
- broad Numerai literature review without a concrete completed experiment

Use:

- `experiment-design` for round planning before results exist
- `numereng-experiment-ops` for experiment contract/layout questions
- `store-ops` for drift, reset, reindex, or destructive cleanup

## Hard Gates

Before writing anything, confirm all of the following:

- the experiment exists
- at least one completed / scored run exists
- there are no active experiment run jobs
- required artifacts exist for analyzed runs:
  - `run.json`
  - `resolved.json`
  - `metrics.json`
  - `results.json`
  - `score_provenance.json`
- the experiment is not obviously draft-only or zero-result

If any gate fails:

- stop
- report the exact blocker(s)
- do not update research memory

## Canonical Metric Basis

Use the current numereng metric contract:

- primary: `bmc_last_200_eras.mean`
- tie-break: `bmc.mean`
- supporting: `corr.mean`, `mmc.mean`, `cwmm.mean`

Do not use payout-estimate framing as the canonical basis in new research-memory notes.

## Source Priority

Read in this order:

1. `experiment.json`
2. run artifacts:
   - `run.json`
   - `metrics.json`
   - `resolved.json`
   - `results.json`
   - `score_provenance.json`
3. `EXPERIMENT.pack.md` if present
4. `EXPERIMENT.md`
5. `uv run numereng experiment details --id <id> --format json`
6. `uv run numereng experiment report --id <id> --metric bmc_last_200_eras.mean --format json`

Secondary priors are allowed only after executed experiment evidence is understood:

- `.numereng/notes/NUMERAI_KEY_DYNAMICS/*`
- `.numereng/notes/NUMERAI_RESEARCH_STRATEGY/*`
- relevant files under `.numereng/notes/research/research-briefs/*`
- `.numereng/notes/__RESEARCH_MEMORY__/legacy-progression/*`

## Evidence Classes

Every analyzed experiment must be assigned exactly one class:

- `frontier`
  - direct frontier-shaping evidence on a shared strong evaluation surface
- `scout`
  - directional evidence useful within-family or within-surface, but not direct frontier proof
- `supporting`
  - incomplete, mixed, legacy, or otherwise non-comparable evidence

Default heuristics:

- full `v5.2` + strong shared evaluation route such as `purged_walk_forward` -> likely `frontier`
- smoke / `simple` / staged mixed tiers -> likely `scout`
- mixed or degraded evidence that should not move defaults -> `supporting`

## Comparison Policy

Use two comparison passes:

1. Primary relevant subset
- strict on comparability
- prefer same or very similar:
  - target family
  - feature scope
  - model family
  - evaluation surface / profile

2. Secondary broader sweep
- all experiments
- use only for:
  - contradiction checks
  - repeated dead-end detection
  - adjacent supporting evidence

Infer relevance heuristically from:

- experiment tags
- hypothesis text
- run metadata such as:
  - `model.type`
  - `data.feature_set`
  - `data.target_col`
  - dataset version
  - engine / profile

Use strict-primary / loose-secondary surface discipline:

- comparable strong surfaces can move the frontier directly
- smoke / `simple` / staged results can guide direction but must be called out as non-frontier proof

## Update Rules

One successful run of this skill must:

- create the experiment review if missing
- update touched topic ledgers
- refresh `CURRENT.md`
- create a decision note only if the frontier materially changes

V1 topic ledgers:

- `topics/baselines.md`
- `topics/targets.md`
- `topics/feature-scope.md`
- `topics/model-families.md`
- `topics/evaluation-surfaces.md`
- `topics/postprocessing.md`

Experiment review lifecycle:

- first ingest creates the canonical review
- if material evidence later changes:
  - append a dated addendum
  - do not rewrite the original review body
- if nothing material changed:
  - no-op

## Required Output Shapes

Read the templates in `assets/` before writing:

- `assets/CURRENT.template.md`
- `assets/experiment-review.template.md`
- `assets/topic-ledger.template.md`
- `assets/decision-note.template.md`
- `assets/AGENTS.template.md`

When the task needs details on sourcing, comparison, or writing policy, load:

- `references/source-priority.md`
- `references/comparison-policy.md`
- `references/write-contract.md`

## Writing Rules

- Keep notes decision-oriented.
- Prefer links over repeated prose.
- Use note-relative links inside `__RESEARCH_MEMORY__`.
- Use explicit app routes for experiment navigation:
  - `/experiments/<id>`
  - `/experiments/<id>/runs/<run_id>`
- Be explicit about uncertainty and confidence.
- Optimize for “what should we do next?” not exhaustive replay.

## Minimal Workflow

1. Validate gates.
2. Read canonical structured artifacts.
3. Read experiment narrative artifacts.
4. Assign an evidence class.
5. Build the relevant comparison subset.
6. Run the broader contradiction/support check.
7. Write or no-op the experiment review.
8. Update touched topic ledgers.
9. Refresh `CURRENT.md`.
10. If defaults, blocked paths, promoted directions, or the top-ranked next move materially changed, write a decision note.
