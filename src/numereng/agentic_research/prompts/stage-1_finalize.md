<!--
stage-1_finalize.md — closeout phase 1 prompt (tracked, generic; no machine-specific paths).
The runner substitutes {{CONTEXT_JSON}} with the bounded FINALIZE context and calls the LLM with a
strict output schema. Output is ONE JSON object matching the schema below — the agentic decision
memo, written to the closeout-owned file EXPERIMENT.closeout.md. This is NOT the manual finalizer:
no EXPERIMENT.pack.md, no full-summary parity gating. Cover only what the agentic record plus the
optional metrics enrichment supports; name metrics the record lacks as gaps.
-->

# Closeout Phase 1 — FINALIZE

You are finalizing one completed agentic research experiment. You are given a bounded, deterministic
evidence bundle. Produce a single decision memo that a human can act on.

## Inputs

Everything you may use is in the JSON context below. Use ONLY these numbers — never invent metrics.
`evidence_summary` is authoritative and complete for this experiment; where a metric is marked
`"unavailable: run not pulled"` or `"unavailable: metric absent"`, treat it as a gap, not a value.

```json
{{CONTEXT_JSON}}
```

## Standing doctrine (state explicitly where relevant)

- Within-lane BMC200 is a **candidate-ranker**, not a deploy signal.
- Scout-tier outputs are **candidates, not results**; a scout's findings are confirmed only on full
  data by a successor experiment.
- Every claim carries an **evidence label**: verified artifact, computed metric, supported
  inference, or hypothesis / next-step.

## Candidate wording ladder

Candidate wording must stay evidence-level accurate. Use the weakest label the evidence supports:

- `best single run` — the top row on the primary metric, nothing more.
- `candidate family` — seed or family evidence supports follow-up on a recipe rather than one row.
- `stabilizer candidate` — until a scored blend actually proves stabilization.
- `ensemble candidate` — until an ensemble artifact is built and scored.
- `champion` — reserved for production-ready evidence with the handoff checks in place.
- `no champion` — the default here. State `no champion` whenever the evidence is validation-only,
  single-row selected, or missing ensemble, correlation, or production-readiness checks.

Keep special-case candidates separate from production-ready ones.

## Metric conflict severity

Describe conflicts with severity, never with binary language:

- BMC: `strong`, `moderate`, `weak`, or `mixed`.
- MMC: `strong`, `positive but marginal`, `mixed`, `weak`, or `missing`.
- FNC: `positive`, `mixed`, `negative`, or `missing`.
- Drawdown: `clean`, `target-dependent`, or `warning`.
- Exposure: `measured`, `missing`, or `promotion gate`.
- Coverage/comparability: call out limited coverage, missing full summaries, target preselection,
  feature-set differences, model-recipe changes, and post-selection effects.

Never write an unqualified `no major conflict` when a supporting metric is small, mixed,
seed-sensitive, or coverage-limited. If a reading rests on `mmc_coverage_ratio_rows`, explain what
that ratio means or mark the MMC interpretation as coverage-limited.

## Required output

Return exactly one JSON object:

```json
{"files": [{"path": "EXPERIMENT.closeout.md", "content": "<the full memo, GitHub-flavored markdown>"}],
 "notes": "<one-line summary of the verdict>"}
```

The `content` memo MUST include these level-2 sections, in this order, each non-trivial:

1. `## Verdict` — hypothesis supported / partially supported / rejected, with the one-line reason.
2. `## Evidence Status And Caveats` — what the record proves vs. what is missing (name the
   "unavailable" metrics as gaps); dataset tier and scope.
3. `## Candidate Hierarchy` — `believed_best` vs the mechanical champion vs any parsimony ties, with
   trio statistics and the observed seed-noise floor; use the candidate wording ladder above (not
   "winner"), and state the champion / no-champion decision explicitly.
4. `## Metric Conflicts` — where BMC200, FNC, and benchmark-correlation disagree; rate severity with
   the vocabulary above; name unavailable metrics as gaps.
5. `## Sweep Discipline Audit` — read from the deterministic counts (coverage ranges, parentage,
   SWEEP ABANDONED rounds, duplicate skips, failure taxonomy): was the search disciplined?
6. `## Design-Space Roles` — a table of the axes this experiment varied / controlled / left
   untested and what each role tells us.
7. `## Implications For Future Work` — what the next experiment should test or avoid, including any
   hidden selection pressure from target, recipe, metric, or prior-experiment choices.
8. `## Master-Ledger Update` — concrete recommendations for the six research-memory ledgers
   (recommendations only; this phase writes no memory).

Reference the experiment id and the `believed_best` config filename explicitly in the memo. Aim for
a thorough memo (well over 3,000 characters). Output only the JSON object, nothing else.
