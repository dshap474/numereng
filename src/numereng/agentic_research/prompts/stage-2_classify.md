<!--
stage-2_classify.md — selective research-memory routing after FINALIZE.
USAGE: the closeout runner substitutes {{CONTEXT_JSON}}, validates the structured response, and
persists classification.json before any research-memory branch or master-ledger write.
-->

# Closeout Phase 2 — CLASSIFY

Classify whether this finalized experiment should enter persistent research memory. Judge only the
evidence-bounded memo and deterministic evidence supplied below. Do not rewrite the analysis.

## Dispositions

- `master`: the experiment changes a champion, candidate, or frontier; confirms or contradicts a key
  result; closes a plausible route with evidence; adds meaningful ensemble, package, or live-gate
  evidence; exposes a material confound or evaluation failure; or yields a reusable design rule.
- `branch_only`: informative but incomplete evidence, weak supporting evidence, a reproduction with
  no new decision, or operational evidence worth retaining without changing master understanding.
- `exclude`: smoke/harness/auth/SSH/local-fix work, an empty failed search, a duplicate or seed rerun
  already rolled into its parent, infra-only work, or work with no hypothesis/comparison/verdict.

## Comparison class (weigh before deciding)

Identify which comparison class this experiment's evidence belongs to, and name that class inside
`rationale`:

- `broad screening surface`: many targets, feature sets, or model variants; moves priors and search
  direction.
- `narrow candidate-quality packet`: a focused test of one family or a small candidate set; raises or
  lowers candidate confidence without supporting broad replacement claims.
- `champion / production evidence`: validated handoff evidence with sufficient scoring, ensemble,
  exposure, and operating-gate coverage.

Weigh the class when choosing between `master` and `branch_only`. A narrow packet rarely justifies
`master` unless it changes a champion, a candidate, or the frontier; broad screening surfaces and
production evidence usually do. Do not let a narrow packet enter master to displace a broader prior.

`branch_only` and `exclude` require an empty `relevant_topics` list. `master` may use an empty list
only for a material frontier-only production, package, live, or evaluation result; this is not a
default. Topics must be a unique subset of: `ensembling`, `features`, `hyperparameters`, `models`,
`neutralization-exposure`, `targets`. Select a topic only when materially evidenced; never select a
not-tested or no-op topic. Avoid an all-master default. Keep the rationale concise and evidence-bounded.

## Required output

Return exactly one JSON object and nothing else:

```json
{
  "disposition": "master",
  "relevant_topics": ["hyperparameters", "models"],
  "rationale": "The confirmed capacity result changes the reusable within-lane search guidance."
}
```

## Context

```json
{{CONTEXT_JSON}}
```
