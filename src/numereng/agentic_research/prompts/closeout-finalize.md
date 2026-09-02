# Closeout: Finalize

You are writing the decision memo for one completed agentic research experiment. The evidence
bundle below is complete and authoritative. Use only its numbers. Where a metric is marked
unavailable, name it as a gap rather than a value.

```json
{{CONTEXT_JSON}}
```

## Doctrine

Within-lane BMC200 ranks candidates; it is not a deploy signal. Scout-tier outputs are candidates,
not results, until a full-data successor confirms them. Label every claim as verified artifact,
computed metric, supported inference, or hypothesis.

Use the weakest candidate label the evidence supports: `best single run`; `candidate family` when
seed or family evidence supports a recipe rather than one row; `stabilizer candidate` until a scored
blend proves stabilization; `ensemble candidate` until an ensemble is built and scored; `champion`
only with production-ready evidence and the handoff checks in place; or `no champion`, the default
whenever evidence is validation-only, single-row, or missing ensemble, correlation, or production
checks. Keep special-case candidates separate from production-ready ones.

Rate metric conflicts by severity, never as a binary. BMC is strong, moderate, weak, or mixed. MMC is
strong, positive but marginal, mixed, weak, or missing. FNC is positive, mixed, negative, or missing.
Drawdown is clean, target-dependent, or warning. Exposure is measured, missing, or a promotion gate.
Call out limited coverage, missing full summaries, target preselection, feature-set differences,
recipe changes, and post-selection effects. Never write an unqualified "no conflict" when a
supporting metric is small, mixed, seed-sensitive, or coverage-limited.

## The Memo

Return the memo directly as GitHub-flavored markdown, with no preamble and no surrounding code
fence. Name the experiment id and the `believed_best` config filename. Use these level-2 sections in
order, each substantive:

1. `## Verdict`: hypothesis supported, partially supported, or rejected, with the one-line reason.
2. `## Evidence And Gaps`: what the record proves, what is missing, and the dataset tier and scope.
3. `## Candidates`: `believed_best` against the mechanical champion and any parsimony ties, with
   trio statistics and the observed seed-noise floor, using the labels above, and an explicit
   champion or no-champion decision.
4. `## Metric Conflicts`: where BMC200, FNC, and benchmark correlation disagree, with severity.
5. `## Search Audit`: from the deterministic counts, coverage, parentage, abandoned sweeps,
   duplicate skips, and failures, whether the search was disciplined.
6. `## Design-Space Roles`: a table of the axes varied, controlled, and left untested, and what
   each tells us.
7. `## Implications`: what the next experiment should test or avoid, including hidden selection
   pressure from target, recipe, metric, or prior-experiment choices.
8. `## Memory Notes`: concrete recommendations for the six research-memory ledgers. This memo
   writes no memory itself.
