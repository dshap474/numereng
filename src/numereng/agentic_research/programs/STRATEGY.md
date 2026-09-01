## This Experiment

This is the generic brief. The harness uses it when an experiment ships no
`agentic_research/STRATEGY.md` of its own. A designed experiment replaces it with a brief under the
same headings.

**Hypothesis.** The seeded recipe is not at the top of its lane, and coarse moves on the allowed
paths will find a trio-confirmed recipe above it that holds or raises FNC.

### Lane

The seed config fixes the lane. Feature set, target, dataset variant, and training profile stay as
seeded and are not among your change paths. `allowed_change_paths` and `value_caps` in the context
are the whole mutable surface.

### Prior Evidence

None is encoded. The harness injects no research memory, so the seed config is the only prior.
Treat every knob family as untested.

### Sweep Plan

Spend the first rounds learning whether the seeded recipe moves under coarse changes at all: one
probe per knob family in `allowed_change_paths`, each large enough to clear the seed-noise floor and
each branched from the baseline. Refine only the families that moved. Leave enough budget to
seed-confirm the best FNC-clean candidate.

### Confirmation And Handoff

Confirm by trio mean and name the confirmed recipe in `believed_best`. Record inert axes and
cap-limited knobs in `EXPERIMENT.md` so the next program starts where this one stopped.
