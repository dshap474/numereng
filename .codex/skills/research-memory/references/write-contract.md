# Write Contract

The goal is decision usefulness with explicit precedence.

## CURRENT.md

`CURRENT.md` is the sole canonical present-tense decision layer.

Must include:

- summary snapshot
- current promoted defaults
- active frontier
- what we are not doing
- top 5 ranked next experiments
- recent material shifts
- key evidence anchors

Each promoted default must include:

- claim
- scope conditions
- confidence
- freshness
- key evidence links

Each blocked path must include:

- blocked path
- why blocked
- reopen conditions

Each next-experiment card must include:

- title
- hypothesis
- why now
- expected information gain
- expected upside
- success criteria
- minimal run plan
- base-rate check
- evidence links

## Experiment Reviews

Must include:

- original result summary
- current interpretation
- interpretation context
- evidence risk
- experiment question and surface
- aggregate results
- representative runs when useful
- claim quality / scope notes
- what this experiment establishes
- what remains ambiguous
- comparison to relevant prior evidence
- global contradiction / support check
- impact on frontier
- addenda

Reviews preserve provenance, but `Current Interpretation` may be updated over time.

Review rules:

- use aggregate results as the primary unit whenever the experiment’s inferential unit is a target, model, scope, or similar summary rather than a single run
- use top runs only as representative examples, not the main story, unless the experiment is inherently run-level
- keep `Current Interpretation` short and verdict-like
- require `Interpretation context` during chronological rebuilds and allow `current-state ingest` for normal ongoing use
- require `Evidence risk` as a lightweight summary of confounds or limitations

## Topic Ledgers

Use a scoped-claim structure:

- current state
- scoped promoted beliefs
- conditional anti-patterns
- open questions
- evidence log

Each nontrivial belief must include:

- claim
- scope conditions
- confidence
- freshness
- evidence links

Each anti-pattern must include reopen conditions.

## Decision Notes

Create only when a materiality trigger fires.

Must include:

- decision
- prior state
- new state
- why it changed
- evidence
- consequences

## Materiality

Treat a change as material if it changes any of:

- a promoted default in `CURRENT.md`
- the top-ranked next experiment
- a blocked path or its reopen conditions
- the confidence or freshness of a major current belief
- the interpretation of a previously cited key evidence anchor

## Links

- use note-relative links for notes under `__RESEARCH_MEMORY__`
- use explicit app routes for experiment navigation
- avoid raw filesystem paths unless no app route exists
