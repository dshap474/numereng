"""Tracked marker for the round prompt and the generic experiment brief."""

# Programs

Three files here are tracked:

- `PROGRAM.md` — the round prompt: the harness contract and the generic research doctrine, carrying
  the `{{STRATEGY}}` placeholder for the experiment brief and `{{CONTEXT_JSON}}` for the bounded
  context. Editing it reaches every run at its next round.
- `STRATEGY.md` — the generic brief, used when an experiment ships none of its own.
- `README.md` — this file.

Everything else in this directory is local-only: `archive/` holds the self-contained programs from
the single-file era as history, and nothing here is loaded at run time.

## Authoring A Brief

An experiment's brief lives at `.numereng/experiments/<id>/agentic_research/STRATEGY.md`. The
filename is fixed, so the runner finds it and remote experiment sync carries it. Write it when the
experiment is created; `prompts/INIT-PROGRAM.md` does exactly that as its last stage.

Follow the headings of `STRATEGY.md`: `## This Experiment` opening with the hypothesis, then
`### Lane`, `### Prior Evidence`, `### Sweep Plan`, and `### Confirmation And Handoff`. Keep it to
what differs per experiment. The harness injects no research memory, so the prior evidence the model
needs — closed lanes, retired claims, inert axes, anchors, the calibration stance — has to be
written out here as standalone prose, along with any substrate facts for the model family.

A brief carries neither placeholder.
