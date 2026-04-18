# Agentic Research

Use `numereng research` when you want numereng to run an autonomous experiment loop that mutates one parent config, trains one child config at a time, and persists the decision trail inside one root experiment.

## Use This When

- you already have an experiment with at least one viable parent config
- you want numereng to keep searching instead of hand-editing each child config
- you want round-by-round artifacts, lineage, and program state persisted under the experiment

## Prerequisites

- a root experiment under `.numereng/experiments/<experiment_id>/`
- at least one config to mutate under that experiment
- a selected research program from `src/numereng/features/agentic_research/programs/`
- working local training for the underlying config family

## Inspect The Program Catalog

```bash
uv run numereng research program list
uv run numereng research program show --program numerai-experiment-loop
```

The checked-in default program is `numerai-experiment-loop`.

## Initialize A Research Session

```bash
uv run numereng research init \
  --experiment-id 2026-04-18_research-root \
  --program numerai-experiment-loop
```

This persists the selected program snapshot into the experiment so later runs resume from the exact initialized program, not the live source file.

## Run The Loop

```bash
uv run numereng research run \
  --experiment-id 2026-04-18_research-root \
  --max-rounds 3 \
  --max-paths 2
```

Useful inspection commands:

```bash
uv run numereng research status --experiment-id 2026-04-18_research-root
uv run numereng experiment report --id 2026-04-18_research-root
```

## What Numereng Persists

Under `.numereng/experiments/<experiment_id>/agentic_research/`:

- `program.json`
- `session_program.md`
- `lineage.json`
- `rounds/rN/round.json`
- `rounds/rN/round.md`
- `rounds/rN/llm_trace.jsonl`
- `rounds/rN/llm_trace.md`

These files are the durable record of what program was selected, which parent config was chosen, what change set was proposed, and how each round scored.

## High-Risk Gotchas

- `research init` requires `--program`; there is no implicit default at the command boundary.
- Resume uses the persisted program snapshot, not the current markdown in the source tree.
- The default numerai program is config-centric: each autonomous round proposes one small mutation to one parent config and trains exactly one child config.
- Planner backend selection is controlled by `ACTIVE_MODEL_SOURCE=codex-exec|openrouter`.
- `research run` still relies on the normal training/scoring stack, so broken configs or missing datasets fail the same way they would in manual workflows.

## Read Next

- [Experiments](experiments.md)
- [Hyperparameter Optimization](optimization.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
