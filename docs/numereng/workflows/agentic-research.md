# Agentic Research

Use `numereng research` when you want numereng to run a simple autonomous config loop:

1. run a deterministic ML round
2. send the resulting configs, metrics, notes, and recent research ledger to the LLM
3. let the LLM choose one small config mutation
4. validate, train, score, and record the next round

The LLM does not edit Python code. It only returns a structured decision that mutates an experiment config.

## Use This When

- you already have an experiment with at least one viable config under `configs/`
- you want numereng to search configs instead of hand-editing every child config
- you want a round-by-round decision trail under the experiment

## Research Prompt

The prompt policy lives at:

```text
src/numereng/features/agentic_research/PROGRAM.md
```

It defines the objective, allowed config paths, and required JSON decision format. The deterministic Python runner renders that prompt with live experiment context.

## Run The Loop

```bash
uv run numereng research run \
  --experiment-id 2026-04-18_research-root \
  --max-rounds 3
```

`research run` initializes its own state on first use. If the experiment has no scored primary-metric rows yet, the first round copies and trains the first existing config as a baseline before asking the LLM for mutations.

Useful inspection commands:

```bash
uv run numereng research status --experiment-id 2026-04-18_research-root
uv run numereng experiment report --id 2026-04-18_research-root
```

## What Numereng Persists

Under `.numereng/experiments/<experiment_id>/agentic_research/`:

- `state.json`
- `ledger.jsonl`
- `rounds/rNNN/decision.json`
- `rounds/rNNN/context.json` for LLM rounds
- `rounds/rNNN/notes.md`
- `rounds/rNNN/debug/*` only on LLM/Codex failure

JSON files are the machine contract and replay surface. Markdown files are the human-readable audit trail.

## High-Risk Gotchas

- The mutable research surface is config JSON, not Python source.
- Python rejects changes outside the allowed paths in `PROGRAM.md` / runner context.
- Candidate configs must validate as `TrainingConfig`.
- Duplicate configs are rejected before training.
- Planner backend selection is controlled by `ACTIVE_MODEL_SOURCE=codex-exec|openrouter`.
- `research run` still relies on the normal training/scoring stack, so broken configs or missing datasets fail the same way they would in manual workflows.

## Read Next

- [Experiments](experiments.md)
- [Hyperparameter Optimization](optimization.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
