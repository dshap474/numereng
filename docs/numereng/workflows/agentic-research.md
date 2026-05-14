# Agentic Research

Use `numereng research` when you want numereng to run a simple autonomous config loop:

1. run a deterministic ML round
2. send the resulting configs, metrics, latest rolling memo, and recent decision log to the LLM
3. let the LLM choose one small config mutation and update the rolling memo
4. validate, train, score, and record the next round

The LLM does not edit Python code. It returns a structured decision form plus cumulative markdown
research state. Python converts the form into the strict machine decision, mutates an experiment
config, and appends exact execution results.

## Use This When

- you already have an experiment with at least one viable config under `configs/`
- you want numereng to search configs instead of hand-editing every child config
- you want a round-by-round decision trail under the experiment

## Research Prompt

The prompt policy lives at:

```text
src/numereng/features/agentic_research/PROGRAM.md
```

It defines the objective, allowed config paths, and required `decision_form + round_markdown`
response. Treat it as the base/default program contract. Serious experiments should usually use a
focused custom program that declares one hypothesis, fixed surface, allowed variation axis, and
stop/confirmation rule.

Custom prompt programs belong in:

```text
src/numereng/features/agentic_research/custom_programs/
```

That directory is local-only and gitignored except for its small `README.md`. Experiments opt into one by setting `metadata.agentic_research_program` to the custom Markdown filename.
The tracked `src/numereng/features/agentic_research/AGENTS.md` explains the local program template and usage contract.

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
- `trace.jsonl` as the append-only prompt/response/event trace for debugging
- `rounds/decision.json` as an append-only JSON-lines decision/result log
- `rounds/rNNN.md` as the cumulative human-readable research state after each round
- `rounds/rNNN.debug.*` only on LLM/Codex failure

`trace.jsonl` is for debugging and improving the loop; it is not fed back into future prompts by default. `rounds/decision.json` is the compact machine memory. The latest `rounds/rNNN.md` is the rolling human-readable memory loaded into the next LLM prompt.

## High-Risk Gotchas

- The mutable research surface is config JSON, not Python source.
- Python rejects changes outside the allowed paths in `PROGRAM.md` / runner context.
- Codex CLI runs with a JSON output schema so successful responses contain `decision_form` and `round_markdown`.
- Candidate configs must validate as `TrainingConfig`.
- Duplicate configs are rejected before training.
- Planner backend selection is controlled by `ACTIVE_MODEL_SOURCE=codex-exec|openrouter`.
- The default prompt is tracked as `PROGRAM.md`; an experiment can set `metadata.agentic_research_program` to a local file under `custom_programs/`.
- Custom programs are ignored local files; use `src/numereng/features/agentic_research/AGENTS.md` as the tracked template.
- `research run` still relies on the normal training/scoring stack, so broken configs or missing datasets fail the same way they would in manual workflows.

## Read Next

- [Experiments](experiments.md)
- [Hyperparameter Optimization](optimization.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
