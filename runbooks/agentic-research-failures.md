# Agentic research failures

## Symptoms

- `numereng research run` stops after planner errors
- a round bundle is missing `round.json` or `round.md`
- planner traces exist but the child config or follow-on train step did not materialize

## First checks

1. Inspect `.numereng/experiments/<id>/agentic_research/program.json`.
2. Inspect `.numereng/experiments/<id>/agentic_research/lineage.json`.
3. Inspect `.numereng/experiments/<id>/agentic_research/llm_trace.jsonl` and `llm_trace.md`.
4. Inspect the most recent `rounds/rN/round.json` and `round.md`.

## Contract reminders

- Numerai rounds are config-centric and materialize at most one child config per autonomous iteration.
- Planner traces are append-only.
- Round bundles are canonical; older transport-specific artifacts are not the durable contract.

## Common recovery paths

- If the planner response is invalid, fix the parser/prompt contract rather than hand-editing round artifacts.
- If the child config was not written, inspect validation feedback and duplicate-detection paths.
- If train/score steps failed after planner success, diagnose those flows with the training/store runbooks.
