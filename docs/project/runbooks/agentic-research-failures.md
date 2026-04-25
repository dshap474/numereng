# Agentic research failures

## Symptoms

- `numereng research run` stops after planner errors
- a round bundle is missing `round.json`, `decision.json`, or `learning.md`
- a prompt/decision exists but the child config or follow-on train step did not materialize

## First checks

1. Inspect `experiments/<id>/agentic_research/state.json`.
2. Inspect `experiments/<id>/agentic_research/ledger.jsonl`.
3. Inspect the most recent `rounds/rN/context.json`, `prompt.md`, `decision.json`, and `round.json`.

## Contract reminders

- Numerai rounds are config-centric and materialize at most one child config per autonomous iteration.
- The mutable surface is config JSON, not Python code.
- Round bundles are canonical; `ledger.jsonl` is the compact cross-round history.

## Common recovery paths

- If the planner response is invalid, fix the parser/prompt contract rather than hand-editing round artifacts.
- If the child config was not written, inspect validation feedback and duplicate-detection paths.
- If train/score steps failed after planner success, diagnose those flows with the training/store runbooks.
