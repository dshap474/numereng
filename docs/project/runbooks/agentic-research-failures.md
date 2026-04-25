# Agentic research failures

## Symptoms

- `numereng research run` stops after planner errors
- `rounds/decision.json` or a `rounds/rNNN.md` notes file is missing
- a prompt/decision exists but the child config or follow-on train step did not materialize

## First checks

1. Inspect `experiments/<id>/agentic_research/state.json`.
2. Inspect `experiments/<id>/agentic_research/trace.jsonl`.
3. Inspect `experiments/<id>/agentic_research/rounds/decision.json`.
4. Inspect the most recent `rounds/rNNN.md` notes file.
5. If the planner failed, inspect `rounds/rNNN.debug.*`.

## Contract reminders

- Numerai rounds are config-centric and materialize at most one child config per autonomous iteration.
- The mutable surface is config JSON, not Python code.
- `trace.jsonl` is the append-only prompt/response/event trace for debugging.
- `rounds/decision.json` is the append-only mechanical contract.
- `rounds/rNNN.md` is the readable round summary.

## Common recovery paths

- If the planner response is invalid, fix the parser/prompt contract rather than hand-editing round artifacts.
- If the child config was not written, inspect validation feedback and duplicate-detection paths.
- If train/score steps failed after planner success, diagnose those flows with the training/store runbooks.
