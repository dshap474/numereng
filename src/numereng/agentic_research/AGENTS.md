"""Agent guidance for the agentic research feature — code-editing rules only."""

# Agentic Research — Agent Guide

Everything about *using* this feature (running the loop, the prompt, closeout, interpreting
results, the deploy gate) lives in [README.md](README.md). This file is only what an agent must
know before changing code in this folder.

## Layout

- `engine/` — all harness code: `types.py`, `memory.py`, `aggregate.py`, `boundary.py`, `llm.py`,
  `context.py`, `loop.py`, plus `engine/closeout/` (`evidence.py`, `runner.py`, `types.py`), which
  turns a finished run into an evidence bundle and one decision memo.
- `prompts/` — `closeout-finalize.md`, the one closeout LLM call, plus the pre-run
  `INIT-PROGRAM.md` playbook that designs and stages the next experiment from research memory.
- `programs/` — `PROGRAM.md` (the tracked round prompt), `STRATEGY.md` (the generic experiment
  brief used when an experiment ships none), and `README.md`. Everything else here, including
  `archive/`, is local-only.
- Runtime state: `.numereng/experiments/<id>/agentic_research/` (`STRATEGY.md`, `state.json`,
  `journal.jsonl`, `rounds/`, `closeout/`).

## The One Boundary

The LLM proposes one `decision_form` per round; Python validates, executes, and records — it never
edits a proposal. The harness holds **no research strategy**: what to try, when to confirm, when to
diversify all live in `PROGRAM.md` and the experiment's brief. If a change adds strategy to Python,
it is in the wrong place.

## Invariants To Preserve

- **Bounded context.** No term assembled by `engine/context.py` may grow with round count. This
  killed a 500-round run once (prompts grew to ~890 KB and the API stream-disconnected).
- **No auto-stop.** The only LLM action is `"run"`. Runs end on CLI budget, human stop, or the
  5-consecutive-failure bail (resumable).
- **Reject whole, never clamp.** Boundary violations (path allowlist, value caps, horizon/target
  match, strict `TrainingConfig`) fail the round with a stable error token; see `engine/boundary.py`.
  The one in-round retry hands that token back as `last_error` and re-asks; it never repairs a
  proposal, and a second failure is recorded and counted exactly as a single failure was.
- **The prompt is `PROGRAM.md` plus the experiment's `STRATEGY.md`**, composed at run time by two
  placeholder substitutions in `engine/llm.py`. Editing `PROGRAM.md` reaches every run, live ones
  included, at its next round: there is no per-experiment copy and no re-splice step. Keep both
  placeholders present exactly once and keep briefs free of them.
- **`data.dataset_variant` and `training.engine.*` are not LLM-mutable** — deliberately absent
  from `ALLOWED_CHANGE_PATHS` so one experiment can never mix downsampled and full-data metrics or
  move its own evaluator (profile, window, embargo). Manifests may only narrow the allowlist.
- **The seed path is data, not a name.** `agentic_research_seed_path` (manifest) drives seed
  injection, journal seed extraction (`loop._config_seed`), and recipe grouping
  (`aggregate.recipe_key`); do not hard-code `random_state`/`seed` anywhere new.
- **The payout target is owned by the scoring layer.** Context mirrors
  `features.scoring.metrics.DEFAULT_PAYOUT_TARGET_COL`; do not reintroduce a shadow constant here.
- **Closeout never launches anything.** It ends at the evidence bundle and the memo; research-memory
  writes belong to the `research-memory-update` skill and next-experiment design to
  `prompts/INIT-PROGRAM.md`, both behind a human gate. Do not add auto-launch.
- **BMC200 is the search objective, not a deploy signal.** Never wire deploy automation to the
  within-lane champion.

## Verify

```bash
uv run pytest tests/unit/numereng -k agentic_research -q
uv run pytest tests/unit/numereng/agentic_research -q
```
