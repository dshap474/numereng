"""Agent guidance for the agentic research feature — code-editing rules only."""

# Agentic Research — Agent Guide

Everything about *using* this feature (running the loop, closeout, authoring programs, interpreting
results, the deploy gate) lives in [README.md](README.md). This file is only what an agent must
know before changing code in this folder.

## Layout

- `engine/` — all harness code: `types.py`, `memory.py`, `aggregate.py`, `boundary.py`, `llm.py`,
  `context.py`, `loop.py`, `program.py` (CORE drift check / re-splice), plus the `engine/closeout/`
  post-run distillation chain.
- `prompts/` — the closeout phase prompts (`stage-1_finalize.md`, `stage-2_classify.md`,
  `stage-3_extract.md`, `stage-4_synthesize.md`) plus the pre-run `INIT-PROGRAM.md` playbook that
  designs and stages the next experiment from synthesized research memory.
- `programs/` — `PROGRAM.md` (the tracked canonical program) plus local-only custom programs;
  finished-experiment programs go in `programs/archive/`. Only `PROGRAM.md` and `README.md` are
  tracked here.
- Runtime state: `.numereng/experiments/<id>/agentic_research/` (`state.json`, `journal.jsonl`,
  `rounds/`, `closeout/`).

## The One Boundary

The LLM proposes one `decision_form` per round; Python validates, executes, and records — it never
edits a proposal. The harness holds **no research strategy**: what to try, when to confirm, when to
diversify all live in program files. If a change adds strategy to Python, it is in the wrong place.

## Invariants To Preserve

- **Bounded context.** No term assembled by `engine/context.py` may grow with round count. This
  killed a 500-round run once (prompts grew to ~890 KB and the API stream-disconnected).
- **No auto-stop.** The only LLM action is `"run"`. Runs end on CLI budget, human stop, or the
  5-consecutive-failure bail (resumable).
- **Reject whole, never clamp.** Boundary violations (path allowlist, value caps, horizon/target
  match, strict `TrainingConfig`) fail the round with a stable error token; see `engine/boundary.py`.
- **`data.dataset_variant` and `training.engine.*` are not LLM-mutable** — deliberately absent
  from `ALLOWED_CHANGE_PATHS` so one experiment can never mix downsampled and full-data metrics or
  move its own evaluator (profile, window, embargo). Manifests may only narrow the allowlist.
- **Custom programs are self-contained** and their CORE sections must match `programs/PROGRAM.md`
  byte-verbatim: enforced at session start and by
  `tests/unit/numereng/test_agentic_research_program_core.py` (`programs/archive/` is exempt).
- **A CORE edit in `PROGRAM.md` is a change to every custom program, live ones included.** The
  drift check re-runs on every re-entry, so a live run fails at its next resume/bail re-invoke until
  its program is re-spliced. Land the edit together with
  `numereng research program resplice --experiment-id <id>` for every active experiment on every
  host that runs it; never edit CORE while a run is live without that. Never re-implement the
  splice by hand — `engine/program.py` is the one path.
- **The seed path is data, not a name.** `agentic_research_seed_path` (manifest) drives seed
  injection, journal seed extraction (`loop._config_seed`), and recipe grouping
  (`aggregate.recipe_key`); do not hard-code `random_state`/`seed` anywhere new.
- **The payout target is owned by the scoring layer.** Context mirrors
  `features.scoring.metrics.DEFAULT_PAYOUT_TARGET_COL`; do not reintroduce a shadow constant here.
- **Closeout never launches anything.** The chain ends once research memory is synthesized;
  next-experiment design lives in `prompts/INIT-PROGRAM.md` and stays behind a human launch gate.
  Do not add auto-launch.
- **BMC200 is the search objective, not a deploy signal.** Never wire deploy automation to the
  within-lane champion.

## Verify

```bash
uv run pytest tests/unit/numereng -k agentic_research -q
uv run pytest tests/unit/numereng/agentic_research -q
```
