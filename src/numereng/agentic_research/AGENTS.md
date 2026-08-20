"""Agent guidance for the agentic research feature — code-editing rules only."""

# Agentic Research — Agent Guide

Everything about *using* this feature (running the loop, closeout, authoring programs, interpreting
results, the deploy gate) lives in [README.md](README.md). This file is only what an agent must
know before changing code in this folder.

## Layout

- `engine/` — all harness code: `types.py`, `memory.py`, `aggregate.py`, `boundary.py`, `llm.py`,
  `context.py`, `loop.py`, plus the `engine/closeout/` post-run distillation chain.
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
- **`data.dataset_variant` is not LLM-mutable** — deliberately absent from `ALLOWED_CHANGE_PATHS`
  so one experiment can never mix downsampled and full-data metrics.
- **Custom programs are self-contained** and their CORE sections must match `programs/PROGRAM.md`
  byte-verbatim: enforced at session start and by
  `tests/unit/numereng/test_agentic_research_program_core.py` (`programs/archive/` is exempt).
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
