# Agentic Research

Use `numereng research` when you want numereng to run a simple autonomous config loop:

1. run a deterministic ML round
2. send the resulting configs, metrics, latest rolling memo, and recent decision log to the LLM
3. let the LLM propose one config mutation (`run` is the only action)
4. validate, train, score, and record the next round

The LLM does not edit Python code. It returns a structured decision form plus cumulative markdown
research state. Python converts the form into the strict machine decision, mutates an experiment
config, and appends exact execution results.

## Use This When

- you already have an experiment with at least one viable config under `configs/`
- you want numereng to search configs instead of hand-editing every child config
- you want a round-by-round decision trail under the experiment

## Research Prompt

Every round's prompt is composed from two files:

```text
src/numereng/agentic_research/programs/PROGRAM.md   the tracked contract and generic doctrine
.numereng/experiments/<id>/agentic_research/STRATEGY.md   the experiment brief
```

`PROGRAM.md` defines the objective, the evaluator, the champion and seed-trio rules, search
discipline, and the required `decision_form + round_markdown` response. It is tracked and edited in
one place; an edit reaches every run, live ones included, at its next round.

`STRATEGY.md` is the experiment brief: hypothesis, lane, prior evidence, sweep plan, confirmation
plan, and any substrate facts for the model family. The filename is fixed, so remote sync/fetch
carries it with the experiment. Write it at experiment creation. An experiment without one falls
back to the tracked generic brief at `src/numereng/agentic_research/programs/STRATEGY.md`, whose
headings a real brief follows.

`uv run numereng research status --experiment-id <id> --format json` reports the loop's status,
its champion, and whether the closeout memo exists. The tracked
`src/numereng/agentic_research/README.md` explains the full flow.

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

- `STRATEGY.md` — the experiment brief spliced into `PROGRAM.md` each round.
- `state.json` — small session state (schema v2): status, counters, champion, `believed_best`, heartbeat, last error.
- `journal.jsonl` — one append-only line per round attempt (machine-readable), carrying each run's `seed`, `metric` (BMC200), and `fnc`.
- `rounds/rNNN.md` — the model's verbatim round memo with one `## Machine Result` block appended by the harness.
- `rounds/rNNN.debug.*` — written only on an LLM/Codex failure.

Durable history lives in `journal.jsonl` plus `rounds/*.md`. `EXPERIMENT.md` (one level up, next to `configs/`) is the model-curated working set. The latest `rounds/rNNN.md` plus the bounded context assembled from `journal.jsonl` are what the next prompt sees; no context term grows with round count.

## Round Actions

There is exactly one action:

**`run`** — mutate one parent config on allowed paths, train, and score a new model.

The harness never edits the proposal: it validates the `decision_form` against fixed boundaries, materializes one config (rejecting out-of-bounds proposals whole, never clamping), trains, scores, and records the round. There is no `ensemble` action and no `stop` action — `stop_reason` is kept in the schema for shape stability and is ignored. A plateau is a reason to diversify, not to quit; the loop runs until the requested round budget, a human stop, an unhandled failure, or the five-consecutive-failure bail.

Strategy (what to try, when to seed-confirm, when to diversify, what to believe) lives in the prompt and the model, not in Python. To change how the loop behaves, edit `PROGRAM.md` for every experiment or the experiment's own `STRATEGY.md` for one.

## High-Risk Gotchas

- The mutable research surface is config JSON, not Python source.
- Python rejects changes outside the allowed paths in `context.allowed_change_paths`, and never clamps or repairs a proposal.
- Codex CLI runs with a JSON output schema so successful responses contain `decision_form` and `round_markdown`.
- Candidate configs must validate as `TrainingConfig`.
- A rejected proposal or a duplicate config comes back to the model once as `context.last_error` for one in-round retry; the round is only recorded as failed (or skipped, for a duplicate) if the second proposal fails too. The round memo's `## Machine Result` block carries the first token as `retry: <token>`.
- Planner backend selection is controlled by `ACTIVE_MODEL_SOURCE=codex-exec|droid-exec`.
- `PROGRAM.md` must keep the `{{STRATEGY}}` and `{{CONTEXT_JSON}}` placeholders exactly once each, and a brief must contain neither; the run refuses to start otherwise.
- `decision_form.seeds` (optional, 1 to 3 integers) trains the same child recipe once per seed inside one round; the seed is written to the experiment's `agentic_research_seed_path`, which must be an allowed change path.
- `research run` still relies on the normal training/scoring stack, so broken configs or missing datasets fail the same way they would in manual workflows.
- Boundary violations (disallowed change path, out-of-cap value, target/horizon mismatch, invalid `TrainingConfig`, non-`run` action, cross-experiment stale-run reuse) fail the round and count toward the five-consecutive-failure bail; a duplicate-by-hash is a soft skip that does not count.

## Read Next

- [Experiments](experiments.md)
- [Hyperparameter Optimization](optimization.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
