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

The prompt policy lives at:

```text
src/numereng/agentic_research/programs/PROGRAM.md
```

It defines the objective, allowed config paths, and required `decision_form + round_markdown`
response. Treat it as the base/default program contract. Serious experiments should usually use a
focused custom program that declares one hypothesis, fixed surface, allowed variation axis, and
stop/confirmation rule.

A focused experiment's custom program belongs in the experiment's own folder:

```text
.numereng/experiments/<id>/agentic_research/<name>.md
```

Author it there at experiment creation; it then travels with the experiment on remote sync/fetch. Experiments opt into one by setting `metadata.agentic_research_program` to the bare Markdown filename, which is resolved experiment-folder-first and then from the local-only, gitignored legacy fallback `src/numereng/agentic_research/programs/`.
The tracked `src/numereng/agentic_research/README.md` explains the program authoring rules and usage contract.

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

- `state.json` — small session state (schema v2): status, counters, champion, `believed_best`, heartbeat, last error.
- `journal.jsonl` — one append-only line per round attempt (machine-readable), carrying each run's `seed`, `metric` (BMC200), and `fnc`.
- `rounds/rNNN.md` — the model's verbatim round memo with one `## Machine Result` block appended by the harness.
- `rounds/rNNN.debug.*` — written only on an LLM/Codex failure.

Durable history lives in `journal.jsonl` plus `rounds/*.md`. `EXPERIMENT.md` (one level up, next to `configs/`) is the model-curated working set. The latest `rounds/rNNN.md` plus the bounded context assembled from `journal.jsonl` are what the next prompt sees; no context term grows with round count.

## Round Actions

There is exactly one action:

**`run`** — mutate one parent config on allowed paths, train, and score a new model.

The harness never edits the proposal: it validates the `decision_form` against fixed boundaries, materializes one config (rejecting out-of-bounds proposals whole, never clamping), trains, scores, and records the round. There is no `ensemble` action and no `stop` action — `stop_reason` is kept in the schema for shape stability and is ignored. A plateau is a reason to diversify, not to quit; the loop runs until the requested round budget, a human stop, an unhandled failure, or the five-consecutive-failure bail.

Strategy (what to try, when to seed-confirm, when to diversify, what to believe) lives in the program file and the model, not in Python. To change how the loop behaves, edit the program prompt (`PROGRAM.md` or the experiment's custom program).

## High-Risk Gotchas

- The mutable research surface is config JSON, not Python source.
- Python rejects changes outside the allowed paths in `PROGRAM.md` / runner context.
- Codex CLI runs with a JSON output schema so successful responses contain `decision_form` and `round_markdown`.
- Candidate configs must validate as `TrainingConfig`.
- Duplicate configs are rejected before training.
- Planner backend selection is controlled by `ACTIVE_MODEL_SOURCE=codex-exec|openrouter|droid-exec`.
- The default prompt is tracked as `PROGRAM.md`; an experiment sets `metadata.agentic_research_program` to a bare filename resolved from `<experiment_root>/agentic_research/<name>.md` first, then the legacy `programs/` fallback.
- A resolved custom program's CORE sections are checked byte-verbatim against `PROGRAM.md` at session start; start from `src/numereng/agentic_research/programs/PROGRAM.md` and keep its CORE sections verbatim.
- `research run` still relies on the normal training/scoring stack, so broken configs or missing datasets fail the same way they would in manual workflows.
- Boundary violations (disallowed change path, out-of-cap value, target/horizon mismatch, invalid `TrainingConfig`, non-`run` action, cross-experiment stale-run reuse) fail the round and count toward the five-consecutive-failure bail; a duplicate-by-hash is the one exception (soft skip, no count).

## Read Next

- [Experiments](experiments.md)
- [Hyperparameter Optimization](optimization.md)
- [Runtime Artifacts & Paths](../reference/runtime-artifacts.md)
