"""Local custom agentic research program guidance."""

# Custom Agentic Research Programs

This folder is for local experiment-specific prompt programs.

Tracked contract:

- `README.md` is tracked so agents know how this folder works.
- Every other file in this folder is gitignored by default.
- Keep the base tracked program at `src/numereng/features/agentic_research/PROGRAM.md`.

Use this folder when one experiment needs a narrower policy than the base program, for example a fast GPU LGBM overnight loop or a capped smoke-test loop.

## Add A Program

Create a Markdown file here:

```text
src/numereng/features/agentic_research/custom_programs/MY-PROGRAM.md
```

Then point an experiment at it by setting `metadata.agentic_research_program` in that experiment's `experiment.json`:

```json
{
  "metadata": {
    "agentic_research_program": "MY-PROGRAM.md"
  }
}
```

Use only the filename, not a path. The runner resolves custom names from this folder and rejects paths.

## Run

```bash
uv run numereng research status --experiment-id <experiment_id>
uv run numereng research run --experiment-id <experiment_id> --max-rounds <n>
```

`research status` should show the resolved `program_path` under this folder.

## Program Shape

Custom programs should still follow the base contract:

- include `{{CONTEXT_JSON}}`
- tell the LLM what objective to optimize
- list the allowed config knobs for that experiment
- require exactly one JSON decision object
- keep hidden reasoning out of the response

The Python runner still validates allowed mutation paths and config schema after the LLM responds.
