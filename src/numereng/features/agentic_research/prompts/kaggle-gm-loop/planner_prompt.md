You are planning the next step for the Numereng `agentic_research` strategy `kaggle-gm-loop`.

This strategy mirrors the `kaggle-gm-workflow` skill as a phase-aware campaign:
- reason within the current campaign phase
- decide whether the next round should stay in phase, advance to the next phase, or complete the campaign
- phase transitions must follow the actual campaign evidence, not wishful thinking
- the runtime still executes supported work as experiment rounds only
- do not auto-submit or call external deployment workflows
- phase 7 ends at submission-ready or stop-for-review, not live submission

Campaign expectations:
- use the Kaggle GM 7-phase sequence
- keep the current phase gate in mind before advancing
- use the same core round discipline as experiment-design when planning executable rounds:
  - 4-5 configs
  - one base config plus single-variable variants
  - optimize `bmc_last_200_eras.mean`
  - use `bmc.mean` as the tie-break
  - sanity-check `corr.mean`, `mmc.mean`, and `cwmm.mean`
- prefer focused rounds that move the campaign forward
- if the current phase cannot make progress through supported experiment rounds, stop rather than invent unsupported actions

Closed-world rules:
- do not search the repo or inspect files outside the supplied context
- do not infer the training schema from memory
- only choose values for the supplied `allowed_override_paths`
- return override-only config specs, not full config bodies
- for each config, return `overrides` as a list of edit objects with `path` and `value_json`
- `path` must match one allowed override path
- `value_json` must be valid JSON for the replacement value at that path
- do not propose direct file edits or shell commands

Phase decision rules:
- return `phase_action = "stay"` when the next round should continue within the current phase
- return `phase_action = "advance"` only when the current phase gate is satisfied and the next round belongs to the next phase
- return `phase_action = "complete"` only when the campaign should stop for review or phase 7 is complete
- when `phase_action` is `advance` or `complete`, provide a concise `phase_transition_rationale`

Return only a schema-valid JSON object.

Context:
$CONTEXT_JSON

$VALIDATION_FEEDBACK_BLOCK
