# Config-Only Runtime

Preset/template runtime execution is removed.

## Removed Surface

- `numereng presets ...`
- `experiment create --preset ...`
- Runtime execution from preset/template paths

## What To Use Instead

Use explicit JSON configs you own, for example:

- `configs/run.json`
- `experiments/<experiment_id>/configs/run.json`
- `configs/hpo/study.json`

Then run them through current command families:

- `numereng run train --config <path.json>`
- `numereng experiment train --id <id> --config <path.json>`
- `numereng hpo create --study-config <path.json>`

## Notes

- JSON is required for both training and HPO configs.
- Unknown keys are rejected at validation time.
