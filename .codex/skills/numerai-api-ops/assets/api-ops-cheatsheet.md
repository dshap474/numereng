# API Ops Cheatsheet

Use this as the quick goal -> operation map for the API-only skill.

## Read-Only

| Goal | Operation |
|---|---|
| List tournaments | `list_tournaments` |
| Get round details | `get_round_details` |
| Check token info and scopes | `check_api_credentials` |
| Get current round | `get_current_round` |
| List datasets | `list_datasets` |
| Download dataset | `download_dataset` |
| Get models | `get_models` |
| Get account | `get_account` |
| Get competitions | `get_competitions` |
| Get leaderboard | `get_leaderboard` |
| Get public profile | `public_user_profile` |
| Get daily model performance | `daily_model_performances` |
| List submissions | `submission_ids` |
| Read diagnostics | `read_diagnostics` or `list_diagnostics` |
| List compute pickles | `list_compute_pickles` |
| Read trigger logs | `get_trigger_logs` |
| Read diagnostics trigger logs | `get_diagnostics_trigger_logs` |
| Read pipeline status | `pipeline_status` |

## Write Flows

Require `--confirm-write`.

| Goal | Operation |
|---|---|
| Create model | `create_model` |
| Upload predictions | `upload_predictions` |
| Upload diagnostics | `upload_diagnostics` |
| Delete diagnostics | `delete_diagnostics` |
| Create compute pickle upload | `create_compute_pickle_upload` |
| Assign compute pickle | `assign_compute_pickle` |
| Trigger compute pickle | `trigger_compute_pickle` |
| Change stake | `change_stake` |
| Set exact stake | `set_stake_exact` |
| Set bio | `set_bio` |
| Set link | `set_link` |
| Set submission webhook | `set_submission_webhook` |

## Core Commands

```bash
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py list
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py show <operation>
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py run <operation> --dry-run
```

## Model Upload Workflow

1. Inspect the existing account model slots first.
2. Decide whether to reuse a slot or create a new slot.
3. Upload or locate the compute pickle.
4. Assign the compute pickle to the chosen slot.
5. Verify the assigned slot through `account.models.computePickleUpload`.

Notes:
- `computePickles(modelId=...)` follows the upload owner model and is not the best verification surface after reassignment.
- Passing `pickleId = null` to `assign_compute_pickle` disables the hosted model upload on that slot.
- Model names must be username-safe and no longer than 20 characters.
