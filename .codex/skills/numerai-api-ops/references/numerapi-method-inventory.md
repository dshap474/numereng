# NumerAPI Method Inventory

Inventory source:
- vendored client code under `vendor/numerapi/numerapi/*.py`

## Shared `Api` Methods

- `check_new_round`
- `check_round_open`
- `diagnostics`
- `download_dataset`
- `download_submission`
- `get_account`
- `get_account_leaderboard`
- `get_current_round`
- `get_models`
- `intra_round_scores`
- `list_datasets`
- `model_upload`
- `model_upload_data_versions`
- `model_upload_docker_images`
- `modelid_to_modelname`
- `models_of_account`
- `pipeline_status`
- `raw_query`
- `round_model_performances`
- `round_model_performances_v2`
- `set_bio`
- `set_global_data_dir`
- `set_link`
- `set_submission_webhook`
- `stake_change`
- `stake_decrease`
- `stake_drain`
- `stake_increase`
- `submission_ids`
- `upload_diagnostics`
- `upload_predictions`
- `wallet_transactions`

## `NumerAPI` (Classic) Additional Methods

- `daily_model_performances`
- `get_competitions`
- `get_leaderboard`
- `get_submission_filenames`
- `public_user_profile`
- `stake_get`
- `stake_set`

## `SignalsAPI` Additional Methods

- `daily_model_performances`
- `get_leaderboard`
- `public_user_profile`
- `ticker_universe`
- `stake_get`

## `CryptoAPI` Additional Methods

- `get_leaderboard`

## Compatibility Notes

- The vendored Classic submission upload path still references older snake_case GraphQL roots:
  - `submission_upload_auth`
  - `create_submission`
- The live GraphQL schema confirmed on 2026-03-21 uses camelCase roots instead:
  - `submissionUploadAuth`
  - `createSubmission`
- This skill therefore prefers direct GraphQL helpers for Classic/Crypto submission upload even though `upload_predictions` exists in the vendored wrapper.

## Write-Capable Wrapper Methods

Require explicit user confirmation before these:

- `upload_predictions`
- `upload_diagnostics`
- `model_upload`
- `stake_change`
- `stake_increase`
- `stake_decrease`
- `stake_drain`
- `stake_set`
- `set_bio`
- `set_link`
- `set_submission_webhook`
- `raw_query` when used with mutation semantics
