# API Parity Matrix

This matrix is the operator-facing lookup table for the API-only skill. Use it to choose between vendored `numerapi`, a direct GraphQL helper, or a `schema-unconfirmed` status.

Status values:
- `numerapi-native`: use vendored `numerapi` through `scripts/numerai_api_ops.py`
- `graphql-helper`: use direct GraphQL through `scripts/numerai_api_ops.py`
- `schema-unconfirmed`: do not advertise as supported until confirmed by `scripts/introspect_graphql_schema.py`

## Supported Capabilities

| Capability | Status | Preferred Path | Script Operation | Root / Method | MCP-Equivalent | Notes |
|---|---|---|---|---|---|---|
| List tournaments | `graphql-helper` | direct GraphQL | `list_tournaments` | `tournaments` | `get_tournaments` | Live schema confirmed 2026-03-21 |
| Get round details | `graphql-helper` | direct GraphQL | `get_round_details` | `roundDetails` | `get_round_details` | Live schema confirmed 2026-03-21 |
| Check API credentials and scopes | `graphql-helper` | direct GraphQL | `check_api_credentials` | `apiTokenInfo`, `apiTokenScopes` | `check_api_credentials` | Auth required |
| Get current round | `numerapi-native` | vendored `numerapi` | `get_current_round` | `Api.get_current_round` | `get_current_round` | Any surface |
| List datasets | `numerapi-native` | vendored `numerapi` | `list_datasets` | `Api.list_datasets` | `list_datasets` | Any surface |
| Download dataset | `numerapi-native` | vendored `numerapi` | `download_dataset` | `Api.download_dataset` | none | Any surface |
| Get account | `numerapi-native` | vendored `numerapi` | `get_account` | `Api.get_account` | none | Auth required |
| Get models | `numerapi-native` | vendored `numerapi` | `get_models` | `Api.get_models` | none | Auth required |
| Get Classic competitions / rounds | `numerapi-native` | vendored `numerapi` | `get_competitions` | `NumerAPI.get_competitions` | partial `get_round_details` parity | Classic only |
| Get leaderboard | `numerapi-native` | vendored `numerapi` | `get_leaderboard` | `get_leaderboard` | `get_leaderboard` | Surface-specific |
| Get public profile | `numerapi-native` | vendored `numerapi` | `public_user_profile` | `public_user_profile` | partial `get_model_profile` parity | Classic or Signals |
| Get daily model performance | `numerapi-native` | vendored `numerapi` | `daily_model_performances` | `daily_model_performances` | partial `get_model_performance` parity | Classic or Signals |
| Get round model performance v2 | `numerapi-native` | vendored `numerapi` | `round_model_performances_v2` | `Api.round_model_performances_v2` | partial `get_model_performance` parity | Auth may be required depending on model |
| Get intra-round scores | `numerapi-native` | vendored `numerapi` | `intra_round_scores` | `Api.intra_round_scores` | none | Auth required |
| List submissions | `numerapi-native` | vendored `numerapi` | `submission_ids` | `Api.submission_ids` | none | Auth required |
| Download submission | `numerapi-native` | vendored `numerapi` | `download_submission` | `Api.download_submission` | none | Auth required |
| Upload Classic/Crypto predictions | `graphql-helper` | direct GraphQL | `upload_predictions` | `submissionUploadAuth`, `createSubmission` | none | Preferred over vendored Classic wrapper due live camelCase schema |
| Upload Signals predictions | `graphql-helper` | direct GraphQL | `upload_predictions` | `submissionUploadSignalsAuth`, `createSignalsSubmission` | none | Direct GraphQL keeps one consistent submission helper |
| List diagnostics | `numerapi-native` | vendored `numerapi` | `list_diagnostics` | `Api.diagnostics(model_id, id=None)` | `run_diagnostics.list` | Auth required |
| Read diagnostics by ID | `numerapi-native` | vendored `numerapi` | `read_diagnostics` | `Api.diagnostics` | `run_diagnostics.get` | Auth required |
| Upload diagnostics | `numerapi-native` | vendored `numerapi` | `upload_diagnostics` | `Api.upload_diagnostics` | `run_diagnostics.create` | Auth required |
| Delete diagnostics | `graphql-helper` | direct GraphQL | `delete_diagnostics` | `deleteDiagnostics` | `run_diagnostics.delete` | Live schema confirmed 2026-03-21 |
| Create compute pickle upload | `numerapi-native` | vendored `numerapi` | `create_compute_pickle_upload` | `Api.model_upload` | `upload_model.create` | Auth required |
| List compute pickles | `graphql-helper` | direct GraphQL | `list_compute_pickles` | `computePickles` | `upload_model.list` | Live schema confirmed 2026-03-21 |
| List compute data versions | `numerapi-native` | vendored `numerapi` | `list_model_upload_data_versions` | `Api.model_upload_data_versions` | `upload_model.list_data_versions` | Auth required |
| List compute docker images | `numerapi-native` | vendored `numerapi` | `list_model_upload_docker_images` | `Api.model_upload_docker_images` | `upload_model.list_docker_images` | Auth required |
| Assign compute pickle to model | `graphql-helper` | direct GraphQL | `assign_compute_pickle` | `assignPickleToModel` | `upload_model.assign` | Live schema confirmed 2026-03-21 |
| Trigger compute pickle | `graphql-helper` | direct GraphQL | `trigger_compute_pickle` | `triggerComputePickleUpload` | `upload_model.trigger` | Live schema confirmed 2026-03-21 |
| Read trigger logs | `graphql-helper` | direct GraphQL | `get_trigger_logs` | `triggerLogs` | `upload_model.get_logs` | Live schema confirmed 2026-03-21 |
| Read diagnostics trigger logs | `graphql-helper` | direct GraphQL | `get_diagnostics_trigger_logs` | `diagnosticsTriggerLogs` | none | Live schema confirmed 2026-03-21 |
| Create model | `graphql-helper` | direct GraphQL | `create_model` | `addModel` | `create_model` | Current mutation name is `addModel`, not legacy `createModel` |
| Pipeline status | `numerapi-native` | vendored `numerapi` | `pipeline_status` | `Api.pipeline_status` | none | Read-only |
| Signals ticker universe | `numerapi-native` | vendored `numerapi` | `ticker_universe` | `SignalsAPI.ticker_universe` | none | Signals only |
| Increase / decrease stake | `numerapi-native` | vendored `numerapi` | `change_stake` | `stake_increase`, `stake_decrease` | none | Auth required |
| Set stake exactly | `numerapi-native` | vendored `numerapi` | `set_stake_exact` | `NumerAPI.stake_set` | none | Classic only |
| Set bio | `numerapi-native` | vendored `numerapi` | `set_bio` | `Api.set_bio` | none | Auth required |
| Set link | `numerapi-native` | vendored `numerapi` | `set_link` | `Api.set_link` | none | Auth required |
| Set submission webhook | `numerapi-native` | vendored `numerapi` | `set_submission_webhook` | `Api.set_submission_webhook` | none | Auth required |

## Explicitly Not Advertised Yet

| Capability | Status | Reason |
|---|---|---|
| Download compute pickle | `schema-unconfirmed` | Root existence is visible in schema, but this skill does not yet ship a confirmed helper or template for it |
| Archive / unarchive model | `schema-unconfirmed` | Not part of the parity target for replacing routine MCP usage |
| Rename model or account | `schema-unconfirmed` | Not part of the parity target for routine operations |
