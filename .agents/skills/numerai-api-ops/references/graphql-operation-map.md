# GraphQL Operation Map

This map is for direct GraphQL helper selection inside the API-only skill. Use it when vendored `numerapi` does not already provide the required operation.

Endpoint:
- `https://api-tournament.numer.ai`

Auth header for protected operations:
- `Authorization: Token PUBLIC_KEY$PRIVATE_KEY`

Live schema verification:
- Root names and argument shapes below were checked against unauthenticated live introspection on 2026-03-21.

## Root Fields Used By This Skill

| Domain | Root Field(s) | Status | Preferred Path | Notes |
|---|---|---|---|---|
| tournaments | `tournaments` | `graphql-helper` | direct GraphQL | No args |
| round detail | `roundDetails` | `graphql-helper` | direct GraphQL | Args: `roundNumber`, `tournament` |
| API token info | `apiTokenInfo`, `apiTokenScopes` | `graphql-helper` | direct GraphQL | Auth probe and scope listing |
| rounds/current round | `rounds` | `numerapi-native` | vendored wrapper | Use direct GraphQL only if wrapper path is insufficient |
| datasets | `listDatasets`, `dataset` | `numerapi-native` | vendored wrapper | Stable read paths |
| account/models | `account`, `accountProfile`, `model` | mixed | wrapper first | `model` arg is `modelId` in live schema; `account.models.computePickleUpload` is the preferred read-back for slot assignment state |
| submissions list/download | `submissions`, `submissionDownloadAuth` | `numerapi-native` | vendored wrapper | Auth required |
| Classic/Crypto submission upload | `submissionUploadAuth`, `createSubmission` | `graphql-helper` | direct GraphQL | Preferred over vendored Classic wrapper due schema drift |
| Signals submission upload | `submissionUploadSignalsAuth`, `createSignalsSubmission` | `graphql-helper` | direct GraphQL | One consistent helper path |
| diagnostics create/read | `diagnosticsUploadAuth`, `createDiagnostics`, `diagnostics` | mixed | wrapper first | `diagnostics` returns a list |
| diagnostics delete | `deleteDiagnostics` | `graphql-helper` | direct GraphQL | Arg is `v2DiagnosticsIds` |
| leaderboard | `v2Leaderboard`, `signalsLeaderboard`, `cryptosignalsLeaderboard`, `accountLeaderboard` | `numerapi-native` | vendored wrapper | Stable read paths |
| performance history | `v2RoundModelPerformances`, `v3UserProfile`, `v2SignalsProfile` | `numerapi-native` | vendored wrapper | Use wrapper unless a custom query is needed |
| staking | `v2ChangeStake`, `releaseStake` | `numerapi-native` | vendored wrapper | Auth required |
| compute upload create | `computePickleUploadAuth`, `createComputePickleUpload` | `numerapi-native` | vendored wrapper | Stable enough for helper use |
| compute list | `computePickles` | `graphql-helper` | direct GraphQL | Covers MCP `upload_model.list` parity; `modelId` filters by upload owner model |
| compute assign | `assignPickleToModel` | `graphql-helper` | direct GraphQL | MCP-equivalent assign path; verify reassignment through `account.models.computePickleUpload` |
| compute trigger | `triggerComputePickleUpload` | `graphql-helper` | direct GraphQL | MCP-equivalent trigger path |
| compute logs | `triggerLogs`, `diagnosticsTriggerLogs` | `graphql-helper` | direct GraphQL | Direct log retrieval parity |
| create model | `addModel` | `graphql-helper` | direct GraphQL | Current mutation name is `addModel`; model names must be username-safe and <= 20 chars |
| profile settings | `setUserBio`, `setUserLink`, `setSubmissionWebhook` | `numerapi-native` | vendored wrapper | Auth required |
| pipeline | `pipelineStatus` | `numerapi-native` | vendored wrapper | Read-only |

## Compatibility Notes

- Older docs and wrappers may reference:
  - `createModel`
  - `submission_upload_auth`
  - `create_submission`
- For this skill, the live schema names above are authoritative for direct GraphQL helpers.

## Grouped Templates

- auth: `assets/graphql-templates/auth.graphql`
- compute: `assets/graphql-templates/compute.graphql`
- datasets: `assets/graphql-templates/datasets.graphql`
- diagnostics: `assets/graphql-templates/diagnostics.graphql`
- introspection: `assets/graphql-templates/introspection.graphql`
- leaderboard: `assets/graphql-templates/leaderboard.graphql`
- models: `assets/graphql-templates/models.graphql`
- rounds: `assets/graphql-templates/rounds.graphql`
- staking: `assets/graphql-templates/staking.graphql`
- submissions: `assets/graphql-templates/submissions.graphql`
- tournaments: `assets/graphql-templates/tournaments.graphql`
