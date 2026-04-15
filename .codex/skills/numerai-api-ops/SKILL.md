---
name: numerai-api-ops
description: "API-only Numerai operations playbook using vendored numerapi plus direct GraphQL parity helpers."
user-invocable: true
argument-hint: <operation intent> (e.g., "list tournaments", "upload classic predictions", "assign a compute pickle")
---

# Numerai API Ops

Use this skill when the user wants official Numerai API operations through vendored `numerapi` or direct GraphQL helpers.

Run from:
- `<workspace>`

## Use When

Use this skill when the task is to:
- inspect rounds, tournaments, datasets, leaderboards, accounts, diagnostics, submissions, or compute uploads
- determine whether vendored `numerapi` already covers a requested Numerai operation
- execute a schema-confirmed direct GraphQL helper when no stable vendored wrapper exists

## Do Not Use When

- Do not use this skill for numereng training, experiments, experiment design, or store maintenance workflows.
- Do not use this skill for numereng-specific experiment winner handoff or predictions submission through the package surface; use `numereng-experiment-ops`.
- Do not use this skill for general Numerai research, community review, or documentation synthesis that is not an operational API task.

## Scope

This skill is API-only. It covers official Numerai operations through:
- vendored `numerapi` clients (`NumerAPI`, `SignalsAPI`, `CryptoAPI`)
- direct GraphQL calls to `https://api-tournament.numer.ai`

## Non-Goals

- This skill does not manage numereng experiments, training, or run orchestration.
- This skill does not guess undocumented GraphQL fields or argument shapes.
- Any operation that is not `numerapi-native` or live-confirmed as a `graphql-helper` remains `schema-unconfirmed`.

## Hard Rules

- Prefer the highest-level stable API path:
  1. vendored `numerapi` wrapper if the wrapper matches the live schema
  2. direct GraphQL helper if no stable wrapper exists or the wrapper has schema drift
- Treat live GraphQL introspection as the source of truth for direct GraphQL helpers.
- Do not advertise an operation as supported unless it is either:
  - `numerapi-native`, or
  - `graphql-helper` confirmed by live schema introspection.
- Mark anything else as `schema-unconfirmed`.
- Require explicit user confirmation before write actions.
- After write actions, always record:
  - interface used
  - exact method or GraphQL root used
  - core IDs returned
  - read-back verification result
- Distinguish these objects before giving write advice:
  - model slot: the persistent Numerai model identity
  - compute pickle: the uploaded hosted-inference artifact
  - assigned pickle: the compute pickle currently attached to a model slot
- For model-upload workflows, check account model capacity before assuming the user must reuse an existing slot.
- For compute-pickle assignment read-back, verify against `account.models.computePickleUpload` first. Do not rely only on `computePickles(modelId=...)`, because that filter tracks the upload owner model and can miss a reassigned slot.

## Network And Auth

- Live Numerai network access is required for direct GraphQL helpers and authenticated account operations.
- Missing auth must stop write actions rather than falling back to guessed or partial behavior.
- Offline or unauthenticated schema gaps must be reported as unverified, not inferred.

## Auth Matrix

### Direct GraphQL

- Endpoint: `https://api-tournament.numer.ai`
- Preferred env header:
  - `NUMERAI_API_AUTH="Token PUBLIC_KEY$PRIVATE_KEY"`
- Accepted fallback envs:
  - `NUMERAI_MCP_AUTH`
  - `NUMERAI_PUBLIC_ID` + `NUMERAI_SECRET_KEY`

### NumerAPI

- Python clients:
  - `numerapi.NumerAPI()` for Classic
  - `numerapi.SignalsAPI()` for Signals
  - `numerapi.CryptoAPI()` for Crypto
- The helper scripts in `scripts/` resolve credentials from the same env sources above.

## Reference Loading Guide

Load this reference only when the task matches the condition below.

| Task condition | Load this reference |
|---|---|
| The task is capability discovery, parity checking, or API-path selection. | `references/api-parity-matrix.md` |
| The task is checking whether vendored `numerapi` already exposes a method. | `references/numerapi-method-inventory.md` |
| The task needs a direct GraphQL helper or a live-confirmed root name. | `references/graphql-operation-map.md` |
| The task requires official Numerai source links or source attribution. | `references/official-sources.md` |

## Asset Usage Guide

Use this asset only when the task matches the condition below.

| Task condition | Use this asset |
|---|---|
| The task needs a quick operator command or common workflow shortcut. | `assets/api-ops-cheatsheet.md` |
| The task needs raw HTTP or `curl` request structure. | `assets/http-examples.md` |
| The task needs API token info or scopes GraphQL shapes. | `assets/graphql-templates/auth.graphql` |
| The task needs compute pickle list, assign, trigger, or logs GraphQL shapes. | `assets/graphql-templates/compute.graphql` |
| The task needs dataset GraphQL examples for inspection only. | `assets/graphql-templates/datasets.graphql` |
| The task needs diagnostics upload, read, or delete GraphQL shapes. | `assets/graphql-templates/diagnostics.graphql` |
| The task needs schema introspection query text. | `assets/graphql-templates/introspection.graphql` |
| The task needs leaderboard GraphQL examples for inspection only. | `assets/graphql-templates/leaderboard.graphql` |
| The task needs account, model, or model-creation GraphQL shapes. | `assets/graphql-templates/models.graphql` |
| The task needs round or round-detail GraphQL shapes. | `assets/graphql-templates/rounds.graphql` |
| The task needs staking GraphQL examples for inspection only. | `assets/graphql-templates/staking.graphql` |
| The task needs submission upload or download GraphQL shapes. | `assets/graphql-templates/submissions.graphql` |
| The task needs tournament-list GraphQL shapes. | `assets/graphql-templates/tournaments.graphql` |
| The task needs live schema verification of root names or argument shapes. | `scripts/introspect_graphql_schema.py` |
| The task needs deterministic execution of a read or write-capable helper. | `scripts/numerai_api_ops.py` |

## Execution Policy

Use this decision rule:

1. Look up the requested capability in `references/api-parity-matrix.md`.
2. If status is `numerapi-native`, prefer `scripts/numerai_api_ops.py`.
3. If status is `graphql-helper`, use the direct GraphQL helper in `scripts/numerai_api_ops.py` and the matching template under `assets/graphql-templates/`.
4. If status is `schema-unconfirmed`, run `scripts/introspect_graphql_schema.py` first and only then build the operation.

## Expected Outputs

- For read-only tasks, report the interface used, the exact method or GraphQL root used, the key IDs or fields returned, and any verification performed.
- For write-intent tasks before confirmation, report the intended mutation, required arguments, the planned verification query, and that explicit user confirmation is required.
- For completed write tasks, report the returned identifiers, the read-back verification result, and any unresolved caveats.

## Quickstart

### Inspect supported operations

```bash
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py list
```

### Show one operation

```bash
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py show list_tournaments
```

### Dry-run one operation

```bash
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py run \
  list_tournaments \
  --dry-run
```

### Execute a read operation

```bash
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py run \
  get_current_round \
  --json-args '{"surface":"classic"}'
```

### Execute a write operation

```bash
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py run \
  create_model \
  --json-args '{"name":"example-model","tournament":8}' \
  --confirm-write
```

## Live Schema Workflow

When GraphQL root names or args are uncertain:

```bash
uv run python .agents/skills/numerai-api-ops/scripts/introspect_graphql_schema.py \
  --section both
```

To inspect only a few roots:

```bash
uv run python .agents/skills/numerai-api-ops/scripts/introspect_graphql_schema.py \
  --section both \
  --field tournaments \
  --field roundDetails \
  --field addModel
```

## Compatibility Notes

- The live GraphQL schema currently uses camelCase roots such as:
  - `submissionUploadAuth`
  - `createSubmission`
  - `addModel`
- Some vendored `numerapi` write flows still reference older snake_case names in Classic submission upload paths.
- In this skill, direct GraphQL helpers are the source of truth for parity-critical operations where the vendored wrapper drifts from the live schema.

## Gotchas

- Wrapper parity is incomplete. Use direct GraphQL helpers for confirmed gaps instead of stretching an unstable wrapper path.
- `schema-unconfirmed` means the operation must not be advertised or executed as supported until live schema verification succeeds.
- All write actions require explicit user confirmation, even if a helper exists.
- Model names created through `addModel` must be username-safe and no longer than 20 characters.
- The safest model-upload operator sequence is:
  1. inspect existing account models
  2. decide whether to reuse or create a slot
  3. create or upload the compute pickle for the target slot
  4. assign the compute pickle only when it already belongs to that slot, or when disabling with `pickleId = null`
  5. verify the assigned slot through `account.models.computePickleUpload`
- Disabling a hosted model upload is the same assignment mutation with `pickleId = null`.
- Do not treat cross-slot compute-pickle reassignment as a safe production migration path. If a model should run under a new slot, upload a fresh compute pickle directly to that slot.

## Validation

Run these checks after changing the skill package:

```bash
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py list
uv run python .agents/skills/numerai-api-ops/scripts/numerai_api_ops.py show list_tournaments
```

## Done Criteria

A task is complete when:
- the requested operation was mapped to one supported API path
- the tool or helper used is reproducible from this skill alone
- write actions were explicitly confirmed
- read-back verification was recorded for writes
