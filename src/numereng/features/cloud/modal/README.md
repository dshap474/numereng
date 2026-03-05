# numereng Modal Adapter

This package provides a provider-local Modal adapter for running `numereng` training remotely.

## What Is Included

- Typed contracts for request/response/state payloads.
- Adapter protocols and a Modal SDK-backed adapter implementation.
- Runtime helper that executes `run_training` in-process on a Modal worker.
- State store for chained lifecycle commands.
- Deploy orchestration for ECR-backed Modal app/function updates.
- Data sync orchestration for uploading config-required datasets into a Modal Volume.
- Service orchestration for `submit`, `status`, `logs`, `cancel`, and `pull`.
- `logs` and `pull` currently return metadata only (no direct artifact/log stream download).
- API wiring at `numereng.api.cloud.modal` and CLI wiring at `numereng.cli.commands.cloud_modal`.

## What Is Not Included

- No dependency manifest changes are included (Modal SDK remains optional at runtime).

## Remote Function Contract

The adapter expects a deployed Modal function (default: app `numereng-train`, function `train_remote`)
that accepts one payload dictionary compatible with `ModalRuntimePayload` and returns a dictionary
compatible with `ModalRuntimeResult`.

`runtime.run_training_payload` is the canonical helper to execute that contract.

`cloud modal deploy` can provision that function directly from ECR using local AWS credentials
(default credential chain with optional `--aws-profile`).

`cloud modal data sync` uploads config-required dataset files into a Modal Volume and
`cloud modal deploy --data-volume-name <name>` mounts that volume at
`/app/.numereng/datasets` in the remote function.
