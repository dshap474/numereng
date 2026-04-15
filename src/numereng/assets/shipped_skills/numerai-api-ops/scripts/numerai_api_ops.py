#!/usr/bin/env python3
"""Deterministic Numerai API operations helper.

This script is the executable surface for the API-only `numerai-api-ops`
skill. It prefers vendored `numerapi` wrappers where they match the current
live API, and it uses direct GraphQL helpers for parity-critical operations
that are missing or stale in the vendored wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[4]
VENDOR_NUMERAPI_ROOT = REPO_ROOT / "vendor" / "numerapi"
if str(VENDOR_NUMERAPI_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_NUMERAPI_ROOT))

from numerapi.base_api import API_TOURNAMENT_URL  # type: ignore  # noqa: E402
from numerapi.cryptoapi import CryptoAPI  # type: ignore  # noqa: E402
from numerapi.numerapi import NumerAPI  # type: ignore  # noqa: E402
from numerapi.signalsapi import SignalsAPI  # type: ignore  # noqa: E402

GRAPHQL_ENDPOINT = API_TOURNAMENT_URL
DEFAULT_TIMEOUT = 600
SURFACES = ("classic", "signals", "crypto")


@dataclass(frozen=True)
class OperationSpec:
    name: str
    status: str
    interface: str
    summary: str
    auth_required: bool
    write: bool
    mcp_equivalent: str | None = None
    default_surface: str | None = None
    allowed_surfaces: tuple[str, ...] | None = None
    method: str | None = None
    notes: str | None = None


def _spec(
    name: str,
    status: str,
    interface: str,
    summary: str,
    auth_required: bool,
    write: bool,
    *,
    mcp_equivalent: str | None = None,
    default_surface: str | None = None,
    allowed_surfaces: tuple[str, ...] | None = None,
    method: str | None = None,
    notes: str | None = None,
) -> OperationSpec:
    return OperationSpec(
        name=name,
        status=status,
        interface=interface,
        summary=summary,
        auth_required=auth_required,
        write=write,
        mcp_equivalent=mcp_equivalent,
        default_surface=default_surface,
        allowed_surfaces=allowed_surfaces,
        method=method,
        notes=notes,
    )


OPERATIONS: dict[str, OperationSpec] = {
    "list_tournaments": _spec(
        "list_tournaments",
        "graphql-helper",
        "graphql",
        "List active tournaments from the live GraphQL schema.",
        False,
        False,
        mcp_equivalent="get_tournaments",
    ),
    "get_round_details": _spec(
        "get_round_details",
        "graphql-helper",
        "graphql",
        "Fetch detailed round metadata by tournament and round number.",
        False,
        False,
        mcp_equivalent="get_round_details",
    ),
    "check_api_credentials": _spec(
        "check_api_credentials",
        "graphql-helper",
        "graphql",
        "Read token metadata and available scopes from the live GraphQL API.",
        True,
        False,
        mcp_equivalent="check_api_credentials",
    ),
    "create_model": _spec(
        "create_model",
        "graphql-helper",
        "graphql",
        "Create a model slot using the current `addModel` mutation.",
        True,
        True,
        mcp_equivalent="create_model",
    ),
    "list_compute_pickles": _spec(
        "list_compute_pickles",
        "graphql-helper",
        "graphql",
        "List compute pickle uploads and their lifecycle state.",
        True,
        False,
        mcp_equivalent="upload_model.list",
    ),
    "assign_compute_pickle": _spec(
        "assign_compute_pickle",
        "graphql-helper",
        "graphql",
        "Assign a compute pickle to a model slot.",
        True,
        True,
        mcp_equivalent="upload_model.assign",
    ),
    "trigger_compute_pickle": _spec(
        "trigger_compute_pickle",
        "graphql-helper",
        "graphql",
        "Trigger a compute pickle upload for execution or validation.",
        True,
        True,
        mcp_equivalent="upload_model.trigger",
    ),
    "get_trigger_logs": _spec(
        "get_trigger_logs",
        "graphql-helper",
        "graphql",
        "Fetch invocation logs for a trigger ID.",
        True,
        False,
        mcp_equivalent="upload_model.get_logs",
    ),
    "get_diagnostics_trigger_logs": _spec(
        "get_diagnostics_trigger_logs",
        "graphql-helper",
        "graphql",
        "Fetch diagnostics-trigger logs for a pickle ID.",
        True,
        False,
    ),
    "delete_diagnostics": _spec(
        "delete_diagnostics",
        "graphql-helper",
        "graphql",
        "Delete diagnostics jobs by ID.",
        True,
        True,
        mcp_equivalent="run_diagnostics.delete",
    ),
    "upload_predictions": _spec(
        "upload_predictions",
        "graphql-helper",
        "graphql",
        "Upload predictions using the live camelCase submission GraphQL roots.",
        True,
        True,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        notes="Preferred over vendored Classic upload because the live schema uses camelCase roots.",
    ),
    "get_current_round": _spec(
        "get_current_round",
        "numerapi-native",
        "numerapi",
        "Get the current round for the chosen surface.",
        False,
        False,
        mcp_equivalent="get_current_round",
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="get_current_round",
    ),
    "list_datasets": _spec(
        "list_datasets",
        "numerapi-native",
        "numerapi",
        "List datasets for the chosen surface.",
        False,
        False,
        mcp_equivalent="list_datasets",
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="list_datasets",
    ),
    "download_dataset": _spec(
        "download_dataset",
        "numerapi-native",
        "numerapi",
        "Download a dataset file for the chosen surface.",
        False,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="download_dataset",
    ),
    "get_account": _spec(
        "get_account",
        "numerapi-native",
        "numerapi",
        "Fetch account information through the vendored client.",
        True,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="get_account",
    ),
    "get_models": _spec(
        "get_models",
        "numerapi-native",
        "numerapi",
        "Fetch account model mappings for the chosen surface.",
        True,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="get_models",
    ),
    "get_competitions": _spec(
        "get_competitions",
        "numerapi-native",
        "numerapi",
        "Fetch Classic competition rounds.",
        False,
        False,
        default_surface="classic",
        allowed_surfaces=("classic",),
        method="get_competitions",
    ),
    "get_leaderboard": _spec(
        "get_leaderboard",
        "numerapi-native",
        "numerapi",
        "Fetch the surface-specific leaderboard.",
        False,
        False,
        mcp_equivalent="get_leaderboard",
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="get_leaderboard",
    ),
    "public_user_profile": _spec(
        "public_user_profile",
        "numerapi-native",
        "numerapi",
        "Fetch a public Classic or Signals profile.",
        False,
        False,
        default_surface="classic",
        allowed_surfaces=("classic", "signals"),
        method="public_user_profile",
    ),
    "daily_model_performances": _spec(
        "daily_model_performances",
        "numerapi-native",
        "numerapi",
        "Fetch daily model performance history for Classic or Signals.",
        False,
        False,
        default_surface="classic",
        allowed_surfaces=("classic", "signals"),
        method="daily_model_performances",
    ),
    "round_model_performances_v2": _spec(
        "round_model_performances_v2",
        "numerapi-native",
        "numerapi",
        "Fetch round-level performance history for one model.",
        False,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="round_model_performances_v2",
    ),
    "intra_round_scores": _spec(
        "intra_round_scores",
        "numerapi-native",
        "numerapi",
        "Fetch intra-round scores for one model.",
        False,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="intra_round_scores",
    ),
    "submission_ids": _spec(
        "submission_ids",
        "numerapi-native",
        "numerapi",
        "List submission IDs for one model.",
        True,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="submission_ids",
    ),
    "download_submission": _spec(
        "download_submission",
        "numerapi-native",
        "numerapi",
        "Download a submission by ID or latest model submission.",
        True,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="download_submission",
    ),
    "list_diagnostics": _spec(
        "list_diagnostics",
        "numerapi-native",
        "numerapi",
        "List diagnostics for a model by omitting the diagnostics ID.",
        True,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="diagnostics",
    ),
    "read_diagnostics": _spec(
        "read_diagnostics",
        "numerapi-native",
        "numerapi",
        "Read one diagnostics record by ID.",
        True,
        False,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="diagnostics",
    ),
    "upload_diagnostics": _spec(
        "upload_diagnostics",
        "numerapi-native",
        "numerapi",
        "Upload diagnostics input CSV and start diagnostics.",
        True,
        True,
        mcp_equivalent="run_diagnostics.create",
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="upload_diagnostics",
    ),
    "create_compute_pickle_upload": _spec(
        "create_compute_pickle_upload",
        "numerapi-native",
        "numerapi",
        "Upload a compute pickle and create the server-side pickle record.",
        True,
        True,
        mcp_equivalent="upload_model.create",
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="model_upload",
    ),
    "list_model_upload_data_versions": _spec(
        "list_model_upload_data_versions",
        "numerapi-native",
        "numerapi",
        "List compute pickle data versions.",
        True,
        False,
        mcp_equivalent="upload_model.list_data_versions",
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="model_upload_data_versions",
    ),
    "list_model_upload_docker_images": _spec(
        "list_model_upload_docker_images",
        "numerapi-native",
        "numerapi",
        "List compute pickle docker image options.",
        True,
        False,
        mcp_equivalent="upload_model.list_docker_images",
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="model_upload_docker_images",
    ),
    "pipeline_status": _spec(
        "pipeline_status",
        "numerapi-native",
        "numerapi",
        "Fetch pipeline status by date.",
        False,
        False,
        default_surface="classic",
        allowed_surfaces=("classic", "signals"),
        method="pipeline_status",
    ),
    "ticker_universe": _spec(
        "ticker_universe",
        "numerapi-native",
        "numerapi",
        "Fetch accepted Signals tickers.",
        False,
        False,
        default_surface="signals",
        allowed_surfaces=("signals",),
        method="ticker_universe",
    ),
    "change_stake": _spec(
        "change_stake",
        "numerapi-native",
        "numerapi",
        "Increase or decrease stake using vendored helpers.",
        True,
        True,
        default_surface="classic",
        allowed_surfaces=SURFACES,
    ),
    "set_stake_exact": _spec(
        "set_stake_exact",
        "numerapi-native",
        "numerapi",
        "Set exact stake value for a Classic model.",
        True,
        True,
        default_surface="classic",
        allowed_surfaces=("classic",),
        method="stake_set",
    ),
    "set_bio": _spec(
        "set_bio",
        "numerapi-native",
        "numerapi",
        "Set a model bio.",
        True,
        True,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="set_bio",
    ),
    "set_link": _spec(
        "set_link",
        "numerapi-native",
        "numerapi",
        "Set a model link.",
        True,
        True,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="set_link",
    ),
    "set_submission_webhook": _spec(
        "set_submission_webhook",
        "numerapi-native",
        "numerapi",
        "Set a model submission webhook.",
        True,
        True,
        default_surface="classic",
        allowed_surfaces=SURFACES,
        method="set_submission_webhook",
    ),
}

LIST_TOURNAMENTS_QUERY = """
query ListTournaments {
  tournaments {
    id
    name
    tournament
    active
  }
}
"""

ROUND_DETAILS_QUERY = """
query RoundDetails($roundNumber: Int!, $tournament: Int!) {
  roundDetails(roundNumber: $roundNumber, tournament: $tournament) {
    roundNumber
    tournament
    status
    openTime
    closeTime
    scoreTime
    roundResolveTime
    totalSubmitted
    totalStakes
    totalPayout
  }
}
"""

API_TOKEN_QUERY = """
query ApiTokenInfoAndScopes {
  apiTokenInfo {
    accountUsername
    name
    publicId
    scopes
  }
  apiTokenScopes {
    name
    description
  }
}
"""

ADD_MODEL_MUTATION = """
mutation AddModel($name: String!, $tournament: Int!) {
  addModel(name: $name, tournament: $tournament) {
    id
    name
    tournament
  }
}
"""

MODEL_BY_ID_QUERY = """
query ModelById($modelId: ID!) {
  model(modelId: $modelId) {
    id
    name
    tournament
    computeEnabled
    submissionWebhook
    archived
    username
  }
}
"""

ACCOUNT_MODELS_QUERY = """
query AccountModels {
  account {
    models {
      id
      name
      tournament
      username
      computePickleUpload {
        id
        filename
        modelId
        assignedModelSlots
        validationStatus
        diagnosticsStatus
        triggerStatus
        insertedAt
        updatedAt
      }
    }
  }
}
"""

COMPUTE_PICKLES_QUERY = """
query ListComputePickles($id: ID, $modelId: ID, $unassigned: Boolean!) {
  computePickles(id: $id, modelId: $modelId, unassigned: $unassigned) {
    id
    filename
    modelId
    assignedModelSlots
    validationStatus
    diagnosticsStatus
    triggerStatus
    insertedAt
    updatedAt
  }
}
"""

ASSIGN_PICKLE_MUTATION = """
mutation AssignPickleToModel($modelId: String, $pickleId: ID) {
  assignPickleToModel(modelId: $modelId, pickleId: $pickleId)
}
"""

TRIGGER_PICKLE_MUTATION = """
mutation TriggerComputePickleUpload(
  $modelId: ID,
  $pickleId: ID,
  $triggerValidation: Boolean
) {
  triggerComputePickleUpload(
    modelId: $modelId,
    pickleId: $pickleId,
    triggerValidation: $triggerValidation
  ) {
    id
    modelId
    triggerStatus
    diagnosticsStatus
    updatedAt
  }
}
"""

TRIGGER_LOGS_QUERY = """
query TriggerLogs($triggerId: ID!) {
  triggerLogs(triggerId: $triggerId) {
    timestamp
    message
  }
}
"""

DIAGNOSTICS_TRIGGER_LOGS_QUERY = """
query DiagnosticsTriggerLogs($pickleId: ID!) {
  diagnosticsTriggerLogs(pickleId: $pickleId) {
    timestamp
    message
  }
}
"""

DELETE_DIAGNOSTICS_MUTATION = """
mutation DeleteDiagnostics($ids: [String!]!) {
  deleteDiagnostics(v2DiagnosticsIds: $ids)
}
"""

DIAGNOSTICS_BY_ID_QUERY = """
query DiagnosticsById($id: ID!, $modelId: ID) {
  diagnostics(id: $id, modelId: $modelId) {
    id
    status
    updatedAt
    validationCorrMean
    validationMmcMean
    validationCorrPlusMmcSharpe
  }
}
"""

LIST_SUBMISSIONS_QUERY = """
query ListSubmissions($modelId: ID) {
  submissions(modelId: $modelId) {
    id
    filename
    insertedAt
    selected
    triggerId
  }
}
"""

SUBMISSION_UPLOAD_AUTH_CLASSIC = """
query SubmissionUploadAuthClassic($filename: String!, $tournament: Int!, $modelId: ID) {
  submissionUploadAuth(filename: $filename, tournament: $tournament, modelId: $modelId) {
    accelerated
    filename
    url
  }
}
"""

SUBMISSION_UPLOAD_AUTH_SIGNALS = """
query SubmissionUploadAuthSignals($filename: String!, $tournament: Int, $modelId: ID) {
  submissionUploadSignalsAuth(filename: $filename, tournament: $tournament, modelId: $modelId) {
    accelerated
    filename
    url
  }
}
"""

CREATE_SUBMISSION_CLASSIC = """
mutation CreateSubmissionClassic(
  $filename: String!,
  $tournament: Int!,
  $modelId: ID,
  $triggerId: ID,
  $dataDatestamp: Int
) {
  createSubmission(
    filename: $filename,
    tournament: $tournament,
    modelId: $modelId,
    triggerId: $triggerId,
    source: "manual",
    dataDatestamp: $dataDatestamp
  ) {
    id
    filename
    insertedAt
    triggerId
  }
}
"""

CREATE_SUBMISSION_SIGNALS = """
mutation CreateSubmissionSignals(
  $filename: String!,
  $tournament: Int,
  $modelId: ID,
  $triggerId: ID,
  $dataDatestamp: Int
) {
  createSignalsSubmission(
    filename: $filename,
    tournament: $tournament,
    modelId: $modelId,
    triggerId: $triggerId,
    source: "manual",
    dataDatestamp: $dataDatestamp
  ) {
    id
    filename
    insertedAt
    triggerId
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List supported operations.")

    show_parser = subparsers.add_parser("show", help="Show one operation.")
    show_parser.add_argument("operation", choices=sorted(OPERATIONS))

    run_parser = subparsers.add_parser("run", help="Run or dry-run one operation.")
    run_parser.add_argument("operation", choices=sorted(OPERATIONS))
    run_parser.add_argument("--json-args", default="{}", help="JSON object of keyword args.")
    run_parser.add_argument("--dry-run", action="store_true", help="Print execution plan only.")
    run_parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required to execute write operations.",
    )
    run_parser.add_argument(
        "--auth-header",
        default=None,
        help="Override Authorization header, e.g. 'Token PUBLIC$SECRET'.",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [jsonify(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def print_json(value: Any) -> None:
    print(json.dumps(jsonify(value), indent=2, sort_keys=True, default=json_default))


def resolve_auth_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for env_name in ("NUMERAI_API_AUTH", "NUMERAI_MCP_AUTH"):
        value = os.getenv(env_name)
        if value:
            return value
    public_id = os.getenv("NUMERAI_PUBLIC_ID")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    if public_id and secret_key:
        return f"Token {public_id}${secret_key}"
    return None


def resolve_public_secret(auth_header: str | None = None) -> tuple[str | None, str | None]:
    public_id = os.getenv("NUMERAI_PUBLIC_ID")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    if public_id and secret_key:
        return public_id, secret_key
    token = resolve_auth_token(auth_header)
    if not token or not token.startswith("Token "):
        return None, None
    payload = token.removeprefix("Token ").strip()
    if "$" not in payload:
        return None, None
    public_id, secret_key = payload.split("$", 1)
    return public_id, secret_key


def build_client(surface: str, auth_header: str | None = None):
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}")
    public_id, secret_key = resolve_public_secret(auth_header)
    common = {"public_id": public_id, "secret_key": secret_key, "verbosity": "ERROR"}
    if surface == "classic":
        return NumerAPI(**common)
    if surface == "signals":
        return SignalsAPI(**common)
    return CryptoAPI(**common)


def graphql_request(
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    auth_required: bool = False,
    auth_header: str | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if auth_required:
        token = resolve_auth_token(auth_header)
        if not token:
            raise ValueError("Authentication required: set NUMERAI_API_AUTH or related env vars.")
        headers["Authorization"] = token
    response = requests.post(
        GRAPHQL_ENDPOINT,
        headers=headers,
        json={"query": query, "variables": variables or {}},
        timeout=DEFAULT_TIMEOUT,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ValueError(f"GraphQL HTTP error {response.status_code}: {response.text}") from exc
    payload = response.json()
    if "errors" in payload:
        raise ValueError(payload["errors"])
    return payload["data"]


def upload_file(url: str, file_path: str, *, headers: dict[str, str] | None = None) -> None:
    with open(file_path, "rb") as handle:
        response = requests.put(
            url,
            data=handle.read(),
            headers=headers or {},
            timeout=DEFAULT_TIMEOUT,
        )
    response.raise_for_status()


def require_write_confirmation(spec: OperationSpec, confirm_write: bool) -> None:
    if spec.write and not confirm_write:
        raise ValueError(f"{spec.name} is write-capable; rerun with --confirm-write.")


def load_json_args(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--json-args must decode to a JSON object.")
    return parsed


def validate_surface(spec: OperationSpec, args: dict[str, Any]) -> str:
    surface = args.get("surface") or spec.default_surface
    if not surface:
        raise ValueError(f"{spec.name} requires a surface.")
    if spec.allowed_surfaces and surface not in spec.allowed_surfaces:
        raise ValueError(f"{spec.name} only supports surfaces {spec.allowed_surfaces}.")
    args["surface"] = surface
    return surface


def dry_run_output(spec: OperationSpec, args: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": spec.name,
        "status": spec.status,
        "interface": spec.interface,
        "write": spec.write,
        "auth_required": spec.auth_required,
        "args": args,
        "plan": plan,
    }


def run_list_tournaments(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    plan = {"query": LIST_TOURNAMENTS_QUERY.strip(), "variables": {}}
    if dry_run:
        return dry_run_output(OPERATIONS["list_tournaments"], args, plan)
    return {
        "result": graphql_request(LIST_TOURNAMENTS_QUERY, auth_required=False, auth_header=auth_header)["tournaments"]
    }


def run_get_round_details(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    variables = {"roundNumber": args["round_number"], "tournament": args["tournament"]}
    plan = {"query": ROUND_DETAILS_QUERY.strip(), "variables": variables}
    if dry_run:
        return dry_run_output(OPERATIONS["get_round_details"], args, plan)
    return {
        "result": graphql_request(ROUND_DETAILS_QUERY, variables, auth_required=False, auth_header=auth_header)[
            "roundDetails"
        ]
    }


def run_check_api_credentials(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    plan = {"query": API_TOKEN_QUERY.strip(), "variables": {}}
    if dry_run:
        return dry_run_output(OPERATIONS["check_api_credentials"], args, plan)
    return {"result": graphql_request(API_TOKEN_QUERY, auth_required=True, auth_header=auth_header)}


def _get_account_models(auth_header: str | None) -> list[dict[str, Any]]:
    account = graphql_request(ACCOUNT_MODELS_QUERY, auth_required=True, auth_header=auth_header)["account"]
    models = account.get("models")
    if not isinstance(models, list):
        raise RuntimeError("GraphQL account.models response was not a list.")
    return models


def _find_account_model(models: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for model in models:
        if model.get("id") == model_id:
            return model
    raise RuntimeError(f"Model {model_id} was not present in authenticated account model list.")


def run_create_model(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    variables = {"name": args["name"], "tournament": args.get("tournament", 8)}
    plan = {
        "mutation": ADD_MODEL_MUTATION.strip(),
        "variables": variables,
        "verify_query": ACCOUNT_MODELS_QUERY.strip(),
    }
    if dry_run:
        return dry_run_output(OPERATIONS["create_model"], args, plan)
    data = graphql_request(ADD_MODEL_MUTATION, variables, auth_required=True, auth_header=auth_header)
    model = data["addModel"]
    verification = _find_account_model(_get_account_models(auth_header), model["id"])
    return {"result": model, "verification": verification}


def run_list_compute_pickles(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    variables = {
        "id": args.get("id"),
        "modelId": args.get("model_id"),
        "unassigned": args.get("unassigned", False),
    }
    plan = {"query": COMPUTE_PICKLES_QUERY.strip(), "variables": variables}
    if dry_run:
        return dry_run_output(OPERATIONS["list_compute_pickles"], args, plan)
    return {
        "result": graphql_request(COMPUTE_PICKLES_QUERY, variables, auth_required=True, auth_header=auth_header)[
            "computePickles"
        ]
    }


def run_assign_compute_pickle(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    variables = {"modelId": args["model_id"], "pickleId": args.get("pickle_id")}
    verify_vars = {"id": args.get("pickle_id"), "modelId": None, "unassigned": False}
    plan = {
        "mutation": ASSIGN_PICKLE_MUTATION.strip(),
        "variables": variables,
        "verify_query": {
            "account_model": ACCOUNT_MODELS_QUERY.strip(),
            "pickle_record": COMPUTE_PICKLES_QUERY.strip(),
        },
        "verify_variables": {
            "account_model_id": args["model_id"],
            "pickle_record": verify_vars,
        },
    }
    if dry_run:
        return dry_run_output(OPERATIONS["assign_compute_pickle"], args, plan)
    result = graphql_request(ASSIGN_PICKLE_MUTATION, variables, auth_required=True, auth_header=auth_header)
    verification: dict[str, Any] = {"model": _find_account_model(_get_account_models(auth_header), args["model_id"])}
    if args.get("pickle_id") is not None:
        verification["pickles"] = graphql_request(
            COMPUTE_PICKLES_QUERY,
            verify_vars,
            auth_required=True,
            auth_header=auth_header,
        )["computePickles"]
        if verification["pickles"]:
            owner_model_id = verification["pickles"][0].get("modelId")
            if owner_model_id != args["model_id"]:
                verification["warnings"] = [
                    "assigned_pickle_owner_mismatch:existing pickle belongs to a different model slot;"
                    " use a fresh compute pickle upload for the target slot instead of cross-slot reassignment"
                ]
    return {"result": result["assignPickleToModel"], "verification": verification}


def run_trigger_compute_pickle(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    variables = {
        "modelId": args.get("model_id"),
        "pickleId": args.get("pickle_id"),
        "triggerValidation": args.get("trigger_validation", False),
    }
    verify_vars = {"id": args.get("pickle_id"), "modelId": args.get("model_id"), "unassigned": False}
    plan = {
        "mutation": TRIGGER_PICKLE_MUTATION.strip(),
        "variables": variables,
        "verify_query": COMPUTE_PICKLES_QUERY.strip(),
        "verify_variables": verify_vars,
    }
    if dry_run:
        return dry_run_output(OPERATIONS["trigger_compute_pickle"], args, plan)
    result = graphql_request(TRIGGER_PICKLE_MUTATION, variables, auth_required=True, auth_header=auth_header)
    verification = graphql_request(COMPUTE_PICKLES_QUERY, verify_vars, auth_required=True, auth_header=auth_header)
    return {"result": result["triggerComputePickleUpload"], "verification": verification["computePickles"]}


def run_get_trigger_logs(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    variables = {"triggerId": args["trigger_id"]}
    plan = {"query": TRIGGER_LOGS_QUERY.strip(), "variables": variables}
    if dry_run:
        return dry_run_output(OPERATIONS["get_trigger_logs"], args, plan)
    return {
        "result": graphql_request(TRIGGER_LOGS_QUERY, variables, auth_required=True, auth_header=auth_header)[
            "triggerLogs"
        ]
    }


def run_get_diagnostics_trigger_logs(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    variables = {"pickleId": args["pickle_id"]}
    plan = {"query": DIAGNOSTICS_TRIGGER_LOGS_QUERY.strip(), "variables": variables}
    if dry_run:
        return dry_run_output(OPERATIONS["get_diagnostics_trigger_logs"], args, plan)
    return {
        "result": graphql_request(
            DIAGNOSTICS_TRIGGER_LOGS_QUERY,
            variables,
            auth_required=True,
            auth_header=auth_header,
        )["diagnosticsTriggerLogs"]
    }


def run_delete_diagnostics(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    ids = args["ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("delete_diagnostics requires ids as a non-empty list.")
    model_id = args.get("model_id")
    plan = {
        "mutation": DELETE_DIAGNOSTICS_MUTATION.strip(),
        "variables": {"ids": ids},
        "verify_query": DIAGNOSTICS_BY_ID_QUERY.strip(),
        "verify_variables": [{"id": item, "modelId": model_id} for item in ids],
    }
    if dry_run:
        return dry_run_output(OPERATIONS["delete_diagnostics"], args, plan)
    result = graphql_request(DELETE_DIAGNOSTICS_MUTATION, {"ids": ids}, auth_required=True, auth_header=auth_header)
    verification = [
        graphql_request(
            DIAGNOSTICS_BY_ID_QUERY,
            {"id": item, "modelId": model_id},
            auth_required=True,
            auth_header=auth_header,
        )["diagnostics"]
        for item in ids
    ]
    return {"result": result["deleteDiagnostics"], "verification": verification}


def run_upload_predictions(args: dict[str, Any], auth_header: str | None, dry_run: bool) -> dict[str, Any]:
    surface = args.get("surface", "classic")
    model_id = args["model_id"]
    file_path = args["file_path"]
    tournament = args.get("tournament")
    if tournament is None:
        tournament = 11 if surface == "signals" else 8 if surface == "classic" else 12
    trigger_id = args.get("trigger_id") or os.getenv("TRIGGER_ID")
    data_datestamp = args.get("data_datestamp")
    auth_query = SUBMISSION_UPLOAD_AUTH_SIGNALS if surface == "signals" else SUBMISSION_UPLOAD_AUTH_CLASSIC
    create_query = CREATE_SUBMISSION_SIGNALS if surface == "signals" else CREATE_SUBMISSION_CLASSIC
    auth_vars = {"filename": Path(file_path).name, "tournament": tournament, "modelId": model_id}
    create_vars = {
        "filename": "<UPLOADED_FILENAME>",
        "tournament": tournament,
        "modelId": model_id,
        "triggerId": trigger_id,
        "dataDatestamp": data_datestamp,
    }
    plan = {
        "auth_query": auth_query.strip(),
        "auth_variables": auth_vars,
        "upload_file_path": file_path,
        "create_mutation": create_query.strip(),
        "create_variables": create_vars,
        "verify_query": LIST_SUBMISSIONS_QUERY.strip(),
        "verify_variables": {"modelId": model_id},
    }
    if dry_run:
        return dry_run_output(OPERATIONS["upload_predictions"], args, plan)
    auth_data = graphql_request(auth_query, auth_vars, auth_required=True, auth_header=auth_header)
    upload_key = "submissionUploadSignalsAuth" if surface == "signals" else "submissionUploadAuth"
    upload_auth = auth_data[upload_key]
    headers = {"x_compute_id": os.getenv("NUMERAI_COMPUTE_ID")} if os.getenv("NUMERAI_COMPUTE_ID") else {}
    upload_file(upload_auth["url"], file_path, headers=headers)
    create_vars["filename"] = upload_auth["filename"]
    create_data = graphql_request(create_query, create_vars, auth_required=True, auth_header=auth_header)
    create_key = "createSignalsSubmission" if surface == "signals" else "createSubmission"
    submission = create_data[create_key]
    verification = graphql_request(
        LIST_SUBMISSIONS_QUERY,
        {"modelId": model_id},
        auth_required=True,
        auth_header=auth_header,
    )["submissions"]
    return {"result": submission, "verification": verification}


def run_numerapi_method(
    spec: OperationSpec,
    args: dict[str, Any],
    auth_header: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    surface = validate_surface(spec, args)
    method_name = spec.method
    if spec.name == "change_stake":
        method_name = None
    if not method_name and spec.name != "change_stake":
        raise ValueError(f"No numerapi method configured for {spec.name}.")
    call_args = dict(args)
    call_args.pop("surface", None)

    if spec.name == "list_diagnostics":
        call_args["diagnostics_id"] = None
    if spec.name == "read_diagnostics":
        if "diagnostics_id" not in call_args:
            raise ValueError("read_diagnostics requires diagnostics_id.")
    if spec.name == "change_stake":
        action = call_args.pop("action")
        plan = {
            "client_surface": surface,
            "method": f"stake_{action}",
            "kwargs": call_args,
            "verification": "wrapper receipt only",
        }
        if dry_run:
            return dry_run_output(spec, args, plan)
        client = build_client(surface, auth_header)
        if action == "increase":
            result = client.stake_increase(call_args["amount_nmr"], call_args["model_id"])
        elif action == "decrease":
            result = client.stake_decrease(call_args["amount_nmr"], call_args["model_id"])
        else:
            raise ValueError("change_stake action must be 'increase' or 'decrease'.")
        return {"result": result, "verification": "wrapper receipt only"}

    plan = {"client_surface": surface, "method": method_name, "kwargs": call_args}
    if spec.write:
        if spec.name == "upload_diagnostics":
            plan["verification_query"] = DIAGNOSTICS_BY_ID_QUERY.strip()
        elif spec.name == "create_compute_pickle_upload":
            plan["verification_query"] = COMPUTE_PICKLES_QUERY.strip()
        elif spec.name == "set_submission_webhook":
            plan["verification_query"] = MODEL_BY_ID_QUERY.strip()
    if dry_run:
        return dry_run_output(spec, args, plan)

    client = build_client(surface, auth_header)
    method = getattr(client, method_name)
    result = method(**call_args)
    verification: Any = None
    if spec.name == "upload_diagnostics":
        verification = graphql_request(
            DIAGNOSTICS_BY_ID_QUERY,
            {"id": result, "modelId": call_args["model_id"]},
            auth_required=True,
            auth_header=auth_header,
        )["diagnostics"]
    elif spec.name == "create_compute_pickle_upload":
        verification = graphql_request(
            COMPUTE_PICKLES_QUERY,
            {"id": result, "modelId": call_args.get("model_id"), "unassigned": False},
            auth_required=True,
            auth_header=auth_header,
        )["computePickles"]
    elif spec.name == "set_submission_webhook":
        verification = graphql_request(
            MODEL_BY_ID_QUERY,
            {"modelId": call_args["model_id"]},
            auth_required=False,
            auth_header=auth_header,
        )["model"]
    return {"result": result, "verification": verification}


GRAPHQL_RUNNERS: dict[str, Callable[[dict[str, Any], str | None, bool], dict[str, Any]]] = {
    "list_tournaments": run_list_tournaments,
    "get_round_details": run_get_round_details,
    "check_api_credentials": run_check_api_credentials,
    "create_model": run_create_model,
    "list_compute_pickles": run_list_compute_pickles,
    "assign_compute_pickle": run_assign_compute_pickle,
    "trigger_compute_pickle": run_trigger_compute_pickle,
    "get_trigger_logs": run_get_trigger_logs,
    "get_diagnostics_trigger_logs": run_get_diagnostics_trigger_logs,
    "delete_diagnostics": run_delete_diagnostics,
    "upload_predictions": run_upload_predictions,
}


def run_operation(
    spec: OperationSpec,
    args: dict[str, Any],
    *,
    auth_header: str | None,
    dry_run: bool,
    confirm_write: bool,
) -> dict[str, Any]:
    require_write_confirmation(spec, confirm_write)
    if spec.interface == "graphql":
        return GRAPHQL_RUNNERS[spec.name](args, auth_header, dry_run)
    return run_numerapi_method(spec, args, auth_header, dry_run)


def list_operations() -> None:
    rows = []
    for spec in OPERATIONS.values():
        rows.append(
            {
                "name": spec.name,
                "status": spec.status,
                "interface": spec.interface,
                "write": spec.write,
                "mcp_equivalent": spec.mcp_equivalent,
            }
        )
    print_json(rows)


def show_operation(name: str) -> None:
    spec = OPERATIONS[name]
    print_json(asdict(spec))


def main() -> int:
    args = parse_args()
    if args.command == "list":
        list_operations()
        return 0
    if args.command == "show":
        show_operation(args.operation)
        return 0

    spec = OPERATIONS[args.operation]
    call_args = load_json_args(args.json_args)
    result = run_operation(
        spec,
        call_args,
        auth_header=args.auth_header,
        dry_run=args.dry_run,
        confirm_write=args.confirm_write,
    )
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
