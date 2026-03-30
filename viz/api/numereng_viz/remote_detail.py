"""SSH-backed read-only detail fetches for viz experiment and run pages."""

from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from numereng.platform import load_remote_targets
from numereng.platform.remotes.contracts import SshRemoteTargetProfile
from numereng.platform.remotes.ssh import build_remote_python_command, build_ssh_command

_REMOTE_DETAIL_METHODS: tuple[str, ...] = (
    "get_experiment",
    "list_experiment_configs",
    "list_experiment_runs",
    "list_experiment_round_results",
    "list_run_jobs",
    "get_run_job",
    "list_run_job_events",
    "list_run_job_logs",
    "list_run_job_samples",
    "get_run_lifecycle",
    "get_run_manifest",
    "get_run_metrics",
    "get_run_events",
    "get_run_resources",
    "get_per_era_corr",
    "get_scoring_dashboard",
    "get_trials",
    "get_best_params",
    "get_resolved_config",
    "get_diagnostics_sources",
    "get_run_bundle",
    "get_study",
    "get_study_trials",
    "get_experiment_studies",
    "get_ensemble",
    "get_ensemble_correlations",
    "get_ensemble_artifacts",
    "get_experiment_ensembles",
    "get_experiment_doc",
    "get_run_doc",
)
_REMOTE_DETAIL_SCRIPT = f"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

from numereng_viz.services import VizService
from numereng_viz.store_adapter import VizStoreAdapter, VizStoreConfig

ALLOWED_METHODS = {list(_REMOTE_DETAIL_METHODS)!r}
try:
    request = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    method_name = request["method"]
    if method_name not in ALLOWED_METHODS:
        raise ValueError(f"unsupported remote viz detail method: {{method_name}}")

    adapter = VizStoreAdapter(
        VizStoreConfig(
            store_root=Path(request["store_root"]),
            repo_root=Path(request["repo_root"]),
        )
    )
    service = VizService(adapter)
    method = getattr(service, method_name)
    payload = method(*request.get("args", []), **request.get("kwargs", {{}}))
    if asyncio.iscoroutine(payload):
        payload = asyncio.run(payload)
except Exception as exc:
    print(
        json.dumps(
            {{
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }},
            default=str,
        )
    )
    sys.exit(0)
print(json.dumps({{"ok": True, "payload": payload}}, default=str))
""".strip()


class RemoteDetailCoordinator:
    """Run whitelisted viz detail reads against one SSH target."""

    def __init__(self, *, store_root: str | Path = ".numereng") -> None:
        self._store_root = Path(store_root).expanduser().resolve()

    def call(
        self,
        method: str,
        *,
        source_kind: str,
        source_id: str,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        target = self._resolve_target(source_kind=source_kind, source_id=source_id)
        request = json.dumps(
            {
                "method": method,
                "args": list(args),
                "kwargs": dict(kwargs or {}),
                "repo_root": target.repo_root,
                "store_root": target.store_root,
            },
            separators=(",", ":"),
        )
        encoded_request = base64.b64encode(request.encode("utf-8")).decode("ascii")
        command = build_ssh_command(
            target,
            build_remote_python_command(
                target,
                _REMOTE_DETAIL_SCRIPT,
                args=(encoded_request,),
                cwd=target.repo_root,
            ),
        )
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=target.command_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"remote viz detail command timed out for source {source_id}") from exc
        if result.returncode != 0:
            stdout = (result.stdout or "").strip()
            if stdout:
                payload = _extract_remote_json(stdout)
                if not payload.get("ok"):
                    _raise_remote_error(
                        error_type=str(payload.get("error_type") or "ValueError"),
                        message=str(payload.get("error_message") or "remote viz detail command failed"),
                    )
            stderr = (result.stderr or "").strip()
            detail = stderr or f"remote viz detail command failed for source {source_id}"
            if "FileNotFoundError" in detail:
                raise FileNotFoundError(detail)
            if "LookupError" in detail or "KeyError" in detail:
                raise LookupError(detail)
            raise ValueError(detail)

        payload = _extract_remote_json(result.stdout)
        if not payload.get("ok"):
            _raise_remote_error(
                error_type=str(payload.get("error_type") or "ValueError"),
                message=str(payload.get("error_message") or "remote viz detail command failed"),
            )
        return payload.get("payload")

    def _resolve_target(self, *, source_kind: str, source_id: str) -> SshRemoteTargetProfile:
        normalized_kind = (source_kind or "").strip() or "local"
        normalized_id = (source_id or "").strip()
        if normalized_kind != "ssh":
            raise ValueError(f"Unsupported source_kind: {source_kind}")
        if not normalized_id:
            raise ValueError("source_id is required for remote experiment detail requests")

        targets = {target.id: target for target in load_remote_targets(strict=True)}
        target = targets.get(normalized_id)
        if target is None:
            raise LookupError(f"Unknown remote source_id: {normalized_id}")
        return target


def _extract_remote_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ValueError("remote viz detail command returned no JSON")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("remote viz detail command returned invalid JSON") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("remote viz detail command returned non-object JSON")
    return payload


def _raise_remote_error(*, error_type: str, message: str) -> None:
    if error_type in {"LookupError", "KeyError"}:
        raise LookupError(message)
    if error_type == "FileNotFoundError":
        raise FileNotFoundError(message)
    raise ValueError(message)


__all__ = ["RemoteDetailCoordinator"]
