"""Thin wrapper around the OpenRouter HTTP API."""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from numereng.platform.errors import OpenRouterClientError

JsonDict = dict[str, Any]
_INFERENCE_PATHS = {"/chat/completions", "/completions", "/responses"}
_ACTIVE_MODEL_PATH = Path(__file__).resolve().parents[2] / "config" / "openrouter" / "active-model.py"
OpenRouterModelSource = Literal["codex-exec", "openrouter"]


@dataclass(frozen=True)
class OpenRouterConfig:
    """Repository-local OpenRouter runtime settings."""

    active_model_source: OpenRouterModelSource
    active_model: str | None


def load_openrouter_config() -> OpenRouterConfig:
    """Load the repository-local OpenRouter model and compute-source settings."""
    if not _ACTIVE_MODEL_PATH.exists():
        return OpenRouterConfig(active_model_source="codex-exec", active_model=None)

    spec = importlib.util.spec_from_file_location("numereng_openrouter_active_model", _ACTIVE_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise OpenRouterClientError("openrouter_config_load_failed")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise OpenRouterClientError("openrouter_config_load_failed") from exc

    source = getattr(module, "ACTIVE_MODEL_SOURCE", "codex-exec")
    if source not in {"codex-exec", "openrouter"}:
        raise OpenRouterClientError("openrouter_active_model_source_invalid")

    active_model = getattr(module, "ACTIVE_MODEL", None)
    if active_model is None:
        return OpenRouterConfig(active_model_source=source, active_model=None)
    if not isinstance(active_model, str) or not active_model.strip():
        raise OpenRouterClientError("openrouter_active_model_invalid")
    return OpenRouterConfig(active_model_source=source, active_model=active_model.strip())


def active_model_source() -> OpenRouterModelSource:
    """Return the configured planner/model compute source."""
    return load_openrouter_config().active_model_source


@dataclass(frozen=True)
class OpenRouterStreamEvent:
    """One parsed SSE event from an OpenRouter streaming response."""

    event: str | None
    data: Any


class OpenRouterClient:
    """Minimal OpenRouter adapter over the documented HTTP endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        referer: str | None = None,
        title: str | None = None,
        categories: str | Sequence[str] | None = None,
        default_model: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise OpenRouterClientError("openrouter_api_key_missing")

        self._api_key = resolved_api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        resolved_config = load_openrouter_config()
        self._default_model = default_model if default_model is not None else resolved_config.active_model
        self._extra_headers = dict(extra_headers or {})

        default_headers: dict[str, str] = {}
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-OpenRouter-Title"] = title
        if categories:
            if isinstance(categories, str):
                categories_value = categories
            else:
                categories_value = ",".join(categories)
            default_headers["X-OpenRouter-Categories"] = categories_value
        self._default_headers = default_headers

    def request(
        self,
        *,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JsonDict:
        request = self._build_request(
            method=method,
            path=path,
            payload=payload,
            params=params,
            headers=headers,
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            raise self._map_http_error(exc) from exc
        except URLError as exc:
            raise OpenRouterClientError("openrouter_request_failed") from exc

        decoded = self._decode_json(body, error_code="openrouter_response_invalid_json")
        if not isinstance(decoded, dict):
            raise OpenRouterClientError("openrouter_response_invalid_json")
        return decoded

    def stream(
        self,
        *,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[OpenRouterStreamEvent]:
        request = self._build_request(
            method=method,
            path=path,
            payload=payload,
            params=params,
            headers=headers,
        )

        try:
            response = urlopen(request, timeout=self._timeout_seconds)
        except HTTPError as exc:
            raise self._map_http_error(exc) from exc
        except URLError as exc:
            raise OpenRouterClientError("openrouter_request_failed") from exc

        return self._iter_sse(response)

    def chat_completions(self, *, payload: Mapping[str, Any]) -> JsonDict:
        return self.request(method="POST", path="/chat/completions", payload=payload)

    def stream_chat_completions(self, *, payload: Mapping[str, Any]) -> Iterator[OpenRouterStreamEvent]:
        return self.stream(method="POST", path="/chat/completions", payload=payload)

    def completions(self, *, payload: Mapping[str, Any]) -> JsonDict:
        return self.request(method="POST", path="/completions", payload=payload)

    def stream_completions(self, *, payload: Mapping[str, Any]) -> Iterator[OpenRouterStreamEvent]:
        return self.stream(method="POST", path="/completions", payload=payload)

    def responses(self, *, payload: Mapping[str, Any]) -> JsonDict:
        return self.request(method="POST", path="/responses", payload=payload)

    def stream_responses(self, *, payload: Mapping[str, Any]) -> Iterator[OpenRouterStreamEvent]:
        return self.stream(method="POST", path="/responses", payload=payload)

    def list_models(self, *, filters: Mapping[str, object] | None = None) -> JsonDict:
        return self.request(method="GET", path="/models", params=filters)

    def list_user_models(self, *, filters: Mapping[str, object] | None = None) -> JsonDict:
        return self.request(method="GET", path="/models/user", params=filters)

    def get_model_endpoints(self, *, author: str, slug: str) -> JsonDict:
        return self.request(method="GET", path=f"/models/{author}/{slug}/endpoints")

    def get_generation(self, *, generation_id: str) -> JsonDict:
        return self.request(method="GET", path="/generation", params={"id": generation_id})

    def get_key(self) -> JsonDict:
        return self.request(method="GET", path="/key")

    def _build_request(
        self,
        *,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        params: Mapping[str, object] | None,
        headers: Mapping[str, str] | None,
    ) -> Request:
        request_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._default_headers,
            **self._extra_headers,
            **dict(headers or {}),
        }
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(self._prepare_payload(path=path, payload=payload)).encode("utf-8")
        return Request(
            url=self._build_url(path=path, params=params),
            data=data,
            headers=request_headers,
            method=method.upper(),
        )

    def _prepare_payload(self, *, path: str, payload: Mapping[str, Any]) -> JsonDict:
        normalized_payload = dict(payload)
        if path in _INFERENCE_PATHS and "model" not in normalized_payload:
            if self._default_model is None:
                raise OpenRouterClientError("openrouter_active_model_missing")
            normalized_payload["model"] = self._default_model
        return normalized_payload

    def _build_url(self, *, path: str, params: Mapping[str, object] | None) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        base_url = f"{self._base_url}{normalized_path}"
        if not params:
            return base_url
        query = urlencode(self._normalize_params(params), doseq=True)
        if not query:
            return base_url
        return f"{base_url}?{query}"

    def _normalize_params(self, params: Mapping[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                normalized[key] = "true" if value else "false"
                continue
            if isinstance(value, (list, tuple)):
                normalized[key] = [self._normalize_param_item(item) for item in value]
                continue
            normalized[key] = self._normalize_param_item(value)
        return normalized

    def _normalize_param_item(self, value: object) -> object:
        if isinstance(value, bool):
            return "true" if value else "false"
        return value

    def _iter_sse(self, response: Any) -> Iterator[OpenRouterStreamEvent]:
        def iterator() -> Iterator[OpenRouterStreamEvent]:
            with response:
                event_name: str | None = None
                data_lines: list[str] = []
                for raw_line in response:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line:
                        event = self._finalize_sse_event(event_name=event_name, data_lines=data_lines)
                        if event is not None:
                            yield event
                        event_name = None
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line.removeprefix("event:").strip() or None
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line.removeprefix("data:").lstrip())
                event = self._finalize_sse_event(event_name=event_name, data_lines=data_lines)
                if event is not None:
                    yield event

        return iterator()

    def _finalize_sse_event(
        self,
        *,
        event_name: str | None,
        data_lines: list[str],
    ) -> OpenRouterStreamEvent | None:
        if not data_lines:
            return None
        raw_payload = "\n".join(data_lines)
        if raw_payload == "[DONE]":
            return None

        payload = self._decode_json(raw_payload.encode("utf-8"), error_code="openrouter_stream_invalid_json")
        if isinstance(payload, dict) and "error" in payload:
            error = payload.get("error")
            message = "openrouter_stream_error"
            if isinstance(error, dict):
                code = error.get("code")
                detail = error.get("message")
                if code is not None:
                    message = f"{message}:{code}"
                if detail:
                    message = f"{message}:{detail}"
            raise OpenRouterClientError(message)
        return OpenRouterStreamEvent(event=event_name, data=payload)

    def _decode_json(self, body: bytes, *, error_code: str) -> Any:
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenRouterClientError(error_code) from exc

    def _map_http_error(self, exc: HTTPError) -> OpenRouterClientError:
        raw_body = exc.read()
        message = f"openrouter_http_error:{exc.code}"
        if raw_body:
            try:
                payload = self._decode_json(raw_body, error_code="openrouter_response_invalid_json")
            except OpenRouterClientError:
                return OpenRouterClientError(message)
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    detail = error.get("message")
                    if detail:
                        message = f"{message}:{detail}"
        return OpenRouterClientError(message)
