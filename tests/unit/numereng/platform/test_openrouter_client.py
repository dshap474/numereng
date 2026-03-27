from __future__ import annotations

import io
import json
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

import numereng.platform.clients.openrouter as openrouter_module
from numereng.platform.clients.openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterStreamEvent,
    active_model_source,
    load_openrouter_config,
)
from numereng.platform.errors import OpenRouterClientError


class _JsonResponse:
    def __init__(self, body: object) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _StreamResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _header_map(request: Any) -> dict[str, str]:
    return {key.lower(): value for key, value in request.header_items()}


def test_openrouter_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(OpenRouterClientError, match="openrouter_api_key_missing"):
        OpenRouterClient()


def test_load_openrouter_config_reads_active_source_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "active-model.py"
    config_path.write_text(
        'ACTIVE_MODEL_SOURCE = "codex-exec"\nACTIVE_MODEL = "openai/gpt-4o-mini"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(openrouter_module, "_ACTIVE_MODEL_PATH", config_path)

    assert load_openrouter_config() == OpenRouterConfig(
        active_model_source="codex-exec",
        active_model="openai/gpt-4o-mini",
    )
    assert active_model_source() == "codex-exec"


def test_openrouter_client_list_models_uses_env_key_and_query_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _JsonResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _JsonResponse({"data": [{"id": "openai/gpt-4o-mini"}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "key-123")
    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(
        referer="https://numereng.local",
        title="numereng",
        categories=("research", "internal"),
        timeout_seconds=12.5,
    )

    response = client.list_models(filters={"category": "chat", "use_rss": True})

    assert response == {"data": [{"id": "openai/gpt-4o-mini"}]}
    assert captured["timeout"] == 12.5
    assert captured["request"].full_url == ("https://openrouter.ai/api/v1/models?category=chat&use_rss=true")
    headers = _header_map(captured["request"])
    assert headers["authorization"] == "Bearer key-123"
    assert headers["http-referer"] == "https://numereng.local"
    assert headers["x-openrouter-title"] == "numereng"
    assert headers["x-openrouter-categories"] == "research,internal"


def test_openrouter_client_chat_completions_posts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _JsonResponse:
        _ = timeout
        captured["request"] = request
        return _JsonResponse({"id": "chatcmpl-1", "choices": []})

    monkeypatch.setattr(
        openrouter_module,
        "load_openrouter_config",
        lambda: OpenRouterConfig(active_model_source="openrouter", active_model="nvidia/default-model"),
    )
    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="key-123")

    response = client.chat_completions(
        payload={
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert response["id"] == "chatcmpl-1"
    assert captured["request"].full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert json.loads(captured["request"].data.decode("utf-8")) == {
        "model": "nvidia/default-model",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_openrouter_client_explicit_model_overrides_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _JsonResponse:
        _ = timeout
        captured["request"] = request
        return _JsonResponse({"id": "chatcmpl-1", "choices": []})

    monkeypatch.setattr(
        openrouter_module,
        "load_openrouter_config",
        lambda: OpenRouterConfig(active_model_source="openrouter", active_model="nvidia/default-model"),
    )
    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="key-123")

    client.chat_completions(
        payload={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert json.loads(captured["request"].data.decode("utf-8"))["model"] == "openai/gpt-4o-mini"


def test_openrouter_client_stream_chat_completions_parses_sse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _StreamResponse:
        _ = (request, timeout)
        return _StreamResponse(
            [
                b": OPENROUTER PROCESSING\n",
                b"\n",
                b'data: {"id":"chunk-1","choices":[{"delta":{"content":"hi"}}]}\n',
                b"\n",
                b'data: {"id":"chunk-1","choices":[],"usage":{"total_tokens":3}}\n',
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ]
        )

    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="key-123")

    events = list(client.stream_chat_completions(payload={"model": "openai/gpt-4o-mini", "stream": True}))

    assert events == [
        OpenRouterStreamEvent(
            event=None,
            data={"id": "chunk-1", "choices": [{"delta": {"content": "hi"}}]},
        ),
        OpenRouterStreamEvent(
            event=None,
            data={"id": "chunk-1", "choices": [], "usage": {"total_tokens": 3}},
        ),
    ]


def test_openrouter_client_stream_responses_preserves_event_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _StreamResponse:
        _ = (request, timeout)
        return _StreamResponse(
            [
                b"event: response.completed\n",
                b'data: {"id":"resp-1","status":"completed"}\n',
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ]
        )

    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="key-123")

    events = list(client.stream_responses(payload={"model": "openai/gpt-4o-mini", "input": "hi"}))

    assert events == [
        OpenRouterStreamEvent(
            event="response.completed",
            data={"id": "resp-1", "status": "completed"},
        )
    ]


def test_openrouter_client_http_errors_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _JsonResponse:
        _ = (request, timeout)
        raise HTTPError(
            url="https://openrouter.ai/api/v1/models",
            code=429,
            msg="Too Many Requests",
            hdrs=Message(),
            fp=io.BytesIO(b'{"error":{"message":"Rate limited"}}'),
        )

    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="key-123")

    with pytest.raises(OpenRouterClientError, match="openrouter_http_error:429:Rate limited"):
        client.list_models()


def test_openrouter_client_stream_errors_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    error_chunk = (
        b'data: {"error":{"code":503,"message":"No providers available"},"choices":[{"finish_reason":"error"}]}\n'
    )

    def fake_urlopen(request: Any, timeout: float) -> _StreamResponse:
        _ = (request, timeout)
        return _StreamResponse([error_chunk, b"\n"])

    monkeypatch.setattr(openrouter_module, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="key-123")

    with pytest.raises(OpenRouterClientError, match="openrouter_stream_error:503:No providers available"):
        list(client.stream_chat_completions(payload={"model": "openai/gpt-4o-mini", "stream": True}))
