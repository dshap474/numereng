from __future__ import annotations

from typing import Any

import pytest

import numereng.platform.numerai_client as numerai_client_module
from numereng.platform.errors import NumeraiClientError
from numereng.platform.numerai_client import NumeraiClient


class _SuccessfulNumerAPI:
    def __init__(self, **_: Any) -> None:
        pass

    def list_datasets(self, round_num: int | None = None) -> list[str]:
        return [f"round_{round_num}"] if round_num is not None else ["default"]

    def download_dataset(
        self,
        filename: str,
        dest_path: str | None = None,
        round_num: int | None = None,
    ) -> str:
        _ = round_num
        return dest_path or filename

    def get_models(self) -> dict[object, object]:
        return {"main": "model-1", 22: 99}

    def upload_predictions(self, *, file_path: str, model_id: str) -> str:
        _ = (file_path, model_id)
        return "submission-1"

    def get_current_round(self) -> int:
        return 555

    def model_upload(
        self,
        *,
        file_path: str,
        model_id: str,
        data_version: str | None = None,
        docker_image: str | None = None,
    ) -> str:
        _ = (file_path, model_id, data_version, docker_image)
        return "pickle-1"

    def model_upload_data_versions(self) -> list[str]:
        return ["v5.2"]

    def model_upload_docker_images(self) -> list[str]:
        return ["ghcr.io/numerai/numerai-inference:py3.11"]


class _FailingUploadNumerAPI:
    def __init__(self, **_: Any) -> None:
        pass

    def upload_predictions(self, *, file_path: str, model_id: str) -> str:
        _ = (file_path, model_id)
        raise RuntimeError("boom")

    def model_upload(
        self,
        *,
        file_path: str,
        model_id: str,
        data_version: str | None = None,
        docker_image: str | None = None,
    ) -> str:
        _ = (file_path, model_id, data_version, docker_image)
        raise RuntimeError("boom")


class _SignalsNumerAPI(_SuccessfulNumerAPI):
    def list_datasets(self, round_num: int | None = None) -> list[str]:
        _ = round_num
        return ["signals/v2.1/live.parquet"]


class _CryptoNumerAPI(_SuccessfulNumerAPI):
    def list_datasets(self, round_num: int | None = None) -> list[str]:
        _ = round_num
        return ["crypto/v2.0/live.parquet"]


def test_numerai_client_get_models_normalizes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numerai_client_module, "NumerAPI", _SuccessfulNumerAPI)
    client = NumeraiClient(public_id="abc", secret_key="def", show_progress_bars=False)

    assert client.get_models() == {"main": "model-1", "22": "99"}
    assert client.get_current_round() == 555


def test_numerai_client_list_datasets_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numerai_client_module, "NumerAPI", _SuccessfulNumerAPI)
    client = NumeraiClient(show_progress_bars=False)

    assert client.list_datasets(round_num=2) == ["round_2"]


def test_numerai_client_upload_error_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numerai_client_module, "NumerAPI", _FailingUploadNumerAPI)
    client = NumeraiClient(show_progress_bars=False)

    with pytest.raises(NumeraiClientError, match="numerai_upload_predictions_failed"):
        client.upload_predictions(file_path="preds.csv", model_id="model-1")


def test_numerai_client_model_upload_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numerai_client_module, "NumerAPI", _SuccessfulNumerAPI)
    client = NumeraiClient(show_progress_bars=False)

    assert client.model_upload(file_path="model.pkl", model_id="model-1", data_version="v5.2") == "pickle-1"
    assert client.model_upload_data_versions() == ["v5.2"]
    assert client.model_upload_docker_images() == ["ghcr.io/numerai/numerai-inference:py3.11"]


def test_numerai_client_model_upload_error_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numerai_client_module, "NumerAPI", _FailingUploadNumerAPI)
    client = NumeraiClient(show_progress_bars=False)

    with pytest.raises(NumeraiClientError, match="numerai_model_upload_failed"):
        client.model_upload(file_path="model.pkl", model_id="model-1")


def test_numerai_client_signals_tournament(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numerai_client_module, "SignalsAPI", _SignalsNumerAPI)
    client = NumeraiClient(tournament="signals", show_progress_bars=False)

    assert client.list_datasets() == ["signals/v2.1/live.parquet"]


def test_numerai_client_crypto_tournament(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numerai_client_module, "CryptoAPI", _CryptoNumerAPI)
    client = NumeraiClient(tournament="crypto", show_progress_bars=False)

    assert client.list_datasets() == ["crypto/v2.0/live.parquet"]


def test_numerai_client_invalid_tournament() -> None:
    with pytest.raises(NumeraiClientError, match="numerai_tournament_not_supported"):
        NumeraiClient(tournament="invalid", show_progress_bars=False)  # type: ignore[arg-type]
