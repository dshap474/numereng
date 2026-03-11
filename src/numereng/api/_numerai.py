"""Numerai API handlers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from numereng.api.contracts import (
    NumeraiCurrentRoundRequest,
    NumeraiCurrentRoundResponse,
    NumeraiDatasetDownloadRequest,
    NumeraiDatasetDownloadResponse,
    NumeraiDatasetListRequest,
    NumeraiDatasetListResponse,
    NumeraiModelsRequest,
    NumeraiModelsResponse,
)
from numereng.platform.errors import ForumScraperError, NumeraiClientError, PackageError


def list_numerai_datasets(
    request: NumeraiDatasetListRequest | None = None,
) -> NumeraiDatasetListResponse:
    """List available Numerai datasets for the configured account."""
    from numereng import api as api_module

    list_request = NumeraiDatasetListRequest() if request is None else request
    try:
        datasets = api_module._create_numerai_client(tournament=list_request.tournament).list_datasets(
            round_num=list_request.round_num
        )
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return NumeraiDatasetListResponse(datasets=datasets)


def download_numerai_dataset(request: NumeraiDatasetDownloadRequest) -> NumeraiDatasetDownloadResponse:
    """Download one Numerai dataset file."""
    from numereng import api as api_module

    resolved_dest_path = request.dest_path or api_module._default_dataset_dest_path(request.filename)
    try:
        path = api_module._create_numerai_client(tournament=request.tournament).download_dataset(
            filename=request.filename,
            dest_path=resolved_dest_path,
            round_num=request.round_num,
        )
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return NumeraiDatasetDownloadResponse(path=path)


def list_numerai_models(request: NumeraiModelsRequest | None = None) -> NumeraiModelsResponse:
    """List account model name to model-id mapping."""
    from numereng import api as api_module

    models_request = NumeraiModelsRequest() if request is None else request
    try:
        models = api_module._create_numerai_client(tournament=models_request.tournament).get_models()
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return NumeraiModelsResponse(models=models)


def get_numerai_current_round(request: NumeraiCurrentRoundRequest | None = None) -> NumeraiCurrentRoundResponse:
    """Return the current Numerai round if available."""
    from numereng import api as api_module

    round_request = NumeraiCurrentRoundRequest() if request is None else request
    try:
        round_num = api_module._create_numerai_client(tournament=round_request.tournament).get_current_round()
    except NumeraiClientError as exc:
        raise PackageError(str(exc)) from exc
    return NumeraiCurrentRoundResponse(round_num=round_num)


class NumeraiForumScrapeResponse(BaseModel):
    """Public API response for forum scrape runs."""

    output_dir: str
    posts_dir: str
    index_path: str
    manifest_path: str
    state_path: str
    mode: Literal["incremental", "full"]
    pages_fetched: int
    fetched_posts: int
    new_posts: int
    total_posts: int
    latest_post_id: int | None
    oldest_post_id: int | None
    started_at: str
    completed_at: str


def scrape_numerai_forum(
    *,
    output_dir: str = "docs/numerai/forum",
    state_path: str | None = None,
    full_refresh: bool = False,
) -> NumeraiForumScrapeResponse:
    """Scrape all visible forum posts and write markdown outputs."""
    from numereng import api as api_module

    try:
        payload = api_module.scrape_forum_posts(
            output_dir=output_dir,
            state_path=state_path,
            full_refresh=full_refresh,
        )
    except ForumScraperError as exc:
        raise PackageError(str(exc)) from exc
    return NumeraiForumScrapeResponse.model_validate(payload)


__all__ = [
    "download_numerai_dataset",
    "get_numerai_current_round",
    "scrape_numerai_forum",
    "list_numerai_datasets",
    "list_numerai_models",
    "NumeraiForumScrapeResponse",
]
