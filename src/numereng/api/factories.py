"""Internal service and client factories for API handlers."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from numereng.api.contracts import NumeraiTournament
from numereng.features.cloud.aws import CloudAwsManagedService, CloudEc2Service
from numereng.features.cloud.modal import CloudModalService
from numereng.platform import NumeraiClient


def _create_numerai_client(*, tournament: NumeraiTournament = "classic") -> NumeraiClient:
    return NumeraiClient(tournament=tournament)


def _create_cloud_ec2_service() -> CloudEc2Service:
    return CloudEc2Service()


def _create_cloud_aws_managed_service() -> CloudAwsManagedService:
    return CloudAwsManagedService()


def _create_cloud_modal_service() -> CloudModalService:
    return CloudModalService()


def _default_dataset_dest_path(filename: str) -> str:
    posix_path = PurePosixPath(filename)
    return str(Path(".numereng") / "datasets" / Path(*posix_path.parts))


__all__ = [
    "_create_cloud_aws_managed_service",
    "_create_cloud_ec2_service",
    "_create_cloud_modal_service",
    "_create_numerai_client",
    "_default_dataset_dest_path",
]
