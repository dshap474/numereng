"""Checked-in SageMaker runtime image catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SageMakerRuntimeImage:
    """Canonical default SageMaker image for one runtime profile."""

    runtime_profile: str
    repository: str
    tag: str
    label: str


_SAGEMAKER_RUNTIME_IMAGES: dict[str, SageMakerRuntimeImage] = {
    "standard": SageMakerRuntimeImage(
        runtime_profile="standard",
        repository="numereng-training",
        tag="sagemaker-standard-current",
        label="SageMaker standard CPU runtime",
    ),
    "lgbm-cuda": SageMakerRuntimeImage(
        runtime_profile="lgbm-cuda",
        repository="numereng-training",
        tag="sagemaker-lgbm-cuda-current",
        label="SageMaker LightGBM CUDA runtime",
    ),
}


def get_sagemaker_runtime_image(runtime_profile: str) -> SageMakerRuntimeImage:
    """Return the checked-in default SageMaker image for one runtime profile."""

    candidate = runtime_profile.strip().lower()
    image = _SAGEMAKER_RUNTIME_IMAGES.get(candidate)
    if image is None:
        raise ValueError(f"unknown_sagemaker_runtime_profile:{runtime_profile}")
    return image

