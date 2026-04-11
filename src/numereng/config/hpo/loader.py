"""HPO study config loading and canonical JSON validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from numereng.config.hpo.contracts import HpoStudyConfig, canonicalize_hpo_study_payload

_JSON_EXTENSION = ".json"
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "hpo_study_config.schema.json"


class HpoConfigLoaderError(ValueError):
    """Raised when one HPO study config payload fails canonical validation."""


def ensure_json_config_path(value: str, *, field_name: str = "study_config_path") -> str:
    """Ensure one config path points to a `.json` file."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    if not stripped.lower().endswith(_JSON_EXTENSION):
        raise ValueError(f"{field_name} must reference a .json file")
    return stripped


def load_hpo_study_config_json(config_path: Path) -> dict[str, object]:
    """Load and validate one canonical HPO study config JSON file."""
    if not config_path.exists():
        raise HpoConfigLoaderError(f"hpo_study_config_file_not_found:{config_path}")

    if config_path.suffix.lower() != _JSON_EXTENSION:
        raise HpoConfigLoaderError("hpo_study_config_json_required")

    try:
        raw_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HpoConfigLoaderError("hpo_study_config_json_invalid") from exc

    if not isinstance(raw_payload, dict):
        raise HpoConfigLoaderError("hpo_study_config_not_dict")

    try:
        validated = HpoStudyConfig.model_validate(raw_payload)
    except ValidationError as exc:
        detail = _first_validation_error(exc)
        raise HpoConfigLoaderError(f"hpo_study_config_schema_invalid:{detail}") from exc

    dumped = validated.model_dump(mode="python")
    return canonicalize_hpo_study_payload(dumped)


def export_hpo_study_config_schema(path: Path) -> None:
    """Export canonical JSON schema for the HPO study config contract."""
    schema = HpoStudyConfig.model_json_schema()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


def canonical_schema_path() -> Path:
    """Return path for the checked-in canonical JSON schema."""
    return _SCHEMA_PATH


def _first_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid payload"

    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "invalid value"))
    if not location:
        return message
    return f"{location}:{message}"


__all__ = [
    "HpoConfigLoaderError",
    "canonical_schema_path",
    "ensure_json_config_path",
    "export_hpo_study_config_schema",
    "load_hpo_study_config_json",
]
