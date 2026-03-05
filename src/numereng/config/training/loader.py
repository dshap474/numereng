"""Training config loading and canonical JSON validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from numereng.config.training.contracts import TrainingConfig

_JSON_EXTENSION = ".json"
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "training_config.schema.json"


class TrainingConfigLoaderError(ValueError):
    """Raised when one training config payload fails canonical validation."""


def ensure_json_config_path(value: str, *, field_name: str = "config_path") -> str:
    """Ensure one config path points to a `.json` file."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    if not stripped.lower().endswith(_JSON_EXTENSION):
        raise ValueError(f"{field_name} must reference a .json file")
    return stripped


def ensure_json_config_uri(value: str, *, field_name: str = "config_s3_uri") -> str:
    """Ensure one config S3 URI points to a `.json` object."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    if not stripped.startswith("s3://"):
        raise ValueError(f"{field_name} must be an s3:// URI")
    if not stripped.lower().endswith(_JSON_EXTENSION):
        raise ValueError(f"{field_name} must reference a .json object")
    return stripped


def load_training_config_json(config_path: Path) -> dict[str, object]:
    """Load and validate one canonical training config JSON file."""
    if not config_path.exists():
        raise TrainingConfigLoaderError(f"training_config_file_not_found:{config_path}")

    if config_path.suffix.lower() != _JSON_EXTENSION:
        raise TrainingConfigLoaderError("training_config_json_required")

    try:
        raw_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingConfigLoaderError("training_config_json_invalid") from exc

    if not isinstance(raw_payload, dict):
        raise TrainingConfigLoaderError("training_config_not_dict")

    try:
        validated = TrainingConfig.model_validate(raw_payload)
    except ValidationError as exc:
        detail = _first_validation_error(exc)
        raise TrainingConfigLoaderError(f"training_config_schema_invalid:{detail}") from exc

    dumped = validated.model_dump(mode="python", exclude_none=True)
    return {str(key): value for key, value in dumped.items()}


def export_training_config_schema(path: Path) -> None:
    """Export canonical JSON schema for the training config contract."""
    schema = TrainingConfig.model_json_schema()
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
    "TrainingConfigLoaderError",
    "canonical_schema_path",
    "ensure_json_config_path",
    "ensure_json_config_uri",
    "export_training_config_schema",
    "load_training_config_json",
]
