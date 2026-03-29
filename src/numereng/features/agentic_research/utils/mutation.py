"""Mutation-program helpers for agentic research."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from numereng.config.training import TrainingConfig
from numereng.features.agentic_research.utils.programs import (
    render_prompt_template,
    render_validation_feedback_block,
)
from numereng.features.agentic_research.utils.types import (
    MutationChange,
    MutationProposal,
    ResearchProgramDefinition,
)
from numereng.features.experiments import (
    ExperimentRecord,
    ExperimentReport,
    ExperimentReportRow,
)
from numereng.features.training.run_store import compute_config_hash

_FILENAME_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)
_DEFAULT_MUTATION_ALLOWED_PATHS = (
    "data.feature_set",
    "data.target_col",
    "data.scoring_targets",
    "data.target_horizon",
    "preprocessing.nan_missing_all_twos",
    "preprocessing.missing_value",
    "model.type",
    "model.device",
    "model.params.*",
    "model.x_groups",
    "model.data_needed",
    "model.target_transform.*",
    "training.engine.profile",
    "training.engine.window_size_eras",
    "training.engine.embargo_eras",
    "training.resources.parallel_folds",
    "training.resources.max_threads_per_worker",
)


@dataclass(frozen=True)
class ParentConfigSelection:
    """The currently selected parent config and the metrics tied to it."""

    config_filename: str
    config_payload: dict[str, object]
    run_id: str | None
    bmc_last_200_eras_mean: float | None
    bmc_mean: float | None
    corr_mean: float | None
    source: str


@dataclass(frozen=True)
class MaterializedMutationConfig:
    """One validated child config derived from a parent config."""

    filename: str
    payload: dict[str, object]
    change_set: list[dict[str, object]]


def render_mutation_prompt(
    *,
    definition: ResearchProgramDefinition,
    parent: ParentConfigSelection,
    recent_round_summaries: list[dict[str, object]],
    validation_feedback: str | None,
) -> str:
    """Render the compact numerai config-mutation prompt."""
    return render_prompt_template(
        definition.prompt_template,
        {
            "CONTEXT_JSON": json.dumps(
                {
                    "program": {
                        "id": definition.program_id,
                        "title": definition.title,
                        "description": definition.description,
                        "planner_contract": definition.planner_contract,
                    },
                    "metric_policy": {
                        "primary": definition.metric_policy.primary,
                        "tie_break": definition.metric_policy.tie_break,
                        "sanity_checks": list(definition.metric_policy.sanity_checks),
                    },
                    "config_policy": {
                        "allowed_paths": list(definition.config_policy.allowed_paths),
                        "min_changes": definition.config_policy.min_changes,
                        "max_changes": definition.config_policy.max_changes,
                        "max_candidate_configs": definition.config_policy.max_candidate_configs,
                    },
                    "parent_config": {
                        "filename": parent.config_filename,
                        "source": parent.source,
                        "run_id": parent.run_id,
                        "metrics": {
                            "bmc_last_200_eras_mean": parent.bmc_last_200_eras_mean,
                            "bmc_mean": parent.bmc_mean,
                            "corr_mean": parent.corr_mean,
                        },
                        "payload": parent.config_payload,
                    },
                    "recent_round_summaries": recent_round_summaries[-3:],
                },
                indent=2,
                sort_keys=True,
            ),
            "VALIDATION_FEEDBACK_BLOCK": render_validation_feedback_block(validation_feedback),
        },
        source=definition.source_path or definition.program_id,
    )


def select_parent_config(
    *,
    root: Path,
    experiment: ExperimentRecord,
    report: ExperimentReport | None,
    config_dirs: list[Path],
) -> ParentConfigSelection:
    """Pick the best available parent config for the next mutation step."""
    if report is not None:
        row = _best_row(report.rows)
        if row is not None:
            selected = _selection_from_run(root=root, experiment=experiment, row=row)
            if selected is not None:
                return selected
    selected = _selection_from_config_dirs(config_dirs=config_dirs)
    if selected is not None:
        return selected
    raise ValueError(f"agentic_research_parent_config_missing:{experiment.experiment_id}")


def parse_mutation_response(
    response_text: str,
    *,
    definition: ResearchProgramDefinition | None = None,
) -> MutationProposal:
    """Parse the mutation planner text response into a validated proposal."""
    stripped = _strip_code_fence(response_text).strip()
    if not stripped:
        raise ValueError("agentic_research_mutation_response_empty")
    rationale, change_lines = _split_sections(stripped)
    allowed_paths = (
        definition.config_policy.allowed_paths if definition is not None else _DEFAULT_MUTATION_ALLOWED_PATHS
    )
    changes: list[MutationChange] = []
    seen_paths: set[str] = set()
    for raw_line in change_lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if "=" not in line:
            raise ValueError(f"agentic_research_mutation_change_invalid:{raw_line}")
        path_text, value_text = line.split("=", 1)
        path = path_text.strip()
        if not path.startswith("config."):
            raise ValueError(f"agentic_research_mutation_path_invalid:{path}")
        config_path = path.removeprefix("config.").strip()
        if not config_path:
            raise ValueError("agentic_research_mutation_path_missing")
        if not _mutation_path_allowed(config_path, allowed_paths=allowed_paths):
            raise ValueError(f"agentic_research_mutation_path_not_allowed:{config_path}")
        if config_path in seen_paths:
            raise ValueError(f"agentic_research_mutation_path_duplicate:{config_path}")
        try:
            value = json.loads(value_text.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"agentic_research_mutation_value_invalid:{config_path}") from exc
        seen_paths.add(config_path)
        changes.append(MutationChange(path=config_path, value=value))
    if not changes:
        raise ValueError("agentic_research_mutation_changes_missing")
    if definition is not None:
        min_changes = definition.config_policy.min_changes
        max_changes = definition.config_policy.max_changes
        if min_changes is not None and len(changes) < min_changes:
            raise ValueError("agentic_research_mutation_changes_too_few")
        if max_changes is not None and len(changes) > max_changes:
            raise ValueError("agentic_research_mutation_changes_too_many")
    return MutationProposal(rationale=rationale, changes=tuple(changes))


def materialize_mutation_config(
    *,
    round_label: str,
    config_dir: Path,
    parent: ParentConfigSelection,
    proposal: MutationProposal,
    comparison_dirs: list[Path],
) -> MaterializedMutationConfig:
    """Clone a parent config, apply mutations, validate it, and derive a filename."""
    payload = deepcopy(parent.config_payload)
    for change in proposal.changes:
        _assign_dotted_path(payload, change.path.split("."), deepcopy(change.value))
    validated = TrainingConfig.model_validate(payload).model_dump(mode="python", exclude_none=True)
    identity_hash = compute_config_hash(_identity_payload(validated))
    existing_hashes = existing_identity_hashes(config_dirs=[config_dir, *comparison_dirs])
    if identity_hash in existing_hashes:
        raise ValueError(f"agentic_research_candidate_duplicate:{identity_hash[:12]}")
    filename = build_candidate_filename(
        round_label=round_label,
        config_dir=config_dir,
        parent_filename=parent.config_filename,
        proposal=proposal,
    )
    return MaterializedMutationConfig(
        filename=filename,
        payload=validated,
        change_set=[{"path": item.path, "value": deepcopy(item.value)} for item in proposal.changes],
    )


def write_materialized_config(*, config_dir: Path, candidate: MaterializedMutationConfig) -> Path:
    """Persist one materialized child config to disk."""
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / candidate.filename
    path.write_text(json.dumps(candidate.payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_candidate_filename(
    *,
    round_label: str,
    config_dir: Path,
    parent_filename: str,
    proposal: MutationProposal,
) -> str:
    """Derive a deterministic child-config filename from the parent and change set."""
    parent_slug = _slugify(Path(parent_filename).stem)
    change_slug = "__".join(_change_slug(change) for change in proposal.changes) or "mutation"
    stem = f"{round_label}_{parent_slug}__{change_slug}"[:160].rstrip("_-")
    candidate = f"{stem}.json"
    suffix = 2
    while (config_dir / candidate).exists():
        candidate = f"{stem}_{suffix}.json"
        suffix += 1
    return candidate


def existing_identity_hashes(*, config_dirs: list[Path]) -> set[str]:
    """Load normalized identity hashes for existing configs across the lineage."""
    hashes: set[str] = set()
    for config_dir in config_dirs:
        if not config_dir.is_dir():
            continue
        for config_path in sorted(config_dir.glob("*.json")):
            validated = _load_validated_config(config_path)
            if validated is None:
                continue
            hashes.add(compute_config_hash(_identity_payload(validated)))
    return hashes


def _selection_from_run(
    *,
    root: Path,
    experiment: ExperimentRecord,
    row: ExperimentReportRow,
) -> ParentConfigSelection | None:
    run_path = root / "runs" / row.run_id / "run.json"
    try:
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    config_info = manifest.get("config")
    if not isinstance(config_info, dict):
        return None
    config_path_value = config_info.get("path")
    if not isinstance(config_path_value, str) or not config_path_value.strip():
        return None
    config_path = Path(config_path_value).expanduser().resolve()
    if not config_path.is_file():
        return None
    return _selection_from_config_path(
        config_path=config_path,
        run_id=row.run_id,
        bmc_last_200_eras_mean=row.bmc_last_200_eras_mean,
        bmc_mean=row.bmc_mean,
        corr_mean=row.corr_mean,
        source=f"best_run:{experiment.experiment_id}:{row.run_id}",
    )


def _selection_from_config_dirs(config_dirs: list[Path]) -> ParentConfigSelection | None:
    for config_dir in config_dirs:
        if not config_dir.is_dir():
            continue
        for config_path in sorted(config_dir.glob("*.json"), key=lambda item: item.name, reverse=True):
            selection = _selection_from_config_path(
                config_path=config_path,
                run_id=None,
                bmc_last_200_eras_mean=None,
                bmc_mean=None,
                corr_mean=None,
                source=f"seed_config:{config_path}",
            )
            if selection is not None:
                return selection
    return None


def _selection_from_config_path(
    *,
    config_path: Path,
    run_id: str | None,
    bmc_last_200_eras_mean: float | None,
    bmc_mean: float | None,
    corr_mean: float | None,
    source: str,
) -> ParentConfigSelection | None:
    validated = _load_validated_config(config_path)
    if validated is None:
        return None
    return ParentConfigSelection(
        config_filename=config_path.name,
        config_payload=validated,
        run_id=run_id,
        bmc_last_200_eras_mean=bmc_last_200_eras_mean,
        bmc_mean=bmc_mean,
        corr_mean=corr_mean,
        source=source,
    )


def _best_row(rows: tuple[ExperimentReportRow, ...]) -> ExperimentReportRow | None:
    best: ExperimentReportRow | None = None
    best_metrics: tuple[float, float, float] | None = None
    for row in rows:
        if row.bmc_last_200_eras_mean is None:
            continue
        row_metrics = (
            row.bmc_last_200_eras_mean,
            row.bmc_mean or float("-inf"),
            row.corr_mean or float("-inf"),
        )
        if best_metrics is None or row_metrics > best_metrics:
            best = row
            best_metrics = row_metrics
    return best


def _split_sections(response_text: str) -> tuple[str, list[str]]:
    rationale_lines: list[str] = []
    change_lines: list[str] = []
    section: str | None = None
    for raw_line in response_text.splitlines():
        stripped = raw_line.strip()
        upper = stripped.upper()
        if upper.startswith("RATIONALE:"):
            section = "rationale"
            remainder = raw_line.split(":", 1)[1].strip()
            if remainder:
                rationale_lines.append(remainder)
            continue
        if upper.startswith("CHANGES:"):
            section = "changes"
            remainder = raw_line.split(":", 1)[1].strip()
            if remainder:
                change_lines.append(remainder)
            continue
        if section == "rationale":
            rationale_lines.append(raw_line.rstrip())
            continue
        if section == "changes":
            change_lines.append(raw_line.rstrip())
    rationale = "\n".join(line for line in rationale_lines).strip()
    if not rationale:
        raise ValueError("agentic_research_mutation_rationale_missing")
    if not change_lines:
        raise ValueError("agentic_research_mutation_changes_missing")
    return rationale, change_lines


def _assign_dotted_path(target: dict[str, object], parts: list[str], value: object) -> None:
    current: dict[str, object] = target
    for token in parts[:-1]:
        existing = current.get(token)
        if existing is None:
            existing = {}
            current[token] = existing
        if not isinstance(existing, dict):
            dotted = ".".join(parts)
            raise ValueError(f"agentic_research_mutation_target_not_mapping:{dotted}")
        current = existing
    current[parts[-1]] = value


def _identity_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = deepcopy(payload)
    output = normalized.get("output")
    if isinstance(output, dict):
        output.pop("predictions_name", None)
        output.pop("results_name", None)
        output.pop("output_dir", None)
    return normalized


def _mutation_path_allowed(path: str, *, allowed_paths: tuple[str, ...]) -> bool:
    tokens = tuple(path.split("."))
    for allowed in allowed_paths:
        allowed_tokens = tuple(allowed.split("."))
        if len(tokens) != len(allowed_tokens):
            continue
        if all(
            expected == "*" or expected == current for expected, current in zip(allowed_tokens, tokens, strict=True)
        ):
            return True
    return False


def _change_slug(change: MutationChange) -> str:
    path_slug = _slugify(change.path.replace(".", "-"))
    value_slug = _value_slug(change.value)
    return f"{path_slug}-{value_slug}".strip("-")


def _value_slug(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _slugify(str(value).replace(".", "p").replace("-", "neg"))
    if isinstance(value, str):
        return _slugify(value)
    return _slugify(json.dumps(value, sort_keys=True))


def _slugify(value: str) -> str:
    return _FILENAME_SLUG_RE.sub("-", value.lower()).strip("-") or "value"


def _strip_code_fence(value: str) -> str:
    return _CODE_FENCE_RE.sub("", value).strip()


def _load_validated_config(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        validated = TrainingConfig.model_validate(payload)
    except Exception:
        return None
    return validated.model_dump(mode="python", exclude_none=True)
