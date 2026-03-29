"""Program definitions, catalog loading, and validation for agentic research."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from numereng.features.agentic_research.utils.types import (
    ResearchPhaseState,
    ResearchProgramCatalogEntry,
    ResearchProgramConfigPolicy,
    ResearchProgramDefinition,
    ResearchProgramDetails,
    ResearchProgramMetricPolicy,
    ResearchProgramPhase,
    ResearchProgramRoundPolicy,
    ResearchProgramSource,
)

_PROGRAMS_DIR = Path(__file__).resolve().parents[1] / "programs"
_DEFAULT_TRACKED_PROGRAM_IDS = frozenset({"numerai-experiment-loop"})
_ENV_PROGRAMS_DIR = "NUMERENG_AGENTIC_RESEARCH_PROGRAMS_DIR"
_PROGRAM_GLOB = "*.md"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<frontmatter>.*?)\n---\s*\n(?P<body>.*)$", re.DOTALL)
_PROGRAM_PLACEHOLDERS = ("$CONTEXT_JSON", "$VALIDATION_FEEDBACK_BLOCK")
_PLACEHOLDER_RE = re.compile(r"\$([A-Z0-9_]+)")


class _MetricPolicyModel(BaseModel):
    primary: str = Field(min_length=1)
    tie_break: str = Field(min_length=1)
    sanity_checks: list[str] = Field(default_factory=list)


class _RoundPolicyModel(BaseModel):
    plateau_non_improving_rounds: int = Field(ge=1)
    require_scale_confirmation: bool
    scale_confirmation_rounds: int = Field(ge=0)


class _ConfigPolicyModel(BaseModel):
    allowed_paths: list[str] = Field(min_length=1)
    max_candidate_configs: int = Field(ge=1)
    min_candidate_configs: int | None = Field(default=None, ge=1)
    min_changes: int | None = Field(default=None, ge=1)
    max_changes: int | None = Field(default=None, ge=1)


class _PhaseModel(BaseModel):
    phase_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = ""
    gate: str = ""


class _ProgramFrontmatterModel(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    planner_contract: str = Field(min_length=1)
    scoring_stage: str = Field(min_length=1)
    metric_policy: _MetricPolicyModel
    round_policy: _RoundPolicyModel
    improvement_threshold_default: float = Field(gt=0.0)
    config_policy: _ConfigPolicyModel
    phases: list[_PhaseModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_contract_shape(self) -> _ProgramFrontmatterModel:
        if self.planner_contract not in {"config_mutation", "structured_json"}:
            raise ValueError("agentic_research_program_planner_contract_invalid")
        if self.scoring_stage not in {"post_training_core", "post_training_full"}:
            raise ValueError("agentic_research_program_scoring_stage_invalid")
        if self.planner_contract == "config_mutation":
            if self.config_policy.max_candidate_configs != 1:
                raise ValueError("agentic_research_program_max_candidate_configs_invalid")
            if self.config_policy.min_changes is None or self.config_policy.max_changes is None:
                raise ValueError("agentic_research_program_change_bounds_missing")
            if self.config_policy.min_changes > self.config_policy.max_changes:
                raise ValueError("agentic_research_program_change_bounds_invalid")
            if self.config_policy.min_candidate_configs is not None:
                raise ValueError("agentic_research_program_min_candidate_configs_unexpected")
        else:
            if self.config_policy.min_candidate_configs is None:
                raise ValueError("agentic_research_program_min_candidate_configs_missing")
            if self.config_policy.min_candidate_configs > self.config_policy.max_candidate_configs:
                raise ValueError("agentic_research_program_candidate_bounds_invalid")
            if self.config_policy.min_changes is not None or self.config_policy.max_changes is not None:
                raise ValueError("agentic_research_program_change_bounds_unexpected")
        if self.round_policy.require_scale_confirmation:
            if self.round_policy.scale_confirmation_rounds < 1:
                raise ValueError("agentic_research_program_scale_confirmation_rounds_invalid")
        elif self.round_policy.scale_confirmation_rounds != 0:
            raise ValueError("agentic_research_program_scale_confirmation_rounds_unexpected")
        return self


def render_prompt_template(template: str | Path, replacements: dict[str, str], *, source: str = "<inline>") -> str:
    """Render one prompt template with direct placeholder substitution only."""
    if isinstance(template, Path):
        source = str(template)
        template = template.read_text(encoding="utf-8")
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"${key}", value)
    unresolved = sorted({match.group(1) for match in _PLACEHOLDER_RE.finditer(rendered)})
    if unresolved:
        joined = ",".join(unresolved)
        raise ValueError(f"agentic_research_prompt_placeholders_unresolved:{source}:{joined}")
    return rendered.strip() + "\n"


def render_validation_feedback_block(validation_feedback: str | None) -> str:
    """Render the optional validation feedback block."""
    if validation_feedback is None or not validation_feedback.strip():
        return ""
    return (
        "Validation feedback from the last attempt:\n"
        f"{validation_feedback.strip()}\n"
        "Fix the issue and return a valid response."
    )


def resolve_user_programs_dir(path: str | Path | None = None) -> Path:
    """Resolve the active programs directory."""
    if path is not None:
        return Path(path).expanduser().resolve()
    env_value = os.getenv(_ENV_PROGRAMS_DIR)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return _PROGRAMS_DIR


def program_markdown_sha256(markdown_text: str) -> str:
    """Return the canonical sha256 for one raw program markdown file."""
    return hashlib.sha256(markdown_text.encode("utf-8")).hexdigest()


def list_program_catalog(*, user_dir: str | Path | None = None) -> tuple[ResearchProgramCatalogEntry, ...]:
    """Return the active repo-local plus optional override program catalog."""
    entries: list[ResearchProgramCatalogEntry] = []
    seen_ids: set[str] = set()
    directory = resolve_user_programs_dir(user_dir)
    if not directory.is_dir():
        return ()
    for path in sorted(directory.glob(_PROGRAM_GLOB)):
        if not path.is_file() or path.name.startswith(".") or path.name == "README.md":
            continue
        source = _program_source_for(path=path, program_id=path.stem, directory=directory)
        details = _load_program_details(path=path, source=source)
        program_id = details.definition.program_id
        if program_id in seen_ids:
            raise ValueError(f"agentic_research_program_id_duplicate:{program_id}")
        seen_ids.add(program_id)
        entries.append(
            ResearchProgramCatalogEntry(
                program_id=program_id,
                title=details.definition.title,
                description=details.definition.description,
                source=details.definition.source,
                planner_contract=details.definition.planner_contract,
                phase_aware=bool(details.definition.phases),
                source_path=details.definition.source_path,
            )
        )
    return tuple(entries)


def load_program_definition(program_id: str, *, user_dir: str | Path | None = None) -> ResearchProgramDefinition:
    """Load one resolved program definition by id."""
    return load_program_details(program_id, user_dir=user_dir).definition


def load_program_details(program_id: str, *, user_dir: str | Path | None = None) -> ResearchProgramDetails:
    """Load one resolved program definition and its raw markdown source by id."""
    directory = resolve_user_programs_dir(user_dir)
    path = directory / f"{program_id}.md"
    if path.is_file():
        source = _program_source_for(path=path, program_id=program_id, directory=directory)
        return _load_program_details(path=path, source=source)
    raise ValueError(f"agentic_research_program_not_found:{program_id}")


def legacy_builtin_program(program_id: str) -> ResearchProgramDefinition:
    """Load one built-in program for a legacy strategy-backed session."""
    definition = load_program_definition(program_id, user_dir=_PROGRAMS_DIR)
    if definition.source != "builtin":
        raise ValueError(f"agentic_research_legacy_program_invalid:{program_id}")
    return replace(definition, source="legacy_builtin")


def initial_phase_state(definition: ResearchProgramDefinition, *, now_iso: str) -> ResearchPhaseState | None:
    """Return the initial persisted phase state for one phase-aware program."""
    if not definition.phases:
        return None
    first = definition.phases[0]
    return ResearchPhaseState(
        phase_id=first.phase_id,
        phase_title=first.title,
        status="active",
        round_count=0,
        transition_rationale=None,
        started_at=now_iso,
        updated_at=now_iso,
    )


def get_phase_definition(definition: ResearchProgramDefinition, phase_id: str | None) -> ResearchProgramPhase | None:
    """Return the matching configured phase, if any."""
    if phase_id is None:
        return None
    for phase in definition.phases:
        if phase.phase_id == phase_id:
            return phase
    return None


def next_phase_definition(definition: ResearchProgramDefinition, phase_id: str | None) -> ResearchProgramPhase | None:
    """Return the next configured phase after the given one."""
    if phase_id is None:
        return definition.phases[0] if definition.phases else None
    for index, phase in enumerate(definition.phases):
        if phase.phase_id == phase_id:
            next_index = index + 1
            if next_index < len(definition.phases):
                return definition.phases[next_index]
            return None
    return None


def render_session_program_markdown(definition: ResearchProgramDefinition) -> str:
    """Render one persisted program snapshot as markdown with YAML frontmatter."""
    frontmatter = {
        "id": definition.program_id,
        "title": definition.title,
        "description": definition.description,
        "planner_contract": definition.planner_contract,
        "scoring_stage": definition.scoring_stage,
        "metric_policy": {
            "primary": definition.metric_policy.primary,
            "tie_break": definition.metric_policy.tie_break,
            "sanity_checks": list(definition.metric_policy.sanity_checks),
        },
        "round_policy": {
            "plateau_non_improving_rounds": definition.round_policy.plateau_non_improving_rounds,
            "require_scale_confirmation": definition.round_policy.require_scale_confirmation,
            "scale_confirmation_rounds": definition.round_policy.scale_confirmation_rounds,
        },
        "improvement_threshold_default": definition.improvement_threshold_default,
        "config_policy": {
            "allowed_paths": list(definition.config_policy.allowed_paths),
            "max_candidate_configs": definition.config_policy.max_candidate_configs,
            "min_candidate_configs": definition.config_policy.min_candidate_configs,
            "min_changes": definition.config_policy.min_changes,
            "max_changes": definition.config_policy.max_changes,
        },
        "phases": [
            {
                "phase_id": phase.phase_id,
                "title": phase.title,
                "summary": phase.summary,
                "gate": phase.gate,
            }
            for phase in definition.phases
        ],
    }
    rendered_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    return f"---\n{rendered_frontmatter}\n---\n{definition.prompt_template.strip()}\n"


def _load_program_details(*, path: Path, source: ResearchProgramSource) -> ResearchProgramDetails:
    raw_markdown = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw_markdown)
    if match is None:
        raise ValueError(f"agentic_research_program_frontmatter_missing:{path}")
    try:
        payload = yaml.safe_load(match.group("frontmatter"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"agentic_research_program_frontmatter_invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"agentic_research_program_frontmatter_invalid:{path}")
    model = _ProgramFrontmatterModel.model_validate(payload)
    body = match.group("body").strip()
    if not body:
        raise ValueError(f"agentic_research_program_prompt_missing:{path}")
    for placeholder in _PROGRAM_PLACEHOLDERS:
        if placeholder not in body:
            raise ValueError(f"agentic_research_program_placeholder_missing:{path}:{placeholder}")
    if path.stem != model.id:
        raise ValueError(f"agentic_research_program_id_filename_mismatch:{path}")
    definition = ResearchProgramDefinition(
        program_id=model.id,
        title=model.title,
        description=model.description,
        source=source,
        planner_contract=model.planner_contract,
        scoring_stage=model.scoring_stage,
        metric_policy=ResearchProgramMetricPolicy(
            primary=model.metric_policy.primary,
            tie_break=model.metric_policy.tie_break,
            sanity_checks=tuple(model.metric_policy.sanity_checks),
        ),
        round_policy=ResearchProgramRoundPolicy(
            plateau_non_improving_rounds=model.round_policy.plateau_non_improving_rounds,
            require_scale_confirmation=model.round_policy.require_scale_confirmation,
            scale_confirmation_rounds=model.round_policy.scale_confirmation_rounds,
        ),
        improvement_threshold_default=model.improvement_threshold_default,
        config_policy=ResearchProgramConfigPolicy(
            allowed_paths=tuple(model.config_policy.allowed_paths),
            min_candidate_configs=model.config_policy.min_candidate_configs,
            max_candidate_configs=model.config_policy.max_candidate_configs,
            min_changes=model.config_policy.min_changes,
            max_changes=model.config_policy.max_changes,
        ),
        prompt_template=body + "\n",
        phases=tuple(
            ResearchProgramPhase(
                phase_id=item.phase_id,
                title=item.title,
                summary=item.summary,
                gate=item.gate,
            )
            for item in model.phases
        ),
        source_path=str(path),
    )
    return ResearchProgramDetails(definition=definition, raw_markdown=raw_markdown)


def _program_source_for(*, path: Path, program_id: str, directory: Path) -> ResearchProgramSource:
    if directory == _PROGRAMS_DIR and path.parent == _PROGRAMS_DIR and program_id in _DEFAULT_TRACKED_PROGRAM_IDS:
        return "builtin"
    return "user"


__all__ = [
    "get_phase_definition",
    "initial_phase_state",
    "legacy_builtin_program",
    "list_program_catalog",
    "load_program_definition",
    "load_program_details",
    "next_phase_definition",
    "program_markdown_sha256",
    "render_prompt_template",
    "render_session_program_markdown",
    "render_validation_feedback_block",
    "resolve_user_programs_dir",
]
