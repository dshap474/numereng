"""Strategy registry and asset helpers for agentic research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from numereng.features.agentic_research.contracts import (
    ResearchPhaseState,
    ResearchStrategyId,
)
from numereng.features.agentic_research.state import utc_now_iso

_ASSETS_ROOT = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class ResearchStrategyPhase:
    """One configured phase inside one research strategy."""

    phase_id: str
    title: str
    summary: str
    gate: str


@dataclass(frozen=True)
class ResearchStrategyDefinition:
    """Loaded strategy metadata plus resolved asset paths."""

    strategy_id: ResearchStrategyId
    title: str
    description: str
    phase_aware: bool
    prompt_path: Path
    schema_path: Path
    phases: tuple[ResearchStrategyPhase, ...] = ()


_SUPPORTED_STRATEGIES: Final[tuple[ResearchStrategyId, ...]] = (
    "numerai-experiment-loop",
    "kaggle-gm-loop",
)


def list_strategy_ids() -> tuple[ResearchStrategyId, ...]:
    """Return supported agentic research strategy ids."""
    return _SUPPORTED_STRATEGIES


def get_strategy_definition(strategy_id: str) -> ResearchStrategyDefinition:
    """Resolve one strategy definition from feature assets."""
    if strategy_id not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"agentic_research_strategy_invalid:{strategy_id}")
    asset_dir = _ASSETS_ROOT / strategy_id
    metadata_path = asset_dir / "metadata.json"
    prompt_path = asset_dir / "planner_prompt.txt"
    schema_path = asset_dir / "planner_output.schema.json"
    if not metadata_path.is_file() or not prompt_path.is_file() or not schema_path.is_file():
        raise ValueError(f"agentic_research_strategy_assets_missing:{strategy_id}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"agentic_research_strategy_metadata_invalid:{metadata_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"agentic_research_strategy_metadata_invalid:{metadata_path}")
    title = str(payload.get("title") or strategy_id)
    description = str(payload.get("description") or title)
    phase_aware = bool(payload.get("phase_aware"))
    phases = _phases_from_payload(payload.get("phases"))
    return ResearchStrategyDefinition(
        strategy_id=strategy_id,
        title=title,
        description=description,
        phase_aware=phase_aware,
        prompt_path=prompt_path,
        schema_path=schema_path,
        phases=phases,
    )


def initial_phase_state(definition: ResearchStrategyDefinition) -> ResearchPhaseState | None:
    """Return the initial phase state for one strategy."""
    if not definition.phase_aware or not definition.phases:
        return None
    first = definition.phases[0]
    now = utc_now_iso()
    return ResearchPhaseState(
        phase_id=first.phase_id,
        phase_title=first.title,
        status="active",
        round_count=0,
        transition_rationale=None,
        started_at=now,
        updated_at=now,
    )


def get_phase_definition(
    definition: ResearchStrategyDefinition,
    phase_id: str | None,
) -> ResearchStrategyPhase | None:
    """Return the matching configured phase, if any."""
    if phase_id is None:
        return None
    for phase in definition.phases:
        if phase.phase_id == phase_id:
            return phase
    return None


def next_phase_definition(
    definition: ResearchStrategyDefinition,
    phase_id: str | None,
) -> ResearchStrategyPhase | None:
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


def _phases_from_payload(value: object) -> tuple[ResearchStrategyPhase, ...]:
    if not isinstance(value, list):
        return ()
    phases: list[ResearchStrategyPhase] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        phase_id = str(item.get("phase_id") or "").strip()
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        gate = str(item.get("gate") or "").strip()
        if not phase_id or not title:
            continue
        phases.append(
            ResearchStrategyPhase(
                phase_id=phase_id,
                title=title,
                summary=summary,
                gate=gate,
            )
        )
    return tuple(phases)
