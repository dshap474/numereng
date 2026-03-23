"""Contracts for agentic research supervisor workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ResearchProgramStatus = Literal["initialized", "running", "interrupted", "stopped", "failed"]
ResearchPathStatus = Literal["active", "pivoted", "stopped"]
ResearchRoundStatus = Literal["planning", "planned", "running", "scored", "completed", "failed"]
ResearchDecisionAction = Literal["continue", "scale", "pivot", "stop"]
ResearchStrategyId = Literal["numerai-experiment-loop", "kaggle-gm-loop"]
ResearchPhaseStatus = Literal["active", "completed"]
ResearchPhaseAction = Literal["stay", "advance", "complete"]


@dataclass(frozen=True)
class ResearchBestRun:
    """Best run summary tracked across the full supervisor program."""

    experiment_id: str | None = None
    run_id: str | None = None
    bmc_last_200_eras_mean: float | None = None
    bmc_mean: float | None = None
    corr_mean: float | None = None
    mmc_mean: float | None = None
    cwmm_mean: float | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class ResearchPathState:
    """One active or historical line of inquiry."""

    path_id: str
    experiment_id: str
    parent_experiment_id: str | None
    generation: int
    hypothesis: str
    status: ResearchPathStatus
    pivot_reason: str | None
    source_round: str | None
    rounds_completed: int
    plateau_streak: int
    scale_confirmation_used: bool
    needs_scale_confirmation: bool
    best_run_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchRoundState:
    """Checkpointed state for the current in-flight round."""

    round_number: int
    round_label: str
    experiment_id: str
    path_id: str
    status: ResearchRoundStatus
    next_config_index: int
    config_filenames: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    decision_action: ResearchDecisionAction | None = None
    experiment_question: str | None = None
    winner_criteria: str | None = None
    decision_rationale: str | None = None
    decision_path_hypothesis: str | None = None
    decision_path_slug: str | None = None
    phase_id: str | None = None
    phase_action: ResearchPhaseAction | None = None
    phase_transition_rationale: str | None = None
    started_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class ResearchPhaseState:
    """Current strategy phase tracked for phase-aware strategies."""

    phase_id: str
    phase_title: str
    status: ResearchPhaseStatus
    round_count: int
    transition_rationale: str | None
    started_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchProgramState:
    """Canonical persisted program state for the supervisor."""

    root_experiment_id: str
    program_experiment_id: str
    strategy: ResearchStrategyId
    strategy_description: str
    status: ResearchProgramStatus
    active_path_id: str
    active_experiment_id: str
    next_round_number: int
    total_rounds_completed: int
    total_paths_created: int
    improvement_threshold: float
    scoring_stage: Literal["post_training_core", "post_training_full"]
    codex_command: list[str]
    last_checkpoint: str
    stop_reason: str | None
    current_round: ResearchRoundState | None
    current_phase: ResearchPhaseState | None
    best_overall: ResearchBestRun
    paths: list[ResearchPathState]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResearchLineageLink:
    """One experiment lineage record created by the supervisor."""

    path_id: str
    experiment_id: str
    parent_experiment_id: str | None
    generation: int
    source_round: str | None
    pivot_reason: str | None
    created_at: str


@dataclass(frozen=True)
class ResearchLineageState:
    """Persisted experiment lineage graph for one supervisor program."""

    root_experiment_id: str
    program_experiment_id: str
    active_path_id: str
    links: list[ResearchLineageLink]


@dataclass(frozen=True)
class CodexConfigPayload:
    """One planned config emitted by headless Codex."""

    filename: str
    rationale: str
    overrides: dict[str, object]


@dataclass(frozen=True)
class CodexDecision:
    """Structured planning response returned by headless Codex."""

    experiment_question: str
    winner_criteria: str
    decision_rationale: str
    next_action: ResearchDecisionAction
    path_hypothesis: str
    path_slug: str
    configs: list[CodexConfigPayload]
    phase_action: ResearchPhaseAction | None = None
    phase_transition_rationale: str | None = None


@dataclass(frozen=True)
class CodexPlannerAttempt:
    """Telemetry captured for one headless Codex planner attempt."""

    attempt_number: int
    elapsed_seconds: float
    returncode: int
    thread_id: str | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    stdout_line_count: int = 0
    validation_feedback: str | None = None


@dataclass(frozen=True)
class CodexPlannerExecution:
    """One completed planner execution plus persisted telemetry artifacts."""

    decision: CodexDecision
    attempts: list[CodexPlannerAttempt]
    stdout_jsonl: str
    stderr_text: str
    last_message: dict[str, object]


@dataclass(frozen=True)
class ResearchInitResult:
    """Result payload for initializing one supervisor program."""

    root_experiment_id: str
    strategy: ResearchStrategyId
    strategy_description: str
    status: ResearchProgramStatus
    active_experiment_id: str
    active_path_id: str
    improvement_threshold: float
    current_phase: ResearchPhaseState | None
    agentic_research_dir: Path
    program_path: Path
    lineage_path: Path


@dataclass(frozen=True)
class ResearchStatusResult:
    """Public status payload for the supervisor."""

    root_experiment_id: str
    strategy: ResearchStrategyId
    strategy_description: str
    status: ResearchProgramStatus
    active_experiment_id: str
    active_path_id: str
    next_round_number: int
    total_rounds_completed: int
    total_paths_created: int
    improvement_threshold: float
    last_checkpoint: str
    stop_reason: str | None
    best_overall: ResearchBestRun
    current_round: ResearchRoundState | None
    current_phase: ResearchPhaseState | None
    program_path: Path
    lineage_path: Path


@dataclass(frozen=True)
class ResearchRunResult:
    """Result payload after one foreground research loop exits."""

    root_experiment_id: str
    strategy: ResearchStrategyId
    strategy_description: str
    status: ResearchProgramStatus
    active_experiment_id: str
    active_path_id: str
    next_round_number: int
    total_rounds_completed: int
    total_paths_created: int
    last_checkpoint: str
    stop_reason: str | None
    current_phase: ResearchPhaseState | None
    interrupted: bool
