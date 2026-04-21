"""Shared response mappers for agentic research API handlers."""

from __future__ import annotations

from numereng.api.contracts import ResearchPhaseResponse, ResearchRoundResponse


def phase_response(current_phase: object) -> ResearchPhaseResponse | None:
    if current_phase is None:
        return None
    return ResearchPhaseResponse(
        phase_id=current_phase.phase_id,
        phase_title=current_phase.phase_title,
        status=current_phase.status,
        round_count=current_phase.round_count,
        transition_rationale=current_phase.transition_rationale,
        started_at=current_phase.started_at,
        updated_at=current_phase.updated_at,
    )


def round_response(current_round: object) -> ResearchRoundResponse | None:
    if current_round is None:
        return None
    return ResearchRoundResponse(
        round_number=current_round.round_number,
        round_label=current_round.round_label,
        experiment_id=current_round.experiment_id,
        path_id=current_round.path_id,
        status=current_round.status,
        next_config_index=current_round.next_config_index,
        config_filenames=list(current_round.config_filenames),
        run_ids=list(current_round.run_ids),
        decision_action=current_round.decision_action,
        experiment_question=current_round.experiment_question,
        winner_criteria=current_round.winner_criteria,
        decision_rationale=current_round.decision_rationale,
        decision_path_hypothesis=current_round.decision_path_hypothesis,
        decision_path_slug=current_round.decision_path_slug,
        phase_id=current_round.phase_id,
        phase_action=current_round.phase_action,
        phase_transition_rationale=current_round.phase_transition_rationale,
        started_at=current_round.started_at,
        updated_at=current_round.updated_at,
    )
