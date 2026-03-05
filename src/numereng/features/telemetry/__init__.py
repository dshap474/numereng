"""Public surface for local run telemetry helpers."""

from numereng.features.telemetry.context import LaunchMetadata, bind_launch_metadata, get_launch_metadata
from numereng.features.telemetry.contracts import LocalResourceSampler, LocalRunTelemetrySession, ResourceSample
from numereng.features.telemetry.service import (
    append_log_line,
    append_resource_sample,
    begin_local_training_session,
    capture_local_resource_sample,
    emit_job_event,
    emit_metric_event,
    emit_stage_event,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
    mark_job_starting,
    start_local_resource_sampler,
    stop_local_resource_sampler,
)

__all__ = [
    "LaunchMetadata",
    "LocalResourceSampler",
    "LocalRunTelemetrySession",
    "ResourceSample",
    "append_log_line",
    "append_resource_sample",
    "begin_local_training_session",
    "bind_launch_metadata",
    "capture_local_resource_sample",
    "emit_job_event",
    "emit_metric_event",
    "emit_stage_event",
    "get_launch_metadata",
    "mark_job_completed",
    "mark_job_failed",
    "mark_job_running",
    "mark_job_starting",
    "start_local_resource_sampler",
    "stop_local_resource_sampler",
]
