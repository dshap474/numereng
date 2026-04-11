"""Public surface for remote SSH sync and launch workflows with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "RemoteBootstrapStatus",
    "RemoteConfigPushResult",
    "RemoteDoctorResult",
    "RemoteExperimentLaunchResult",
    "RemoteExperimentMaintainResult",
    "RemoteExperimentPullFailure",
    "RemoteExperimentPullResult",
    "RemoteExperimentStatusResult",
    "RemoteExperimentStopResult",
    "RemoteExecutionError",
    "RemoteExperimentSyncResult",
    "RemoteOpsError",
    "RemoteRepoSyncResult",
    "RemoteSyncPolicy",
    "RemoteTargetNotFoundError",
    "RemoteTargetRecord",
    "RemoteTrainLaunchResult",
    "RemoteValidationError",
    "RemoteVizBootstrapResult",
    "RemoteVizBootstrapTargetResult",
    "bootstrap_viz_remotes",
    "doctor_remote_target",
    "load_viz_bootstrap_state",
    "list_remote_targets",
    "remote_experiment_status",
    "remote_launch_experiment",
    "remote_maintain_experiment",
    "pull_remote_experiment",
    "push_remote_config",
    "remote_run_train",
    "remote_stop_experiment",
    "remote_viz_bootstrap_state_path",
    "sync_remote_experiment",
    "sync_remote_repo",
]

_EXPORT_MODULES = {
    "RemoteBootstrapStatus": "numereng.features.remote_ops.contracts",
    "RemoteConfigPushResult": "numereng.features.remote_ops.contracts",
    "RemoteDoctorResult": "numereng.features.remote_ops.contracts",
    "RemoteExperimentLaunchResult": "numereng.features.remote_ops.contracts",
    "RemoteExperimentMaintainResult": "numereng.features.remote_ops.contracts",
    "RemoteExperimentPullFailure": "numereng.features.remote_ops.contracts",
    "RemoteExperimentPullResult": "numereng.features.remote_ops.contracts",
    "RemoteExperimentStatusResult": "numereng.features.remote_ops.contracts",
    "RemoteExperimentStopResult": "numereng.features.remote_ops.contracts",
    "RemoteExperimentSyncResult": "numereng.features.remote_ops.contracts",
    "RemoteRepoSyncResult": "numereng.features.remote_ops.contracts",
    "RemoteSyncPolicy": "numereng.features.remote_ops.contracts",
    "RemoteTargetRecord": "numereng.features.remote_ops.contracts",
    "RemoteTrainLaunchResult": "numereng.features.remote_ops.contracts",
    "RemoteVizBootstrapResult": "numereng.features.remote_ops.contracts",
    "RemoteVizBootstrapTargetResult": "numereng.features.remote_ops.contracts",
    "load_viz_bootstrap_state": "numereng.features.remote_ops.bootstrap_state",
    "remote_viz_bootstrap_state_path": "numereng.features.remote_ops.bootstrap_state",
    "RemoteExecutionError": "numereng.features.remote_ops.service",
    "RemoteOpsError": "numereng.features.remote_ops.service",
    "RemoteTargetNotFoundError": "numereng.features.remote_ops.service",
    "RemoteValidationError": "numereng.features.remote_ops.service",
    "bootstrap_viz_remotes": "numereng.features.remote_ops.service",
    "doctor_remote_target": "numereng.features.remote_ops.service",
    "list_remote_targets": "numereng.features.remote_ops.service",
    "remote_experiment_status": "numereng.features.remote_ops.service",
    "remote_launch_experiment": "numereng.features.remote_ops.service",
    "remote_maintain_experiment": "numereng.features.remote_ops.service",
    "pull_remote_experiment": "numereng.features.remote_ops.service",
    "push_remote_config": "numereng.features.remote_ops.service",
    "remote_run_train": "numereng.features.remote_ops.service",
    "remote_stop_experiment": "numereng.features.remote_ops.service",
    "sync_remote_experiment": "numereng.features.remote_ops.service",
    "sync_remote_repo": "numereng.features.remote_ops.service",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
