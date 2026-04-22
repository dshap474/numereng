"""Remote SSH ops API facade preserving the historical private import path."""

from __future__ import annotations

from numereng.api._remote.launch import (
    remote_experiment_launch,
    remote_experiment_maintain,
    remote_experiment_status,
    remote_experiment_stop,
    remote_train_launch,
)
from numereng.api._remote.sync import (
    remote_config_push,
    remote_experiment_pull,
    remote_experiment_sync,
    remote_repo_sync,
)
from numereng.api._remote.targets import remote_bootstrap_viz, remote_doctor, remote_list_targets
from numereng.features.remote_ops import bootstrap_viz_remotes as bootstrap_viz_remotes_record
from numereng.features.remote_ops import doctor_remote_target as doctor_remote_target_record
from numereng.features.remote_ops import list_remote_targets as list_remote_targets_record
from numereng.features.remote_ops import pull_remote_experiment as pull_remote_experiment_record
from numereng.features.remote_ops import push_remote_config as push_remote_config_record
from numereng.features.remote_ops import remote_experiment_status as remote_experiment_status_record
from numereng.features.remote_ops import remote_launch_experiment as remote_launch_experiment_record
from numereng.features.remote_ops import remote_maintain_experiment as remote_maintain_experiment_record
from numereng.features.remote_ops import remote_run_train as remote_run_train_record
from numereng.features.remote_ops import remote_stop_experiment as remote_stop_experiment_record
from numereng.features.remote_ops import sync_remote_experiment as sync_remote_experiment_record
from numereng.features.remote_ops import sync_remote_repo as sync_remote_repo_record

__all__ = [
    "remote_experiment_launch",
    "remote_experiment_maintain",
    "remote_config_push",
    "remote_bootstrap_viz",
    "remote_doctor",
    "remote_experiment_status",
    "remote_experiment_stop",
    "remote_experiment_pull",
    "remote_experiment_sync",
    "remote_list_targets",
    "remote_repo_sync",
    "remote_train_launch",
]
