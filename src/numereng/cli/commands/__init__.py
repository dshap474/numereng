"""CLI command handler namespace with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "handle_baseline_command",
    "handle_cloud_command",
    "handle_dataset_tools_command",
    "handle_ensemble_command",
    "handle_experiment_command",
    "handle_hpo_command",
    "handle_monitor_command",
    "handle_neutralize_command",
    "handle_numerai_command",
    "handle_remote_command",
    "handle_research_command",
    "handle_run_command",
    "handle_serve_command",
    "handle_store_command",
    "handle_viz_command",
]

_COMMAND_MODULES = {
    "handle_baseline_command": "numereng.cli.commands.baseline",
    "handle_cloud_command": "numereng.cli.commands.cloud",
    "handle_dataset_tools_command": "numereng.cli.commands.dataset_tools",
    "handle_ensemble_command": "numereng.cli.commands.ensemble",
    "handle_experiment_command": "numereng.cli.commands.experiment",
    "handle_hpo_command": "numereng.cli.commands.hpo",
    "handle_monitor_command": "numereng.cli.commands.monitor",
    "handle_neutralize_command": "numereng.cli.commands.neutralize",
    "handle_numerai_command": "numereng.cli.commands.numerai",
    "handle_remote_command": "numereng.cli.commands.remote",
    "handle_research_command": "numereng.cli.commands.research",
    "handle_run_command": "numereng.cli.commands.run",
    "handle_serve_command": "numereng.cli.commands.serve",
    "handle_store_command": "numereng.cli.commands.store",
    "handle_viz_command": "numereng.cli.commands.viz",
}


def __getattr__(name: str) -> Any:
    module_name = _COMMAND_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
