"""CLI command handlers."""

from numereng.cli.commands.baseline import handle_baseline_command
from numereng.cli.commands.cloud import handle_cloud_command
from numereng.cli.commands.dataset_tools import handle_dataset_tools_command
from numereng.cli.commands.ensemble import handle_ensemble_command
from numereng.cli.commands.experiment import handle_experiment_command
from numereng.cli.commands.hpo import handle_hpo_command
from numereng.cli.commands.init import handle_init_command
from numereng.cli.commands.monitor import handle_monitor_command
from numereng.cli.commands.neutralize import handle_neutralize_command
from numereng.cli.commands.numerai import handle_numerai_command
from numereng.cli.commands.remote import handle_remote_command
from numereng.cli.commands.research import handle_research_command
from numereng.cli.commands.run import handle_run_command
from numereng.cli.commands.store import handle_store_command

__all__ = [
    "handle_baseline_command",
    "handle_cloud_command",
    "handle_dataset_tools_command",
    "handle_ensemble_command",
    "handle_experiment_command",
    "handle_hpo_command",
    "handle_init_command",
    "handle_monitor_command",
    "handle_neutralize_command",
    "handle_numerai_command",
    "handle_remote_command",
    "handle_research_command",
    "handle_run_command",
    "handle_store_command",
]
