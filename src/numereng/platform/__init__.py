"""Shared infrastructure surface for numereng."""

from numereng.platform.clients.openrouter import OpenRouterClient, OpenRouterStreamEvent
from numereng.platform.errors import (
    ForumScraperError,
    NumeraiClientError,
    NumeraiMcpAuthError,
    OpenRouterClientError,
    PackageError,
)
from numereng.platform.forum_scraper import scrape_forum_posts
from numereng.platform.numerai_client import NumeraiClient
from numereng.platform.remotes.contracts import RemoteTargetError, SshRemoteTargetProfile
from numereng.platform.remotes.loader import (
    default_remote_profiles_dir,
    load_remote_targets,
    resolve_remote_profiles_dir,
)

__all__ = [
    "ForumScraperError",
    "NumeraiClient",
    "NumeraiClientError",
    "NumeraiMcpAuthError",
    "OpenRouterClient",
    "OpenRouterClientError",
    "OpenRouterStreamEvent",
    "PackageError",
    "RemoteTargetError",
    "SshRemoteTargetProfile",
    "default_remote_profiles_dir",
    "load_remote_targets",
    "resolve_remote_profiles_dir",
    "scrape_forum_posts",
]
