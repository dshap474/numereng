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

__all__ = [
    "ForumScraperError",
    "NumeraiClient",
    "NumeraiClientError",
    "NumeraiMcpAuthError",
    "OpenRouterClient",
    "OpenRouterClientError",
    "OpenRouterStreamEvent",
    "PackageError",
    "scrape_forum_posts",
]
