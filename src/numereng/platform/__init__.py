"""Shared infrastructure surface for numereng."""

from numereng.platform.errors import ForumScraperError, NumeraiClientError, PackageError
from numereng.platform.forum_scraper import scrape_forum_posts
from numereng.platform.numerai_client import NumeraiClient

__all__ = [
    "ForumScraperError",
    "NumeraiClient",
    "NumeraiClientError",
    "PackageError",
    "scrape_forum_posts",
]
