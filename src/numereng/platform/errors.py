"""Shared package error types."""


class PackageError(Exception):
    """Consumer-facing error type for numereng APIs and CLI."""


class NumeraiClientError(PackageError):
    """Raised when Numerai API client operations fail."""


class ForumScraperError(PackageError):
    """Raised when forum scraping operations fail."""
