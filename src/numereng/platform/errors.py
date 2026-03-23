"""Shared package error types."""


class PackageError(Exception):
    """Consumer-facing error type for numereng APIs and CLI."""


class NumeraiClientError(PackageError):
    """Raised when Numerai API client operations fail."""


class OpenRouterClientError(PackageError):
    """Raised when OpenRouter API client operations fail."""


class NumeraiMcpAuthError(PackageError):
    """Raised when Numerai MCP auth cannot be resolved for shell export."""


class ForumScraperError(PackageError):
    """Raised when forum scraping operations fail."""
