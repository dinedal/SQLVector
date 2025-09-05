from typing import Any


class SQLRAGError(Exception):
    """Base exception for SQL RAG operations."""
    pass


class ConfigurationError(SQLRAGError):
    """Raised when configuration is invalid."""
    pass


class EmbeddingError(SQLRAGError):
    """Raised when embedding operations fail."""
    pass


class QueryError(SQLRAGError):
    """Raised when query operations fail."""
    pass


class LoaderError(SQLRAGError):
    """Raised when document loading operations fail."""
    pass