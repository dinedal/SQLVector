"""PostgreSQL backend for SQL RAG with pgvector support.

This module provides a complete PostgreSQL-based RAG implementation with:
- Native pgvector extension for efficient vector operations
- HNSW and IVFFlat indexing for fast similarity search
- Async operations with asyncpg connection pooling
- SQLAlchemy integration for flexibility
- JSONB metadata filtering for complex queries
- Batch operations for efficient data loading
- Support for cosine, euclidean, and inner product similarity

Main Components:
- PostgresRAG: High-level interface for document loading and querying
- PostgresConfig: Configuration for PostgreSQL backend with pgvector
- PostgresLoader: Document and embedding loading with batch operations
- PostgresQuerier: Vector similarity querying with pgvector operators
"""

from .rag import PostgresRAG
from .config import PostgresConfig
from .loader import PostgresLoader
from .querier import PostgresQuerier
from .models import PostgresDocument, PostgresEmbedding, PostgresQueryResult

__all__ = [
    "PostgresRAG",
    "PostgresConfig", 
    "PostgresLoader",
    "PostgresQuerier",
    "PostgresDocument",
    "PostgresEmbedding",
    "PostgresQueryResult",
]