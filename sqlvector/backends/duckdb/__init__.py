"""DuckDB backend for SQL RAG with high-performance vector operations."""

from .config import DuckDBConfig
from .models import DuckDBDocument, DuckDBEmbedding
from .loader import DuckDBLoader
from .querier import DuckDBQuerier
from .rag import DuckDBRAG

__all__ = [
    "DuckDBConfig",
    "DuckDBDocument", 
    "DuckDBEmbedding",
    "DuckDBLoader",
    "DuckDBQuerier", 
    "DuckDBRAG"
]