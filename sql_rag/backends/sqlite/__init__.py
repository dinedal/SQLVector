"""SQLite backend for SQL RAG with VSS support.

This module provides a complete SQLite-based RAG implementation with:
- Binary embedding storage for space efficiency  
- Custom SQLite functions for vector similarity
- Optional sqlite-vss extension for accelerated search
- Batch operations and efficient data loading
- Support for metadata filtering and complex queries

Main Components:
- SQLiteRAG: High-level interface for document loading and querying
- SQLiteConfig: Configuration for SQLite backend with VSS support
- SQLiteLoader: Document and embedding loading with batch operations
- SQLiteQuerier: Vector similarity querying with VSS optimization
"""

from .rag import SQLiteRAG
from .config import SQLiteConfig
from .loader import SQLiteLoader
from .querier import SQLiteQuerier
from .models import SQLiteDocument, SQLiteEmbedding

__all__ = [
    "SQLiteRAG",
    "SQLiteConfig", 
    "SQLiteLoader",
    "SQLiteQuerier",
    "SQLiteDocument",
    "SQLiteEmbedding",
]