# Configure logging on import
from .logger import configure_logging
configure_logging()

from .config import RAGConfig, SyncRAGConfig
from .models import Document, Embedding, Base
from .embedding import (
    EmbeddingProvider,
    DefaultEmbeddingProvider,
    EmbeddingService,
    DefaultSyncEmbeddingProvider,
    SyncEmbeddingService,
)
from .query import QueryInterface, SyncQueryInterface
from .loader import LoaderInterface, DocumentData, SyncLoaderInterface
from .protocols import (
    EmbeddingProvider as EmbeddingProviderProtocol,
    DatabaseConfigProtocol,
    DocumentLoaderProtocol,
    DocumentQuerierProtocol,
    RAGSystemProtocol,
    DocumentProtocol,
    EmbeddingProtocol,
    QueryResultProtocol,
)
from .exceptions import (
    SQLRAGError,
    ConfigurationError,
    EmbeddingError,
    QueryError,
    LoaderError,
)

# Backend imports (optional dependencies)
try:
    from .backends.duckdb import DuckDBRAG, DuckDBConfig

    __duckdb_available__ = True
except ImportError:
    __duckdb_available__ = False

try:
    from .backends.sqlite import SQLiteRAG, SQLiteConfig

    __sqlite_available__ = True
except ImportError:
    __sqlite_available__ = False

try:
    from .backends.postgres import PostgresRAG, PostgresConfig

    __postgres_available__ = True
except ImportError:
    __postgres_available__ = False

from typing import Optional, List, Dict, Any, Any
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import Engine


class SQLRAG:
    """Main interface for SQL RAG system."""

    def __init__(
        self,
        engine: AsyncEngine,
        documents_table: str = "documents",
        embeddings_table: str = "embeddings",
        embedding_dimension: int = 768,
        embeddings_column: str = "vector",
        embedding_provider: Optional[EmbeddingProvider] = None,
        # Column name parameters
        documents_id_column: str = "id",
        documents_content_column: str = "content",
        documents_metadata_column: Optional[str] = "doc_metadata",
        embeddings_id_column: str = "id",
        embeddings_document_id_column: str = "document_id",
        embeddings_model_column: Optional[str] = "model_name",
    ) -> None:
        self.config = RAGConfig(
            engine=engine,
            documents_table=documents_table,
            embeddings_table=embeddings_table,
            embedding_dimension=embedding_dimension,
            embeddings_column=embeddings_column,
            documents_id_column=documents_id_column,
            documents_content_column=documents_content_column,
            documents_metadata_column=documents_metadata_column,
            embeddings_id_column=embeddings_id_column,
            embeddings_document_id_column=embeddings_document_id_column,
            embeddings_model_column=embeddings_model_column,
        )

        self.embedding_service = EmbeddingService(
            provider=embedding_provider, dimension=embedding_dimension
        )

        self.query_interface = QueryInterface(self.config, self.embedding_service)
        self.loader_interface = LoaderInterface(self.config, self.embedding_service)

    async def create_tables(self) -> None:
        """Create database tables for documents and embeddings."""
        async with self.config.engine.begin() as conn:
            await conn.run_sync(self.config.metadata.create_all)

    async def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> str:
        """Load a single document."""
        doc_data = DocumentData(content, metadata, document_id)
        return await self.loader_interface.load_document(doc_data, generate_embedding)

    async def load_documents(
        self, documents: List[Dict[str, Any]], generate_embeddings: bool = True
    ) -> List[str]:
        """Load multiple documents."""
        docs_data = [
            DocumentData(
                content=doc["content"],
                metadata=doc.get("metadata"),
                document_id=doc.get("document_id"),
            )
            for doc in documents
        ]
        return await self.loader_interface.load_documents_batch(
            docs_data, generate_embeddings
        )

    async def query(
        self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity."""
        return await self.query_interface.query(query_text, top_k, similarity_threshold)

    async def query_batch(
        self, query_texts: List[str], top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        return await self.query_interface.query_batch(
            query_texts, top_k, similarity_threshold
        )

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return await self.loader_interface.get_document(document_id)

    async def get_documents(self, document_ids: List[str]) -> List[Document]:
        """Get multiple documents by IDs."""
        return await self.loader_interface.get_documents_batch(document_ids)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document."""
        return await self.loader_interface.delete_document(document_id)

    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        regenerate_embedding: bool = True,
    ) -> bool:
        """Update a document."""
        return await self.loader_interface.update_document(
            document_id, content, metadata, regenerate_embedding
        )


class SyncSQLRAG:
    """Synchronous interface for SQL RAG system."""

    def __init__(
        self,
        engine: Engine,
        documents_table: str = "documents",
        embeddings_table: str = "embeddings",
        embedding_dimension: int = 768,
        embeddings_column: str = "vector",
        embedding_provider: Optional[Any] = None,
        # Column name parameters
        documents_id_column: str = "id",
        documents_content_column: str = "content",
        documents_metadata_column: Optional[str] = "doc_metadata",
        embeddings_id_column: str = "id",
        embeddings_document_id_column: str = "document_id",
        embeddings_model_column: Optional[str] = "model_name",
    ) -> None:
        self.config = SyncRAGConfig(
            engine=engine,
            documents_table=documents_table,
            embeddings_table=embeddings_table,
            embedding_dimension=embedding_dimension,
            embeddings_column=embeddings_column,
            documents_id_column=documents_id_column,
            documents_content_column=documents_content_column,
            documents_metadata_column=documents_metadata_column,
            embeddings_id_column=embeddings_id_column,
            embeddings_document_id_column=embeddings_document_id_column,
            embeddings_model_column=embeddings_model_column,
        )

        self.embedding_service = SyncEmbeddingService(
            provider=embedding_provider, dimension=embedding_dimension
        )

        self.query_interface = SyncQueryInterface(self.config, self.embedding_service)
        self.loader_interface = SyncLoaderInterface(self.config, self.embedding_service)

    def create_tables(self) -> None:
        """Create database tables for documents and embeddings."""
        with self.config.engine.begin() as conn:
            self.config.metadata.create_all(conn)

    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> str:
        """Load a single document."""
        doc_data = DocumentData(content, metadata, document_id)
        return self.loader_interface.load_document(doc_data, generate_embedding)

    def load_documents(
        self, documents: List[Dict[str, Any]], generate_embeddings: bool = True
    ) -> List[str]:
        """Load multiple documents."""
        docs_data = [
            DocumentData(
                content=doc["content"],
                metadata=doc.get("metadata"),
                document_id=doc.get("document_id"),
            )
            for doc in documents
        ]
        return self.loader_interface.load_documents_batch(
            docs_data, generate_embeddings
        )

    def query(
        self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity."""
        return self.query_interface.query(query_text, top_k, similarity_threshold)

    def query_batch(
        self, query_texts: List[str], top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        return self.query_interface.query_batch(
            query_texts, top_k, similarity_threshold
        )

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.loader_interface.get_document(document_id)

    def get_documents(self, document_ids: List[str]) -> List[Document]:
        """Get multiple documents by IDs."""
        return self.loader_interface.get_documents_batch(document_ids)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document."""
        return self.loader_interface.delete_document(document_id)

    def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        regenerate_embedding: bool = True,
    ) -> bool:
        """Update a document."""
        return self.loader_interface.update_document(
            document_id, content, metadata, regenerate_embedding
        )


__all__ = [
    "SQLRAG",
    "SyncSQLRAG",
    "RAGConfig",
    "SyncRAGConfig",
    "Document",
    "Embedding",
    "Base",
    "EmbeddingProvider",
    "DefaultEmbeddingProvider",
    "EmbeddingService",
    "DefaultSyncEmbeddingProvider",
    "SyncEmbeddingService",
    "QueryInterface",
    "SyncQueryInterface",
    "LoaderInterface",
    "SyncLoaderInterface",
    "DocumentData",
    # Protocols
    "EmbeddingProviderProtocol",
    "DatabaseConfigProtocol",
    "DocumentLoaderProtocol",
    "DocumentQuerierProtocol",
    "RAGSystemProtocol",
    "DocumentProtocol",
    "EmbeddingProtocol",
    "QueryResultProtocol",
    # Exceptions
    "SQLRAGError",
    "ConfigurationError",
    "EmbeddingError",
    "QueryError",
    "LoaderError",
]

# Add DuckDB exports if available
if __duckdb_available__:
    __all__.extend(["DuckDBRAG", "DuckDBConfig"])

# Add SQLite exports if available
if __sqlite_available__:
    __all__.extend(["SQLiteRAG", "SQLiteConfig"])

# Add PostgreSQL exports if available
if __postgres_available__:
    __all__.extend(["PostgresRAG", "PostgresConfig"])
