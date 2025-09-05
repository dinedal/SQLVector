"""Protocol definitions for SQLVector interfaces."""

from typing import List, Dict, Any, Optional, Union, Protocol, runtime_checkable
from pathlib import Path

# Import types that will be used in protocols but avoid circular imports
# These will be forward references that get resolved at runtime


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...


@runtime_checkable
class DatabaseConfigProtocol(Protocol):
    """Protocol for database configuration classes."""
    
    # Configuration attributes
    db_path: str
    documents_table: str
    embeddings_table: str
    embedding_dimension: int
    batch_size: int
    
    def get_connection(self) -> Any:
        """Get a database connection."""
        ...
    
    def get_documents_schema(self) -> str:
        """Get the CREATE TABLE SQL for documents."""
        ...
    
    def get_embeddings_schema(self) -> str:
        """Get the CREATE TABLE SQL for embeddings."""
        ...
    
    def setup_database(self, conn: Any) -> None:
        """Set up the database schema and functions."""
        ...


@runtime_checkable
class DocumentLoaderProtocol(Protocol):
    """Protocol for document loading operations."""
    
    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True
    ) -> str:
        """Load a single document."""
        ...
    
    def load_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True
    ) -> List[str]:
        """Load multiple documents efficiently."""
        ...
    
    def get_document(self, document_id: str) -> Any:
        """Get a single document by ID."""
        ...
    
    def get_documents_batch(self, document_ids: List[str]) -> List[Any]:
        """Get multiple documents by IDs."""
        ...
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        ...
    
    def create_index(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        **kwargs
    ) -> bool:
        """Create an index on embeddings for accelerated similarity search."""
        ...
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an existing index."""
        ...


@runtime_checkable
class DocumentQuerierProtocol(Protocol):
    """Protocol for document querying operations."""
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Any]:
        """Query documents by similarity to query text."""
        ...
    
    def query_with_precomputed_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Any]:
        """Query using a precomputed embedding vector."""
        ...
    
    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[List[Any]]:
        """Query multiple texts in batch."""
        ...
    
    def query_by_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Any]:
        """Query documents with metadata filters and optional similarity search."""
        ...
    
    def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Any]:
        """Find documents similar to a given document."""
        ...


@runtime_checkable
class RAGSystemProtocol(Protocol):
    """Protocol for complete RAG system implementations."""
    
    # Document Loading Methods
    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> str:
        """Load a single document."""
        ...

    def load_documents(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load multiple documents efficiently."""
        ...

    # Document Retrieval Methods
    def get_document(self, document_id: str) -> Any:
        """Get a single document by ID."""
        ...

    def get_documents(self, document_ids: List[str]) -> List[Any]:
        """Get multiple documents by IDs."""
        ...

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        ...

    # Index Management Methods
    def create_index(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        **kwargs
    ) -> bool:
        """Create an index on embeddings for accelerated similarity search."""
        ...

    def delete_index(self, index_name: str) -> bool:
        """Delete an existing index."""
        ...

    # Query Methods
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Any]:
        """Query documents by similarity to query text."""
        ...

    def query_with_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Any]:
        """Query using a precomputed embedding vector."""
        ...

    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[List[Any]]:
        """Query multiple texts in batch."""
        ...

    def query_with_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Any]:
        """Query documents with metadata filters and optional similarity search."""
        ...

    def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Any]:
        """Find documents similar to a given document."""
        ...

    # Analytics and Statistics Methods
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        ...


@runtime_checkable
class DocumentProtocol(Protocol):
    """Protocol for document model classes."""
    
    id: str
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentProtocol":
        """Create document from dictionary."""
        ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for embedding model classes."""
    
    id: str
    document_id: str
    
    def get_vector(self) -> List[float]:
        """Get embedding vector as list of floats."""
        ...
    
    def set_vector(self, vector: List[float]) -> None:
        """Set embedding vector from list of floats."""
        ...


@runtime_checkable
class QueryResultProtocol(Protocol):
    """Protocol for query result classes."""
    
    document_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query result to dictionary."""
        ...