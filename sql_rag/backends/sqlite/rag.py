"""Main SQLite RAG interface combining loader and querier functionality."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ...embedding import EmbeddingService, EmbeddingProvider, DefaultEmbeddingProvider

try:
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import create_engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
from .config import SQLiteConfig
from .loader import SQLiteLoader
from .querier import SQLiteQuerier
from .models import SQLiteDocument, SQLiteEmbedding


class SQLiteRAG:
    """High-performance RAG system using SQLite with VSS integration.
    
    Features:
    - Efficient batch operations using native SQLite
    - Vector similarity functions using custom SQLite functions
    - Optional sqlite-vss extension for accelerated similarity search
    - Support for metadata filtering and complex queries
    - Binary storage of embeddings for space efficiency
    """
    
    def __init__(
        self,
        db_path: Union[str, Path] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_dimension: int = 768,
        batch_size: int = 1000,
        documents_table: str = "documents",
        embeddings_table: str = "embeddings",
        vss_table: str = "vss_embeddings",
        enable_vss_extension: bool = False,
        vss_factory_string: str = "Flat",
        engine: Optional[Union[AsyncEngine, Any]] = None,
        use_sqlalchemy: bool = False,
        # Column name mappings for custom schemas
        documents_id_column: str = "id",
        documents_content_column: str = "content",
        documents_metadata_column: Optional[str] = "metadata",
        embeddings_id_column: str = "id",
        embeddings_document_id_column: str = "document_id",
        embeddings_model_column: Optional[str] = "model_name",
        embeddings_column: str = "embedding",
    ):
        """Initialize SQLite RAG system.

        Args:
            db_path: Path to SQLite database file (or ":memory:" for in-memory). Optional if using SQLAlchemy engine.
            embedding_provider: Custom embedding provider (defaults to basic provider)
            embedding_dimension: Dimension of embedding vectors
            batch_size: Batch size for processing operations
            documents_table: Name of documents table
            embeddings_table: Name of embeddings table
            vss_table: Name of VSS virtual table
            enable_vss_extension: Enable sqlite-vss extension for accelerated search
            vss_factory_string: Faiss factory string for VSS index configuration
            engine: SQLAlchemy engine (optional, for SQLAlchemy-based connections with StaticPool)
            use_sqlalchemy: Whether to use SQLAlchemy engine instead of native connections
            documents_id_column: Name of the documents ID column
            documents_content_column: Name of the documents content column
            documents_metadata_column: Name of the documents metadata column (None to disable)
            embeddings_id_column: Name of the embeddings ID column
            embeddings_document_id_column: Name of the embeddings document_id column
            embeddings_model_column: Name of the embeddings model column (None to disable)
            embeddings_column: Name of the embeddings vector column
        """
        # Validate parameters
        if not use_sqlalchemy and db_path is None:
            raise ValueError("db_path is required when not using SQLAlchemy")
        if use_sqlalchemy and engine is None and db_path is None:
            raise ValueError("Either engine or db_path must be provided when using SQLAlchemy")
        
        self.config = SQLiteConfig(
            db_path=str(db_path) if db_path else ":memory:",
            documents_table=documents_table,
            embeddings_table=embeddings_table,
            vss_table=vss_table,
            embedding_dimension=embedding_dimension,
            batch_size=batch_size,
            enable_vss_extension=enable_vss_extension,
            vss_factory_string=vss_factory_string,
            engine=engine,
            use_sqlalchemy=use_sqlalchemy,
            documents_id_column=documents_id_column,
            documents_content_column=documents_content_column,
            documents_metadata_column=documents_metadata_column,
            embeddings_id_column=embeddings_id_column,
            embeddings_document_id_column=embeddings_document_id_column,
            embeddings_model_column=embeddings_model_column,
            embeddings_column=embeddings_column,
        )

        self.embedding_service = EmbeddingService(
            provider=embedding_provider
            or DefaultEmbeddingProvider(embedding_dimension),
            dimension=embedding_dimension,
        )

        self.loader = SQLiteLoader(self.config, self.embedding_service)
        self.querier = SQLiteQuerier(self.config, self.embedding_service)

        # Initialize database schema
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize the database schema."""
        with self.config.get_connection_context() as conn:
            self.config.setup_database(conn)

    # Document Loading Methods

    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> str:
        """Load a single document.

        Args:
            content: Document text content
            metadata: Optional metadata dictionary
            document_id: Optional custom document ID
            generate_embedding: Whether to generate embedding

        Returns:
            Document ID
        """
        return self.loader.load_document(
            content=content,
            metadata=metadata,
            document_id=document_id,
            generate_embedding=generate_embedding,
        )

    def load_documents(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load multiple documents efficiently.

        Args:
            documents: List of document dicts with 'content', optional 'metadata', 'document_id'
            generate_embeddings: Whether to generate embeddings
            show_progress: Whether to show progress bar

        Returns:
            List of document IDs
        """
        return self.loader.load_documents_batch(
            documents=documents,
            generate_embeddings=generate_embeddings,
            show_progress=show_progress,
        )

    # Document Retrieval Methods

    def get_document(self, document_id: str) -> Optional[SQLiteDocument]:
        """Get a single document by ID."""
        return self.loader.get_document(document_id)

    def get_documents(self, document_ids: List[str]) -> List[SQLiteDocument]:
        """Get multiple documents by IDs."""
        return self.loader.get_documents_batch(document_ids)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        return self.loader.delete_document(document_id)

    # Index Management Methods

    def create_index(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        factory_string: Optional[str] = None,
    ) -> bool:
        """Create a VSS index on embeddings for accelerated similarity search.

        Args:
            index_name: Name for the VSS index (currently only one index supported)
            similarity_function: Similarity function (for compatibility, not used in SQLite-VSS)
            factory_string: Custom Faiss factory string (e.g., "IVF4096,Flat,IDMap2")

        Returns:
            True if index was created successfully

        Raises:
            LoaderError: If VSS extension is not enabled or index creation fails

        Note:
            - Requires enable_vss_extension=True when creating SQLiteRAG instance
            - sqlite-vss uses Faiss factory strings for index configuration
            - For large datasets, consider training with IVF factory strings

        Example:
            >>> rag = SQLiteRAG("mydb.db", enable_vss_extension=True)
            >>> # Load documents...
            >>> rag.create_index("my_idx", factory_string="IVF1024,Flat,IDMap2")
        """
        return self.loader.create_index(
            index_name=index_name,
            similarity_function=similarity_function,
            factory_string=factory_string,
        )

    def delete_index(self, index_name: str) -> bool:
        """Delete VSS index (recreates table with default factory).

        Args:
            index_name: Name of the index to delete

        Returns:
            True if index was deleted successfully
        """
        return self.loader.delete_index(index_name)

    def train_index(self, training_data_limit: Optional[int] = None) -> bool:
        """Train VSS index with existing embeddings (for IVF factory strings).

        Args:
            training_data_limit: Limit on number of training vectors (None for all)

        Returns:
            True if training was successful
        """
        return self.loader.train_index(training_data_limit)

    # Query Methods

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_vss_optimization: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text.

        Args:
            query_text: Text to search for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            similarity_function: Similarity function ("cosine", "inner_product", "euclidean")
            use_vss_optimization: Use VSS index optimization if available

        Returns:
            List of query results sorted by similarity
        """
        return self.querier.query(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            use_vss_optimization=use_vss_optimization,
        )

    def query_with_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_vss_optimization: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector."""
        return self.querier.query_with_precomputed_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            use_vss_optimization=use_vss_optimization,
        )

    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_vss_optimization: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        return self.querier.query_batch(
            query_texts=query_texts,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            use_vss_optimization=use_vss_optimization,
        )

    def query_with_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_vss_optimization: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters and optional similarity search.

        Args:
            filters: Dictionary of metadata filters
            query_text: Optional text query for similarity search
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            similarity_function: Similarity function
            use_vss_optimization: Use VSS index optimization if available

        Returns:
            List of filtered query results
        """
        return self.querier.query_by_filters(
            filters=filters,
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            use_vss_optimization=use_vss_optimization,
        )

    def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_vss_optimization: bool = False,
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        return self.querier.get_similar_documents(
            document_id=document_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            use_vss_optimization=use_vss_optimization,
        )

    # Analytics and Statistics Methods

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.config.get_connection_context() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {self.config.documents_table}")
            doc_count = cursor.fetchone()[0]
            
            cursor = conn.execute(f"SELECT COUNT(*) FROM {self.config.embeddings_table}")
            emb_count = cursor.fetchone()[0]

            return {
                "total_documents": doc_count,
                "total_embeddings": emb_count,
                "embedding_dimension": self.config.embedding_dimension,
                "batch_size": self.config.batch_size,
                "vss_enabled": self.config.enable_vss_extension,
                "vss_factory_string": self.config.vss_factory_string,
            }

    def export_to_dict(
        self, 
        include_embeddings: bool = False, 
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Export documents to a list of dictionaries.

        Args:
            include_embeddings: Whether to include embedding vectors
            document_ids: Optional list of document IDs to filter

        Returns:
            List of document dictionaries
        """
        with self.config.get_connection_context() as conn:
            if include_embeddings:
                model_col = f"e.{self.config.embeddings_model_column}" if self.config.embeddings_model_column else "NULL as model_name"
                query = f"""
                SELECT d.{self.config.documents_id_column}, d.{self.config.documents_content_column}, d.{self.config.documents_metadata_column or 'NULL as metadata'}, d.hash, e.{self.config.embeddings_column}, {model_col}
                FROM {self.config.documents_table} d
                LEFT JOIN {self.config.embeddings_table} e ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                """
            else:
                metadata_col = f"d.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                query = f"""
                SELECT d.{self.config.documents_id_column}, d.{self.config.documents_content_column}, {metadata_col}, d.hash
                FROM {self.config.documents_table} d
                """

            if document_ids:
                placeholders = ",".join(["?" for _ in document_ids])
                query += f" WHERE d.{self.config.documents_id_column} IN ({placeholders})"
                cursor = conn.execute(query, document_ids)
            else:
                cursor = conn.execute(query)

            results = []
            for row in cursor.fetchall():
                row_dict = {
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "hash": row[3],
                }
                
                if include_embeddings and len(row) > 4:
                    if row[4]:  # embedding blob
                        row_dict["embedding"] = SQLiteEmbedding._deserialize_embedding(row[4])
                    else:
                        row_dict["embedding"] = []
                    row_dict["model_name"] = row[5]
                
                results.append(row_dict)

            return results

    # Context Manager Support

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Close shared connection for in-memory databases
        self.config.close_shared_connection()