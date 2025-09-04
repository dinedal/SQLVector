"""Main DuckDB RAG interface combining loader and querier functionality."""

from typing import List, Dict, Any, Optional, Union
import polars as pl
from pathlib import Path

from ...embedding import (
    EmbeddingService,
    EmbeddingProvider,
    DefaultEmbeddingProvider,
    SyncEmbeddingService,
)
# QueryResult removed - using raw dictionaries

try:
    from sqlalchemy import Engine, create_engine
    from sqlalchemy.pool import StaticPool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
from .config import DuckDBConfig
from .loader import DuckDBLoader
from .querier import DuckDBQuerier
from .models import DuckDBDocument


class DuckDBRAG:
    """High-performance RAG system using DuckDB with Polars integration.

    Features:
    - Efficient batch operations using Polars DataFrames
    - Native vector similarity functions in DuckDB
    - Optimized for large-scale document processing
    - Support for metadata filtering and complex queries
    """

    def __init__(
        self,
        # SyncSQLRAG-compatible parameters (primary interface)
        engine: Optional[Engine] = None,
        documents_table: str = "documents",
        embeddings_table: str = "embeddings",
        embedding_dimension: int = 768,
        embeddings_column: str = "embedding",
        embedding_provider: Optional[Any] = None,
        # Column name parameters
        documents_id_column: str = "id",
        documents_content_column: str = "content",
        documents_metadata_column: Optional[str] = "metadata",
        embeddings_id_column: str = "id",
        embeddings_document_id_column: str = "document_id",
        embeddings_model_column: Optional[str] = "model_name",
        # DuckDB-specific parameters
        db_path: Optional[Union[str, Path]] = None,
        batch_size: int = 1000,
        enable_vss_extension: bool = False,
        vss_enable_persistence: bool = True,
        use_sqlalchemy: Optional[bool] = None,
    ):
        """Initialize DuckDB RAG system.

        This constructor is compatible with SyncSQLRAG parameters for drop-in replacement,
        while also supporting DuckDB-specific features.

        SyncSQLRAG-compatible Args:
            engine: SQLAlchemy engine for database connection
            documents_table: Name of documents table
            embeddings_table: Name of embeddings table
            embedding_dimension: Dimension of embedding vectors
            embeddings_column: Name of embeddings column
            embedding_provider: Custom embedding provider
            documents_id_column: Name of document ID column
            documents_content_column: Name of document content column
            documents_metadata_column: Name of document metadata column
            embeddings_id_column: Name of embedding ID column
            embeddings_document_id_column: Name of embedding document ID column
            embeddings_model_column: Name of embedding model column

        DuckDB-specific Args:
            db_path: Path to DuckDB database file (or ":memory:" for in-memory)
            batch_size: Batch size for processing operations
            enable_vss_extension: Enable Vector Similarity Search extension for HNSW indexes
            vss_enable_persistence: Enable experimental persistence for HNSW indexes
            use_sqlalchemy: Whether to use SQLAlchemy engine instead of native connections
        """
        # Determine connection method
        if engine is not None:
            # Engine provided - use SQLAlchemy mode
            if use_sqlalchemy is None:
                use_sqlalchemy = True
            # Extract db_path from engine URL if not provided
            if db_path is None:
                db_url = str(engine.url)
                if "duckdb:///" in db_url:
                    db_path = db_url.replace("duckdb:///", "")
                else:
                    db_path = ":memory:"
        else:
            # No engine - use native DuckDB connections
            if use_sqlalchemy is None:
                use_sqlalchemy = False
            if db_path is None:
                db_path = ":memory:"

        # Store column mappings for compatibility
        self.column_mappings = {
            "documents_id_column": documents_id_column,
            "documents_content_column": documents_content_column,
            "documents_metadata_column": documents_metadata_column,
            "embeddings_id_column": embeddings_id_column,
            "embeddings_document_id_column": embeddings_document_id_column,
            "embeddings_model_column": embeddings_model_column,
            "embeddings_column": embeddings_column,
        }

        self.config = DuckDBConfig(
            db_path=str(db_path),
            documents_table=documents_table,
            embeddings_table=embeddings_table,
            embedding_dimension=embedding_dimension,
            batch_size=batch_size,
            enable_vss_extension=enable_vss_extension,
            vss_enable_persistence=vss_enable_persistence,
            engine=engine,
            use_sqlalchemy=use_sqlalchemy,
            # Pass custom column mappings
            documents_id_column=documents_id_column,
            documents_content_column=documents_content_column,
            documents_metadata_column=documents_metadata_column,
            embeddings_id_column=embeddings_id_column,
            embeddings_document_id_column=embeddings_document_id_column,
            embeddings_model_column=embeddings_model_column,
            embeddings_column=embeddings_column,
        )

        # Use SyncEmbeddingService for compatibility with SyncSQLRAG
        self.embedding_service = SyncEmbeddingService(
            provider=embedding_provider,
            dimension=embedding_dimension,
        )

        # Create async version for internal DuckDB operations
        self._async_embedding_service = EmbeddingService(
            provider=embedding_provider
            or DefaultEmbeddingProvider(embedding_dimension),
            dimension=embedding_dimension,
        )

        # Create and store a single connection for in-memory databases
        self._connection = None
        if str(db_path) == ":memory:":
            self._connection = self.config.get_connection()
            self.config.setup_database(self._connection)

        # Use async embedding service for internal DuckDB operations
        self.loader = DuckDBLoader(
            self.config,
            self._async_embedding_service,
            shared_connection=self._connection,
        )
        self.querier = DuckDBQuerier(
            self.config,
            self._async_embedding_service,
            shared_connection=self._connection,
        )

        # Initialize database schema for file-based databases
        if self._connection is None:
            self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize the database schema."""
        conn = self.config.get_connection()
        try:
            self.config.setup_database(conn)
        finally:
            if hasattr(conn, "close"):
                conn.close()

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

    def load_from_polars(
        self,
        df: pl.DataFrame,
        content_column: str = "content",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load documents directly from a Polars DataFrame.

        Args:
            df: Polars DataFrame containing documents
            content_column: Name of content column
            id_column: Name of ID column (auto-generated if None)
            metadata_columns: List of columns to include as metadata
            generate_embeddings: Whether to generate embeddings
            show_progress: Whether to show progress bar

        Returns:
            List of document IDs
        """
        return self.loader.load_from_polars(
            df=df,
            content_column=content_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
            generate_embeddings=generate_embeddings,
            show_progress=show_progress,
        )

    def load_from_csv(
        self,
        csv_path: Union[str, Path],
        content_column: str = "content",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load documents from a CSV file.

        Args:
            csv_path: Path to CSV file
            content_column: Name of content column
            id_column: Name of ID column (auto-generated if None)
            metadata_columns: List of columns to include as metadata
            generate_embeddings: Whether to generate embeddings
            show_progress: Whether to show progress bar

        Returns:
            List of document IDs
        """
        df = pl.read_csv(csv_path)
        return self.load_from_polars(
            df=df,
            content_column=content_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
            generate_embeddings=generate_embeddings,
            show_progress=show_progress,
        )

    def load_from_parquet(
        self,
        parquet_path: Union[str, Path],
        content_column: str = "content",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load documents from a Parquet file.

        Args:
            parquet_path: Path to Parquet file
            content_column: Name of content column
            id_column: Name of ID column (auto-generated if None)
            metadata_columns: List of columns to include as metadata
            generate_embeddings: Whether to generate embeddings
            show_progress: Whether to show progress bar

        Returns:
            List of document IDs
        """
        df = pl.read_parquet(parquet_path)
        return self.load_from_polars(
            df=df,
            content_column=content_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
            generate_embeddings=generate_embeddings,
            show_progress=show_progress,
        )

    # Document Retrieval Methods

    def get_document(self, document_id: str) -> Optional[DuckDBDocument]:
        """Get a single document by ID."""
        return self.loader.get_document(document_id)

    def get_documents(self, document_ids: List[str]) -> List[DuckDBDocument]:
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
        ef_construction: int = 128,
        ef_search: int = 64,
        M: int = 16,
        M0: Optional[int] = None,
    ) -> bool:
        """Create an HNSW index on embeddings for accelerated similarity search.

        This method creates a Hierarchical Navigable Small World (HNSW) index
        using DuckDB's Vector Similarity Search (VSS) extension. The index should
        be created after loading a large batch of documents for optimal performance.

        Args:
            index_name: Name for the HNSW index
            similarity_function: Similarity function ("cosine", "inner_product", "euclidean")
            ef_construction: Number of candidate vertices during index construction (default: 128)
            ef_search: Number of candidate vertices during search (default: 64)
            M: Maximum number of neighbors per vertex (default: 16)
            M0: Base connectivity for level 0 (default: 2 * M)

        Returns:
            True if index was created successfully

        Raises:
            LoaderError: If VSS extension is not enabled or index creation fails

        Note:
            - Requires enable_vss_extension=True when creating DuckDBRAG instance
            - For best performance, create index after loading all documents
            - Index creation can take significant time for large datasets

        Example:
            >>> rag = DuckDBRAG("mydb.db", enable_vss_extension=True)
            >>> # Load documents...
            >>> rag.create_index("my_cosine_idx", similarity_function="cosine")
        """
        return self.loader.create_index(
            index_name=index_name,
            similarity_function=similarity_function,
            ef_construction=ef_construction,
            ef_search=ef_search,
            M=M,
            M0=M0,
        )

    def delete_index(self, index_name: str) -> bool:
        """Delete an existing HNSW index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if index was deleted successfully
        """
        return self.loader.delete_index(index_name)

    def compact_index(self, index_name: str) -> bool:
        """Compact an HNSW index to remove deleted items.

        This should be called after a significant number of document deletions
        to improve index performance and query quality.

        Args:
            index_name: Name of the index to compact

        Returns:
            True if index was compacted successfully
        """
        return self.loader.compact_index(index_name)

    # Raw dictionary methods (no conversion needed)

    # Query Methods (SyncSQLRAG-compatible interface)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_hnsw_optimization: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text.

        Returns raw dictionary data with all available columns.

        Args:
            query_text: Text to search for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            similarity_function: Similarity function ("cosine", "inner_product", "euclidean")
            use_hnsw_optimization: Use HNSW index optimization if available

        Returns:
            List of raw dictionaries with complete row data
        """
        return self.querier.query(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            use_hnsw_optimization=use_hnsw_optimization,
        )

    def query_batch(
        self, query_texts: List[str], top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        return [self.query(text, top_k, similarity_threshold) for text in query_texts]

    def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        regenerate_embedding: bool = True,
    ) -> bool:
        """Update a document (SyncSQLRAG-compatible)."""
        # Note: DuckDB backend doesn't have a native update method,
        # so we delete and re-add the document
        if self.delete_document(document_id):
            if content is not None:
                self.load_document(
                    content=content,
                    metadata=metadata,
                    document_id=document_id,
                    generate_embedding=regenerate_embedding,
                )
                return True
        return False

    def create_tables(self) -> None:
        """Create database tables for documents and embeddings (SyncSQLRAG-compatible)."""
        # DuckDB tables are created automatically by the config
        pass

    # DuckDB-specific query methods (return raw dictionaries)

    def query_with_embedding_duckdb(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector."""
        return self.querier.query_with_precomputed_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
        )

    def query_batch_duckdb(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch (returns raw dictionaries)."""
        return self.querier.query_batch(
            query_texts=query_texts,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
        )

    def query_with_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters and optional similarity search.

        Args:
            filters: Dictionary of metadata filters
            query_text: Optional text query for similarity search
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            similarity_function: Similarity function

        Returns:
            List of filtered query results
        """
        return self.querier.query_by_filters(
            filters=filters,
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
        )

    def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        return self.querier.get_similar_documents(
            document_id=document_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
        )

    # Analytics and Statistics Methods

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._connection if self._connection else self.config.get_connection()

        # Use connection directly for shared connections, or context manager for file-based
        if self._connection:
            doc_count = conn.execute(
                f"SELECT COUNT(*) FROM {self.config.documents_table}"
            ).fetchone()[0]
            emb_count = conn.execute(
                f"SELECT COUNT(*) FROM {self.config.embeddings_table}"
            ).fetchone()[0]

            return {
                "total_documents": doc_count,
                "total_embeddings": emb_count,
                "embedding_dimension": self.config.embedding_dimension,
                "batch_size": self.config.batch_size,
            }
        else:
            with conn as c:
                doc_count = c.execute(
                    f"SELECT COUNT(*) FROM {self.config.documents_table}"
                ).fetchone()[0]
                emb_count = c.execute(
                    f"SELECT COUNT(*) FROM {self.config.embeddings_table}"
                ).fetchone()[0]

                return {
                    "total_documents": doc_count,
                    "total_embeddings": emb_count,
                    "embedding_dimension": self.config.embedding_dimension,
                    "batch_size": self.config.batch_size,
                }

    def export_to_polars(
        self, include_embeddings: bool = False, document_ids: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Export documents to a Polars DataFrame.

        Args:
            include_embeddings: Whether to include embedding vectors
            document_ids: Optional list of document IDs to filter

        Returns:
            Polars DataFrame with documents
        """
        conn = self._connection if self._connection else self.config.get_connection()

        # Helper function to execute the query
        def _execute_query(c):
            if include_embeddings:
                model_col = f"e.{self.config.embeddings_model_column}" if self.config.embeddings_model_column else "NULL as model_name"
                metadata_col = f"d.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                
                query = f"""
                SELECT d.{self.config.documents_id_column}, d.{self.config.documents_content_column}, {metadata_col}, e.{self.config.embeddings_column}, {model_col}
                FROM {self.config.documents_table} d
                LEFT JOIN {self.config.embeddings_table} e ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                """
            else:
                metadata_col = f"d.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                
                query = f"""
                SELECT d.{self.config.documents_id_column}, d.{self.config.documents_content_column}, {metadata_col}
                FROM {self.config.documents_table} d
                """

            if document_ids:
                # Use prepared statement for filtering
                placeholders = ",".join(["?" for _ in document_ids])
                query += f" WHERE d.{self.config.documents_id_column} IN ({placeholders})"
                
                schema = ["id", "content", "metadata"]
                if include_embeddings:
                    schema.extend(["embedding", "model_name"])
                
                return pl.DataFrame(
                    c.execute(query, document_ids).fetchall(),
                    orient="row",
                    schema=schema,
                )
            else:
                schema = ["id", "content", "metadata"]
                if include_embeddings:
                    schema.extend(["embedding", "model_name"])
                
                return pl.DataFrame(
                    c.execute(query).fetchall(),
                    orient="row",
                    schema=schema,
                )

        if self._connection:
            return _execute_query(conn)
        else:
            with conn as c:
                return _execute_query(c)

    # Context Manager Support

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # DuckDB connections are automatically closed
        pass
