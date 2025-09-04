"""DuckDB-specific loader implementation with efficient batch operations."""

import uuid
from typing import List, Dict, Any, Optional, Union
import polars as pl
import numpy as np
from tqdm import tqdm

from ...embedding import EmbeddingService
from ...exceptions import LoaderError
from ...logger import get_logger
from .config import DuckDBConfig
from .models import DuckDBDocument, DuckDBEmbedding

# Get logger for this module
logger = get_logger(__name__)


class DuckDBLoader:
    """High-performance loader for DuckDB backend using Polars and batch operations."""

    def __init__(
        self,
        config: DuckDBConfig,
        embedding_service: EmbeddingService,
        shared_connection=None,
    ):
        self.config = config
        self.embedding_service = embedding_service
        self.shared_connection = shared_connection

    def _get_connection_context(self):
        """Get connection context manager."""
        if self.shared_connection:
            # For shared connections, return a dummy context manager
            class DummyContext:
                def __init__(self, connection):
                    self.connection = connection

                def __enter__(self):
                    return self.connection

                def __exit__(self, *args):
                    pass

            return DummyContext(self.shared_connection)
        else:
            # Handle SQLAlchemy connections that don't support context manager
            conn = self.config.get_connection()

            class ConnectionContext:
                def __init__(self, connection):
                    self.connection = connection

                def __enter__(self):
                    return self.connection

                def __exit__(self, *args):
                    if hasattr(self.connection, "close"):
                        self.connection.close()

            return ConnectionContext(conn)

    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> str:
        """Load a single document."""
        return self.load_documents_batch(
            [{"content": content, "metadata": metadata, "document_id": document_id}],
            generate_embeddings=generate_embedding,
        )[0]

    def load_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load multiple documents efficiently using batch operations."""
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)

                # Prepare documents
                doc_records = []
                for doc_data in documents:
                    doc_id = doc_data.get("document_id") or str(uuid.uuid4())
                    doc = DuckDBDocument(
                        id=doc_id,
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata"),
                    )
                    doc_records.append(doc.to_dict())

                # Create Polars DataFrame for efficient batch insertion
                docs_df = pl.DataFrame(doc_records)

                # Insert documents using prepared statements to avoid PyArrow dependency
                metadata_col = f"{self.config.documents_metadata_column}, " if self.config.documents_metadata_column else ""
                metadata_val = "?, " if self.config.documents_metadata_column else ""
                metadata_update = f"{self.config.documents_metadata_column} = EXCLUDED.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else ""
                update_clause = f", {metadata_update}" if metadata_update else ""
                
                for doc in doc_records:
                    values = [doc["id"], doc["content"]]
                    if self.config.documents_metadata_column:
                        values.append(doc["metadata"])
                    
                    conn.execute(
                        f"""
                        INSERT INTO {self.config.documents_table} ({self.config.documents_id_column}, {self.config.documents_content_column}{', ' + metadata_col.rstrip(', ') if metadata_col else ''})
                        VALUES (?, ?{', ?' if self.config.documents_metadata_column else ''})
                        ON CONFLICT ({self.config.documents_id_column}) DO UPDATE SET 
                            {self.config.documents_content_column} = EXCLUDED.{self.config.documents_content_column}{update_clause}
                    """,
                        values,
                    )

                document_ids = [doc["id"] for doc in doc_records]

                if generate_embeddings:
                    self._generate_embeddings_batch(
                        conn, docs_df, show_progress=show_progress
                    )

                return document_ids

        except Exception as e:
            raise LoaderError(f"Failed to load documents batch: {e}")

    def load_from_polars(
        self,
        df: pl.DataFrame,
        content_column: str = "content",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load documents directly from a Polars DataFrame."""
        try:
            # Prepare the DataFrame
            if id_column is None:
                # Generate unique IDs for each row
                uuids = [str(uuid.uuid4()) for _ in range(len(df))]
                df = df.with_columns(pl.Series("id", uuids))
                id_column = "id"

            # Create metadata column if specified
            if metadata_columns:
                df = df.with_columns(
                    pl.struct(metadata_columns)
                    .map_elements(
                        lambda x: x if isinstance(x, dict) else {},
                        return_dtype=pl.Object,
                    )
                    .alias("metadata_dict")
                )
            else:
                df = df.with_columns(pl.lit({}).alias("metadata_dict"))

            # Select and rename columns - no hash column needed anymore
            columns_to_select = [
                pl.col(id_column).alias("id"),
                pl.col(content_column).alias("content"),
            ]
            
            if self.config.documents_metadata_column:
                columns_to_select.append(
                    pl.col("metadata_dict")
                    .map_elements(
                        lambda x: __import__("json").dumps(x) if x else "{}",
                        return_dtype=pl.String,
                    )
                    .alias("metadata")
                )
            
            df = df.select(columns_to_select)

            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)

                # Insert documents using iterator to avoid PyArrow dependency
                metadata_col = f"{self.config.documents_metadata_column}, " if self.config.documents_metadata_column else ""
                metadata_update = f"{self.config.documents_metadata_column} = EXCLUDED.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else ""
                update_clause = f", {metadata_update}" if metadata_update else ""
                
                for row in df.iter_rows(named=True):
                    values = [row["id"], row["content"]]
                    if self.config.documents_metadata_column:
                        values.append(row["metadata"])
                    
                    conn.execute(
                        f"""
                        INSERT INTO {self.config.documents_table} ({self.config.documents_id_column}, {self.config.documents_content_column}{', ' + metadata_col.rstrip(', ') if metadata_col else ''})
                        VALUES (?, ?{', ?' if self.config.documents_metadata_column else ''})
                        ON CONFLICT ({self.config.documents_id_column}) DO UPDATE SET 
                            {self.config.documents_content_column} = EXCLUDED.{self.config.documents_content_column}{update_clause}
                    """,
                        values,
                    )

                if generate_embeddings:
                    self._generate_embeddings_batch(
                        conn, df, show_progress=show_progress
                    )

                return df.select("id").to_series().to_list()

        except Exception as e:
            raise LoaderError(f"Failed to load from Polars DataFrame: {e}")

    def _generate_embeddings_batch(
        self, conn, docs_df: pl.DataFrame, show_progress: bool = True
    ) -> None:
        """Generate embeddings in batches for loaded documents."""
        # Get document IDs that don't have embeddings yet
        doc_ids = [row["id"] for row in docs_df.iter_rows(named=True)]

        if not doc_ids:
            return

        placeholders = ",".join(["?" for _ in doc_ids])

        existing_docs = conn.execute(
            f"""
            SELECT {self.config.embeddings_document_id_column} FROM {self.config.embeddings_table}
            WHERE {self.config.embeddings_document_id_column} IN ({placeholders})
        """,
            doc_ids,
        ).fetchall()
        existing_doc_set = {row[0] for row in existing_docs}

        # Filter to only unembedded documents
        unembedded_rows = [
            row
            for row in docs_df.iter_rows(named=True)
            if row["id"] not in existing_doc_set
        ]

        if not unembedded_rows:
            return
        
        logger.info(f"Generating embeddings for {len(unembedded_rows)} documents")

        # Process in batches
        total_docs = len(unembedded_rows)
        batch_size = self.config.batch_size

        progress_bar = (
            tqdm(total=total_docs, desc="Generating embeddings")
            if show_progress
            else None
        )

        try:
            for i in range(0, total_docs, batch_size):
                batch_rows = unembedded_rows[i : i + batch_size]

                if not batch_rows:
                    break

                # Extract content for embedding
                contents = [row["content"] for row in batch_rows]

                # Generate embeddings using the async service (run synchronously here)
                import asyncio

                embeddings = asyncio.run(
                    self.embedding_service.create_embeddings_batch(contents)
                )

                # Convert to numpy array for efficiency
                embeddings_array = np.array(embeddings, dtype=np.float32)

                # Insert embeddings directly
                model_col = f"{self.config.embeddings_model_column}, " if self.config.embeddings_model_column else ""
                model_val = ", ?" if self.config.embeddings_model_column else ""
                
                for row, embedding in zip(batch_rows, embeddings_array):
                    values = [
                        str(uuid.uuid4()),
                        row["id"],
                        embedding.tolist(),
                    ]
                    if self.config.embeddings_model_column:
                        values.append(getattr(self.embedding_service.provider, "model_name", "default"))
                    
                    conn.execute(
                        f"""
                        INSERT INTO {self.config.embeddings_table} 
                        ({self.config.embeddings_id_column}, {self.config.embeddings_document_id_column}, {self.config.embeddings_column}{', ' + model_col.rstrip(', ') if model_col else ''})
                        VALUES (?, ?, ?{model_val})
                    """,
                        values,
                    )

                if progress_bar:
                    progress_bar.update(len(batch_rows))

        finally:
            if progress_bar:
                progress_bar.close()

    def get_document(self, document_id: str) -> Optional[DuckDBDocument]:
        """Get a single document by ID."""
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)
                metadata_col = f"{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                
                result = conn.execute(
                    f"""
                    SELECT {self.config.documents_id_column}, {self.config.documents_content_column}, {metadata_col}
                    FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} = ?
                """,
                    [document_id],
                ).fetchone()

                if result:
                    return DuckDBDocument.from_dict(
                        {
                            "id": result[0],
                            "content": result[1],
                            "metadata": result[2],
                        }
                    )
                return None

        except Exception as e:
            raise LoaderError(f"Failed to get document: {e}")

    def get_documents_batch(self, document_ids: List[str]) -> List[DuckDBDocument]:
        """Get multiple documents by IDs."""
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)
                # Use prepared statement for batch query
                placeholders = ",".join(["?" for _ in document_ids])
                metadata_col = f"{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                
                result = conn.execute(
                    f"""
                    SELECT {self.config.documents_id_column}, {self.config.documents_content_column}, {metadata_col}
                    FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} IN ({placeholders})
                """,
                    document_ids,
                ).fetchall()

                return [
                    DuckDBDocument.from_dict(
                        {
                            "id": row[0],
                            "content": row[1],
                            "metadata": row[2],
                        }
                    )
                    for row in result
                ]

        except Exception as e:
            raise LoaderError(f"Failed to get documents batch: {e}")

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)

                # Check if document exists first
                check_result = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} = ?
                """,
                    [document_id],
                ).fetchone()

                if check_result[0] == 0:
                    return False

                # Delete embeddings first
                conn.execute(
                    f"""
                    DELETE FROM {self.config.embeddings_table}
                    WHERE {self.config.embeddings_document_id_column} = ?
                """,
                    [document_id],
                )

                # Delete document
                conn.execute(
                    f"""
                    DELETE FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} = ?
                """,
                    [document_id],
                )

                # Verify deletion
                verify_result = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} = ?
                """,
                    [document_id],
                ).fetchone()

                return verify_result[0] == 0

        except Exception as e:
            raise LoaderError(f"Failed to delete document: {e}")


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
            - Requires enable_vss_extension=True in DuckDBConfig
            - For best performance, create index after loading all documents
            - Index creation can take significant time for large datasets
        """
        if not self.config.enable_vss_extension:
            raise LoaderError(
                "VSS extension is not enabled. Set enable_vss_extension=True in DuckDBConfig."
            )

        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)

                # Map similarity functions to VSS metrics
                metric_map = {
                    "cosine": "cosine",
                    "inner_product": "ip",
                    "euclidean": "l2sq",
                }

                metric = metric_map.get(similarity_function, "cosine")
                if M0 is None:
                    M0 = 2 * M

                # Create HNSW index
                logger.info(f"Creating HNSW index '{index_name}' with metric '{metric}'")
                index_sql = f"""
                CREATE INDEX {index_name} 
                ON {self.config.embeddings_table} 
                USING HNSW (embedding)
                WITH (
                    metric = '{metric}',
                    ef_construction = {ef_construction},
                    ef_search = {ef_search},
                    M = {M},
                    M0 = {M0}
                )
                """

                conn.execute(index_sql)
                logger.info(f"HNSW index '{index_name}' created successfully")
                return True

        except Exception as e:
            raise LoaderError(f"Failed to create HNSW index '{index_name}': {e}")

    def delete_index(self, index_name: str) -> bool:
        """Delete an existing HNSW index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if index was deleted successfully
        """
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)

                conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                return True

        except Exception as e:
            raise LoaderError(f"Failed to delete index '{index_name}': {e}")

    def compact_index(self, index_name: str) -> bool:
        """Compact an HNSW index to remove deleted items.

        This should be called after a significant number of document deletions
        to improve index performance and query quality.

        Args:
            index_name: Name of the index to compact

        Returns:
            True if index was compacted successfully
        """
        if not self.config.enable_vss_extension:
            raise LoaderError(
                "VSS extension is not enabled. Set enable_vss_extension=True in DuckDBConfig."
            )

        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)

                conn.execute(f"PRAGMA hnsw_compact_index('{index_name}')")
                return True

        except Exception as e:
            raise LoaderError(f"Failed to compact index '{index_name}': {e}")
