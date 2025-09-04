"""DuckDB-specific querier implementation with efficient vector operations."""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
import polars as pl

from ...embedding import EmbeddingService
from ...exceptions import QueryError
from ...logger import get_logger
from .config import DuckDBConfig
# DuckDBQueryResult removed - using raw dictionaries

# Get logger for this module
logger = get_logger(__name__)


class DuckDBQuerier:
    """High-performance querier for DuckDB backend with vector similarity functions."""

    def __init__(
        self,
        config: DuckDBConfig,
        embedding_service: EmbeddingService,
        shared_connection=None,
    ):
        self.config = config
        self.embedding_service = embedding_service
        self.shared_connection = shared_connection
        self._embedding_function_registered = False

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

    def setup_vector_functions(self, conn) -> None:
        """Set up vector similarity functions in DuckDB."""
        # For shared connections, only register once
        if self.shared_connection and self._embedding_function_registered:
            return

        # Create embedding function for real-time queries
        def embed_text(text: str) -> List[float]:
            import asyncio

            embedding = asyncio.run(self.embedding_service.create_embedding(text))
            # Convert to float32 to match DuckDB FLOAT[] type expectations
            return [float(x) for x in embedding]

        # Register the embedding function
        conn.create_function(
            "embed",
            embed_text,
            ["VARCHAR"],
            f"FLOAT[{self.config.embedding_dimension}]",
        )

        # Create vector similarity functions
        def array_inner_product(vec1: List[float], vec2: List[float]) -> float:
            """Compute inner product of two vectors."""
            if len(vec1) != len(vec2):
                return 0.0
            return sum(a * b for a, b in zip(vec1, vec2))

        def array_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """Compute cosine similarity between two vectors."""
            if len(vec1) != len(vec2):
                return 0.0

            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        def array_euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
            """Compute Euclidean distance between two vectors."""
            if len(vec1) != len(vec2):
                return float("inf")
            return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5

        # Register vector functions with specific types (drop and recreate to avoid conflicts)
        try:
            conn.execute("DROP FUNCTION IF EXISTS array_inner_product")
            conn.execute("DROP FUNCTION IF EXISTS array_cosine_similarity")
            conn.execute("DROP FUNCTION IF EXISTS array_euclidean_distance")
        except:
            pass  # Functions may not exist yet

        conn.create_function(
            "array_inner_product", array_inner_product, ["FLOAT[]", "FLOAT[]"], "FLOAT"
        )

        conn.create_function(
            "array_cosine_similarity",
            array_cosine_similarity,
            ["FLOAT[]", "FLOAT[]"],
            "FLOAT",
        )

        conn.create_function(
            "array_euclidean_distance",
            array_euclidean_distance,
            ["FLOAT[]", "FLOAT[]"],
            "FLOAT",
        )

        # Only set the flag for shared connections to avoid re-registration
        if self.shared_connection:
            self._embedding_function_registered = True

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_hnsw_optimization: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text using DuckDB vector functions."""
        logger.debug(f"Querying with DuckDB backend, similarity_function={similarity_function}, use_hnsw={use_hnsw_optimization}")
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)
                self.setup_vector_functions(conn)

                # Choose similarity function
                sim_func_map = {
                    "cosine": "array_cosine_similarity",
                    "inner_product": "array_inner_product",
                    "euclidean": "array_euclidean_distance",
                }

                sim_func = sim_func_map.get(
                    similarity_function, "array_cosine_similarity"
                )

                # For euclidean distance, we want smaller values (so invert the ordering)
                order_direction = (
                    "ASC" if similarity_function == "euclidean" else "DESC"
                )

                # Use HNSW-optimized query if enabled and VSS extension is available
                if use_hnsw_optimization and self.config.enable_vss_extension:
                    # Map to DuckDB VSS functions
                    vss_func_map = {
                        "cosine": "array_cosine_distance",
                        "inner_product": "array_negative_inner_product",
                        "euclidean": "array_distance",
                    }
                    vss_func = vss_func_map.get(
                        similarity_function, "array_cosine_distance"
                    )

                    # Select documents table columns with similarity calculation
                    query_sql = f"""
                    SELECT d.*, {vss_func}(e.{self.config.embeddings_column}, embed(?)) as similarity
                    FROM {self.config.embeddings_table} e
                    JOIN {self.config.documents_table} d ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                    ORDER BY {vss_func}(e.{self.config.embeddings_column}, embed(?))
                    LIMIT ?
                    """

                    # Execute query with HNSW optimization (no threshold filtering in VSS mode)
                    result = conn.execute(
                        query_sql, [query_text, query_text, top_k]
                    ).fetchall()
                else:
                    # Select documents table columns with similarity calculation
                    query_sql = f"""
                    SELECT d.*, {sim_func}(e.{self.config.embeddings_column}, embed(?)) as similarity
                    FROM {self.config.documents_table} d
                    JOIN {self.config.embeddings_table} e ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                    WHERE {sim_func}(e.{self.config.embeddings_column}, embed(?)) {'<=' if similarity_function == 'euclidean' else '>='} ?
                    ORDER BY similarity {order_direction}
                    LIMIT ?
                    """

                    # Execute query
                    result = conn.execute(
                        query_sql, [query_text, query_text, similarity_threshold, top_k]
                    ).fetchall()

                # Convert to raw dictionaries with all column data
                results = []
                for row in result:
                    # Get column names from the connection description
                    columns = [desc[0] for desc in conn.description]
                    row_dict = dict(zip(columns, row))
                    results.append(row_dict)

                return results

        except Exception as e:
            raise QueryError(f"Query failed: {e}")

    def query_with_precomputed_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector."""
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)
                self.setup_vector_functions(conn)

                # Create a temporary approach without using register (to avoid PyArrow dependency)
                # We'll pass the embedding directly in the query

                # Choose similarity function
                sim_func_map = {
                    "cosine": "array_cosine_similarity",
                    "inner_product": "array_inner_product",
                    "euclidean": "array_euclidean_distance",
                }

                sim_func = sim_func_map.get(
                    similarity_function, "array_cosine_similarity"
                )
                order_direction = (
                    "ASC" if similarity_function == "euclidean" else "DESC"
                )

                # Select documents table columns with similarity calculation
                query_sql = f"""
                SELECT d.*, {sim_func}(e.{self.config.embeddings_column}, ?::FLOAT[{self.config.embedding_dimension}]) as similarity
                FROM {self.config.documents_table} d
                JOIN {self.config.embeddings_table} e ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                WHERE {sim_func}(e.{self.config.embeddings_column}, ?::FLOAT[{self.config.embedding_dimension}]) {'<=' if similarity_function == 'euclidean' else '>='} ?
                ORDER BY similarity {order_direction}
                LIMIT ?
                """

                # Convert query embedding to float32 to match DuckDB FLOAT[] type
                query_embedding_f32 = [float(x) for x in query_embedding]

                result = conn.execute(
                    query_sql,
                    [
                        query_embedding_f32,
                        query_embedding_f32,
                        similarity_threshold,
                        top_k,
                    ],
                ).fetchall()

                # Convert to raw dictionaries with all column data
                results = []
                for row in result:
                    # Get column names from the connection description
                    columns = [desc[0] for desc in conn.description]
                    row_dict = dict(zip(columns, row))
                    results.append(row_dict)

                return results

        except Exception as e:
            raise QueryError(f"Query with precomputed embedding failed: {e}")

    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        try:
            # For now, process sequentially - could be optimized with parallel execution
            results = []
            for query_text in query_texts:
                result = self.query(
                    query_text, top_k, similarity_threshold, similarity_function
                )
                results.append(result)

            return results

        except Exception as e:
            raise QueryError(f"Batch query failed: {e}")

    def query_by_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters and optional similarity search."""
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)

                # Build WHERE clause for filters
                where_conditions = []
                params = []

                for key, value in filters.items():
                    # Use JSON extraction for metadata filtering if metadata column exists
                    if self.config.documents_metadata_column:
                        # JSON_EXTRACT_STRING returns the string value, not quoted JSON
                        where_conditions.append(
                            f"JSON_EXTRACT_STRING(d.{self.config.documents_metadata_column}, '$.{key}') = ?"
                        )
                        params.append(str(value))
                    else:
                        # If no metadata column, this filter will never match
                        where_conditions.append("1=0")  # Always false condition

                where_clause = (
                    " AND ".join(where_conditions) if where_conditions else "1=1"
                )

                if query_text:
                    self.setup_vector_functions(conn)

                    sim_func_map = {
                        "cosine": "array_cosine_similarity",
                        "inner_product": "array_inner_product",
                        "euclidean": "array_euclidean_distance",
                    }

                    sim_func = sim_func_map.get(
                        similarity_function, "array_cosine_similarity"
                    )
                    order_direction = (
                        "ASC" if similarity_function == "euclidean" else "DESC"
                    )

                    # Select documents table columns with similarity calculation
                    query_sql = f"""
                    SELECT d.*, {sim_func}(e.{self.config.embeddings_column}, embed(?)) as similarity
                    FROM {self.config.documents_table} d
                    JOIN {self.config.embeddings_table} e ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                    WHERE ({where_clause}) 
                      AND {sim_func}(e.{self.config.embeddings_column}, embed(?)) {'<=' if similarity_function == 'euclidean' else '>='} ?
                    ORDER BY similarity {order_direction}
                    LIMIT ?
                    """

                    params = (
                        [query_text]
                        + params
                        + [query_text, similarity_threshold, top_k]
                    )

                else:
                    # Use SELECT * for documents only (no embeddings for non-similarity query)
                    query_sql = f"""
                    SELECT d.*, 0.0 as similarity
                    FROM {self.config.documents_table} d
                    WHERE {where_clause}
                    LIMIT ?
                    """

                    params.append(top_k)

                result = conn.execute(query_sql, params).fetchall()

                # Convert to raw dictionaries with all column data
                results = []
                for row in result:
                    # Get column names from the connection description
                    columns = [desc[0] for desc in conn.description]
                    row_dict = dict(zip(columns, row))
                    results.append(row_dict)

                return results

        except Exception as e:
            raise QueryError(f"Filtered query failed: {e}")

    def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        try:
            with self._get_connection_context() as conn:
                if not self.shared_connection:  # Only setup if not already done
                    self.config.setup_database(conn)
                self.setup_vector_functions(conn)

                sim_func_map = {
                    "cosine": "array_cosine_similarity",
                    "inner_product": "array_inner_product",
                    "euclidean": "array_euclidean_distance",
                }

                sim_func = sim_func_map.get(
                    similarity_function, "array_cosine_similarity"
                )
                order_direction = (
                    "ASC" if similarity_function == "euclidean" else "DESC"
                )

                # Select documents table columns with similarity calculation
                query_sql = f"""
                SELECT d2.*, {sim_func}(e1.{self.config.embeddings_column}, e2.{self.config.embeddings_column}) as similarity
                FROM {self.config.documents_table} d1
                JOIN {self.config.embeddings_table} e1 ON d1.{self.config.documents_id_column} = e1.{self.config.embeddings_document_id_column}
                JOIN {self.config.embeddings_table} e2 ON e1.{self.config.embeddings_id_column} != e2.{self.config.embeddings_id_column}
                JOIN {self.config.documents_table} d2 ON e2.{self.config.embeddings_document_id_column} = d2.{self.config.documents_id_column}
                WHERE d1.{self.config.documents_id_column} = ?
                  AND {sim_func}(e1.{self.config.embeddings_column}, e2.{self.config.embeddings_column}) {'<=' if similarity_function == 'euclidean' else '>='} ?
                ORDER BY similarity {order_direction}
                LIMIT ?
                """

                result = conn.execute(
                    query_sql, [document_id, similarity_threshold, top_k]
                ).fetchall()

                # Convert to raw dictionaries with all column data
                results = []
                for row in result:
                    # Get column names from the connection description
                    columns = [desc[0] for desc in conn.description]
                    row_dict = dict(zip(columns, row))
                    results.append(row_dict)

                return results

        except Exception as e:
            raise QueryError(f"Similar documents query failed: {e}")
