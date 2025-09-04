"""SQLite-specific querier implementation with efficient vector operations and VSS support."""

import sqlite3
import math
from typing import List, Dict, Any, Optional, Callable

from ...embedding import EmbeddingService
from ...exceptions import QueryError
from ...logger import get_logger
from .config import SQLiteConfig
from .models import SQLiteEmbedding

# Get logger for this module
logger = get_logger(__name__)


class SQLiteQuerier:
    """High-performance querier for SQLite backend with vector similarity functions and VSS support."""
    
    def __init__(self, config: SQLiteConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
    
    def setup_similarity_functions(self, conn: sqlite3.Connection) -> None:
        """Set up vector similarity functions in SQLite."""
        # Note: SQLite functions must be registered per connection,
        # so we register them for each new connection
        
        def cosine_similarity(blob1: bytes, blob2: bytes) -> float:
            """Compute cosine similarity between two embedding blobs."""
            if not blob1 or not blob2:
                return 0.0
            
            vec1 = SQLiteEmbedding._deserialize_embedding(blob1)
            vec2 = SQLiteEmbedding._deserialize_embedding(blob2)
            
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        def inner_product(blob1: bytes, blob2: bytes) -> float:
            """Compute inner product between two embedding blobs."""
            if not blob1 or not blob2:
                return 0.0
            
            vec1 = SQLiteEmbedding._deserialize_embedding(blob1)
            vec2 = SQLiteEmbedding._deserialize_embedding(blob2)
            
            if len(vec1) != len(vec2):
                return 0.0
            
            return sum(a * b for a, b in zip(vec1, vec2))
        
        def euclidean_distance(blob1: bytes, blob2: bytes) -> float:
            """Compute Euclidean distance between two embedding blobs."""
            if not blob1 or not blob2:
                return float('inf')
            
            vec1 = SQLiteEmbedding._deserialize_embedding(blob1)
            vec2 = SQLiteEmbedding._deserialize_embedding(blob2)
            
            if len(vec1) != len(vec2):
                return float('inf')
            
            return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
        
        def embed_text(text: str) -> bytes:
            """Generate embedding for text and return as blob."""
            import asyncio
            embedding = asyncio.run(self.embedding_service.create_embedding(text))
            return SQLiteEmbedding._serialize_embedding(embedding)
        
        # Register similarity functions
        conn.create_function("cosine_similarity", 2, cosine_similarity)
        conn.create_function("inner_product", 2, inner_product)
        conn.create_function("euclidean_distance", 2, euclidean_distance)
        conn.create_function("embed_text", 1, embed_text)
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_vss_optimization: bool = False
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text using SQLite vector functions or VSS."""
        try:
            with self.config.get_connection_context() as conn:
                # Use VSS-optimized query if enabled and extension is available
                if use_vss_optimization and self.config.enable_vss_extension:
                    return self._query_with_vss(conn, query_text, top_k, similarity_threshold)
                else:
                    return self._query_standard(conn, query_text, top_k, similarity_threshold, similarity_function)
                    
        except Exception as e:
            raise QueryError(f"Query failed: {e}")
    
    def _query_with_vss(
        self,
        conn: sqlite3.Connection,
        query_text: str,
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Query using VSS extension for optimized vector search."""
        logger.debug("Using VSS-optimized query")
        # Generate query embedding
        import asyncio
        import json
        query_embedding = asyncio.run(self.embedding_service.create_embedding(query_text))
        
        # Convert embedding to JSON format for VSS (required by sqlite-vss)
        query_json = json.dumps(query_embedding)
        
        try:
            # Try with LIMIT clause first (SQLite 3.41+)
            try:
                cursor = conn.execute(f"""
                    SELECT d.*, vss.distance as similarity
                    FROM {self.config.vss_table} vss
                    JOIN {self.config.embeddings_table} e ON e.rowid = vss.rowid
                    JOIN {self.config.documents_table} d ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                    WHERE vss_search(vss.embedding, ?)
                    LIMIT {top_k}
                """, (query_json,))
            except sqlite3.OperationalError as e:
                if "LIMIT required" in str(e) or "k > 0" in str(e):
                    # Fallback to vss_search_params for older SQLite or compatibility issues
                    cursor = conn.execute(f"""
                        SELECT d.*, vss.distance as similarity
                        FROM {self.config.vss_table} vss
                        JOIN {self.config.embeddings_table} e ON e.rowid = vss.rowid
                        JOIN {self.config.documents_table} d ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                        WHERE vss_search(vss.embedding, vss_search_params(?, {top_k}))
                    """, (query_json,))
                else:
                    raise
            
            results = []
            for row in cursor.fetchall():
                # Get column names and create dictionary
                columns = [desc[0] for desc in cursor.description]
                row_dict = dict(zip(columns, row))
                
                # VSS returns distance, convert to similarity (assuming cosine distance)
                # Cosine distance ranges from 0 to 2, where 0 is most similar
                if "similarity" in row_dict:
                    distance = float(row_dict["similarity"])
                    row_dict["similarity"] = max(0.0, 1.0 - (distance / 2.0))
                
                # Apply similarity threshold
                if row_dict.get("similarity", 0) >= similarity_threshold:
                    results.append(row_dict)
            
            return results
            
        except sqlite3.OperationalError as e:
            # Fall back to standard query if VSS fails
            print(f"VSS query failed, falling back to standard: {e}")
            return self._query_standard(conn, query_text, top_k, similarity_threshold, "cosine")
    
    def _query_standard(
        self,
        conn: sqlite3.Connection,
        query_text: str,
        top_k: int,
        similarity_threshold: float,
        similarity_function: str
    ) -> List[Dict[str, Any]]:
        """Query using standard SQLite vector functions."""
        self.setup_similarity_functions(conn)
        
        # Choose similarity function
        sim_func_map = {
            "cosine": "cosine_similarity",
            "inner_product": "inner_product",
            "euclidean": "euclidean_distance"
        }
        
        sim_func = sim_func_map.get(similarity_function, "cosine_similarity")
        
        # For euclidean distance, we want smaller values (so invert the ordering)
        order_direction = "ASC" if similarity_function == "euclidean" else "DESC"
        threshold_op = "<=" if similarity_function == "euclidean" else ">="
        
        # Select documents table columns with similarity calculation
        query_sql = f"""
        SELECT d.*, {sim_func}(e.{self.config.embeddings_column}, embed_text(?)) as similarity
        FROM {self.config.documents_table} d
        JOIN {self.config.embeddings_table} e ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
        WHERE {sim_func}(e.{self.config.embeddings_column}, embed_text(?)) {threshold_op} ?
        ORDER BY similarity {order_direction}
        LIMIT ?
        """
        
        # Execute query
        cursor = conn.execute(query_sql, [query_text, query_text, similarity_threshold, top_k])
        
        # Convert to raw dictionaries with all column data
        results = []
        for row in cursor.fetchall():
            # Get column names and create dictionary
            columns = [desc[0] for desc in cursor.description]
            row_dict = dict(zip(columns, row))
            results.append(row_dict)
        
        return results
    
    def query_with_precomputed_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        use_vss_optimization: bool = False
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector."""
        try:
            with self.config.get_connection_context() as conn:
                # Use VSS-optimized query if enabled
                if use_vss_optimization and self.config.enable_vss_extension:
                    try:
                        # Convert embedding to JSON format for VSS
                        import json
                        query_json = json.dumps(query_embedding)
                        
                        model_col = f"e.{self.config.embeddings_model_column}" if self.config.embeddings_model_column else "NULL as model_name"
                        metadata_col = f"d.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                        
                        cursor = conn.execute(f"""
                            SELECT 
                                d.{self.config.documents_id_column},
                                d.{self.config.documents_content_column},
                                {metadata_col},
                                d.hash,
                                e.{self.config.embeddings_id_column} as embedding_id,
                                e.{self.config.embeddings_column},
                                {model_col},
                                vss.distance as similarity
                            FROM {self.config.vss_table} vss
                            JOIN {self.config.embeddings_table} e ON e.rowid = vss.rowid
                            JOIN {self.config.documents_table} d ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                            WHERE vss_search(vss.embedding, ?)
                            LIMIT {top_k}
                        """, (query_json,))
                        
                        results = []
                        for row in cursor.fetchall():
                            # Convert distance to similarity (cosine distance ranges 0-2)
                            distance = row[7]
                            similarity = max(0, 1.0 - (distance / 2.0))
                            
                            row_dict = {
                                "id": row[0],
                                "content": row[1],
                                "metadata": row[2],
                                "hash": row[3],
                                "embedding_id": row[4],
                                "embedding": row[5] or b"",
                                "model_name": row[6],
                                "similarity": similarity
                            }
                            
                            if row_dict["similarity"] >= similarity_threshold:
                                results.append(row_dict)
                        
                        return results
                        
                    except sqlite3.OperationalError:
                        pass  # Fall through to standard query
                
                # Standard query with precomputed embedding
                self.setup_similarity_functions(conn)
                
                # Serialize query embedding
                query_blob = SQLiteEmbedding._serialize_embedding(query_embedding)
                
                sim_func_map = {
                    "cosine": "cosine_similarity",
                    "inner_product": "inner_product",
                    "euclidean": "euclidean_distance"
                }
                
                sim_func = sim_func_map.get(similarity_function, "cosine_similarity")
                order_direction = "ASC" if similarity_function == "euclidean" else "DESC"
                threshold_op = "<=" if similarity_function == "euclidean" else ">="
                
                model_col = f"e.{self.config.embeddings_model_column}" if self.config.embeddings_model_column else "NULL as model_name"
                metadata_col = f"d.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                
                query_sql = f"""
                SELECT 
                    d.{self.config.documents_id_column},
                    d.{self.config.documents_content_column},
                    {metadata_col},
                    d.hash,
                    e.{self.config.embeddings_id_column} as embedding_id,
                    e.{self.config.embeddings_column},
                    {model_col},
                    {sim_func}(e.{self.config.embeddings_column}, ?) as similarity
                FROM {self.config.documents_table} d
                JOIN {self.config.embeddings_table} e ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                WHERE {sim_func}(e.{self.config.embeddings_column}, ?) {threshold_op} ?
                ORDER BY similarity {order_direction}
                LIMIT ?
                """
                
                cursor = conn.execute(query_sql, [query_blob, query_blob, similarity_threshold, top_k])
                
                results = []
                for row in cursor.fetchall():
                    row_dict = {
                        "id": row[0],
                        "content": row[1],
                        "metadata": row[2],
                        "hash": row[3],
                        "embedding_id": row[4],
                        "embedding": row[5],
                        "model_name": row[6],
                        "similarity": row[7]
                    }
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
        use_vss_optimization: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        try:
            results = []
            for query_text in query_texts:
                result = self.query(
                    query_text, 
                    top_k, 
                    similarity_threshold, 
                    similarity_function,
                    use_vss_optimization
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
        use_vss_optimization: bool = False
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters and optional similarity search."""
        try:
            with self.config.get_connection_context() as conn:
                # Build WHERE clause for filters
                where_conditions = []
                params = []
                
                for key, value in filters.items():
                    # Use JSON extraction for metadata filtering
                    if self.config.documents_metadata_column:
                        where_conditions.append(f"JSON_EXTRACT(d.{self.config.documents_metadata_column}, '$.{key}') = ?")
                        params.append(str(value))
                    else:
                        # Skip filtering if metadata column is disabled
                        pass
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                if query_text:
                    # Use VSS optimization if available and requested
                    if use_vss_optimization and self.config.enable_vss_extension:
                        try:
                            import asyncio
                            import json
                            query_embedding = asyncio.run(self.embedding_service.create_embedding(query_text))
                            query_json = json.dumps(query_embedding)
                            
                            # Create a temporary view with filtered documents
                            view_sql = f"""
                            CREATE TEMP VIEW filtered_docs AS
                            SELECT d.{self.config.documents_id_column} FROM {self.config.documents_table} d
                            WHERE {where_clause}
                            """
                            conn.execute(view_sql, params)
                            
                            model_col = f"e.{self.config.embeddings_model_column}" if self.config.embeddings_model_column else "NULL as model_name"
                            metadata_col = f"d.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                            
                            cursor = conn.execute(f"""
                                SELECT 
                                    d.{self.config.documents_id_column},
                                    d.{self.config.documents_content_column},
                                    {metadata_col},
                                    d.hash,
                                    e.{self.config.embeddings_id_column} as embedding_id,
                                    e.{self.config.embeddings_column},
                                    {model_col},
                                    vss.distance as similarity
                                FROM {self.config.vss_table} vss
                                JOIN {self.config.embeddings_table} e ON e.rowid = vss.rowid
                                JOIN {self.config.documents_table} d ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                                JOIN filtered_docs f ON f.{self.config.documents_id_column} = d.{self.config.documents_id_column}
                                WHERE vss_search(vss.embedding, ?)
                                LIMIT {top_k}
                            """, (query_json,))
                            
                            results = []
                            for row in cursor.fetchall():
                                # Convert distance to similarity (cosine distance ranges 0-2)
                                distance = row[7]
                                similarity = max(0, 1.0 - (distance / 2.0))
                                
                                row_dict = {
                                    "id": row[0],
                                    "content": row[1],
                                    "metadata": row[2],
                                    "hash": row[3],
                                    "embedding_id": row[4],
                                    "embedding": row[5] or b"",
                                    "model_name": row[6],
                                    "similarity": similarity
                                }
                                
                                if row_dict["similarity"] >= similarity_threshold:
                                    results.append(row_dict)
                            
                            conn.execute("DROP VIEW filtered_docs")
                            return results
                            
                        except sqlite3.OperationalError:
                            pass  # Fall through to standard query
                    
                    # Standard filtered similarity query
                    self.setup_similarity_functions(conn)
                    
                    sim_func_map = {
                        "cosine": "cosine_similarity",
                        "inner_product": "inner_product",
                        "euclidean": "euclidean_distance"
                    }
                    
                    sim_func = sim_func_map.get(similarity_function, "cosine_similarity")
                    order_direction = "ASC" if similarity_function == "euclidean" else "DESC"
                    threshold_op = "<=" if similarity_function == "euclidean" else ">="
                    
                    model_col = f"e.{self.config.embeddings_model_column}" if self.config.embeddings_model_column else "NULL as model_name"
                    metadata_col = f"d.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                    
                    query_sql = f"""
                    SELECT 
                        d.{self.config.documents_id_column},
                        d.{self.config.documents_content_column},
                        {metadata_col},
                        d.hash,
                        e.{self.config.embeddings_id_column} as embedding_id,
                        e.{self.config.embeddings_column},
                        {model_col},
                        {sim_func}(e.{self.config.embeddings_column}, embed_text(?)) as similarity
                    FROM {self.config.documents_table} d
                    JOIN {self.config.embeddings_table} e ON d.{self.config.documents_id_column} = e.{self.config.embeddings_document_id_column}
                    WHERE ({where_clause}) 
                      AND {sim_func}(e.{self.config.embeddings_column}, embed_text(?)) {threshold_op} ?
                    ORDER BY similarity {order_direction}
                    LIMIT ?
                    """
                    
                    all_params = [query_text] + params + [query_text, similarity_threshold, top_k]
                    
                else:
                    # Just filter without similarity
                    metadata_col = f"d.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                    
                    query_sql = f"""
                    SELECT 
                        d.{self.config.documents_id_column},
                        d.{self.config.documents_content_column},
                        {metadata_col},
                        d.hash,
                        NULL as embedding_id,
                        NULL as embedding,
                        NULL as model_name,
                        0.0 as similarity
                    FROM {self.config.documents_table} d
                    WHERE {where_clause}
                    LIMIT ?
                    """
                    
                    all_params = params + [top_k]
                
                cursor = conn.execute(query_sql, all_params)
                
                results = []
                for row in cursor.fetchall():
                    row_dict = {
                        "id": row[0],
                        "content": row[1],
                        "metadata": row[2],
                        "hash": row[3],
                        "embedding_id": row[4],
                        "embedding": row[5] or b"",
                        "model_name": row[6],
                        "similarity": row[7]
                    }
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
        use_vss_optimization: bool = False
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        try:
            with self.config.get_connection_context() as conn:
                self.setup_similarity_functions(conn)
                
                sim_func_map = {
                    "cosine": "cosine_similarity",
                    "inner_product": "inner_product",
                    "euclidean": "euclidean_distance"
                }
                
                sim_func = sim_func_map.get(similarity_function, "cosine_similarity")
                order_direction = "ASC" if similarity_function == "euclidean" else "DESC"
                threshold_op = "<=" if similarity_function == "euclidean" else ">="
                
                model_col = f"e2.{self.config.embeddings_model_column}" if self.config.embeddings_model_column else "NULL as model_name"
                metadata_col = f"d2.{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                
                query_sql = f"""
                SELECT 
                    d2.{self.config.documents_id_column},
                    d2.{self.config.documents_content_column},
                    {metadata_col},
                    d2.hash,
                    e2.{self.config.embeddings_id_column} as embedding_id,
                    e2.{self.config.embeddings_column},
                    {model_col},
                    {sim_func}(e1.{self.config.embeddings_column}, e2.{self.config.embeddings_column}) as similarity
                FROM {self.config.documents_table} d1
                JOIN {self.config.embeddings_table} e1 ON d1.{self.config.documents_id_column} = e1.{self.config.embeddings_document_id_column}
                JOIN {self.config.embeddings_table} e2 ON e1.{self.config.embeddings_id_column} != e2.{self.config.embeddings_id_column}
                JOIN {self.config.documents_table} d2 ON e2.{self.config.embeddings_document_id_column} = d2.{self.config.documents_id_column}
                WHERE d1.{self.config.documents_id_column} = ?
                  AND {sim_func}(e1.{self.config.embeddings_column}, e2.{self.config.embeddings_column}) {threshold_op} ?
                ORDER BY similarity {order_direction}
                LIMIT ?
                """
                
                cursor = conn.execute(query_sql, [document_id, similarity_threshold, top_k])
                
                results = []
                for row in cursor.fetchall():
                    row_dict = {
                        "id": row[0],
                        "content": row[1],
                        "metadata": row[2],
                        "hash": row[3],
                        "embedding_id": row[4],
                        "embedding": row[5],
                        "model_name": row[6],
                        "similarity": row[7]
                    }
                    results.append(row_dict)
                
                return results
                
        except Exception as e:
            raise QueryError(f"Similar documents query failed: {e}")