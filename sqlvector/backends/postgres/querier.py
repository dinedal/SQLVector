"""PostgreSQL-specific querier implementation with pgvector similarity search."""

import asyncio
import json
from typing import List, Dict, Any, Optional

from ...embedding import EmbeddingService
from ...exceptions import QueryError
from ...logger import get_logger
from .config import PostgresConfig
from .models import PostgresEmbedding, PostgresQueryResult

# Get logger for this module
logger = get_logger(__name__)


class PostgresQuerier:
    """High-performance querier for PostgreSQL backend with pgvector similarity search."""
    
    def __init__(self, config: PostgresConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
    
    async def query_async(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text asynchronously."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.create_embedding(query_text)
            
            return await self.query_with_precomputed_embedding_async(
                query_embedding,
                top_k,
                similarity_threshold,
                similarity_function,
                **kwargs
            )
            
        except Exception as e:
            raise QueryError(f"Query failed: {e}")
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text synchronously."""
        return asyncio.run(self.query_async(
            query_text, top_k, similarity_threshold, similarity_function, **kwargs
        ))
    
    async def query_with_precomputed_embedding_async(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector asynchronously."""
        try:
            async with self.config.get_async_connection() as conn:
                # Build and execute similarity query
                results = await self._execute_similarity_query_async(
                    conn,
                    query_embedding,
                    top_k,
                    similarity_threshold,
                    similarity_function,
                    **kwargs
                )
                
                return self._format_results(results)
                
        except Exception as e:
            raise QueryError(f"Query with embedding failed: {e}")
    
    def query_with_precomputed_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector synchronously."""
        return asyncio.run(self.query_with_precomputed_embedding_async(
            query_embedding, top_k, similarity_threshold, similarity_function, **kwargs
        ))
    
    async def _execute_similarity_query_async(
        self,
        conn: Any,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float,
        similarity_function: str,
        **kwargs
    ) -> List[Any]:
        """Execute the similarity query using pgvector."""
        # Format query embedding for PostgreSQL
        embedding_str = PostgresEmbedding.format_vector(query_embedding)
        
        # Map similarity functions to pgvector operators and similarity calculations
        if similarity_function == "cosine":
            operator = "<=>"  # Cosine distance
            similarity_calc = f"1 - (e.{self.config.embeddings_column} <=> $1::vector)"
        elif similarity_function == "euclidean":
            operator = "<->"  # L2 distance
            # Convert distance to similarity (inverse)
            similarity_calc = f"1 / (1 + (e.{self.config.embeddings_column} <-> $1::vector))"
        elif similarity_function == "inner_product":
            operator = "<#>"  # Inner product (negative)
            # pgvector returns negative inner product, so negate it
            similarity_calc = f"-(e.{self.config.embeddings_column} <#> $1::vector)"
        else:
            # Default to cosine
            operator = "<=>"
            similarity_calc = f"1 - (e.{self.config.embeddings_column} <=> $1::vector)"
        
        # Build the query with optional metadata filtering
        metadata_filter = ""
        if kwargs.get("filters"):
            filters = kwargs["filters"]
            # Build JSONB filter conditions
            filter_conditions = []
            for key, value in filters.items():
                if isinstance(value, (list, dict)):
                    filter_conditions.append(
                        f"d.{self.config.documents_metadata_column}->'{key}' = '{json.dumps(value)}'::jsonb"
                    )
                else:
                    filter_conditions.append(
                        f"d.{self.config.documents_metadata_column}->'{key}' = '\"{ value}\"'::jsonb"
                    )
            
            if filter_conditions:
                metadata_filter = f"AND {' AND '.join(filter_conditions)}"
        
        # Build the main query
        query = f"""
            SELECT 
                d.{self.config.documents_id_column} as document_id,
                d.{self.config.documents_content_column} as content,
                d.{self.config.documents_metadata_column} as metadata,
                e.{self.config.embeddings_column} as embedding,
                {similarity_calc} as similarity
            FROM {self.config.embeddings_table} e
            JOIN {self.config.documents_table} d 
                ON e.{self.config.embeddings_document_id_column} = d.{self.config.documents_id_column}
            WHERE 1=1 {metadata_filter}
            ORDER BY e.{self.config.embeddings_column} {operator} $1::vector
            LIMIT $2
        """
        
        # Execute query
        if hasattr(conn, 'fetch'):
            # asyncpg connection
            rows = await conn.fetch(query, embedding_str, top_k)
            results = [dict(row) for row in rows]
        else:
            # SQLAlchemy connection
            from sqlalchemy import text
            result = await conn.execute(
                text(query),
                {"embedding": embedding_str, "limit": top_k}
            )
            results = [dict(row) for row in result.fetchall()]
        
        # Filter by similarity threshold if specified
        if similarity_threshold > 0:
            results = [r for r in results if r["similarity"] >= similarity_threshold]
        
        return results
    
    def _format_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Format query results."""
        formatted = []
        for row in results:
            result = PostgresQueryResult.from_row(row)
            formatted.append(result.to_dict())
        return formatted
    
    async def query_batch_async(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch asynchronously."""
        # Generate embeddings for all queries
        query_embeddings = await self.embedding_service.create_embeddings_batch(query_texts)
        
        # Execute queries in parallel
        tasks = [
            self.query_with_precomputed_embedding_async(
                embedding, top_k, similarity_threshold, similarity_function, **kwargs
            )
            for embedding in query_embeddings
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch synchronously."""
        return asyncio.run(self.query_batch_async(
            query_texts, top_k, similarity_threshold, similarity_function, **kwargs
        ))
    
    async def query_by_filters_async(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters and optional similarity search asynchronously."""
        if query_text:
            # Combine filters with similarity search
            kwargs["filters"] = filters
            return await self.query_async(
                query_text, top_k, similarity_threshold, similarity_function, **kwargs
            )
        else:
            # Metadata filtering only
            async with self.config.get_async_connection() as conn:
                # Build JSONB filter conditions
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, (list, dict)):
                        filter_conditions.append(
                            f"{self.config.documents_metadata_column}->'{key}' = '{json.dumps(value)}'::jsonb"
                        )
                    else:
                        filter_conditions.append(
                            f"{self.config.documents_metadata_column}->'{key}' = '\"{value}\"'::jsonb"
                        )
                
                where_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"
                
                query = f"""
                    SELECT 
                        {self.config.documents_id_column} as document_id,
                        {self.config.documents_content_column} as content,
                        {self.config.documents_metadata_column} as metadata,
                        1.0 as similarity
                    FROM {self.config.documents_table}
                    WHERE {where_clause}
                    LIMIT $1
                """
                
                if hasattr(conn, 'fetch'):
                    # asyncpg connection
                    rows = await conn.fetch(query, top_k)
                    results = [dict(row) for row in rows]
                else:
                    # SQLAlchemy connection
                    from sqlalchemy import text
                    result = await conn.execute(text(query), {"limit": top_k})
                    results = [dict(row) for row in result.fetchall()]
                
                return self._format_results(results)
    
    def query_by_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters and optional similarity search synchronously."""
        return asyncio.run(self.query_by_filters_async(
            filters, query_text, top_k, similarity_threshold, similarity_function, **kwargs
        ))
    
    async def get_similar_documents_async(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document asynchronously."""
        async with self.config.get_async_connection() as conn:
            # Get the embedding for the given document
            query = f"""
                SELECT {self.config.embeddings_column} 
                FROM {self.config.embeddings_table}
                WHERE {self.config.embeddings_document_id_column} = $1
                LIMIT 1
            """
            
            if hasattr(conn, 'fetchrow'):
                # asyncpg connection
                row = await conn.fetchrow(query, document_id)
            else:
                # SQLAlchemy connection
                from sqlalchemy import text
                result = await conn.execute(text(query), {"doc_id": document_id})
                row = result.fetchone()
            
            if not row:
                return []
            
            # Parse the embedding
            embedding = PostgresEmbedding.parse_vector(row[self.config.embeddings_column])
            
            # Find similar documents (excluding the source document)
            kwargs["exclude_document_id"] = document_id
            return await self.query_with_precomputed_embedding_async(
                embedding, top_k + 1, similarity_threshold, similarity_function, **kwargs
            )
    
    def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document synchronously."""
        return asyncio.run(self.get_similar_documents_async(
            document_id, top_k, similarity_threshold, similarity_function, **kwargs
        ))