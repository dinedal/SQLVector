"""PostgreSQL RAG system with pgvector support."""

import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ...embedding import EmbeddingService, EmbeddingProvider, DefaultEmbeddingProvider
from .config import PostgresConfig
from .loader import PostgresLoader
from .querier import PostgresQuerier
from .models import PostgresDocument, PostgresEmbedding


class PostgresRAG:
    """High-performance RAG system using PostgreSQL with pgvector.
    
    Features:
    - Native pgvector support for efficient vector operations
    - HNSW and IVFFlat indexing for fast similarity search
    - Async and sync operation modes
    - SQLAlchemy integration
    - Connection pooling with asyncpg
    - JSONB metadata filtering
    - Batch operations for efficient data loading
    """
    
    def __init__(
        self,
        # Database connection parameters
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        db_url: Optional[str] = None,
        # Embedding configuration
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_dimension: int = 768,
        # Table configuration
        batch_size: int = 1000,
        documents_table: str = "documents",
        embeddings_table: str = "embeddings",
        # pgvector index configuration
        index_type: str = "ivfflat",
        index_lists: int = 100,
        index_m: int = 16,
        index_ef_construction: int = 64,
        # SQLAlchemy support
        engine: Optional[Any] = None,
        use_sqlalchemy: bool = False,
        use_async_sqlalchemy: bool = True,
        # Column name mappings for custom schemas
        documents_id_column: str = "id",
        documents_content_column: str = "content",
        documents_metadata_column: Optional[str] = "metadata",
        embeddings_id_column: str = "id",
        embeddings_document_id_column: str = "document_id",
        embeddings_model_column: Optional[str] = "model_name",
        embeddings_column: str = "embedding",
        # Connection pool settings
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ):
        """Initialize PostgreSQL RAG system."""
        # Validate parameters
        if not db_url and not host:
            raise ValueError("Either db_url or host must be provided")
        
        self.config = PostgresConfig(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            db_url=db_url,
            documents_table=documents_table,
            embeddings_table=embeddings_table,
            embedding_dimension=embedding_dimension,
            batch_size=batch_size,
            index_type=index_type,
            index_lists=index_lists,
            index_m=index_m,
            index_ef_construction=index_ef_construction,
            engine=engine,
            use_sqlalchemy=use_sqlalchemy,
            use_async_sqlalchemy=use_async_sqlalchemy,
            documents_id_column=documents_id_column,
            documents_content_column=documents_content_column,
            documents_metadata_column=documents_metadata_column,
            embeddings_id_column=embeddings_id_column,
            embeddings_document_id_column=embeddings_document_id_column,
            embeddings_model_column=embeddings_model_column,
            embeddings_column=embeddings_column,
            pool_min_size=pool_min_size,
            pool_max_size=pool_max_size,
        )

        self.embedding_service = EmbeddingService(
            provider=embedding_provider or DefaultEmbeddingProvider(embedding_dimension),
            dimension=embedding_dimension,
        )

        self.loader = PostgresLoader(self.config, self.embedding_service)
        self.querier = PostgresQuerier(self.config, self.embedding_service)

        # Initialize database schema
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize the database schema."""
        # Run async initialization
        asyncio.run(self._initialize_database_async())
    
    async def _initialize_database_async(self) -> None:
        """Initialize the database schema asynchronously."""
        async with self.config.get_async_connection() as conn:
            await self.config.setup_database(conn)

    # Document Loading Methods
    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> str:
        """Load a single document."""
        return self.loader.load_document(
            content=content,
            metadata=metadata,
            document_id=document_id,
            generate_embedding=generate_embedding,
        )
    
    async def load_document_async(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> str:
        """Load a single document asynchronously."""
        return await self.loader.load_document_async(
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
        """Load multiple documents efficiently."""
        return self.loader.load_documents_batch(
            documents=documents,
            generate_embeddings=generate_embeddings,
            show_progress=show_progress,
        )
    
    async def load_documents_async(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load multiple documents efficiently asynchronously."""
        return await self.loader.load_documents_batch_async(
            documents=documents,
            generate_embeddings=generate_embeddings,
            show_progress=show_progress,
        )

    # Document Retrieval Methods
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID."""
        return self.loader.get_document(document_id)
    
    async def get_document_async(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID asynchronously."""
        return await self.loader.get_document_async(document_id)

    def get_documents(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents by IDs."""
        return self.loader.get_documents_batch(document_ids)
    
    async def get_documents_async(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents by IDs asynchronously."""
        return await self.loader.get_documents_batch_async(document_ids)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        return self.loader.delete_document(document_id)
    
    async def delete_document_async(self, document_id: str) -> bool:
        """Delete a document and its embeddings asynchronously."""
        return await self.loader.delete_document_async(document_id)

    # Index Management Methods
    def create_index(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        **kwargs
    ) -> bool:
        """Create an index on embeddings."""
        return self.loader.create_index(
            index_name=index_name,
            similarity_function=similarity_function,
            **kwargs
        )
    
    async def create_index_async(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        **kwargs
    ) -> bool:
        """Create an index on embeddings asynchronously."""
        return await self.loader.create_index_async(
            index_name=index_name,
            similarity_function=similarity_function,
            **kwargs
        )

    def delete_index(self, index_name: str) -> bool:
        """Delete an existing index."""
        return self.loader.delete_index(index_name)
    
    async def delete_index_async(self, index_name: str) -> bool:
        """Delete an existing index asynchronously."""
        return await self.loader.delete_index_async(index_name)

    # Query Methods
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text."""
        return self.querier.query(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )
    
    async def query_async(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text asynchronously."""
        return await self.querier.query_async(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    def query_with_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector."""
        return self.querier.query_with_precomputed_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )
    
    async def query_with_embedding_async(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector asynchronously."""
        return await self.querier.query_with_precomputed_embedding_async(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        return self.querier.query_batch(
            query_texts=query_texts,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )
    
    async def query_batch_async(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch asynchronously."""
        return await self.querier.query_batch_async(
            query_texts=query_texts,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    def query_with_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters."""
        return self.querier.query_by_filters(
            filters=filters,
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )
    
    async def query_with_filters_async(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters asynchronously."""
        return await self.querier.query_by_filters_async(
            filters=filters,
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        return self.querier.get_similar_documents(
            document_id=document_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )
    
    async def find_similar_documents_async(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document asynchronously."""
        return await self.querier.get_similar_documents_async(
            document_id=document_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    # Analytics and Statistics Methods
    async def get_statistics_async(self) -> Dict[str, Any]:
        """Get database statistics asynchronously."""
        async with self.config.get_async_connection() as conn:
            # Get document count
            doc_count_query = f"SELECT COUNT(*) FROM {self.config.documents_table}"
            
            # Get embedding count
            emb_count_query = f"SELECT COUNT(*) FROM {self.config.embeddings_table}"
            
            # Get index information
            index_query = """
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = $1
            """
            
            if hasattr(conn, 'fetchval'):
                # asyncpg connection
                doc_count = await conn.fetchval(doc_count_query)
                emb_count = await conn.fetchval(emb_count_query)
                indexes = await conn.fetch(index_query, self.config.embeddings_table)
                index_list = [dict(row) for row in indexes]
            else:
                # SQLAlchemy connection
                from sqlalchemy import text
                doc_result = await conn.execute(text(doc_count_query))
                doc_count = doc_result.scalar()
                
                emb_result = await conn.execute(text(emb_count_query))
                emb_count = emb_result.scalar()
                
                index_result = await conn.execute(
                    text(index_query),
                    {"table": self.config.embeddings_table}
                )
                index_list = [dict(row) for row in index_result.fetchall()]
            
            return {
                "document_count": doc_count,
                "embedding_count": emb_count,
                "embedding_dimension": self.config.embedding_dimension,
                "indexes": index_list,
                "tables": {
                    "documents": self.config.documents_table,
                    "embeddings": self.config.embeddings_table
                }
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return asyncio.run(self.get_statistics_async())
    
    async def close(self):
        """Close the connection pool."""
        await self.config.close_pool()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.close())
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()