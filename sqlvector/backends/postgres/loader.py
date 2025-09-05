"""PostgreSQL-specific loader implementation with efficient batch operations."""

import uuid
import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from ...embedding import EmbeddingService
from ...exceptions import LoaderError
from ...logger import get_logger
from .config import PostgresConfig
from .models import PostgresDocument, PostgresEmbedding

# Get logger for this module
logger = get_logger(__name__)


class PostgresLoader:
    """High-performance loader for PostgreSQL backend with batch operations and pgvector support."""
    
    def __init__(self, config: PostgresConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
    
    async def load_document_async(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True
    ) -> str:
        """Load a single document asynchronously."""
        return await self.load_documents_batch_async([{
            "content": content,
            "metadata": metadata,
            "document_id": document_id
        }], generate_embeddings=generate_embedding)[0]
    
    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True
    ) -> str:
        """Load a single document synchronously."""
        return asyncio.run(self.load_document_async(
            content, metadata, document_id, generate_embedding
        ))
    
    async def load_documents_batch_async(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True
    ) -> List[str]:
        """Load multiple documents efficiently using batch operations (async)."""
        try:
            async with self.config.get_async_connection() as conn:
                # Prepare documents
                doc_records = []
                for doc_data in documents:
                    doc_id = doc_data.get("document_id") or str(uuid.uuid4())
                    doc = PostgresDocument(
                        id=doc_id,
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata")
                    )
                    doc_records.append(doc)
                
                # Insert documents
                await self._insert_documents_async(conn, doc_records)
                
                document_ids = [doc.id for doc in doc_records]
                
                if generate_embeddings:
                    await self._generate_embeddings_batch_async(
                        conn, 
                        doc_records, 
                        show_progress=show_progress
                    )
                
                return document_ids
                
        except Exception as e:
            raise LoaderError(f"Failed to load documents batch: {e}")
    
    def load_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True
    ) -> List[str]:
        """Load multiple documents efficiently using batch operations (sync)."""
        return asyncio.run(self.load_documents_batch_async(
            documents, generate_embeddings, show_progress
        ))
    
    async def _insert_documents_async(
        self, 
        conn: Any, 
        doc_records: List[PostgresDocument]
    ) -> None:
        """Insert documents into database asynchronously."""
        # Build the query based on whether metadata column exists
        if self.config.documents_metadata_column:
            query = f"""
                INSERT INTO {self.config.documents_table} 
                ({self.config.documents_id_column}, 
                 {self.config.documents_content_column}, 
                 {self.config.documents_metadata_column})
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT ({self.config.documents_id_column}) 
                DO UPDATE SET 
                    {self.config.documents_content_column} = EXCLUDED.{self.config.documents_content_column},
                    {self.config.documents_metadata_column} = EXCLUDED.{self.config.documents_metadata_column}
            """
            values = [
                (doc.id, doc.content, json.dumps(doc.metadata or {}))
                for doc in doc_records
            ]
        else:
            query = f"""
                INSERT INTO {self.config.documents_table} 
                ({self.config.documents_id_column}, 
                 {self.config.documents_content_column})
                VALUES ($1, $2)
                ON CONFLICT ({self.config.documents_id_column}) 
                DO UPDATE SET 
                    {self.config.documents_content_column} = EXCLUDED.{self.config.documents_content_column}
            """
            values = [(doc.id, doc.content) for doc in doc_records]
        
        # Use executemany for batch insert
        if hasattr(conn, 'executemany'):
            # asyncpg connection
            await conn.executemany(query, values)
        else:
            # SQLAlchemy connection
            from sqlalchemy import text
            await conn.execute(text(query), values)
    
    async def _generate_embeddings_batch_async(
        self,
        conn: Any,
        doc_records: List[PostgresDocument],
        show_progress: bool = True
    ) -> None:
        """Generate embeddings in batches for loaded documents."""
        logger.info(f"Generating embeddings for {len(doc_records)} documents")
        
        # Process in batches
        total_docs = len(doc_records)
        batch_size = self.config.batch_size
        
        progress_bar = tqdm(total=total_docs, desc="Generating embeddings") if show_progress else None
        
        try:
            for i in range(0, total_docs, batch_size):
                batch_docs = doc_records[i:i + batch_size]
                
                # Extract content for embedding
                contents = [doc.content for doc in batch_docs]
                
                # Generate embeddings
                embeddings = await self.embedding_service.create_embeddings_batch(contents)
                
                # Prepare embedding records
                embedding_records = []
                for doc, embedding in zip(batch_docs, embeddings):
                    embedding_id = str(uuid.uuid4())
                    emb = PostgresEmbedding(
                        id=embedding_id,
                        document_id=doc.id,
                        embedding=embedding,
                        model_name=getattr(
                            self.embedding_service.provider, 'model_name', None
                        )
                    )
                    embedding_records.append(emb)
                
                # Insert embeddings
                await self._insert_embeddings_async(conn, embedding_records)
                
                if progress_bar:
                    progress_bar.update(len(batch_docs))
                    
        finally:
            if progress_bar:
                progress_bar.close()
    
    async def _insert_embeddings_async(
        self, 
        conn: Any, 
        embedding_records: List[PostgresEmbedding]
    ) -> None:
        """Insert embeddings into database asynchronously."""
        # Build the query based on whether model column exists
        if self.config.embeddings_model_column:
            query = f"""
                INSERT INTO {self.config.embeddings_table} 
                ({self.config.embeddings_id_column}, 
                 {self.config.embeddings_document_id_column}, 
                 {self.config.embeddings_column},
                 {self.config.embeddings_model_column})
                VALUES ($1, $2, $3::vector, $4)
                ON CONFLICT ({self.config.embeddings_id_column}) DO NOTHING
            """
            values = [
                (emb.id, emb.document_id, PostgresEmbedding.format_vector(emb.embedding), emb.model_name)
                for emb in embedding_records
            ]
        else:
            query = f"""
                INSERT INTO {self.config.embeddings_table} 
                ({self.config.embeddings_id_column}, 
                 {self.config.embeddings_document_id_column}, 
                 {self.config.embeddings_column})
                VALUES ($1, $2, $3::vector)
                ON CONFLICT ({self.config.embeddings_id_column}) DO NOTHING
            """
            values = [
                (emb.id, emb.document_id, PostgresEmbedding.format_vector(emb.embedding))
                for emb in embedding_records
            ]
        
        # Use executemany for batch insert
        if hasattr(conn, 'executemany'):
            # asyncpg connection
            await conn.executemany(query, values)
        else:
            # SQLAlchemy connection
            from sqlalchemy import text
            await conn.execute(text(query), values)
    
    async def get_document_async(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID asynchronously."""
        async with self.config.get_async_connection() as conn:
            query = f"""
                SELECT * FROM {self.config.documents_table}
                WHERE {self.config.documents_id_column} = $1
            """
            
            if hasattr(conn, 'fetchrow'):
                # asyncpg connection
                row = await conn.fetchrow(query, document_id)
                if row:
                    return dict(row)
            else:
                # SQLAlchemy connection
                from sqlalchemy import text
                result = await conn.execute(text(query), {"id": document_id})
                row = result.fetchone()
                if row:
                    return dict(row)
            
            return None
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID synchronously."""
        return asyncio.run(self.get_document_async(document_id))
    
    async def get_documents_batch_async(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents by IDs asynchronously."""
        async with self.config.get_async_connection() as conn:
            # Create parameterized placeholders
            placeholders = ','.join([f'${i+1}' for i in range(len(document_ids))])
            query = f"""
                SELECT * FROM {self.config.documents_table}
                WHERE {self.config.documents_id_column} IN ({placeholders})
            """
            
            if hasattr(conn, 'fetch'):
                # asyncpg connection
                rows = await conn.fetch(query, *document_ids)
                return [dict(row) for row in rows]
            else:
                # SQLAlchemy connection
                from sqlalchemy import text
                # For SQLAlchemy, we need named parameters
                params = {f"id_{i}": doc_id for i, doc_id in enumerate(document_ids)}
                placeholders_named = ','.join([f':id_{i}' for i in range(len(document_ids))])
                query_named = f"""
                    SELECT * FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} IN ({placeholders_named})
                """
                result = await conn.execute(text(query_named), params)
                return [dict(row) for row in result.fetchall()]
    
    def get_documents_batch(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents by IDs synchronously."""
        return asyncio.run(self.get_documents_batch_async(document_ids))
    
    async def delete_document_async(self, document_id: str) -> bool:
        """Delete a document and its embeddings asynchronously."""
        async with self.config.get_async_connection() as conn:
            # Delete from documents table (embeddings will cascade delete)
            query = f"""
                DELETE FROM {self.config.documents_table}
                WHERE {self.config.documents_id_column} = $1
            """
            
            if hasattr(conn, 'execute'):
                # asyncpg connection
                result = await conn.execute(query, document_id)
                # asyncpg returns a string like "DELETE 1"
                return "1" in result
            else:
                # SQLAlchemy connection
                from sqlalchemy import text
                result = await conn.execute(text(query), {"id": document_id})
                return result.rowcount > 0
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings synchronously."""
        return asyncio.run(self.delete_document_async(document_id))
    
    async def create_index_async(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        **kwargs
    ) -> bool:
        """Create an index on embeddings for accelerated similarity search."""
        try:
            async with self.config.get_async_connection() as conn:
                # Update config with any provided kwargs
                if "index_type" in kwargs:
                    self.config.index_type = kwargs["index_type"]
                if "index_lists" in kwargs:
                    self.config.index_lists = kwargs["index_lists"]
                if "index_m" in kwargs:
                    self.config.index_m = kwargs["index_m"]
                if "index_ef_construction" in kwargs:
                    self.config.index_ef_construction = kwargs["index_ef_construction"]
                
                # Get the index creation SQL
                index_sql = self.config.get_index_schema(index_name, similarity_function)
                
                if hasattr(conn, 'execute'):
                    # asyncpg connection
                    await conn.execute(index_sql)
                else:
                    # SQLAlchemy connection
                    from sqlalchemy import text
                    await conn.execute(text(index_sql))
                
                logger.info(f"Created index {index_name} with {similarity_function} similarity")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def create_index(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        **kwargs
    ) -> bool:
        """Create an index on embeddings for accelerated similarity search."""
        return asyncio.run(self.create_index_async(index_name, similarity_function, **kwargs))
    
    async def delete_index_async(self, index_name: str) -> bool:
        """Delete an existing index asynchronously."""
        try:
            async with self.config.get_async_connection() as conn:
                query = f"DROP INDEX IF EXISTS {index_name}"
                
                if hasattr(conn, 'execute'):
                    # asyncpg connection
                    await conn.execute(query)
                else:
                    # SQLAlchemy connection
                    from sqlalchemy import text
                    await conn.execute(text(query))
                
                logger.info(f"Deleted index {index_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an existing index synchronously."""
        return asyncio.run(self.delete_index_async(index_name))