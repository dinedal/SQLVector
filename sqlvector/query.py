import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .config import RAGConfig, SyncRAGConfig
from .models import Document, Embedding
from .embedding import EmbeddingService, SyncEmbeddingService
from .exceptions import QueryError
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


# QueryResult class removed - queries now return raw dictionaries


class QueryInterface:
    """Interface for querying documents by similarity."""

    def __init__(self, config: RAGConfig, embedding_service: EmbeddingService) -> None:
        self.config = config
        self.embedding_service = embedding_service

    async def query(
        self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text."""
        logger.debug(f"Querying with text: '{query_text[:50]}...', top_k={top_k}, threshold={similarity_threshold}")
        try:
            query_embedding = await self.embedding_service.create_embedding(query_text)

            async with self.config.get_session() as session:
                # Use dynamic table objects from config
                docs_table = self.config.documents_table_obj
                emb_table = self.config.embeddings_table_obj

                # Get documents with their embeddings for similarity calculation
                stmt = select(
                    *[col for col in docs_table.c],
                    emb_table.c[self.config.embeddings_column].label('embedding_vector')
                ).join(
                    emb_table,
                    emb_table.c[self.config.embeddings_document_id_column]
                    == docs_table.c[self.config.documents_id_column],
                )

                result = await session.execute(stmt)

                similarities = []
                for row in result:
                    # Convert row to dictionary with document table columns only
                    row_dict = dict(row._mapping)
                    
                    # Calculate similarity using the embedding
                    if 'embedding_vector' in row_dict and row_dict['embedding_vector']:
                        import json
                        doc_vector = json.loads(row_dict['embedding_vector'])
                        similarity = self.embedding_service.similarity(
                            query_embedding, doc_vector
                        )
                        
                        # Remove the embedding vector from result and add similarity
                        del row_dict['embedding_vector']
                        row_dict['similarity'] = similarity
                        
                        if similarity >= similarity_threshold:
                            similarities.append((row_dict, similarity))

                # Sort by similarity and take top_k
                similarities.sort(key=lambda x: x[1], reverse=True)

                return [row_dict for row_dict, _ in similarities[:top_k]]

        except Exception as e:
            raise QueryError(f"Query failed: {e}")

    async def query_batch(
        self, query_texts: List[str], top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        try:
            tasks = [
                self.query(text, top_k, similarity_threshold) for text in query_texts
            ]
            return await asyncio.gather(*tasks)
        except Exception as e:
            raise QueryError(f"Batch query failed: {e}")

    async def query_by_document_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by document ID."""
        try:
            async with self.config.get_session() as session:
                # Use dynamic table objects from config
                docs_table = self.config.documents_table_obj

                # Select only documents table columns
                stmt = (
                    select(*[col for col in docs_table.c])
                    .where(docs_table.c[self.config.documents_id_column] == document_id)
                )

                result = await session.execute(stmt)
                row = result.first()

                if row:
                    # Convert row to dictionary with all column data
                    row_dict = dict(row._mapping)
                    row_dict['similarity'] = 1.0  # Default similarity for direct document lookup
                    return row_dict
                return None

        except Exception as e:
            raise QueryError(f"Query by document ID failed: {e}")


class SyncQueryInterface:
    """Synchronous interface for querying documents by similarity."""

    def __init__(
        self, config: SyncRAGConfig, embedding_service: SyncEmbeddingService
    ) -> None:
        self.config = config
        self.embedding_service = embedding_service

    def query(
        self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text."""
        logger.debug(f"Querying with text: '{query_text[:50]}...', top_k={top_k}, threshold={similarity_threshold}")
        try:
            query_embedding = self.embedding_service.create_embedding(query_text)

            with self.config.get_session() as session:
                # Use dynamic table objects from config
                docs_table = self.config.documents_table_obj
                emb_table = self.config.embeddings_table_obj

                # Get documents with their embeddings for similarity calculation
                stmt = select(
                    *[col for col in docs_table.c],
                    emb_table.c[self.config.embeddings_column].label('embedding_vector')
                ).join(
                    emb_table,
                    emb_table.c[self.config.embeddings_document_id_column]
                    == docs_table.c[self.config.documents_id_column],
                )

                result = session.execute(stmt)

                similarities = []
                for row in result:
                    # Convert row to dictionary with document table columns only
                    row_dict = dict(row._mapping)
                    
                    # Calculate similarity using the embedding
                    if 'embedding_vector' in row_dict and row_dict['embedding_vector']:
                        import json
                        doc_vector = json.loads(row_dict['embedding_vector'])
                        similarity = self.embedding_service.similarity(
                            query_embedding, doc_vector
                        )
                        
                        # Remove the embedding vector from result and add similarity
                        del row_dict['embedding_vector']
                        row_dict['similarity'] = similarity
                        
                        if similarity >= similarity_threshold:
                            similarities.append((row_dict, similarity))

                # Sort by similarity and take top_k
                similarities.sort(key=lambda x: x[1], reverse=True)

                return [row_dict for row_dict, _ in similarities[:top_k]]

        except Exception as e:
            raise QueryError(f"Query failed: {e}")

    def query_batch(
        self, query_texts: List[str], top_k: int = 5, similarity_threshold: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        try:
            return [
                self.query(text, top_k, similarity_threshold) for text in query_texts
            ]
        except Exception as e:
            raise QueryError(f"Batch query failed: {e}")

    def query_by_document_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document and embedding by document ID."""
        try:
            with self.config.get_session() as session:
                # Use dynamic table objects from config
                docs_table = self.config.documents_table_obj
                emb_table = self.config.embeddings_table_obj

                # Join column logic
                if (
                    self.config.embeddings_document_id_column
                    != self.config.embeddings_id_column
                ):
                    join_column = emb_table.c[self.config.embeddings_document_id_column]
                else:
                    join_column = emb_table.c[self.config.embeddings_id_column]

                # Select all columns from both tables
                stmt = (
                    select(*[col for col in docs_table.c] + [col for col in emb_table.c])
                    .join(
                        emb_table,
                        docs_table.c[self.config.documents_id_column] == join_column,
                    )
                    .where(docs_table.c[self.config.documents_id_column] == document_id)
                )

                result = session.execute(stmt)
                row = result.first()

                if row:
                    # Convert row to dictionary with all column data
                    row_dict = dict(row._mapping)
                    row_dict['similarity'] = 1.0  # Default similarity for direct document lookup
                    return row_dict
                return None

        except Exception as e:
            raise QueryError(f"Query by document ID failed: {e}")


import asyncio
