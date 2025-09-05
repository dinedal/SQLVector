from typing import List, Dict, Any, Optional, Union
import uuid
import asyncio
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .config import RAGConfig, SyncRAGConfig
from .models import Document, Embedding
from .embedding import EmbeddingService, SyncEmbeddingService
from .exceptions import LoaderError
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


class DocumentData:
    """Data structure for document loading."""

    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> None:
        self.content = content
        self.metadata = metadata or {}
        self.document_id = document_id or str(uuid.uuid4())


class LoaderInterface:
    """Interface for loading and managing documents."""

    def __init__(self, config: RAGConfig, embedding_service: EmbeddingService) -> None:
        self.config = config
        self.embedding_service = embedding_service

    async def load_document(
        self, document_data: DocumentData, generate_embedding: bool = True
    ) -> str:
        """Load a single document into the database."""
        logger.info(f"Loading document with ID: {document_data.document_id}")
        try:
            async with self.config.get_session() as session:
                # Use dynamic document model from config
                DocumentModel = self.config.DocumentModel

                # Create document using dynamic model
                document = DocumentModel()
                setattr(
                    document, self.config.documents_id_column, document_data.document_id
                )
                setattr(
                    document,
                    self.config.documents_content_column,
                    document_data.content,
                )
                document.set_metadata(document_data.metadata)

                session.add(document)

                # Generate and store embedding if requested
                if generate_embedding:
                    embedding_vector = await self.embedding_service.create_embedding(
                        document_data.content
                    )

                    # Use dynamic embedding model from config
                    EmbeddingModel = self.config.EmbeddingModel
                    embedding = EmbeddingModel()
                    setattr(
                        embedding, self.config.embeddings_id_column, str(uuid.uuid4())
                    )
                    setattr(
                        embedding,
                        self.config.embeddings_document_id_column,
                        document_data.document_id,
                    )
                    embedding.set_vector(embedding_vector)

                    # Set model name if column exists
                    if self.config.embeddings_model_column:
                        setattr(
                            embedding,
                            self.config.embeddings_model_column,
                            getattr(
                                self.embedding_service.provider, "model_name", "default"
                            ),
                        )

                    session.add(embedding)

                await session.commit()
                return document_data.document_id

        except Exception as e:
            raise LoaderError(f"Failed to load document: {e}")

    async def load_documents_batch(
        self, documents_data: List[DocumentData], generate_embeddings: bool = True
    ) -> List[str]:
        """Load multiple documents in batch."""
        logger.info(f"Loading batch of {len(documents_data)} documents")
        try:
            async with self.config.get_session() as session:
                document_ids = []

                # Create documents
                for doc_data in documents_data:
                    document = Document(
                        id=doc_data.document_id, content=doc_data.content
                    )
                    document.set_metadata(doc_data.metadata)
                    session.add(document)
                    document_ids.append(doc_data.document_id)

                # Generate embeddings in batch if requested
                if generate_embeddings:
                    contents = [doc.content for doc in documents_data]
                    embeddings_vectors = (
                        await self.embedding_service.create_embeddings_batch(contents)
                    )

                    for doc_data, embedding_vector in zip(
                        documents_data, embeddings_vectors
                    ):
                        embedding = Embedding(
                            id=str(uuid.uuid4()),
                            document_id=doc_data.document_id,
                            model_name=getattr(
                                self.embedding_service.provider, "model_name", "default"
                            ),
                        )
                        embedding.set_vector(embedding_vector)
                        session.add(embedding)

                await session.commit()
                return document_ids

        except Exception as e:
            raise LoaderError(f"Failed to load documents batch: {e}")

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a single document by ID."""
        try:
            async with self.config.get_session() as session:
                stmt = select(Document).where(Document.id == document_id)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except Exception as e:
            raise LoaderError(f"Failed to get document: {e}")

    async def get_documents_batch(self, document_ids: List[str]) -> List[Document]:
        """Get multiple documents by IDs."""
        try:
            async with self.config.get_session() as session:
                stmt = select(Document).where(Document.id.in_(document_ids))
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as e:
            raise LoaderError(f"Failed to get documents batch: {e}")

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        try:
            async with self.config.get_session() as session:
                # Delete embeddings first
                await session.execute(
                    delete(Embedding).where(Embedding.document_id == document_id)
                )

                # Delete document
                result = await session.execute(
                    delete(Document).where(Document.id == document_id)
                )

                await session.commit()
                return result.rowcount > 0

        except Exception as e:
            raise LoaderError(f"Failed to delete document: {e}")

    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        regenerate_embedding: bool = True,
    ) -> bool:
        """Update a document and optionally regenerate its embedding."""
        try:
            async with self.config.get_session() as session:
                stmt = select(Document).where(Document.id == document_id)
                result = await session.execute(stmt)
                document = result.scalar_one_or_none()

                if not document:
                    return False

                # Update document
                if content is not None:
                    document.content = content
                if metadata is not None:
                    document.set_metadata(metadata)

                # Regenerate embedding if content changed
                if content is not None and regenerate_embedding:
                    # Delete old embedding
                    await session.execute(
                        delete(Embedding).where(Embedding.document_id == document_id)
                    )

                    # Create new embedding
                    embedding_vector = await self.embedding_service.create_embedding(
                        content
                    )
                    embedding = Embedding(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        model_name=getattr(
                            self.embedding_service.provider, "model_name", "default"
                        ),
                    )
                    embedding.set_vector(embedding_vector)
                    session.add(embedding)

                await session.commit()
                return True

        except Exception as e:
            raise LoaderError(f"Failed to update document: {e}")


class SyncLoaderInterface:
    """Synchronous interface for loading and managing documents."""

    def __init__(
        self, config: SyncRAGConfig, embedding_service: SyncEmbeddingService
    ) -> None:
        self.config = config
        self.embedding_service = embedding_service

    def load_document(
        self, document_data: DocumentData, generate_embedding: bool = True
    ) -> str:
        """Load a single document into the database."""
        try:
            with self.config.get_session() as session:
                # Use dynamic document model from config
                DocumentModel = self.config.DocumentModel

                # Create document using dynamic model
                document = DocumentModel()
                setattr(
                    document, self.config.documents_id_column, document_data.document_id
                )
                setattr(
                    document,
                    self.config.documents_content_column,
                    document_data.content,
                )
                document.set_metadata(document_data.metadata)

                session.add(document)

                # Generate and store embedding if requested
                if generate_embedding:
                    embedding_vector = self.embedding_service.create_embedding(
                        document_data.content
                    )

                    # Use dynamic embedding model from config
                    EmbeddingModel = self.config.EmbeddingModel
                    embedding = EmbeddingModel()
                    setattr(
                        embedding, self.config.embeddings_id_column, str(uuid.uuid4())
                    )
                    setattr(
                        embedding,
                        self.config.embeddings_document_id_column,
                        document_data.document_id,
                    )
                    embedding.set_vector(embedding_vector)

                    # Set model name if column exists
                    if self.config.embeddings_model_column:
                        setattr(
                            embedding,
                            self.config.embeddings_model_column,
                            getattr(
                                self.embedding_service.provider, "model_name", "default"
                            ),
                        )

                    session.add(embedding)

                session.commit()
                return document_data.document_id

        except Exception as e:
            raise LoaderError(f"Failed to load document: {e}")

    def load_documents_batch(
        self, documents_data: List[DocumentData], generate_embeddings: bool = True
    ) -> List[str]:
        """Load multiple documents in batch."""
        try:
            with self.config.get_session() as session:
                document_ids = []

                # Create documents
                for doc_data in documents_data:
                    document = Document(
                        id=doc_data.document_id, content=doc_data.content
                    )
                    document.set_metadata(doc_data.metadata)
                    session.add(document)
                    document_ids.append(doc_data.document_id)

                # Generate embeddings in batch if requested
                if generate_embeddings:
                    contents = [doc.content for doc in documents_data]
                    embeddings_vectors = self.embedding_service.create_embeddings_batch(
                        contents
                    )

                    for doc_data, embedding_vector in zip(
                        documents_data, embeddings_vectors
                    ):
                        embedding = Embedding(
                            id=str(uuid.uuid4()),
                            document_id=doc_data.document_id,
                            model_name=getattr(
                                self.embedding_service.provider, "model_name", "default"
                            ),
                        )
                        embedding.set_vector(embedding_vector)
                        session.add(embedding)

                session.commit()
                return document_ids

        except Exception as e:
            raise LoaderError(f"Failed to load documents batch: {e}")

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a single document by ID."""
        try:
            with self.config.get_session() as session:
                stmt = select(Document).where(Document.id == document_id)
                result = session.execute(stmt)
                return result.scalar_one_or_none()
        except Exception as e:
            raise LoaderError(f"Failed to get document: {e}")

    def get_documents_batch(self, document_ids: List[str]) -> List[Document]:
        """Get multiple documents by IDs."""
        try:
            with self.config.get_session() as session:
                stmt = select(Document).where(Document.id.in_(document_ids))
                result = session.execute(stmt)
                return list(result.scalars().all())
        except Exception as e:
            raise LoaderError(f"Failed to get documents batch: {e}")

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        try:
            with self.config.get_session() as session:
                # Delete embeddings first
                session.execute(
                    delete(Embedding).where(Embedding.document_id == document_id)
                )

                # Delete document
                result = session.execute(
                    delete(Document).where(Document.id == document_id)
                )

                session.commit()
                return result.rowcount > 0

        except Exception as e:
            raise LoaderError(f"Failed to delete document: {e}")

    def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        regenerate_embedding: bool = True,
    ) -> bool:
        """Update a document and optionally regenerate its embedding."""
        try:
            with self.config.get_session() as session:
                stmt = select(Document).where(Document.id == document_id)
                result = session.execute(stmt)
                document = result.scalar_one_or_none()

                if not document:
                    return False

                # Update document
                if content is not None:
                    document.content = content
                if metadata is not None:
                    document.set_metadata(metadata)

                # Regenerate embedding if content changed
                if content is not None and regenerate_embedding:
                    # Delete old embedding
                    session.execute(
                        delete(Embedding).where(Embedding.document_id == document_id)
                    )

                    # Create new embedding
                    embedding_vector = self.embedding_service.create_embedding(content)
                    embedding = Embedding(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        model_name=getattr(
                            self.embedding_service.provider, "model_name", "default"
                        ),
                    )
                    embedding.set_vector(embedding_vector)
                    session.add(embedding)

                session.commit()
                return True

        except Exception as e:
            raise LoaderError(f"Failed to update document: {e}")
