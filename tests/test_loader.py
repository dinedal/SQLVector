import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sql_rag.loader import LoaderInterface, DocumentData
from sql_rag.models import Document, Embedding
from sql_rag.exceptions import LoaderError


class TestDocumentData:
    def test_document_data_creation_minimal(self):
        """Test creating DocumentData with minimal parameters."""
        doc_data = DocumentData(content="Test content")
        
        assert doc_data.content == "Test content"
        assert doc_data.metadata == {}
        assert doc_data.document_id is not None
        assert isinstance(doc_data.document_id, str)
    
    def test_document_data_creation_full(self):
        """Test creating DocumentData with all parameters."""
        metadata = {"key": "value", "number": 42}
        doc_data = DocumentData(
            content="Test content",
            metadata=metadata,
            document_id="custom-id"
        )
        
        assert doc_data.content == "Test content"
        assert doc_data.metadata == metadata
        assert doc_data.document_id == "custom-id"
    
    def test_document_data_none_metadata(self):
        """Test DocumentData with None metadata."""
        doc_data = DocumentData(content="Test content", metadata=None)
        
        assert doc_data.metadata == {}


class TestLoaderInterface:
    async def test_load_document_with_embedding(self, rag_config, embedding_service):
        """Test loading a document with embedding generation."""
        loader = LoaderInterface(rag_config, embedding_service)
        doc_data = DocumentData(
            content="Test document content",
            metadata={"source": "test"}
        )
        
        document_id = await loader.load_document(doc_data, generate_embedding=True)
        
        assert document_id == doc_data.document_id
        
        # Verify document was created
        async with rag_config.get_session() as session:
            stmt = select(Document).where(Document.id == document_id)
            result = await session.execute(stmt)
            document = result.scalar_one_or_none()
            
            assert document is not None
            assert document.content == "Test document content"
            assert document.get_metadata() == {"source": "test"}
            
            # Verify embedding was created
            stmt = select(Embedding).where(Embedding.document_id == document_id)
            result = await session.execute(stmt)
            embedding = result.scalar_one_or_none()
            
            assert embedding is not None
            assert embedding.document_id == document_id
            assert len(embedding.get_vector()) == rag_config.embedding_dimension
    
    async def test_load_document_without_embedding(self, rag_config, embedding_service):
        """Test loading a document without embedding generation."""
        loader = LoaderInterface(rag_config, embedding_service)
        doc_data = DocumentData(
            content="Test document content",
            document_id="test-doc-no-embedding"
        )
        
        document_id = await loader.load_document(doc_data, generate_embedding=False)
        
        assert document_id == "test-doc-no-embedding"
        
        # Verify document was created
        async with rag_config.get_session() as session:
            stmt = select(Document).where(Document.id == document_id)
            result = await session.execute(stmt)
            document = result.scalar_one_or_none()
            
            assert document is not None
            assert document.content == "Test document content"
            
            # Verify no embedding was created
            stmt = select(Embedding).where(Embedding.document_id == document_id)
            result = await session.execute(stmt)
            embedding = result.scalar_one_or_none()
            
            assert embedding is None
    
    async def test_load_documents_batch(self, rag_config, embedding_service):
        """Test loading multiple documents in batch."""
        loader = LoaderInterface(rag_config, embedding_service)
        docs_data = [
            DocumentData(content="Document 1", metadata={"index": 1}),
            DocumentData(content="Document 2", metadata={"index": 2}),
            DocumentData(content="Document 3", metadata={"index": 3})
        ]
        
        document_ids = await loader.load_documents_batch(docs_data, generate_embeddings=True)
        
        assert len(document_ids) == 3
        assert all(doc_id == doc_data.document_id for doc_id, doc_data in zip(document_ids, docs_data))
        
        # Verify all documents were created
        async with rag_config.get_session() as session:
            for i, document_id in enumerate(document_ids):
                stmt = select(Document).where(Document.id == document_id)
                result = await session.execute(stmt)
                document = result.scalar_one_or_none()
                
                assert document is not None
                assert document.content == f"Document {i + 1}"
                assert document.get_metadata() == {"index": i + 1}
                
                # Verify embedding was created
                stmt = select(Embedding).where(Embedding.document_id == document_id)
                result = await session.execute(stmt)
                embedding = result.scalar_one_or_none()
                
                assert embedding is not None
    
    async def test_get_document(self, rag_config, embedding_service):
        """Test retrieving a single document."""
        loader = LoaderInterface(rag_config, embedding_service)
        doc_data = DocumentData(
            content="Test retrieval content",
            document_id="test-retrieval-id"
        )
        
        # Load the document first
        await loader.load_document(doc_data, generate_embedding=False)
        
        # Retrieve the document
        retrieved_doc = await loader.get_document("test-retrieval-id")
        
        assert retrieved_doc is not None
        assert retrieved_doc.id == "test-retrieval-id"
        assert retrieved_doc.content == "Test retrieval content"
    
    async def test_get_document_not_found(self, rag_config, embedding_service):
        """Test retrieving a non-existent document."""
        loader = LoaderInterface(rag_config, embedding_service)
        
        retrieved_doc = await loader.get_document("non-existent-id")
        
        assert retrieved_doc is None
    
    async def test_get_documents_batch(self, rag_config, embedding_service):
        """Test retrieving multiple documents in batch."""
        loader = LoaderInterface(rag_config, embedding_service)
        docs_data = [
            DocumentData(content="Batch doc 1", document_id="batch-1"),
            DocumentData(content="Batch doc 2", document_id="batch-2"),
            DocumentData(content="Batch doc 3", document_id="batch-3")
        ]
        
        # Load documents first
        for doc_data in docs_data:
            await loader.load_document(doc_data, generate_embedding=False)
        
        # Retrieve documents in batch
        document_ids = ["batch-1", "batch-2", "batch-3", "non-existent"]
        retrieved_docs = await loader.get_documents_batch(document_ids)
        
        # Should return 3 documents (non-existent one is filtered out)
        assert len(retrieved_docs) == 3
        retrieved_ids = [doc.id for doc in retrieved_docs]
        assert "batch-1" in retrieved_ids
        assert "batch-2" in retrieved_ids
        assert "batch-3" in retrieved_ids
    
    async def test_delete_document(self, rag_config, embedding_service):
        """Test deleting a document and its embeddings."""
        loader = LoaderInterface(rag_config, embedding_service)
        doc_data = DocumentData(
            content="Document to delete",
            document_id="delete-test-id"
        )
        
        # Load document with embedding
        await loader.load_document(doc_data, generate_embedding=True)
        
        # Verify document and embedding exist
        async with rag_config.get_session() as session:
            stmt = select(Document).where(Document.id == "delete-test-id")
            result = await session.execute(stmt)
            assert result.scalar_one_or_none() is not None
            
            stmt = select(Embedding).where(Embedding.document_id == "delete-test-id")
            result = await session.execute(stmt)
            assert result.scalar_one_or_none() is not None
        
        # Delete the document
        deleted = await loader.delete_document("delete-test-id")
        assert deleted is True
        
        # Verify document and embedding are gone
        async with rag_config.get_session() as session:
            stmt = select(Document).where(Document.id == "delete-test-id")
            result = await session.execute(stmt)
            assert result.scalar_one_or_none() is None
            
            stmt = select(Embedding).where(Embedding.document_id == "delete-test-id")
            result = await session.execute(stmt)
            assert result.scalar_one_or_none() is None
    
    async def test_delete_document_not_found(self, rag_config, embedding_service):
        """Test deleting a non-existent document."""
        loader = LoaderInterface(rag_config, embedding_service)
        
        deleted = await loader.delete_document("non-existent-id")
        assert deleted is False
    
    async def test_update_document_content_only(self, rag_config, embedding_service):
        """Test updating document content only."""
        loader = LoaderInterface(rag_config, embedding_service)
        doc_data = DocumentData(
            content="Original content",
            document_id="update-test-id",
            metadata={"original": True}
        )
        
        # Load document with embedding
        await loader.load_document(doc_data, generate_embedding=True)
        
        # Update content only
        updated = await loader.update_document(
            "update-test-id",
            content="Updated content",
            regenerate_embedding=True
        )
        
        assert updated is True
        
        # Verify update
        retrieved_doc = await loader.get_document("update-test-id")
        assert retrieved_doc.content == "Updated content"
        assert retrieved_doc.get_metadata() == {"original": True}  # Metadata unchanged
    
    async def test_update_document_metadata_only(self, rag_config, embedding_service):
        """Test updating document metadata only."""
        loader = LoaderInterface(rag_config, embedding_service)
        doc_data = DocumentData(
            content="Original content",
            document_id="update-meta-id",
            metadata={"original": True}
        )
        
        # Load document
        await loader.load_document(doc_data, generate_embedding=False)
        
        # Update metadata only
        updated = await loader.update_document(
            "update-meta-id",
            metadata={"updated": True, "version": 2},
            regenerate_embedding=False
        )
        
        assert updated is True
        
        # Verify update
        retrieved_doc = await loader.get_document("update-meta-id")
        assert retrieved_doc.content == "Original content"  # Content unchanged
        assert retrieved_doc.get_metadata() == {"updated": True, "version": 2}
    
    async def test_update_document_not_found(self, rag_config, embedding_service):
        """Test updating a non-existent document."""
        loader = LoaderInterface(rag_config, embedding_service)
        
        updated = await loader.update_document(
            "non-existent-id",
            content="New content"
        )
        
        assert updated is False