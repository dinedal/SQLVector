import pytest
from sqlalchemy import select
from typing import List

from sql_rag import SQLRAG, EmbeddingProvider
from sql_rag.models import Document, Embedding
from sql_rag.exceptions import ConfigurationError


class CustomTestEmbeddingProvider(EmbeddingProvider):
    """Custom embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = "custom-test-model"
    
    async def embed(self, text: str) -> List[float]:
        # Simple embedding based on text characteristics
        text_hash = hash(text) % 1000
        return [float(text_hash % 10)] * self.dimension
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]


class TestSQLRAG:
    async def test_sqlrag_creation_default(self, async_engine):
        """Test creating SQLRAG with default settings."""
        rag = SQLRAG(engine=async_engine)
        
        assert rag.config.engine == async_engine
        assert rag.config.documents_table == "documents"
        assert rag.config.embeddings_table == "embeddings"
        assert rag.config.embedding_dimension == 768
        assert rag.embedding_service is not None
        assert rag.query_interface is not None
        assert rag.loader_interface is not None
    
    async def test_sqlrag_creation_custom_config(self, async_engine):
        """Test creating SQLRAG with custom configuration."""
        rag = SQLRAG(
            engine=async_engine,
            documents_table="my_docs",
            embeddings_table="my_embeddings",
            embedding_dimension=512
        )
        
        assert rag.config.documents_table == "my_docs"
        assert rag.config.embeddings_table == "my_embeddings"
        assert rag.config.embedding_dimension == 512
    
    async def test_sqlrag_creation_custom_provider(self, async_engine):
        """Test creating SQLRAG with custom embedding provider."""
        custom_provider = CustomTestEmbeddingProvider(dimension=256)
        rag = SQLRAG(
            engine=async_engine,
            embedding_dimension=256,
            embedding_provider=custom_provider
        )
        
        assert rag.embedding_service.provider == custom_provider
        assert rag.config.embedding_dimension == 256
    
    async def test_create_tables(self, async_engine):
        """Test creating database tables."""
        rag = SQLRAG(engine=async_engine)
        
        # Tables should be created without error
        await rag.create_tables()
        
        # Verify tables exist by trying to insert data
        async with rag.config.get_session() as session:
            # Should not raise an error
            doc = Document(id="test", content="test")
            session.add(doc)
            await session.commit()
    
    async def test_load_document_simple(self, rag_instance):
        """Test loading a simple document."""
        document_id = await rag_instance.load_document(
            content="Test document content",
            metadata={"source": "test"}
        )
        
        assert document_id is not None
        assert isinstance(document_id, str)
        
        # Verify document was stored
        doc = await rag_instance.get_document(document_id)
        assert doc is not None
        assert doc.content == "Test document content"
        assert doc.get_metadata() == {"source": "test"}
    
    async def test_load_document_with_custom_id(self, rag_instance):
        """Test loading a document with custom ID."""
        custom_id = "my-custom-document-id"
        document_id = await rag_instance.load_document(
            content="Custom ID document",
            document_id=custom_id
        )
        
        assert document_id == custom_id
        
        doc = await rag_instance.get_document(custom_id)
        assert doc is not None
        assert doc.id == custom_id
    
    async def test_load_document_without_embedding(self, rag_instance):
        """Test loading a document without generating embedding."""
        document_id = await rag_instance.load_document(
            content="No embedding document",
            generate_embedding=False
        )
        
        # Verify document exists but no embedding
        async with rag_instance.config.get_session() as session:
            stmt = select(Document).where(Document.id == document_id)
            result = await session.execute(stmt)
            doc = result.scalar_one_or_none()
            assert doc is not None
            
            stmt = select(Embedding).where(Embedding.document_id == document_id)
            result = await session.execute(stmt)
            embedding = result.scalar_one_or_none()
            assert embedding is None
    
    async def test_load_documents_batch(self, rag_instance, sample_documents):
        """Test loading multiple documents in batch."""
        document_ids = await rag_instance.load_documents(sample_documents)
        
        assert len(document_ids) == len(sample_documents)
        assert all(isinstance(doc_id, str) for doc_id in document_ids)
        
        # Verify all documents were loaded
        docs = await rag_instance.get_documents(document_ids)
        assert len(docs) == len(sample_documents)
        
        contents = [doc.content for doc in docs]
        expected_contents = [doc["content"] for doc in sample_documents]
        for content in expected_contents:
            assert content in contents
    
    async def test_query_single_result(self, rag_instance):
        """Test querying with single expected result."""
        # Load some test documents
        await rag_instance.load_document(
            content="The quick brown fox jumps over the lazy dog",
            metadata={"type": "animals"}
        )
        await rag_instance.load_document(
            content="Machine learning algorithms and neural networks",
            metadata={"type": "technology"}
        )
        
        # Query for animal-related content
        results = await rag_instance.query("fox", top_k=5, similarity_threshold=-1.0)
        
        assert len(results) >= 1
        # The fox document should have high similarity
        fox_result = next((r for r in results if "fox" in r["content"]), None)
        assert fox_result is not None
        import json
        assert json.loads(fox_result["doc_metadata"]) == {"type": "animals"}
    
    async def test_query_with_filters(self, rag_instance):
        """Test querying with similarity threshold."""
        # Load documents
        await rag_instance.load_document("Very specific unique content abc123")
        await rag_instance.load_document("Completely different content xyz789")
        
        # Query with high similarity threshold
        results = await rag_instance.query(
            "specific unique content",
            top_k=10,
            similarity_threshold=0.3
        )
        
        # Should filter out dissimilar results
        assert all(r.similarity >= 0.3 for r in results)
    
    async def test_query_batch_multiple(self, rag_instance, sample_documents):
        """Test batch querying."""
        # Load sample documents
        await rag_instance.load_documents(sample_documents)
        
        # Batch query
        queries = ["fox", "machine learning", "programming"]
        batch_results = await rag_instance.query_batch(queries, top_k=2)
        
        assert len(batch_results) == 3
        assert all(isinstance(results, list) for results in batch_results)
        assert all(len(results) <= 2 for results in batch_results)
    
    async def test_get_document_exists(self, rag_instance):
        """Test getting an existing document."""
        document_id = await rag_instance.load_document(
            content="Document for retrieval test",
            document_id="retrieval-test-id"
        )
        
        doc = await rag_instance.get_document("retrieval-test-id")
        
        assert doc is not None
        assert doc.id == "retrieval-test-id"
        assert doc.content == "Document for retrieval test"
    
    async def test_get_document_not_exists(self, rag_instance):
        """Test getting a non-existent document."""
        doc = await rag_instance.get_document("non-existent-id")
        
        assert doc is None
    
    async def test_get_documents_batch(self, rag_instance):
        """Test getting multiple documents by IDs."""
        # Load some documents
        id1 = await rag_instance.load_document("Document 1", document_id="doc-1")
        id2 = await rag_instance.load_document("Document 2", document_id="doc-2")
        id3 = await rag_instance.load_document("Document 3", document_id="doc-3")
        
        # Get documents in batch
        docs = await rag_instance.get_documents([id1, id2, id3, "non-existent"])
        
        # Should return 3 documents (non-existent filtered out)
        assert len(docs) == 3
        doc_ids = [doc.id for doc in docs]
        assert "doc-1" in doc_ids
        assert "doc-2" in doc_ids
        assert "doc-3" in doc_ids
    
    async def test_delete_document(self, rag_instance):
        """Test deleting a document."""
        document_id = await rag_instance.load_document(
            content="Document to be deleted",
            document_id="delete-me"
        )
        
        # Verify document exists
        doc = await rag_instance.get_document(document_id)
        assert doc is not None
        
        # Delete document
        deleted = await rag_instance.delete_document(document_id)
        assert deleted is True
        
        # Verify document is gone
        doc = await rag_instance.get_document(document_id)
        assert doc is None
    
    async def test_delete_document_not_exists(self, rag_instance):
        """Test deleting a non-existent document."""
        deleted = await rag_instance.delete_document("non-existent-id")
        assert deleted is False
    
    async def test_update_document_content(self, rag_instance):
        """Test updating document content."""
        document_id = await rag_instance.load_document(
            content="Original content",
            document_id="update-test",
            metadata={"version": 1}
        )
        
        # Update content
        updated = await rag_instance.update_document(
            document_id,
            content="Updated content",
            regenerate_embedding=True
        )
        
        assert updated is True
        
        # Verify update
        doc = await rag_instance.get_document(document_id)
        assert doc.content == "Updated content"
        assert doc.get_metadata() == {"version": 1}  # Metadata unchanged
    
    async def test_update_document_metadata(self, rag_instance):
        """Test updating document metadata."""
        document_id = await rag_instance.load_document(
            content="Content for metadata update",
            metadata={"version": 1, "type": "test"}
        )
        
        # Update metadata
        updated = await rag_instance.update_document(
            document_id,
            metadata={"version": 2, "type": "updated", "new_field": "value"},
            regenerate_embedding=False
        )
        
        assert updated is True
        
        # Verify update
        doc = await rag_instance.get_document(document_id)
        assert doc.content == "Content for metadata update"  # Content unchanged
        assert doc.get_metadata() == {
            "version": 2, 
            "type": "updated", 
            "new_field": "value"
        }
    
    async def test_update_document_both(self, rag_instance):
        """Test updating both content and metadata."""
        document_id = await rag_instance.load_document(
            content="Original content",
            metadata={"version": 1}
        )
        
        # Update both
        updated = await rag_instance.update_document(
            document_id,
            content="New content",
            metadata={"version": 2, "updated": True}
        )
        
        assert updated is True
        
        # Verify update
        doc = await rag_instance.get_document(document_id)
        assert doc.content == "New content"
        assert doc.get_metadata() == {"version": 2, "updated": True}
    
    async def test_update_document_not_exists(self, rag_instance):
        """Test updating a non-existent document."""
        updated = await rag_instance.update_document(
            "non-existent-id",
            content="New content"
        )
        
        assert updated is False
    
    async def test_end_to_end_workflow(self, rag_instance):
        """Test a complete end-to-end workflow."""
        # 1. Load documents
        docs = [
            {"content": "Python is a programming language", "metadata": {"topic": "programming"}},
            {"content": "Machine learning with neural networks", "metadata": {"topic": "ai"}},
            {"content": "Database design and SQL queries", "metadata": {"topic": "database"}}
        ]
        
        document_ids = await rag_instance.load_documents(docs)
        assert len(document_ids) == 3
        
        # 2. Query for relevant documents
        results = await rag_instance.query("programming", top_k=2)
        assert len(results) >= 1
        
        # 3. Update a document
        python_doc_id = None
        for doc_id in document_ids:
            doc = await rag_instance.get_document(doc_id)
            if "Python" in doc.content:
                python_doc_id = doc_id
                break
        
        assert python_doc_id is not None
        await rag_instance.update_document(
            python_doc_id,
            content="Python is a powerful programming language for AI",
            metadata={"topic": "programming", "updated": True}
        )
        
        # 4. Query again to see updated results
        results = await rag_instance.query("AI programming", top_k=5)
        updated_result = next((r for r in results if r["id"] == python_doc_id), None)
        assert updated_result is not None
        assert "powerful" in updated_result["content"]
        
        # 5. Delete a document
        deleted = await rag_instance.delete_document(document_ids[0])
        assert deleted is True
        
        # 6. Verify deletion
        remaining_docs = await rag_instance.get_documents(document_ids)
        assert len(remaining_docs) == 2
    
    async def test_sqlrag_with_custom_embedding_provider(self, async_engine):
        """Test SQLRAG with custom embedding provider."""
        custom_provider = CustomTestEmbeddingProvider(dimension=128)
        rag = SQLRAG(
            engine=async_engine,
            embedding_dimension=128,
            embedding_provider=custom_provider
        )
        await rag.create_tables()
        
        # Load document
        document_id = await rag.load_document("Test with custom provider")
        
        # Verify embedding was created with custom provider
        async with rag.config.get_session() as session:
            stmt = select(Embedding).where(Embedding.document_id == document_id)
            result = await session.execute(stmt)
            embedding = result.scalar_one_or_none()
            
            assert embedding is not None
            assert len(embedding.get_vector()) == 128
            assert embedding.model_name == "custom-test-model"