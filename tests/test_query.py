import pytest
from sqlalchemy import select

from sql_rag.query import QueryInterface
from sql_rag.loader import LoaderInterface, DocumentData
from sql_rag.models import Document, Embedding
from sql_rag.exceptions import QueryError



class TestQueryInterface:
    async def test_query_single_document(self, rag_config, embedding_service):
        """Test querying with a single document in the database."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        doc_data = DocumentData(
            content="The quick brown fox jumps over the lazy dog",
            metadata={"category": "animals"}
        )
        await loader.load_document(doc_data, generate_embedding=True)
        
        # Query
        results = await query_interface.query("fox", top_k=5, similarity_threshold=-1.0)
        
        assert len(results) == 1
        assert results[0]["content"] == "The quick brown fox jumps over the lazy dog"
        import json
        assert json.loads(results[0]["doc_metadata"]) == {"category": "animals"}
        assert isinstance(results[0]["similarity"], float)
        assert -1 <= results[0]["similarity"] <= 1
    
    async def test_query_multiple_documents(self, rag_config, embedding_service):
        """Test querying with multiple documents."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        docs_data = [
            DocumentData(content="The quick brown fox", metadata={"type": "animal"}),
            DocumentData(content="Machine learning algorithms", metadata={"type": "tech"}),
            DocumentData(content="Python programming language", metadata={"type": "tech"}),
            DocumentData(content="Artificial intelligence systems", metadata={"type": "tech"})
        ]
        
        for doc_data in docs_data:
            await loader.load_document(doc_data, generate_embedding=True)
        
        # Query for technology-related content
        results = await query_interface.query("artificial intelligence", top_k=3, similarity_threshold=-1.0)
        
        assert len(results) <= 3
        assert all(isinstance(r["similarity"], float) for r in results)
        assert all(-1 <= r["similarity"] <= 1 for r in results)
        
        # Results should be sorted by similarity (descending)
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)
    
    async def test_query_with_similarity_threshold(self, rag_config, embedding_service):
        """Test querying with similarity threshold."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        docs_data = [
            DocumentData(content="Very specific unique content abc123"),
            DocumentData(content="Completely different content xyz789"),
            DocumentData(content="Another unrelated document")
        ]
        
        for doc_data in docs_data:
            await loader.load_document(doc_data, generate_embedding=True)
        
        # Query with high similarity threshold
        results = await query_interface.query(
            "specific unique content", 
            top_k=10, 
            similarity_threshold=0.5
        )
        
        # Should filter out low-similarity results
        assert all(r["similarity"] >= 0.5 for r in results)
    
    async def test_query_empty_database(self, rag_config, embedding_service):
        """Test querying an empty database."""
        query_interface = QueryInterface(rag_config, embedding_service)
        
        results = await query_interface.query("any query", top_k=5)
        
        assert results == []
    
    async def test_query_batch(self, rag_config, embedding_service):
        """Test batch querying."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        docs_data = [
            DocumentData(content="Document about cats and animals"),
            DocumentData(content="Programming tutorial with Python"),
            DocumentData(content="Machine learning with neural networks"),
            DocumentData(content="Database systems and SQL queries")
        ]
        
        for doc_data in docs_data:
            await loader.load_document(doc_data, generate_embedding=True)
        
        # Batch query
        query_texts = ["cats", "programming", "machine learning"]
        batch_results = await query_interface.query_batch(query_texts, top_k=2)
        
        assert len(batch_results) == 3
        assert all(isinstance(results, list) for results in batch_results)
        assert all(len(results) <= 2 for results in batch_results)
        
        # Each query should return results
        for results in batch_results:
            assert all(isinstance(r, dict) for r in results)
    
    async def test_query_by_document_id_exists(self, rag_config, embedding_service):
        """Test querying by document ID for existing document."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        doc_data = DocumentData(
            content="Test document for ID query",
            document_id="test-id-query",
            metadata={"source": "test"}
        )
        await loader.load_document(doc_data, generate_embedding=True)
        
        # Query by ID
        result = await query_interface.query_by_document_id("test-id-query")
        
        assert result is not None
        assert result["id"] == "test-id-query"  # Use document id column 
        assert result["content"] == "Test document for ID query"
        import json
        assert json.loads(result["doc_metadata"]) == {"source": "test"}
        assert result["similarity"] == 1.0  # Should be 1.0 for exact match
    
    async def test_query_by_document_id_not_exists(self, rag_config, embedding_service):
        """Test querying by document ID for non-existent document."""
        query_interface = QueryInterface(rag_config, embedding_service)
        
        result = await query_interface.query_by_document_id("non-existent-id")
        
        assert result is None
    
    async def test_query_top_k_limit(self, rag_config, embedding_service):
        """Test that top_k properly limits results."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        # Create more documents than we'll request
        for i in range(10):
            doc_data = DocumentData(content=f"Document number {i}")
            await loader.load_document(doc_data, generate_embedding=True)
        
        # Query with small top_k
        results = await query_interface.query("document", top_k=3)
        
        assert len(results) == 3
    
    async def test_query_zero_top_k(self, rag_config, embedding_service):
        """Test querying with top_k=0."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        doc_data = DocumentData(content="Test document")
        await loader.load_document(doc_data, generate_embedding=True)
        
        # Query with top_k=0
        results = await query_interface.query("test", top_k=0)
        
        assert results == []
    
    async def test_query_batch_empty_list(self, rag_config, embedding_service):
        """Test batch querying with empty query list."""
        query_interface = QueryInterface(rag_config, embedding_service)
        
        batch_results = await query_interface.query_batch([], top_k=5)
        
        assert batch_results == []
    
    async def test_query_batch_single_query(self, rag_config, embedding_service):
        """Test batch querying with single query."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        doc_data = DocumentData(content="Single test document")
        await loader.load_document(doc_data, generate_embedding=True)
        
        # Batch query with single item
        batch_results = await query_interface.query_batch(["test"], top_k=5)
        
        assert len(batch_results) == 1
        assert len(batch_results[0]) == 1
        assert batch_results[0][0]["content"] == "Single test document"
    
    async def test_query_special_characters(self, rag_config, embedding_service):
        """Test querying with special characters."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        doc_data = DocumentData(content="Document with special chars: @#$%^&*()")
        await loader.load_document(doc_data, generate_embedding=True)
        
        # Query with special characters
        results = await query_interface.query("@#$%", top_k=5, similarity_threshold=-1.0)
        
        assert len(results) == 1
        assert "special chars" in results[0]["content"]
    
    async def test_query_unicode_characters(self, rag_config, embedding_service):
        """Test querying with unicode characters."""
        # Set up data
        loader = LoaderInterface(rag_config, embedding_service)
        query_interface = QueryInterface(rag_config, embedding_service)
        
        doc_data = DocumentData(content="Document with unicode: 你好世界 🌍")
        await loader.load_document(doc_data, generate_embedding=True)
        
        # Query with unicode
        results = await query_interface.query("你好", top_k=5)
        
        assert len(results) == 1
        assert "你好世界" in results[0]["content"]