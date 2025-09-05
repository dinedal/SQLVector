"""Tests for SQLite backend."""

import pytest
import tempfile
import json
from pathlib import Path

from sqlvector.backends.sqlite import (
    SQLiteRAG, 
    SQLiteConfig, 
    SQLiteDocument,
    SQLiteEmbedding
)
from sqlvector.embedding import DefaultEmbeddingProvider


class TestSQLiteConfig:
    def test_config_creation(self):
        """Test SQLite config creation."""
        config = SQLiteConfig(
            db_path=":memory:",
            embedding_dimension=384,
            batch_size=500
        )
        
        assert config.db_path == ":memory:"
        assert config.embedding_dimension == 384
        assert config.batch_size == 500
        assert config.documents_table == "documents"
        assert config.embeddings_table == "embeddings"
        assert config.vss_table == "vss_embeddings"
        assert config.enable_vss_extension is False
    
    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            SQLiteConfig(db_path=":memory:", embedding_dimension=0)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SQLiteConfig(db_path=":memory:", batch_size=0)
    
    def test_get_connection(self):
        """Test database connection."""
        config = SQLiteConfig(db_path=":memory:")
        conn = config.get_connection()
        
        assert conn is not None
        
        # Test basic query
        cursor = conn.execute("SELECT 1 as test")
        result = cursor.fetchone()
        assert result[0] == 1
        
        conn.close()
    
    def test_database_setup(self):
        """Test database schema setup."""
        config = SQLiteConfig(db_path=":memory:")
        
        with config.get_connection() as conn:
            config.setup_database(conn)
            
            # Check tables exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            
            table_names = [row[0] for row in cursor.fetchall()]
            assert "documents" in table_names
            assert "embeddings" in table_names


class TestSQLiteDocument:
    def test_document_creation(self):
        """Test document model creation."""
        doc = SQLiteDocument(
            id="test-1",
            content="Test content",
            metadata={"category": "test"}
        )
        
        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.metadata == {"category": "test"}
    
    def test_document_to_dict(self):
        """Test document to dictionary conversion."""
        doc = SQLiteDocument(
            id="test-1",
            content="Test content",
            metadata={"category": "test"}
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["id"] == "test-1"
        assert doc_dict["content"] == "Test content"
        assert doc_dict["metadata"] == '{"category": "test"}'
        assert "hash" in doc_dict
    
    def test_document_from_dict(self):
        """Test document from dictionary creation."""
        doc_dict = {
            "id": "test-1",
            "content": "Test content",
            "metadata": '{"category": "test"}',
            "hash": "abc123"
        }
        
        doc = SQLiteDocument.from_dict(doc_dict)
        
        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.metadata == {"category": "test"}
        assert doc.hash == "abc123"


class TestSQLiteEmbedding:
    def test_embedding_serialization(self):
        """Test embedding serialization and deserialization."""
        original_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Test serialization
        serialized = SQLiteEmbedding._serialize_embedding(original_embedding)
        assert isinstance(serialized, bytes)
        
        # Test deserialization
        deserialized = SQLiteEmbedding._deserialize_embedding(serialized)
        assert len(deserialized) == len(original_embedding)
        
        # Check values are approximately equal (floating point precision)
        for orig, deser in zip(original_embedding, deserialized):
            assert abs(orig - deser) < 1e-6
    
    def test_embedding_to_dict(self):
        """Test embedding to dictionary conversion."""
        embedding = SQLiteEmbedding(
            id="emb-1",
            document_id="doc-1",
            hash="abc123",
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model"
        )
        
        emb_dict = embedding.to_dict()
        
        assert emb_dict["id"] == "emb-1"
        assert emb_dict["document_id"] == "doc-1"
        assert emb_dict["hash"] == "abc123"
        assert isinstance(emb_dict["embedding"], bytes)
        assert emb_dict["model_name"] == "test-model"


class TestSQLiteRAG:
    @pytest.fixture
    def sqlite_rag_instance(self):
        """Create a SQLite RAG instance for testing."""
        provider = DefaultEmbeddingProvider(dimension=384)
        return SQLiteRAG(
            db_path=":memory:",
            embedding_provider=provider,
            embedding_dimension=384,
            batch_size=100
        )
    
    @pytest.fixture
    def vss_rag_instance(self):
        """Create a SQLite RAG instance with VSS extension (may not work in tests)."""
        provider = DefaultEmbeddingProvider(dimension=384)
        return SQLiteRAG(
            db_path=":memory:",
            embedding_provider=provider,
            embedding_dimension=384,
            batch_size=100,
            enable_vss_extension=True  # Will gracefully fall back if not available
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                "content": "The quick brown fox jumps over the lazy dog",
                "metadata": {"category": "animals", "type": "example"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence",
                "metadata": {"category": "technology", "type": "definition"}
            },
            {
                "content": "Python is a popular programming language",
                "metadata": {"category": "technology", "type": "fact"}
            }
        ]
    
    def test_rag_initialization(self, sqlite_rag_instance):
        """Test RAG system initialization."""
        assert sqlite_rag_instance.config.db_path == ":memory:"
        assert sqlite_rag_instance.config.embedding_dimension == 384
        assert sqlite_rag_instance.embedding_service is not None
        assert sqlite_rag_instance.loader is not None
        assert sqlite_rag_instance.querier is not None
    
    def test_load_single_document(self, sqlite_rag_instance):
        """Test loading a single document."""
        document_id = sqlite_rag_instance.load_document(
            content="Test document content",
            metadata={"source": "test"}
        )
        
        assert document_id is not None
        assert isinstance(document_id, str)
        
        # Verify document was stored
        doc = sqlite_rag_instance.get_document(document_id)
        assert doc is not None
        assert doc.content == "Test document content"
        assert doc.metadata == {"source": "test"}
    
    def test_load_documents_batch(self, sqlite_rag_instance, sample_documents):
        """Test loading multiple documents."""
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        assert len(document_ids) == 3
        assert all(isinstance(doc_id, str) for doc_id in document_ids)
        
        # Verify all documents were loaded
        docs = sqlite_rag_instance.get_documents(document_ids)
        assert len(docs) == 3
        
        contents = [doc.content for doc in docs]
        expected_contents = [doc["content"] for doc in sample_documents]
        for content in expected_contents:
            assert content in contents
    
    def test_query_similarity(self, sqlite_rag_instance, sample_documents):
        """Test similarity querying."""
        # Load documents
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Query for animal-related content
        results = sqlite_rag_instance.query(
            query_text="fox animal",
            top_k=5,
            similarity_threshold=-1.0  # Allow all results
        )
        
        assert len(results) >= 1
        assert all(isinstance(r, dict) for r in results)
        assert all(isinstance(r["similarity"], float) for r in results)
        
        # Results should be sorted by similarity (descending for cosine)
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_query_with_precomputed_embedding(self, sqlite_rag_instance, sample_documents):
        """Test querying with precomputed embedding."""
        # Load documents
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Use a simple embedding vector
        query_embedding = [0.1] * 384  # Match the embedding dimension
        
        results = sqlite_rag_instance.query_with_embedding(
            query_embedding=query_embedding,
            top_k=3,
            similarity_threshold=-1.0
        )
        
        assert len(results) >= 1
        assert all(isinstance(r, dict) for r in results)
    
    def test_query_with_filters(self, sqlite_rag_instance, sample_documents):
        """Test filtered querying."""
        # Load documents
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Query with category filter only (no similarity)
        results = sqlite_rag_instance.query_with_filters(
            filters={"category": "technology"},
            top_k=10
        )
        
        assert len(results) >= 1
        import json
        assert all(json.loads(r["metadata"]).get("category") == "technology" for r in results)
        
        # Query with both filter and similarity
        results_with_sim = sqlite_rag_instance.query_with_filters(
            filters={"category": "technology"},
            query_text="machine learning AI",
            top_k=10,
            similarity_threshold=-1.0
        )
        
        assert len(results_with_sim) >= 1
        import json
        assert all(json.loads(r["metadata"]).get("category") == "technology" for r in results_with_sim)
    
    def test_find_similar_documents(self, sqlite_rag_instance, sample_documents):
        """Test finding similar documents."""
        # Load documents
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Find documents similar to the first one
        similar_docs = sqlite_rag_instance.find_similar_documents(
            document_id=document_ids[0],
            top_k=5,
            similarity_threshold=-1.0
        )
        
        # Should find other documents (excluding the queried one)
        assert len(similar_docs) >= 0
        assert all(r["id"] != document_ids[0] for r in similar_docs)
    
    def test_delete_document(self, sqlite_rag_instance):
        """Test document deletion."""
        # Load a document
        document_id = sqlite_rag_instance.load_document(
            content="Document to delete",
            metadata={"temp": True}
        )
        
        # Verify it exists
        doc = sqlite_rag_instance.get_document(document_id)
        assert doc is not None
        
        # Delete it
        deleted = sqlite_rag_instance.delete_document(document_id)
        assert deleted is True
        
        # Verify it's gone
        doc = sqlite_rag_instance.get_document(document_id)
        assert doc is None
    
    def test_statistics(self, sqlite_rag_instance, sample_documents):
        """Test getting database statistics."""
        # Load some documents
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        stats = sqlite_rag_instance.get_statistics()
        
        assert "total_documents" in stats
        assert "total_embeddings" in stats
        assert "embedding_dimension" in stats
        assert "batch_size" in stats
        assert "vss_enabled" in stats
        
        assert stats["total_documents"] == 3
        assert stats["total_embeddings"] == 3
        assert stats["embedding_dimension"] == 384
        assert stats["vss_enabled"] is False  # VSS not enabled by default
    
    def test_export_to_dict(self, sqlite_rag_instance, sample_documents):
        """Test exporting to dictionary."""
        # Load documents
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Export without embeddings
        docs_dict = sqlite_rag_instance.export_to_dict(include_embeddings=False)
        
        assert len(docs_dict) == 3
        assert all("id" in doc for doc in docs_dict)
        assert all("content" in doc for doc in docs_dict)
        assert all("metadata" in doc for doc in docs_dict)
        assert all("embedding" not in doc for doc in docs_dict)
        
        # Export with embeddings
        docs_with_emb = sqlite_rag_instance.export_to_dict(include_embeddings=True)
        
        assert len(docs_with_emb) == 3
        assert all("embedding" in doc for doc in docs_with_emb)
        assert all(isinstance(doc["embedding"], list) for doc in docs_with_emb)
    
    def test_context_manager(self):
        """Test context manager usage."""
        with SQLiteRAG(db_path=":memory:", embedding_dimension=384) as rag:
            document_id = rag.load_document("Test content")
            assert document_id is not None
            
            doc = rag.get_document(document_id)
            assert doc is not None
    
    def test_query_batch(self, sqlite_rag_instance, sample_documents):
        """Test batch querying."""
        # Load documents
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Batch query
        queries = ["fox animal", "machine learning", "python programming"]
        batch_results = sqlite_rag_instance.query_batch(
            query_texts=queries,
            top_k=2,
            similarity_threshold=-1.0
        )
        
        assert len(batch_results) == 3
        assert all(isinstance(results, list) for results in batch_results)
        assert all(len(results) <= 2 for results in batch_results)
    
    def test_different_similarity_functions(self, sqlite_rag_instance, sample_documents):
        """Test different similarity functions."""
        # Load documents
        document_ids = sqlite_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        query_text = "machine learning"
        
        # Test cosine similarity
        cosine_results = sqlite_rag_instance.query(
            query_text, 
            similarity_function="cosine",
            similarity_threshold=-1.0
        )
        
        # Test inner product
        inner_product_results = sqlite_rag_instance.query(
            query_text,
            similarity_function="inner_product", 
            similarity_threshold=-100.0  # Lower threshold for inner product
        )
        
        # Test euclidean distance
        euclidean_results = sqlite_rag_instance.query(
            query_text,
            similarity_function="euclidean",
            similarity_threshold=1000.0  # Higher threshold for distance
        )
        
        # All should return some results
        assert len(cosine_results) >= 1
        assert len(inner_product_results) >= 1
        assert len(euclidean_results) >= 1
    
    def test_vss_optimization_fallback(self, vss_rag_instance, sample_documents):
        """Test VSS optimization with fallback to standard query."""
        # Load documents
        document_ids = vss_rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Try VSS-optimized query (will fall back to standard if VSS not available)
        results = vss_rag_instance.query(
            query_text="machine learning",
            top_k=5,
            use_vss_optimization=True,
            similarity_threshold=-1.0
        )
        
        # Should work regardless of whether VSS extension is available
        assert len(results) >= 1
        assert all(isinstance(r, dict) for r in results)


class TestSQLiteFileOperations:
    def test_file_database_persistence(self):
        """Test that data persists when using file database."""
        # Create temporary database file
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # First session - load data
            with SQLiteRAG(db_path=db_path, embedding_dimension=384) as rag1:
                document_id = rag1.load_document(
                    content="Persistent test document",
                    metadata={"persistent": True}
                )
                stats1 = rag1.get_statistics()
                assert stats1["total_documents"] == 1
            
            # Second session - verify data persists
            with SQLiteRAG(db_path=db_path, embedding_dimension=384) as rag2:
                stats2 = rag2.get_statistics()
                assert stats2["total_documents"] == 1
                
                # Verify we can retrieve the document
                doc = rag2.get_document(document_id)
                assert doc is not None
                assert doc.content == "Persistent test document"
                assert doc.metadata == {"persistent": True}
        
        finally:
            Path(db_path).unlink()  # Clean up temporary file


class TestSQLiteIndexManagement:
    @pytest.fixture
    def sqlite_rag_instance(self):
        """Create a SQLite RAG instance for testing."""
        provider = DefaultEmbeddingProvider(dimension=384)
        return SQLiteRAG(
            db_path=":memory:",
            embedding_provider=provider,
            embedding_dimension=384,
            batch_size=100
        )
    
    @pytest.fixture
    def vss_rag(self):
        """Create RAG instance with VSS enabled (will fall back gracefully)."""
        provider = DefaultEmbeddingProvider(dimension=384)
        return SQLiteRAG(
            db_path=":memory:",
            embedding_provider=provider,
            embedding_dimension=384,
            enable_vss_extension=True
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                "content": "The quick brown fox jumps over the lazy dog",
                "metadata": {"category": "animals", "type": "example"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence",
                "metadata": {"category": "technology", "type": "definition"}
            },
            {
                "content": "Python is a popular programming language",
                "metadata": {"category": "technology", "type": "fact"}
            }
        ]
    
    def test_index_creation_without_vss(self, sqlite_rag_instance):
        """Test index creation fails gracefully when VSS is not enabled."""
        from sqlvector.exceptions import LoaderError
        
        with pytest.raises(LoaderError, match="VSS extension is not enabled"):
            sqlite_rag_instance.create_index("test_index")
    
    def test_index_operations_with_vss(self, vss_rag, sample_documents):
        """Test index creation and deletion (will test fallback if VSS not available)."""
        # Load documents first
        document_ids = vss_rag.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Only test index operations if VSS extension is actually available
        if vss_rag.config.enable_vss_extension:
            # Test index creation
            result = vss_rag.create_index(
                "test_index",
                factory_string="Flat,IDMap2"
            )
            assert result is True
            
            # Test index training (should work even if not strictly needed)
            train_result = vss_rag.train_index(training_data_limit=2)
            assert train_result is True
            
            # Test index deletion
            delete_result = vss_rag.delete_index("test_index")
            assert delete_result is True
        else:
            # If VSS not available, test that LoaderError is raised
            from sqlvector.exceptions import LoaderError
            
            with pytest.raises(LoaderError):
                vss_rag.create_index("test_index")


class TestSQLiteEdgeCases:
    @pytest.fixture
    def sqlite_rag_instance(self):
        """Create a SQLite RAG instance for testing."""
        provider = DefaultEmbeddingProvider(dimension=384)
        return SQLiteRAG(
            db_path=":memory:",
            embedding_provider=provider,
            embedding_dimension=384,
            batch_size=100
        )
    
    def test_empty_database_queries(self):
        """Test queries on empty database."""
        rag = SQLiteRAG(db_path=":memory:", embedding_dimension=384)
        
        # Query empty database
        results = rag.query("test query", top_k=5)
        assert len(results) == 0
        
        # Filter query on empty database
        filter_results = rag.query_with_filters({"category": "test"})
        assert len(filter_results) == 0
        
        # Statistics on empty database
        stats = rag.get_statistics()
        assert stats["total_documents"] == 0
        assert stats["total_embeddings"] == 0
    
    def test_large_metadata(self, sqlite_rag_instance):
        """Test handling of large metadata objects."""
        large_metadata = {
            "description": "A" * 1000,  # Large text field
            "tags": [f"tag_{i}" for i in range(100)],  # Large list
            "nested": {
                "level1": {
                    "level2": {
                        "data": list(range(50))
                    }
                }
            }
        }
        
        document_id = sqlite_rag_instance.load_document(
            content="Document with large metadata",
            metadata=large_metadata
        )
        
        # Verify document was stored correctly
        doc = sqlite_rag_instance.get_document(document_id)
        assert doc is not None
        assert doc.metadata == large_metadata
    
    def test_unicode_content(self, sqlite_rag_instance):
        """Test handling of Unicode content."""
        unicode_content = "Testing Unicode: 你好世界 🌍 café naïve résumé"
        unicode_metadata = {"language": "multi", "emoji": "🚀", "accent": "café"}
        
        document_id = sqlite_rag_instance.load_document(
            content=unicode_content,
            metadata=unicode_metadata
        )
        
        # Verify Unicode is preserved
        doc = sqlite_rag_instance.get_document(document_id)
        assert doc is not None
        assert doc.content == unicode_content
        assert doc.metadata == unicode_metadata
        
        # Test querying with Unicode
        results = sqlite_rag_instance.query("你好", top_k=1)
        assert len(results) >= 0  # Should not crash