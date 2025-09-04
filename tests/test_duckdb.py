"""Tests for DuckDB backend."""

import pytest
import tempfile
import polars as pl
from pathlib import Path

from sql_rag.backends.duckdb import (
    DuckDBRAG, 
    DuckDBConfig, 
    DuckDBDocument
)
from sql_rag.embedding import DefaultEmbeddingProvider
# QueryResult removed - using raw dictionaries


class TestDuckDBConfig:
    def test_config_creation(self):
        """Test DuckDB config creation."""
        config = DuckDBConfig(
            db_path=":memory:",
            embedding_dimension=384,
            batch_size=500
        )
        
        assert config.db_path == ":memory:"
        assert config.embedding_dimension == 384
        assert config.batch_size == 500
        assert config.documents_table == "documents"
        assert config.embeddings_table == "embeddings"
    
    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            DuckDBConfig(db_path=":memory:", embedding_dimension=0)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DuckDBConfig(db_path=":memory:", batch_size=0)
    
    def test_get_connection(self):
        """Test database connection."""
        config = DuckDBConfig(db_path=":memory:")
        conn = config.get_connection()
        
        assert conn is not None
        
        # Test basic query
        result = conn.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1
        
        conn.close()
    
    def test_database_setup(self):
        """Test database schema setup."""
        config = DuckDBConfig(db_path=":memory:")
        
        with config.get_connection() as conn:
            config.setup_database(conn)
            
            # Check tables exist
            tables = conn.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchall()
            
            table_names = [row[0] for row in tables]
            assert "documents" in table_names
            assert "embeddings" in table_names


class TestDuckDBDocument:
    def test_document_creation(self):
        """Test document model creation."""
        doc = DuckDBDocument(
            id="test-1",
            content="Test content",
            metadata={"category": "test"}
        )
        
        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.metadata == {"category": "test"}
    
    def test_document_to_dict(self):
        """Test document to dictionary conversion."""
        doc = DuckDBDocument(
            id="test-1",
            content="Test content",
            metadata={"category": "test"}
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["id"] == "test-1"
        assert doc_dict["content"] == "Test content"
        assert doc_dict["metadata"] == '{"category": "test"}'
    
    def test_document_from_dict(self):
        """Test document from dictionary creation."""
        doc_dict = {
            "id": "test-1",
            "content": "Test content",
            "metadata": '{"category": "test"}'
        }
        
        doc = DuckDBDocument.from_dict(doc_dict)
        
        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.metadata == {"category": "test"}


class TestDuckDBRAG:
    @pytest.fixture
    def rag_instance(self):
        """Create a DuckDB RAG instance for testing."""
        provider = DefaultEmbeddingProvider(dimension=384)
        return DuckDBRAG(
            db_path=":memory:",
            embedding_provider=provider,
            embedding_dimension=384,
            batch_size=100
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
    
    def test_rag_initialization(self, rag_instance):
        """Test RAG system initialization."""
        assert rag_instance.config.db_path == ":memory:"
        assert rag_instance.config.embedding_dimension == 384
        assert rag_instance.embedding_service is not None
        assert rag_instance.loader is not None
        assert rag_instance.querier is not None
    
    def test_load_single_document(self, rag_instance):
        """Test loading a single document."""
        document_id = rag_instance.load_document(
            content="Test document content",
            metadata={"source": "test"}
        )
        
        assert document_id is not None
        assert isinstance(document_id, str)
        
        # Verify document was stored
        doc = rag_instance.get_document(document_id)
        assert doc is not None
        assert doc.content == "Test document content"
        assert doc.metadata == {"source": "test"}
    
    def test_load_documents_batch(self, rag_instance, sample_documents):
        """Test loading multiple documents."""
        document_ids = rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        assert len(document_ids) == 3
        assert all(isinstance(doc_id, str) for doc_id in document_ids)
        
        # Verify all documents were loaded
        docs = rag_instance.get_documents(document_ids)
        assert len(docs) == 3
        
        contents = [doc.content for doc in docs]
        expected_contents = [doc["content"] for doc in sample_documents]
        for content in expected_contents:
            assert content in contents
    
    def test_load_from_polars(self, rag_instance):
        """Test loading from Polars DataFrame."""
        df = pl.DataFrame({
            "content": [
                "First document",
                "Second document", 
                "Third document"
            ],
            "category": ["A", "B", "A"],
            "priority": [1, 2, 3]
        })
        
        document_ids = rag_instance.load_from_polars(
            df=df,
            metadata_columns=["category", "priority"],
            show_progress=False
        )
        
        assert len(document_ids) == 3
        
        # Verify documents with metadata
        docs = rag_instance.get_documents(document_ids)
        assert len(docs) == 3
        
        # Check metadata was stored correctly
        for doc in docs:
            assert "category" in doc.metadata
            assert "priority" in doc.metadata
    
    def test_query_similarity(self, rag_instance, sample_documents):
        """Test similarity querying."""
        # Load documents
        document_ids = rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Query for animal-related content
        results = rag_instance.query(
            query_text="fox animal",
            top_k=5,
            similarity_threshold=-1.0  # Allow all results
        )
        
        assert len(results) >= 1
        assert all(isinstance(r, dict) for r in results)
        assert all(isinstance(r["similarity"], float) for r in results)
        
        # Results should be sorted by similarity
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_query_with_filters(self, rag_instance, sample_documents):
        """Test filtered querying."""
        # Load documents
        document_ids = rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Query with category filter
        results = rag_instance.query_with_filters(
            filters={"category": "technology"},
            top_k=10
        )
        
        assert len(results) >= 1
        import json
        assert all(json.loads(r["metadata"]).get("category") == "technology" for r in results)
    
    def test_find_similar_documents(self, rag_instance, sample_documents):
        """Test finding similar documents."""
        # Load documents
        document_ids = rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Find documents similar to the first one
        similar_docs = rag_instance.find_similar_documents(
            document_id=document_ids[0],
            top_k=5,
            similarity_threshold=-1.0
        )
        
        # Should find other documents (excluding the queried one)
        assert len(similar_docs) >= 0
        assert all(r["id"] != document_ids[0] for r in similar_docs)
    
    def test_delete_document(self, rag_instance):
        """Test document deletion."""
        # Load a document
        document_id = rag_instance.load_document(
            content="Document to delete",
            metadata={"temp": True}
        )
        
        # Verify it exists
        doc = rag_instance.get_document(document_id)
        assert doc is not None
        
        # Delete it
        deleted = rag_instance.delete_document(document_id)
        assert deleted is True
        
        # Verify it's gone
        doc = rag_instance.get_document(document_id)
        assert doc is None
    
    def test_statistics(self, rag_instance, sample_documents):
        """Test getting database statistics."""
        # Load some documents
        document_ids = rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        stats = rag_instance.get_statistics()
        
        assert "total_documents" in stats
        assert "total_embeddings" in stats
        assert "embedding_dimension" in stats
        assert "batch_size" in stats
        
        assert stats["total_documents"] == 3
        assert stats["total_embeddings"] == 3
        assert stats["embedding_dimension"] == 384
    
    def test_export_to_polars(self, rag_instance, sample_documents):
        """Test exporting to Polars DataFrame."""
        # Load documents
        document_ids = rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Export without embeddings
        df = rag_instance.export_to_polars(include_embeddings=False)
        
        assert len(df) == 3
        assert "id" in df.columns
        assert "content" in df.columns
        assert "metadata" in df.columns
        
        # Export with embeddings
        df_with_emb = rag_instance.export_to_polars(include_embeddings=True)
        
        assert len(df_with_emb) == 3
        assert "embedding" in df_with_emb.columns
    
    def test_context_manager(self):
        """Test context manager usage."""
        with DuckDBRAG(db_path=":memory:", embedding_dimension=384) as rag:
            document_id = rag.load_document("Test content")
            assert document_id is not None
            
            doc = rag.get_document(document_id)
            assert doc is not None
    
    def test_query_batch(self, rag_instance, sample_documents):
        """Test batch querying."""
        # Load documents
        document_ids = rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        # Batch query
        queries = ["fox animal", "machine learning", "python programming"]
        batch_results = rag_instance.query_batch(
            query_texts=queries,
            top_k=2,
            similarity_threshold=-1.0
        )
        
        assert len(batch_results) == 3
        assert all(isinstance(results, list) for results in batch_results)
        assert all(len(results) <= 2 for results in batch_results)
    
    def test_different_similarity_functions(self, rag_instance, sample_documents):
        """Test different similarity functions."""
        # Load documents
        document_ids = rag_instance.load_documents(
            sample_documents,
            show_progress=False
        )
        
        query_text = "machine learning"
        
        # Test cosine similarity
        cosine_results = rag_instance.query(
            query_text, 
            similarity_function="cosine",
            similarity_threshold=-1.0
        )
        
        # Test inner product
        inner_product_results = rag_instance.query(
            query_text,
            similarity_function="inner_product", 
            similarity_threshold=-100.0  # Lower threshold for inner product
        )
        
        # Test euclidean distance
        euclidean_results = rag_instance.query(
            query_text,
            similarity_function="euclidean",
            similarity_threshold=1000.0  # Higher threshold for distance
        )
        
        # All should return some results
        assert len(cosine_results) >= 1
        assert len(inner_product_results) >= 1
        assert len(euclidean_results) >= 1


class TestDuckDBFileOperations:
    def test_load_from_csv_file(self):
        """Test loading from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("content,category,priority\n")
            f.write("First document,A,1\n")
            f.write("Second document,B,2\n")
            f.write("Third document,A,3\n")
            csv_path = f.name
        
        try:
            with DuckDBRAG(db_path=":memory:", embedding_dimension=384) as rag:
                document_ids = rag.load_from_csv(
                    csv_path=csv_path,
                    metadata_columns=["category", "priority"],
                    show_progress=False
                )
                
                assert len(document_ids) == 3
                
                # Verify documents were loaded with metadata
                docs = rag.get_documents(document_ids)
                assert len(docs) == 3
                
                for doc in docs:
                    assert "category" in doc.metadata
                    assert "priority" in doc.metadata
        
        finally:
            Path(csv_path).unlink()  # Clean up temporary file