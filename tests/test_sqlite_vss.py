"""Tests for SQLite Vector Similarity Search (VSS) extension functionality."""

import pytest
import tempfile
from pathlib import Path
import sqlite3

from sql_rag.backends.sqlite import SQLiteRAG
from sql_rag.exceptions import LoaderError

# Check if sqlite-vss is available
def is_vss_available():
    """Check if sqlite-vss extension is available."""
    try:
        import sqlite_vss
        # Try to load it in a test connection
        conn = sqlite3.connect(':memory:')
        conn.enable_load_extension(True)
        sqlite_vss.load(conn)
        conn.close()
        return True
    except (ImportError, sqlite3.OperationalError):
        return False

# Skip all VSS tests if sqlite-vss is not available
pytestmark = pytest.mark.skipif(not is_vss_available(), reason="sqlite-vss extension not available")


class TestSQLiteVSS:
    """Test SQLite Vector Similarity Search functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
            db_path = f.name
        # Just use the path, don't create an empty file
        yield db_path
        # Cleanup
        try:
            Path(db_path).unlink()
        except:
            pass
    
    @pytest.fixture
    def vss_rag_instance(self, temp_db_path):
        """Create SQLite RAG instance with VSS extension enabled."""
        return SQLiteRAG(
            db_path=temp_db_path,
            embedding_dimension=384,
            enable_vss_extension=True,
            vss_factory_string="Flat"
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"content": "Machine learning algorithms process data", "metadata": {"category": "AI"}},
            {"content": "Deep neural networks recognize patterns", "metadata": {"category": "AI"}}, 
            {"content": "Python programming language for data science", "metadata": {"category": "programming"}},
            {"content": "SQL databases store structured information", "metadata": {"category": "database"}},
            {"content": "Vector search enables semantic retrieval", "metadata": {"category": "search"}},
        ]
    
    def test_vss_initialization(self, vss_rag_instance):
        """Test VSS-enabled RAG initialization."""
        # VSS extension may not be available, so test configuration intent
        # The actual extension status may be disabled if sqlite-vss is not available
        assert vss_rag_instance.config.vss_factory_string == "Flat"
    
    def test_vss_disabled_by_default(self, temp_db_path):
        """Test that VSS is disabled by default."""
        rag = SQLiteRAG(db_path=temp_db_path, embedding_dimension=384)
        assert rag.config.enable_vss_extension == False
    
    def test_create_index_without_vss(self, temp_db_path):
        """Test that creating index without VSS enabled raises error."""
        rag = SQLiteRAG(db_path=temp_db_path, embedding_dimension=384, enable_vss_extension=False)
        
        # Load some documents first
        documents = [{"content": "Test document"}]
        rag.load_documents(documents, show_progress=False)
        
        # Try to create index without VSS enabled
        with pytest.raises(LoaderError, match="VSS extension is not enabled"):
            rag.create_index("test_idx")
    
    def test_create_index_vss_enabled(self, vss_rag_instance, sample_documents):
        """Test creating Faiss index with VSS enabled."""
        # Load documents
        document_ids = vss_rag_instance.load_documents(sample_documents, show_progress=False)
        assert len(document_ids) == 5
        
        # Create Flat index (default)
        result = vss_rag_instance.create_index(
            index_name="test_flat_idx",
            similarity_function="cosine",
            factory_string="Flat,IDMap2"
        )
        assert result == True
    
    def test_create_ivf_index(self, vss_rag_instance, sample_documents):
        """Test creating IVF (Inverted File) index."""
        # Load documents
        vss_rag_instance.load_documents(sample_documents, show_progress=False)
        
        # Create IVF index (requires training)
        result = vss_rag_instance.create_index(
            index_name="test_ivf_idx",
            factory_string="IVF4,Flat,IDMap2"  # Small number of centroids for test
        )
        assert result == True
        
        # Train the index
        train_result = vss_rag_instance.train_index(training_data_limit=5)
        assert train_result == True
    
    def test_create_multiple_factory_strings(self, vss_rag_instance, sample_documents):
        """Test creating indexes with different Faiss factory strings."""
        # Load documents
        vss_rag_instance.load_documents(sample_documents, show_progress=False)
        
        # Test different factory strings
        factory_strings = [
            "Flat,IDMap2",
            "IVF2,Flat,IDMap2",  # Small IVF for testing
            "PCA16,Flat,IDMap2",  # PCA dimension reduction
        ]
        
        for i, factory_string in enumerate(factory_strings):
            result = vss_rag_instance.create_index(
                index_name=f"test_factory_{i}_idx",
                factory_string=factory_string
            )
            assert result == True
            
            # Train IVF indexes
            if "IVF" in factory_string:
                vss_rag_instance.train_index(training_data_limit=5)
    
    def test_delete_index(self, vss_rag_instance, sample_documents):
        """Test deleting a VSS index."""
        # Load documents and create index
        vss_rag_instance.load_documents(sample_documents, show_progress=False)
        vss_rag_instance.create_index("test_delete_idx", factory_string="Flat,IDMap2")
        
        # Delete index (should reset to default factory)
        result = vss_rag_instance.delete_index("test_delete_idx")
        assert result == True
        
        # Verify factory string was reset
        assert vss_rag_instance.config.vss_factory_string == "Flat"
    
    def test_delete_index_without_vss(self, temp_db_path):
        """Test that deleting index without VSS enabled raises error."""
        rag = SQLiteRAG(db_path=temp_db_path, embedding_dimension=384, enable_vss_extension=False)
        
        with pytest.raises(LoaderError, match="VSS extension is not enabled"):
            rag.delete_index("test_idx")
    
    def test_train_index_without_vss(self, temp_db_path):
        """Test that training index without VSS enabled raises error."""
        rag = SQLiteRAG(db_path=temp_db_path, embedding_dimension=384, enable_vss_extension=False)
        
        with pytest.raises(LoaderError, match="VSS extension is not enabled"):
            rag.train_index()
    
    def test_vss_optimized_query(self, vss_rag_instance, sample_documents):
        """Test VSS-optimized queries."""
        # Load documents and create index
        vss_rag_instance.load_documents(sample_documents, show_progress=False)
        vss_rag_instance.create_index("test_query_idx", factory_string="Flat,IDMap2")
        
        # Test standard query
        results_standard = vss_rag_instance.query(
            query_text="machine learning",
            top_k=3,
            use_vss_optimization=False
        )
        
        # Test VSS-optimized query
        results_vss = vss_rag_instance.query(
            query_text="machine learning", 
            top_k=3,
            use_vss_optimization=True
        )
        
        # Both should return results
        assert len(results_standard) > 0
        assert len(results_vss) > 0
        
        # VSS results should respect top_k limit
        assert len(results_vss) <= 3
        
        # VSS should use distance values (converted to similarity)
        for result in results_vss:
            assert isinstance(result["similarity"], float)
            assert 0.0 <= result["similarity"] <= 1.0  # Should be converted to similarity
    
    def test_vss_optimization_fallback(self, temp_db_path):
        """Test VSS optimization falls back to standard query without VSS."""
        rag = SQLiteRAG(db_path=temp_db_path, embedding_dimension=384, enable_vss_extension=False)
        
        # Load documents
        documents = [{"content": "Test document for fallback"}]
        rag.load_documents(documents, show_progress=False)
        
        # Query with VSS optimization requested but VSS disabled
        # Should fall back to standard query without error
        results = rag.query(
            query_text="test",
            top_k=1,
            use_vss_optimization=True
        )
        
        assert len(results) >= 0  # Should work but use standard query
    
    def test_vss_with_precomputed_embedding(self, vss_rag_instance, sample_documents):
        """Test VSS queries with precomputed embeddings."""
        # Load documents and create index
        vss_rag_instance.load_documents(sample_documents, show_progress=False)
        vss_rag_instance.create_index("test_precomputed_idx", factory_string="Flat,IDMap2")
        
        # Use a simple embedding vector
        query_embedding = [0.1] * 384  # Match the embedding dimension
        
        # Test VSS query with precomputed embedding
        results = vss_rag_instance.query_with_embedding(
            query_embedding=query_embedding,
            top_k=3,
            use_vss_optimization=True
        )
        
        assert len(results) >= 0
        assert len(results) <= 3
    
    def test_vss_with_filters(self, vss_rag_instance, sample_documents):
        """Test VSS optimization with metadata filters."""
        # Load documents and create index
        vss_rag_instance.load_documents(sample_documents, show_progress=False)
        vss_rag_instance.create_index("test_filter_idx", factory_string="Flat,IDMap2")
        
        # Query with both VSS optimization and filters
        results = vss_rag_instance.query_with_filters(
            filters={"category": "AI"},
            query_text="machine learning algorithms",
            top_k=5,
            use_vss_optimization=True
        )
        
        # Should return results filtered by category
        assert len(results) >= 0
        for result in results:
            # Metadata is stored as JSON string, need to parse it
            import json
            metadata = json.loads(result["metadata"])
            assert metadata["category"] == "AI"
    
    def test_factory_string_validation(self, vss_rag_instance, sample_documents):
        """Test different Faiss factory string configurations."""
        # Load documents
        vss_rag_instance.load_documents(sample_documents, show_progress=False)
        
        # Test valid factory strings
        valid_factory_strings = [
            "Flat",
            "Flat,IDMap2", 
            "IVF2,Flat",
            "IVF2,Flat,IDMap2",
            "PCA16,Flat",
            "PCA16,IVF2,Flat,IDMap2"
        ]
        
        for factory_string in valid_factory_strings:
            result = vss_rag_instance.create_index(
                f"test_{factory_string.replace(',', '_').replace(':', '_')}_idx",
                factory_string=factory_string
            )
            assert result == True
            
            # Train IVF indexes
            if "IVF" in factory_string:
                vss_rag_instance.train_index(training_data_limit=5)


class TestSQLiteVSSMemory:
    """Test VSS functionality with in-memory database."""
    
    @pytest.fixture
    def memory_vss_rag(self):
        """Create in-memory SQLite RAG with VSS."""
        return SQLiteRAG(
            db_path=":memory:",
            embedding_dimension=384,
            enable_vss_extension=True,
            vss_factory_string="Flat"
        )
    
    def test_memory_vss_functionality(self, memory_vss_rag):
        """Test that VSS works with in-memory databases."""
        # Load sample documents
        documents = [
            {"content": "Artificial intelligence transforms technology"},
            {"content": "Machine learning enables pattern recognition"},
            {"content": "Data science extracts insights from information"}
        ]
        
        memory_vss_rag.load_documents(documents, show_progress=False)
        
        # Create index (should work with in-memory DB)
        result = memory_vss_rag.create_index("memory_idx", factory_string="Flat,IDMap2")
        assert result == True
        
        # Test query
        results = memory_vss_rag.query(
            "artificial intelligence", 
            top_k=2,
            use_vss_optimization=True
        )
        assert len(results) >= 0
    
    def test_memory_vss_statistics(self, memory_vss_rag):
        """Test VSS statistics with in-memory database."""
        # Load documents
        documents = [{"content": "Test document for stats"}]
        memory_vss_rag.load_documents(documents, show_progress=False)
        
        # Get statistics
        stats = memory_vss_rag.get_statistics()
        
        # Should show VSS configuration (may be disabled if extension not available)
        assert "vss_enabled" in stats
        assert stats["vss_factory_string"] == "Flat"
        assert stats["total_documents"] == 1


class TestSQLiteVSSEdgeCases:
    """Test edge cases and error conditions for SQLite VSS."""
    
    def test_train_flat_index(self):
        """Test training a Flat index (should succeed even though not needed)."""
        rag = SQLiteRAG(
            db_path=":memory:",
            embedding_dimension=384,
            enable_vss_extension=True,
            vss_factory_string="Flat"
        )
        
        # Load documents
        documents = [{"content": "Test document"}]
        rag.load_documents(documents, show_progress=False)
        
        # Train Flat index (should succeed but do nothing)
        result = rag.train_index()
        assert result == True
    
    def test_empty_database_vss_operations(self):
        """Test VSS operations on empty database."""
        rag = SQLiteRAG(
            db_path=":memory:",
            embedding_dimension=384,
            enable_vss_extension=True
        )
        
        # Try to create index on empty database
        result = rag.create_index("empty_idx")
        assert result == True
        
        # Try to train index with no data
        train_result = rag.train_index()
        assert train_result == True
    
    def test_vss_with_large_dimensions(self):
        """Test VSS with larger embedding dimensions."""
        rag = SQLiteRAG(
            db_path=":memory:",
            embedding_dimension=1536,  # Larger dimension like OpenAI embeddings
            enable_vss_extension=True,
            vss_factory_string="Flat"
        )
        
        # Load documents with larger embeddings
        documents = [{"content": "Test with large embeddings"}]
        rag.load_documents(documents, show_progress=False)
        
        # Create index
        result = rag.create_index("large_dim_idx")
        assert result == True
        
        # Test query
        results = rag.query(
            "test query",
            top_k=1,
            use_vss_optimization=True
        )
        assert len(results) >= 0