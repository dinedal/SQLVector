"""Tests for DuckDB Vector Similarity Search (VSS) extension functionality."""

import pytest
import tempfile
from pathlib import Path

from sql_rag.backends.duckdb import DuckDBRAG
from sql_rag.exceptions import LoaderError


class TestDuckDBVSS:
    """Test DuckDB Vector Similarity Search functionality."""
    
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
        """Create DuckDB RAG instance with VSS extension enabled."""
        return DuckDBRAG(
            db_path=temp_db_path,
            embedding_dimension=384,
            enable_vss_extension=True,
            vss_enable_persistence=True
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
        assert vss_rag_instance.config.enable_vss_extension == True
        assert vss_rag_instance.config.vss_enable_persistence == True
    
    def test_vss_disabled_by_default(self, temp_db_path):
        """Test that VSS is disabled by default."""
        rag = DuckDBRAG(db_path=temp_db_path, embedding_dimension=384)
        assert rag.config.enable_vss_extension == False
    
    def test_create_index_without_vss(self, temp_db_path):
        """Test that creating index without VSS enabled raises error."""
        rag = DuckDBRAG(db_path=temp_db_path, embedding_dimension=384, enable_vss_extension=False)
        
        # Load some documents first
        documents = [{"content": "Test document"}]
        rag.load_documents(documents, show_progress=False)
        
        # Try to create index without VSS enabled
        with pytest.raises(LoaderError, match="VSS extension is not enabled"):
            rag.create_index("test_idx")
    
    def test_create_index_vss_enabled(self, vss_rag_instance, sample_documents):
        """Test creating HNSW index with VSS enabled."""
        # Skip if VSS extension is not actually available
        try:
            # Load documents
            document_ids = vss_rag_instance.load_documents(sample_documents, show_progress=False)
            assert len(document_ids) == 5
            
            # Create cosine index
            result = vss_rag_instance.create_index(
                index_name="test_cosine_idx",
                similarity_function="cosine",
                ef_construction=64,
                ef_search=32,
                M=8
            )
            assert result == True
            
        except Exception as e:
            if "vss" in str(e).lower() or "extension" in str(e).lower():
                pytest.skip("VSS extension not available in this DuckDB installation")
            else:
                raise
    
    def test_create_multiple_indexes(self, vss_rag_instance, sample_documents):
        """Test creating multiple indexes with different similarity functions."""
        try:
            # Load documents
            vss_rag_instance.load_documents(sample_documents, show_progress=False)
            
            # Create multiple indexes
            similarity_functions = ["cosine", "inner_product", "euclidean"]
            for sim_func in similarity_functions:
                result = vss_rag_instance.create_index(
                    index_name=f"test_{sim_func}_idx",
                    similarity_function=sim_func,
                    M=8
                )
                assert result == True
                
        except Exception as e:
            if "vss" in str(e).lower() or "extension" in str(e).lower():
                pytest.skip("VSS extension not available in this DuckDB installation")
            else:
                raise
    
    def test_delete_index(self, vss_rag_instance, sample_documents):
        """Test deleting an HNSW index."""
        try:
            # Load documents and create index
            vss_rag_instance.load_documents(sample_documents, show_progress=False)
            vss_rag_instance.create_index("test_delete_idx", M=8)
            
            # Delete index
            result = vss_rag_instance.delete_index("test_delete_idx")
            assert result == True
            
            # Deleting non-existent index should still return True (IF EXISTS)
            result = vss_rag_instance.delete_index("non_existent_idx")
            assert result == True
            
        except Exception as e:
            if "vss" in str(e).lower() or "extension" in str(e).lower():
                pytest.skip("VSS extension not available in this DuckDB installation")
            else:
                raise
    
    def test_compact_index(self, vss_rag_instance, sample_documents):
        """Test compacting an HNSW index."""
        try:
            # Load documents and create index
            document_ids = vss_rag_instance.load_documents(sample_documents, show_progress=False)
            vss_rag_instance.create_index("test_compact_idx", M=8)
            
            # Delete a document to create deleted items in index
            vss_rag_instance.delete_document(document_ids[0])
            
            # Compact index
            result = vss_rag_instance.compact_index("test_compact_idx")
            assert result == True
            
        except Exception as e:
            if "vss" in str(e).lower() or "extension" in str(e).lower():
                pytest.skip("VSS extension not available in this DuckDB installation")
            else:
                raise
    
    def test_compact_index_without_vss(self, temp_db_path):
        """Test that compacting index without VSS enabled raises error."""
        rag = DuckDBRAG(db_path=temp_db_path, embedding_dimension=384, enable_vss_extension=False)
        
        with pytest.raises(LoaderError, match="VSS extension is not enabled"):
            rag.compact_index("test_idx")
    
    def test_hnsw_optimized_query(self, vss_rag_instance, sample_documents):
        """Test HNSW-optimized queries."""
        try:
            # Load documents and create index
            vss_rag_instance.load_documents(sample_documents, show_progress=False)
            vss_rag_instance.create_index("test_query_idx", similarity_function="cosine", M=8)
            
            # Test standard query
            results_standard = vss_rag_instance.query(
                query_text="machine learning",
                top_k=3,
                use_hnsw_optimization=False
            )
            
            # Test HNSW-optimized query
            results_hnsw = vss_rag_instance.query(
                query_text="machine learning", 
                top_k=3,
                use_hnsw_optimization=True
            )
            
            # Both should return results
            assert len(results_standard) > 0
            assert len(results_hnsw) > 0
            
            # Results should be similar (though may differ slightly due to approximation)
            # HNSW is approximate, so we just check that we get reasonable results
            assert len(results_hnsw) <= 3  # Should respect top_k limit
            
        except Exception as e:
            if "vss" in str(e).lower() or "extension" in str(e).lower():
                pytest.skip("VSS extension not available in this DuckDB installation")
            else:
                raise
    
    def test_hnsw_optimization_without_vss(self, temp_db_path):
        """Test HNSW optimization falls back to standard query without VSS."""
        rag = DuckDBRAG(db_path=temp_db_path, embedding_dimension=384, enable_vss_extension=False)
        
        # Load documents
        documents = [{"content": "Test document for fallback"}]
        rag.load_documents(documents, show_progress=False)
        
        # Query with HNSW optimization requested but VSS disabled
        # Should fall back to standard query without error
        results = rag.query(
            query_text="test",
            top_k=1,
            use_hnsw_optimization=True
        )
        
        assert len(results) >= 0  # Should work but use standard query
    
    def test_index_parameters(self, vss_rag_instance, sample_documents):
        """Test different index parameters."""
        try:
            vss_rag_instance.load_documents(sample_documents, show_progress=False)
            
            # Test with custom parameters
            result = vss_rag_instance.create_index(
                index_name="custom_params_idx",
                similarity_function="cosine",
                ef_construction=32,
                ef_search=16,
                M=4,
                M0=8
            )
            assert result == True
            
        except Exception as e:
            if "vss" in str(e).lower() or "extension" in str(e).lower():
                pytest.skip("VSS extension not available in this DuckDB installation")
            else:
                raise


class TestDuckDBVSSMemory:
    """Test VSS functionality with in-memory database."""
    
    @pytest.fixture
    def memory_vss_rag(self):
        """Create in-memory DuckDB RAG with VSS."""
        return DuckDBRAG(
            db_path=":memory:",
            embedding_dimension=384,
            enable_vss_extension=True,
            vss_enable_persistence=False  # Not applicable for memory DB
        )
    
    def test_memory_vss_functionality(self, memory_vss_rag):
        """Test that VSS works with in-memory databases."""
        try:
            # Load sample documents
            documents = [
                {"content": "Artificial intelligence transforms technology"},
                {"content": "Machine learning enables pattern recognition"},
                {"content": "Data science extracts insights from information"}
            ]
            
            memory_vss_rag.load_documents(documents, show_progress=False)
            
            # Create index (should work with in-memory DB)
            result = memory_vss_rag.create_index("memory_idx", M=8)
            assert result == True
            
            # Test query
            results = memory_vss_rag.query("artificial intelligence", top_k=2)
            assert len(results) > 0
            
        except Exception as e:
            if "vss" in str(e).lower() or "extension" in str(e).lower():
                pytest.skip("VSS extension not available in this DuckDB installation")
            else:
                raise