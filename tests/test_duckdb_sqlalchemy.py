"""Tests for DuckDB backend with SQLAlchemy sync Engine integration."""

import pytest
import tempfile
from pathlib import Path
from sqlalchemy import create_engine

from sql_rag.backends.duckdb import DuckDBRAG, DuckDBConfig
from sql_rag.embedding import DefaultEmbeddingProvider


class TestDuckDBSQLAlchemy:
    """Test DuckDB with SQLAlchemy sync Engine integration."""

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

    def test_config_with_sync_engine(self, temp_db_path):
        """Test DuckDBConfig with sync SQLAlchemy engine."""
        # Create sync engine
        engine = create_engine(f"duckdb:///{temp_db_path}")

        config = DuckDBConfig(
            db_path=temp_db_path,
            engine=engine,
            use_sqlalchemy=True,
            embedding_dimension=384,
        )

        assert config.engine is engine
        assert config.use_sqlalchemy is True

        # Test getting SQLAlchemy engine
        retrieved_engine = config.get_sqlalchemy_engine()
        assert retrieved_engine is engine

    def test_config_auto_create_engine(self, temp_db_path):
        """Test DuckDBConfig auto-creating sync engine."""
        config = DuckDBConfig(
            db_path=temp_db_path, use_sqlalchemy=True, embedding_dimension=384
        )

        # Should auto-create engine
        assert config.engine is not None
        assert config.use_sqlalchemy is True

        # Should be able to get connection
        conn = config.get_connection()
        assert conn is not None

    def test_config_memory_db_with_sqlalchemy(self):
        """Test in-memory database with SQLAlchemy."""
        config = DuckDBConfig(
            db_path=":memory:", use_sqlalchemy=True, embedding_dimension=384
        )

        assert config.engine is not None
        assert config.use_sqlalchemy is True

        # Test connection
        conn = config.get_connection()
        assert conn is not None

        # Test basic query
        result = conn.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1

    def test_rag_with_provided_sync_engine(self, temp_db_path):
        """Test DuckDBRAG with user-provided sync engine."""
        # Create sync engine
        engine = create_engine(f"duckdb:///{temp_db_path}")

        rag = DuckDBRAG(
            db_path=temp_db_path,
            engine=engine,
            use_sqlalchemy=True,
            embedding_dimension=384,
        )

        assert rag.config.engine is engine
        assert rag.config.use_sqlalchemy is True

        # Test basic functionality
        doc_id = rag.load_document(
            content="Test document with SQLAlchemy engine", metadata={"source": "test"}
        )

        assert doc_id is not None

        # Verify document retrieval
        doc = rag.get_document(doc_id)
        assert doc is not None
        assert doc.content == "Test document with SQLAlchemy engine"
        assert doc.metadata == {"source": "test"}

    def test_rag_with_auto_engine_creation(self, temp_db_path):
        """Test DuckDBRAG with auto-created sync engine."""
        rag = DuckDBRAG(
            db_path=temp_db_path, use_sqlalchemy=True, embedding_dimension=384
        )

        # Should auto-create engine
        assert rag.config.engine is not None
        assert rag.config.use_sqlalchemy is True

        # Test functionality
        doc_id = rag.load_document(
            content="Test document with auto-created engine", metadata={"type": "auto"}
        )

        assert doc_id is not None

        # Test querying
        results = rag.query("test document", top_k=1, similarity_threshold=-1.0)
        assert len(results) >= 1
        assert results[0]["content"] == "Test document with auto-created engine"

    def test_rag_memory_with_sqlalchemy(self):
        """Test DuckDBRAG in-memory with SQLAlchemy."""
        rag = DuckDBRAG(
            db_path=":memory:", use_sqlalchemy=True, embedding_dimension=384
        )

        # Test basic operations
        doc_ids = rag.load_documents(
            [
                {"content": "First document", "metadata": {"idx": 1}},
                {"content": "Second document", "metadata": {"idx": 2}},
            ],
            show_progress=False,
        )

        assert len(doc_ids) == 2

        # Test query
        results = rag.query("document", top_k=2)
        assert len(results) >= 1  # Should find at least one document

        # Test statistics
        stats = rag.get_statistics()
        assert stats["total_documents"] == 2
        assert stats["total_embeddings"] == 2

    def test_mixed_mode_error_handling(self):
        """Test that DuckDB RAG gracefully handles missing configurations by defaulting to in-memory."""
        # When no db_path or engine provided, should default to in-memory database
        rag = DuckDBRAG(db_path=None, engine=None, use_sqlalchemy=True)
        assert rag.config.db_path == ":memory:"
        
        # When no db_path provided and not using SQLAlchemy, should default to in-memory
        rag2 = DuckDBRAG(db_path=None, use_sqlalchemy=False)
        assert rag2.config.db_path == ":memory:"

    def test_connection_comparison(self, temp_db_path):
        """Test that SQLAlchemy and native connections work similarly."""
        # Create two RAG instances - one with SQLAlchemy, one native
        rag_sqlalchemy = DuckDBRAG(
            db_path=temp_db_path, use_sqlalchemy=True, embedding_dimension=384
        )

        rag_native = DuckDBRAG(
            db_path=":memory:",  # Use different path to avoid conflicts
            use_sqlalchemy=False,
            embedding_dimension=384,
        )

        # Load same documents in both
        test_docs = [
            {"content": "Machine learning algorithms", "metadata": {"category": "AI"}},
            {"content": "Database query optimization", "metadata": {"category": "DB"}},
        ]

        ids_sqlalchemy = rag_sqlalchemy.load_documents(test_docs, show_progress=False)
        ids_native = rag_native.load_documents(test_docs, show_progress=False)

        assert len(ids_sqlalchemy) == 2
        assert len(ids_native) == 2

        # Test queries work similarly
        results_sqlalchemy = rag_sqlalchemy.query(
            "machine learning", top_k=2, similarity_threshold=-1.0
        )
        results_native = rag_native.query(
            "machine learning", top_k=2, similarity_threshold=-1.0
        )

        assert len(results_sqlalchemy) >= 1
        assert len(results_native) >= 1

        # Both should return results (content may vary due to embedding differences)
        # Just verify that we get meaningful results from both approaches
        sqlalchemy_contents = [r["content"].lower() for r in results_sqlalchemy]
        native_contents = [r["content"].lower() for r in results_native]

        # Both should contain documents with similar content types
        assert any(
            "machine" in content or "algorithm" in content
            for content in sqlalchemy_contents
        )
        assert any(
            "machine" in content or "algorithm" in content
            for content in native_contents
        )

    def test_engine_validation_errors(self):
        """Test engine validation error scenarios."""
        config = DuckDBConfig(db_path=":memory:", use_sqlalchemy=False)

        # Should raise error when trying to get SQLAlchemy engine when not configured
        with pytest.raises(ValueError, match="SQLAlchemy engine not configured"):
            config.get_sqlalchemy_engine()

        # Test with None engine but use_sqlalchemy=True without auto-creation path
        config2 = DuckDBConfig(db_path=":memory:", engine=None, use_sqlalchemy=True)
        # This should work because it auto-creates the engine
        assert config2.engine is not None


class TestDuckDBSQLAlchemyAdvanced:
    """Advanced tests for DuckDB SQLAlchemy integration."""

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

    def test_custom_engine_with_connection_pooling(self, temp_db_path):
        """Test custom engine with specific connection pool settings."""
        from sqlalchemy.pool import StaticPool

        # Create engine with custom pool settings
        engine = create_engine(
            f"duckdb:///{temp_db_path}",
            poolclass=StaticPool,
            pool_pre_ping=True,
            echo=False,
        )

        rag = DuckDBRAG(
            db_path=temp_db_path,
            engine=engine,
            use_sqlalchemy=True,
            embedding_dimension=384,
        )

        # Test that it works with custom pool settings
        doc_id = rag.load_document("Test with custom pool settings")
        assert doc_id is not None

        doc = rag.get_document(doc_id)
        assert doc is not None
        assert doc.content == "Test with custom pool settings"

    def test_engine_reuse_across_instances(self, temp_db_path):
        """Test reusing the same engine across multiple RAG instances."""
        engine = create_engine(f"duckdb:///{temp_db_path}")

        # Create two RAG instances with same engine
        rag1 = DuckDBRAG(
            db_path=temp_db_path,
            engine=engine,
            use_sqlalchemy=True,
            embedding_dimension=384,
            documents_table="docs1",
            embeddings_table="emb1",
        )

        rag2 = DuckDBRAG(
            db_path=temp_db_path,
            engine=engine,
            use_sqlalchemy=True,
            embedding_dimension=384,
            documents_table="docs2",
            embeddings_table="emb2",
        )

        # Load documents in each
        doc1_id = rag1.load_document("Document in first RAG")
        doc2_id = rag2.load_document("Document in second RAG")

        # Verify isolation
        assert rag1.get_document(doc1_id) is not None
        assert rag1.get_document(doc2_id) is None  # Should not see doc2

        assert rag2.get_document(doc2_id) is not None
        assert rag2.get_document(doc1_id) is None  # Should not see doc1
