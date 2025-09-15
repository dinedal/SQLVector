"""Tests for PostgreSQL backend."""

import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Dict, List, Any

# Skip all tests if PostgreSQL dependencies are not available
try:
    import asyncpg
    import psycopg2
    from sqlvector.backends.postgres import (
        PostgresRAG,
        PostgresConfig,
        PostgresDocument,
        PostgresEmbedding,
        PostgresQueryResult,
        PostgresLoader,
        PostgresQuerier,
    )
    from sqlvector.embedding import DefaultEmbeddingProvider

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Mark all tests in this file as PostgreSQL tests
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL dependencies not available")
]

# Note: Database URL is now managed by auto-start fixtures


class TestPostgresConfig:
    """Test PostgreSQL configuration."""

    def test_config_creation(self):
        """Test PostgreSQL config creation."""
        config = PostgresConfig(
            host="localhost",
            port=5432,
            user="test",
            password="test",
            database="test_db",
            embedding_dimension=384,
            batch_size=500,
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "test"
        assert config.password == "test"
        assert config.database == "test_db"
        assert config.embedding_dimension == 384
        assert config.batch_size == 500
        assert config.documents_table == "documents"
        assert config.embeddings_table == "embeddings"
        assert "postgresql://test:test@localhost:5432/test_db" in config.db_url

    def test_config_with_url(self):
        """Test config creation with database URL."""
        config = PostgresConfig(
            db_url="postgresql://user:pass@host:5432/db", embedding_dimension=768
        )

        assert config.db_url == "postgresql://user:pass@host:5432/db"
        assert config.embedding_dimension == 768

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            PostgresConfig(host="localhost", database="test", embedding_dimension=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            PostgresConfig(host="localhost", database="test", batch_size=0)

    def test_schema_generation(self):
        """Test SQL schema generation."""
        config = PostgresConfig(
            host="localhost", database="test", embedding_dimension=384
        )

        # Test documents schema
        docs_schema = config.get_documents_schema()
        assert "CREATE TABLE IF NOT EXISTS documents" in docs_schema
        assert "id TEXT PRIMARY KEY" in docs_schema
        assert "content TEXT NOT NULL" in docs_schema
        assert "metadata JSONB" in docs_schema

        # Test embeddings schema
        emb_schema = config.get_embeddings_schema()
        assert "CREATE TABLE IF NOT EXISTS embeddings" in emb_schema
        assert "vector(384)" in emb_schema
        assert "REFERENCES documents(id) ON DELETE CASCADE" in emb_schema

        # Test index schema
        index_schema = config.get_index_schema("test_idx", "cosine")
        assert "CREATE INDEX IF NOT EXISTS test_idx" in index_schema
        assert "vector_cosine_ops" in index_schema


class TestPostgresDocument:
    """Test PostgreSQL document model."""

    def test_document_creation(self):
        """Test document creation."""
        doc = PostgresDocument(
            id="test-id",
            content="Test content",
            metadata={"type": "test", "score": 0.9},
        )

        assert doc.id == "test-id"
        assert doc.content == "Test content"
        assert doc.metadata == {"type": "test", "score": 0.9}

    def test_document_to_dict(self):
        """Test document serialization."""
        doc = PostgresDocument(
            id="test-id", content="Test content", metadata={"type": "test"}
        )

        data = doc.to_dict()
        assert data["id"] == "test-id"
        assert data["content"] == "Test content"
        assert '"type": "test"' in data["metadata"]

    def test_document_from_dict(self):
        """Test document deserialization."""
        data = {
            "id": "test-id",
            "content": "Test content",
            "metadata": '{"type": "test"}',
        }

        doc = PostgresDocument.from_dict(data)
        assert doc.id == "test-id"
        assert doc.content == "Test content"
        assert doc.metadata == {"type": "test"}


class TestPostgresEmbedding:
    """Test PostgreSQL embedding model."""

    def test_embedding_creation(self):
        """Test embedding creation."""
        emb = PostgresEmbedding(
            id="emb-id",
            document_id="doc-id",
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
        )

        assert emb.id == "emb-id"
        assert emb.document_id == "doc-id"
        assert emb.embedding == [0.1, 0.2, 0.3]
        assert emb.model_name == "test-model"

    def test_vector_formatting(self):
        """Test vector formatting for PostgreSQL."""
        embedding = [0.1, 0.2, 0.3]
        formatted = PostgresEmbedding.format_vector(embedding)
        assert formatted == "[0.1,0.2,0.3]"

    def test_vector_parsing(self):
        """Test vector parsing from PostgreSQL."""
        # String format
        parsed = PostgresEmbedding.parse_vector("[0.1,0.2,0.3]")
        assert parsed == [0.1, 0.2, 0.3]

        # List format
        parsed = PostgresEmbedding.parse_vector([0.1, 0.2, 0.3])
        assert parsed == [0.1, 0.2, 0.3]

    def test_embedding_serialization(self):
        """Test embedding serialization."""
        emb = PostgresEmbedding(
            id="emb-id", document_id="doc-id", embedding=[0.1, 0.2, 0.3]
        )

        data = emb.to_dict()
        assert data["id"] == "emb-id"
        assert data["document_id"] == "doc-id"
        assert data["embedding"] == "[0.1,0.2,0.3]"


class TestPostgresQueryResult:
    """Test PostgreSQL query result model."""

    def test_query_result_creation(self):
        """Test query result creation."""
        result = PostgresQueryResult(
            id="doc-id",
            content="Test content",
            similarity=0.95,
            metadata={"type": "test"},
        )

        assert result.id == "doc-id"
        assert result.content == "Test content"
        assert result.similarity == 0.95
        assert result.metadata == {"type": "test"}

    def test_query_result_from_row(self):
        """Test query result from database row."""
        row = {
            "id": "doc-id",
            "content": "Test content",
            "similarity": "0.95",
            "metadata": '{"type": "test"}',
            "embedding": "[0.1,0.2,0.3]",
        }

        result = PostgresQueryResult.from_row(row)
        assert result.id == "doc-id"
        assert result.content == "Test content"
        assert result.similarity == 0.95
        assert result.metadata == {"type": "test"}
        assert result.embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
class TestPostgresRAGWithMocks:
    """Test PostgreSQL RAG with mocked connections."""

    @pytest.fixture
    def mock_config(self):
        """Create a mocked PostgreSQL config."""
        config = PostgresConfig(
            db_url="postgresql://test:test@localhost/test", embedding_dimension=384
        )

        # Mock the connection methods
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetch = AsyncMock()
        mock_conn.fetchrow = AsyncMock()
        mock_conn.fetchval = AsyncMock()
        mock_conn.executemany = AsyncMock()

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_async_connection():
            yield mock_conn

        config.get_async_connection = mock_async_connection
        return config, mock_conn

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mocked embedding service."""
        service = AsyncMock()
        service.create_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        service.create_embeddings_batch = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]
        )
        service.provider = MagicMock()
        service.provider.model_name = "test-model"
        return service

    async def test_loader_load_document(self, mock_config, mock_embedding_service):
        """Test loading a single document."""
        config, mock_conn = mock_config

        loader = PostgresLoader(config, mock_embedding_service)

        # Test loading
        doc_id = await loader.load_document_async(
            content="Test content", metadata={"type": "test"}, generate_embedding=True
        )

        assert doc_id is not None
        assert mock_conn.executemany.called
        assert mock_embedding_service.create_embeddings_batch.called

    async def test_querier_query(self, mock_config, mock_embedding_service):
        """Test querying documents."""
        config, mock_conn = mock_config

        querier = PostgresQuerier(config, mock_embedding_service)

        # Mock query results
        mock_conn.fetch.return_value = [
            {
                "id": "doc-1",
                "content": "Test content 1",
                "similarity": 0.95,
                "metadata": '{"type": "test"}',
                "embedding": "[0.1,0.2,0.3,0.4]",
            }
        ]

        # Test query
        results = await querier.query_async(
            query_text="test query", top_k=5, similarity_function="cosine"
        )

        assert len(results) == 1
        assert results[0]["id"] == "doc-1"
        assert results[0]["content"] == "Test content 1"
        assert results[0]["similarity"] == 0.95
        assert mock_embedding_service.create_embedding.called

    async def test_rag_integration(self, mock_config, mock_embedding_service):
        """Test complete RAG integration."""
        config, mock_conn = mock_config

        # Mock database setup
        config.setup_database = AsyncMock()

        # Create RAG instance with mocked components
        with patch("sqlvector.backends.postgres.rag.asyncio.run") as mock_run:
            mock_run.return_value = None

            rag = PostgresRAG(
                db_url="postgresql://test:test@localhost/test", embedding_dimension=384
            )

            # Replace the embedding service with our mock
            rag.embedding_service = mock_embedding_service
            rag.config = config
            rag.loader.config = config
            rag.querier.config = config

        # Test loading documents
        documents = [
            {"content": "Document 1", "metadata": {"type": "test"}},
            {"content": "Document 2", "metadata": {"type": "test"}},
        ]

        doc_ids = await rag.load_documents_async(documents)
        assert len(doc_ids) == 2

        # Mock query results
        mock_conn.fetch.return_value = [
            {
                "id": doc_ids[0],
                "content": "Document 1",
                "similarity": 0.95,
                "metadata": '{"type": "test"}',
                "embedding": "[0.1,0.2,0.3,0.4]",
            }
        ]

        # Test querying
        results = await rag.query_async("test query", top_k=1)
        assert len(results) == 1
        assert results[0]["content"] == "Document 1"


class TestPostgresRAGIntegration:
    """Integration tests for PostgreSQL RAG (uses auto-managed PostgreSQL)."""

    async def test_full_workflow(self, postgres_rag):
        """Test complete workflow with real database."""
        # Load test documents
        documents = [
            {
                "content": "Machine learning algorithms for data analysis",
                "metadata": {"topic": "ML", "difficulty": "intermediate"},
            },
            {
                "content": "Deep learning neural networks and applications",
                "metadata": {"topic": "DL", "difficulty": "advanced"},
            },
            {
                "content": "Python programming fundamentals and best practices",
                "metadata": {"topic": "Python", "difficulty": "beginner"},
            },
        ]

        # Load documents
        doc_ids = await postgres_rag.load_documents_async(
            documents, show_progress=False
        )
        assert len(doc_ids) == 3

        # Test similarity query
        results = await postgres_rag.query_async(
            "artificial intelligence algorithms", top_k=2
        )
        assert len(results) <= 2
        assert all("similarity" in r for r in results)

        # Test filtered query
        filtered_results = await postgres_rag.query_with_filters_async(
            filters={"topic": "ML"}, query_text="machine learning", top_k=1
        )
        assert len(filtered_results) <= 1

        # Test document retrieval
        doc = await postgres_rag.get_document_async(doc_ids[0])
        assert doc is not None
        assert doc.content == documents[0]["content"]

        # Test similar documents
        similar_docs = await postgres_rag.find_similar_documents_async(
            doc_ids[0], top_k=2
        )
        assert len(similar_docs) <= 2
        # Should not include the source document itself
        assert all(r["id"] != doc_ids[0] for r in similar_docs)

        # Test index creation
        index_created = await postgres_rag.create_index_async(
            "test_cosine_idx", similarity_function="cosine"
        )
        assert index_created

        # Test statistics
        stats = await postgres_rag.get_statistics_async()
        assert stats["document_count"] >= 3
        assert stats["embedding_count"] >= 3
        assert stats["embedding_dimension"] == 384

        # Cleanup - delete documents
        for doc_id in doc_ids:
            deleted = await postgres_rag.delete_document_async(doc_id)
            assert deleted


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
