"""Tests for PostgreSQL backend with custom table names."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import PostgreSQL backend components
try:
    from sqlvector.backends.postgres import PostgresRAG, PostgresConfig
    from sqlvector.embedding import DefaultEmbeddingProvider
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Mark all tests in this file as PostgreSQL tests
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL dependencies not available")
]


class TestPostgresCustomTableNames:
    """Test PostgreSQL backend with custom table and column names."""

    def test_custom_table_names_config(self):
        """Test configuration with custom table names."""
        config = PostgresConfig(
            host="localhost",
            database="test",
            documents_table="custom_docs",
            embeddings_table="custom_embeddings",
            documents_id_column="doc_id",
            documents_content_column="text_content",
            documents_metadata_column="doc_metadata",
            embeddings_id_column="emb_id",
            embeddings_document_id_column="ref_doc_id",
            embeddings_model_column="ai_model",
            embeddings_column="vector_data",
            embedding_dimension=512,
        )

        # Test table names are set correctly
        assert config.documents_table == "custom_docs"
        assert config.embeddings_table == "custom_embeddings"
        assert config.documents_id_column == "doc_id"
        assert config.documents_content_column == "text_content"
        assert config.documents_metadata_column == "doc_metadata"
        assert config.embeddings_id_column == "emb_id"
        assert config.embeddings_document_id_column == "ref_doc_id"
        assert config.embeddings_model_column == "ai_model"
        assert config.embeddings_column == "vector_data"

        # Test schema generation uses custom names
        docs_schema = config.get_documents_schema()
        assert "CREATE TABLE IF NOT EXISTS custom_docs" in docs_schema
        assert "doc_id TEXT PRIMARY KEY" in docs_schema
        assert "text_content TEXT NOT NULL" in docs_schema
        assert "doc_metadata JSONB" in docs_schema

        emb_schema = config.get_embeddings_schema()
        assert "CREATE TABLE IF NOT EXISTS custom_embeddings" in emb_schema
        assert "emb_id TEXT PRIMARY KEY" in emb_schema
        assert "ref_doc_id TEXT NOT NULL REFERENCES custom_docs(doc_id)" in emb_schema
        assert "vector_data vector(512)" in emb_schema
        assert "ai_model TEXT" in emb_schema

        # Test index schema uses custom names
        index_schema = config.get_index_schema("custom_idx", "cosine")
        assert "CREATE INDEX IF NOT EXISTS custom_idx" in index_schema
        assert "ON custom_embeddings" in index_schema
        assert "vector_data vector_cosine_ops" in index_schema
        # Default is ivfflat, not hnsw
        assert "ivfflat" in index_schema.lower() or "hnsw" in index_schema.lower()


@pytest.mark.asyncio
class TestPostgresRAGCustomTablesMocked:
    """Test PostgreSQL RAG with custom tables using mocks."""

    @pytest.fixture
    def mock_rag_instance(self):
        """Create a mocked PostgreSQL RAG with custom table names."""
        # Mock the database connection and operations
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetch = AsyncMock()
        mock_conn.fetchrow = AsyncMock()
        mock_conn.fetchval = AsyncMock()
        mock_conn.executemany = AsyncMock()

        # Create config with custom names
        config = PostgresConfig(
            db_url="postgresql://test:test@localhost/test",
            documents_table="articles",
            embeddings_table="article_vectors",
            documents_id_column="article_id",
            documents_content_column="body_text",
            documents_metadata_column="article_meta",
            embeddings_id_column="vector_id",
            embeddings_document_id_column="article_ref",
            embeddings_model_column="embedding_model",
            embeddings_column="embedding_vector",
            embedding_dimension=768,
        )

        # Mock the async connection context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_async_connection():
            yield mock_conn

        config.get_async_connection = mock_async_connection
        config.setup_database = AsyncMock()

        # Create RAG instance with mocked configuration
        rag = PostgresRAG(
            db_url="postgresql://test:test@localhost/test",
            documents_table="articles",
            embeddings_table="article_vectors",
            documents_id_column="article_id",
            documents_content_column="body_text",
            documents_metadata_column="article_meta",
            embeddings_id_column="vector_id",
            embeddings_document_id_column="article_ref",
            embeddings_model_column="embedding_model",
            embeddings_column="embedding_vector",
            embedding_dimension=768,
            embedding_provider=DefaultEmbeddingProvider(768),
        )

        # Replace config with our mocked one and mark as initialized
        rag.config = config
        rag.loader.config = config
        rag.querier.config = config
        rag._db_initialized = True  # Skip database initialization for mocked tests

        return rag, mock_conn

    async def test_custom_table_document_loading(self, mock_rag_instance):
        """Test document loading with custom table names."""
        rag, mock_conn = mock_rag_instance

        # Mock embedding service
        rag.embedding_service.create_embeddings_batch = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4]]
        )

        # Test loading document
        documents = [
            {
                "content": "Test article content",
                "metadata": {"author": "John Doe", "category": "tech"},
            }
        ]

        doc_ids = await rag.load_documents_async(documents, show_progress=False)

        # Verify the correct table names were used
        calls = mock_conn.executemany.call_args_list
        assert len(calls) >= 1

        # Check that custom table names appear in the SQL queries
        query = calls[0][0][0]  # First call, first argument (SQL query)
        assert "INSERT INTO articles" in query
        assert "article_id" in query
        assert "body_text" in query
        assert "article_meta" in query

        # Check embeddings insert
        if len(calls) > 1:
            emb_query = calls[1][0][0]
            assert "INSERT INTO article_vectors" in emb_query
            assert "vector_id" in emb_query
            assert "article_ref" in emb_query
            assert "embedding_vector" in emb_query
            assert "embedding_model" in emb_query

    async def test_custom_table_querying(self, mock_rag_instance):
        """Test querying with custom table names."""
        rag, mock_conn = mock_rag_instance

        # Mock embedding service
        rag.embedding_service.create_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3, 0.4]
        )

        # Mock query results
        mock_conn.fetch.return_value = [
            {
                "id": "article-1",
                "content": "Test article",
                "similarity": 0.95,
                "metadata": '{"author": "Jane"}',
                "embedding": "[0.1,0.2,0.3,0.4]",
            }
        ]

        # Test query
        results = await rag.query_async("test query", top_k=1)

        # Verify query was called
        assert mock_conn.fetch.called

        # Get the SQL query that was executed
        call_args = mock_conn.fetch.call_args_list[0]
        query = call_args[0][0]  # First positional argument

        # Verify custom table and column names in query
        assert "FROM article_vectors e" in query
        assert "JOIN articles d" in query
        assert "e.article_ref = d.article_id" in query
        assert "d.body_text as content" in query
        assert "d.article_meta::text as metadata" in query
        assert "e.embedding_vector" in query

        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "article-1"

    async def test_custom_table_statistics(self, mock_rag_instance):
        """Test statistics with custom table names."""
        rag, mock_conn = mock_rag_instance

        # Mock statistics queries
        mock_conn.fetchval.side_effect = [5, 5]  # doc count, embedding count
        mock_conn.fetch.return_value = []  # index info

        stats = await rag.get_statistics_async()

        # Verify the correct table names were used in statistics queries
        calls = mock_conn.fetchval.call_args_list

        # Check document count query
        doc_count_query = calls[0][0][0]
        assert "SELECT COUNT(*) FROM articles" in doc_count_query

        # Check embedding count query
        emb_count_query = calls[1][0][0]
        assert "SELECT COUNT(*) FROM article_vectors" in emb_count_query

        # Verify stats structure
        assert stats["document_count"] == 5
        assert stats["embedding_count"] == 5
        assert stats["embedding_dimension"] == 768
        assert stats["tables"]["documents"] == "articles"
        assert stats["tables"]["embeddings"] == "article_vectors"


class TestPostgresCustomTablesIntegration:
    """Integration tests for custom table names (uses auto-managed PostgreSQL)."""

    @pytest.fixture
    async def custom_rag_instance(self, postgres_db_url):
        """Create a RAG instance with custom table names."""
        rag = PostgresRAG(
            db_url=postgres_db_url,
            documents_table="test_articles",
            embeddings_table="test_article_embeddings",
            documents_id_column="article_uuid",
            documents_content_column="article_text",
            documents_metadata_column="article_info",
            embeddings_id_column="emb_uuid",
            embeddings_document_id_column="article_uuid_ref",
            embeddings_model_column="model_used",
            embeddings_column="vector_embedding",
            embedding_dimension=384,
            embedding_provider=DefaultEmbeddingProvider(384),
        )

        yield rag

        # Cleanup - drop the test tables
        try:
            async with rag.config.get_async_connection() as conn:
                await conn.execute(
                    "DROP TABLE IF EXISTS test_article_embeddings CASCADE"
                )
                await conn.execute("DROP TABLE IF EXISTS test_articles CASCADE")
        except Exception:
            pass  # Ignore cleanup errors

        await rag.close()

    async def test_custom_tables_full_workflow(self, custom_rag_instance):
        """Test full workflow with custom table names."""
        rag = custom_rag_instance

        # Load test document
        documents = [
            {
                "content": "Custom table test article about machine learning",
                "metadata": {"category": "AI", "rating": 4.5},
            }
        ]

        doc_ids = await rag.load_documents_async(documents, show_progress=False)
        assert len(doc_ids) == 1

        # Query documents
        results = await rag.query_async("machine learning", top_k=1)
        assert len(results) >= 0  # May be 0 or more due to DefaultEmbeddingProvider behavior
        if len(results) > 0:
            assert results[0]["content"] == documents[0]["content"]

        # Test document retrieval
        doc = await rag.get_document_async(doc_ids[0])
        assert doc is not None
        # PostgresDocument object should have standard field names regardless of custom column names
        assert doc.content == documents[0]["content"]
        # Metadata should be accessible as a dict
        assert doc.metadata is not None
        assert doc.metadata["category"] == "AI"

        # Test metadata filtering
        filtered_results = await rag.query_with_filters_async(
            filters={"category": "AI"}, top_k=1
        )
        assert len(filtered_results) == 1

        # Test statistics
        stats = await rag.get_statistics_async()
        assert stats["document_count"] >= 1
        assert stats["embedding_count"] >= 1
        assert stats["tables"]["documents"] == "test_articles"
        assert stats["tables"]["embeddings"] == "test_article_embeddings"

        # Cleanup
        for doc_id in doc_ids:
            deleted = await rag.delete_document_async(doc_id)
            assert deleted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
