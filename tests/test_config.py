import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.pool import StaticPool

from sqlvector.config import RAGConfig
from sqlvector.exceptions import ConfigurationError


class TestRAGConfig:
    @pytest.fixture
    def async_engine(self) -> AsyncEngine:
        """Create a test async engine."""
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
    
    def test_config_creation_valid(self, async_engine):
        """Test creating a valid configuration."""
        config = RAGConfig(
            engine=async_engine,
            documents_table="docs",
            embeddings_table="embeddings",
            embedding_dimension=512
        )
        
        assert config.engine == async_engine
        assert config.documents_table == "docs"
        assert config.embeddings_table == "embeddings"
        assert config.embedding_dimension == 512
    
    def test_config_creation_defaults(self, async_engine):
        """Test creating configuration with default values."""
        config = RAGConfig(
            engine=async_engine,
            documents_table="documents",
            embeddings_table="embeddings"
        )
        
        assert config.engine == async_engine
        assert config.documents_table == "documents"
        assert config.embeddings_table == "embeddings"
        assert config.embedding_dimension == 768
    
    def test_config_empty_documents_table(self, async_engine):
        """Test configuration with empty documents table name."""
        with pytest.raises(ConfigurationError, match="documents_table is required"):
            RAGConfig(
                engine=async_engine,
                documents_table="",
                embeddings_table="embeddings"
            )
    
    def test_config_empty_embeddings_table(self, async_engine):
        """Test configuration with empty embeddings table name."""
        with pytest.raises(ConfigurationError, match="embeddings_table is required"):
            RAGConfig(
                engine=async_engine,
                documents_table="documents",
                embeddings_table=""
            )
    
    def test_config_none_documents_table(self, async_engine):
        """Test configuration with None documents table name."""
        with pytest.raises(ConfigurationError, match="documents_table is required"):
            RAGConfig(
                engine=async_engine,
                documents_table=None,
                embeddings_table="embeddings"
            )
    
    def test_config_none_embeddings_table(self, async_engine):
        """Test configuration with None embeddings table name."""
        with pytest.raises(ConfigurationError, match="embeddings_table is required"):
            RAGConfig(
                engine=async_engine,
                documents_table="documents",
                embeddings_table=None
            )
    
    def test_config_zero_embedding_dimension(self, async_engine):
        """Test configuration with zero embedding dimension."""
        with pytest.raises(ConfigurationError, match="embedding_dimension must be positive"):
            RAGConfig(
                engine=async_engine,
                documents_table="documents",
                embeddings_table="embeddings",
                embedding_dimension=0
            )
    
    def test_config_negative_embedding_dimension(self, async_engine):
        """Test configuration with negative embedding dimension."""
        with pytest.raises(ConfigurationError, match="embedding_dimension must be positive"):
            RAGConfig(
                engine=async_engine,
                documents_table="documents",
                embeddings_table="embeddings",
                embedding_dimension=-10
            )
    
    async def test_get_session(self, async_engine):
        """Test getting an async session."""
        config = RAGConfig(
            engine=async_engine,
            documents_table="documents",
            embeddings_table="embeddings"
        )
        
        session = config.get_session()
        assert session is not None
        
        # Test that session is usable
        await session.close()
    
    def test_config_special_table_names(self, async_engine):
        """Test configuration with special characters in table names."""
        config = RAGConfig(
            engine=async_engine,
            documents_table="my_documents_table",
            embeddings_table="embeddings_v2",
            embedding_dimension=1024
        )
        
        assert config.documents_table == "my_documents_table"
        assert config.embeddings_table == "embeddings_v2"
        assert config.embedding_dimension == 1024
    
    def test_config_large_embedding_dimension(self, async_engine):
        """Test configuration with large embedding dimension."""
        config = RAGConfig(
            engine=async_engine,
            documents_table="documents",
            embeddings_table="embeddings",
            embedding_dimension=4096
        )
        
        assert config.embedding_dimension == 4096
    
    def test_config_column_name_settings(self, async_engine):
        """Test configuration with custom column name settings."""
        config = RAGConfig(
            engine=async_engine,
            documents_table="documents",
            embeddings_table="embeddings",
            documents_id_column="custom_doc_id",
            embeddings_id_column="custom_emb_id",
            embeddings_document_id_column="custom_doc_ref"
        )
        
        assert config.documents_id_column == "custom_doc_id"
        assert config.embeddings_id_column == "custom_emb_id"
        assert config.embeddings_document_id_column == "custom_doc_ref"