import pytest
import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.pool import StaticPool

from sqlvector import SQLRAG, RAGConfig, EmbeddingService, DefaultEmbeddingProvider
from sqlvector.models import Base

# Import PostgreSQL test utilities
from .postgres_test_utils import (
    get_postgres_manager, 
    POSTGRES_DEPS_AVAILABLE, 
    POSTGRES_BACKEND_AVAILABLE
)

# Import PostgreSQL backend if available
try:
    from sqlvector.backends.postgres import PostgresRAG, PostgresConfig
    from sqlvector.embedding import DefaultEmbeddingProvider as PGEmbeddingProvider
except ImportError:
    PostgresRAG = None
    PostgresConfig = None
    PGEmbeddingProvider = None


@pytest.fixture
async def async_engine() -> AsyncEngine:
    """Create an in-memory SQLite async engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
def rag_config(async_engine: AsyncEngine) -> RAGConfig:
    """Create a RAG configuration for testing."""
    return RAGConfig(
        engine=async_engine,
        documents_table="documents",
        embeddings_table="embeddings",
        embedding_dimension=384
    )


@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Create an embedding service for testing."""
    return EmbeddingService(
        provider=DefaultEmbeddingProvider(dimension=384),
        dimension=384
    )


@pytest.fixture
async def rag_instance(async_engine: AsyncEngine) -> SQLRAG:
    """Create a SQLRAG instance for testing."""
    rag = SQLRAG(
        engine=async_engine,
        embedding_dimension=384
    )
    await rag.create_tables()
    return rag


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"category": "animals", "type": "example"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"category": "technology", "type": "definition"}
        },
        {
            "content": "Python is a popular programming language.",
            "metadata": {"category": "technology", "type": "fact"}
        }
    ]


# PostgreSQL-specific fixtures
@pytest.fixture(scope="session")
def postgres_manager():
    """Get the PostgreSQL test manager."""
    return get_postgres_manager()


@pytest.fixture(scope="session")
def postgres_env(postgres_manager):
    """Set up PostgreSQL test environment for the entire test session."""
    if not postgres_manager.is_available():
        pytest.skip(postgres_manager.get_skip_reason() or "PostgreSQL not available")
    
    # Container should already be started by collection phase
    if not postgres_manager._container_started:
        if not postgres_manager.start_container():
            pytest.skip("Failed to start PostgreSQL container")
    
    yield postgres_manager
    
    # Cleanup happens in pytest_sessionfinish hook


@pytest.fixture
def postgres_db_config(postgres_env):
    """Get PostgreSQL database configuration."""
    return postgres_env.get_db_config()


@pytest.fixture
def postgres_db_url(postgres_env):
    """Get PostgreSQL database URL."""
    return postgres_env.get_db_url()


@pytest.fixture
async def postgres_rag(postgres_db_url):
    """Create a PostgreSQL RAG instance for testing."""
    if not PostgresRAG:
        pytest.skip("PostgreSQL backend not available")
    
    rag = PostgresRAG(
        db_url=postgres_db_url,
        embedding_dimension=384,
        embedding_provider=PGEmbeddingProvider(384)
    )
    
    yield rag
    
    # Cleanup
    await rag.close()


@pytest.fixture
def postgres_config(postgres_db_url):
    """Create a PostgreSQL config for testing."""
    if not PostgresConfig:
        pytest.skip("PostgreSQL backend not available")
    
    return PostgresConfig(
        db_url=postgres_db_url,
        embedding_dimension=384
    )