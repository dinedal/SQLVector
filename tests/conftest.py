import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.pool import StaticPool

from sql_rag import SQLRAG, RAGConfig, EmbeddingService, DefaultEmbeddingProvider
from sql_rag.models import Base


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