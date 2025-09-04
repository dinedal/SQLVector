import pytest
from typing import List
from sql_rag.embedding import (
    EmbeddingProvider,
    DefaultEmbeddingProvider,
    EmbeddingService,
)
from sql_rag.exceptions import EmbeddingError


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384, should_fail: bool = False):
        self.dimension = dimension
        self.should_fail = should_fail
        self.embed_call_count = 0
        self.embed_batch_call_count = 0

    async def embed(self, text: str) -> List[float]:
        self.embed_call_count += 1
        if self.should_fail:
            raise Exception("Mock embedding failure")

        # Return a simple deterministic embedding based on text length
        return [float(len(text) % 10)] * self.dimension

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.embed_batch_call_count += 1
        if self.should_fail:
            raise Exception("Mock batch embedding failure")

        return [[float(len(text) % 10)] * self.dimension for text in texts]


class TestDefaultEmbeddingProvider:
    @pytest.fixture
    def provider(self):
        return DefaultEmbeddingProvider(dimension=384)

    async def test_embed_single_text(self, provider):
        """Test embedding a single text."""
        text = "Hello, world!"
        embedding = await provider.embed(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
        assert all(-1 <= x <= 1 for x in embedding)

    async def test_embed_deterministic(self, provider):
        """Test that embedding is deterministic."""
        text = "Test text for determinism"
        embedding1 = await provider.embed(text)
        embedding2 = await provider.embed(text)

        assert embedding1 == embedding2

    async def test_embed_different_texts(self, provider):
        """Test that different texts produce different embeddings."""
        text1 = "First text"
        text2 = "Second text"

        embedding1 = await provider.embed(text1)
        embedding2 = await provider.embed(text2)

        assert embedding1 != embedding2

    async def test_embed_batch(self, provider):
        """Test batch embedding."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = await provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)

    async def test_embed_batch_consistency(self, provider):
        """Test that batch embedding is consistent with single embedding."""
        texts = ["Text one", "Text two"]

        # Get embeddings individually
        individual_embeddings = []
        for text in texts:
            embedding = await provider.embed(text)
            individual_embeddings.append(embedding)

        # Get embeddings in batch
        batch_embeddings = await provider.embed_batch(texts)

        assert individual_embeddings == batch_embeddings

    async def test_embed_empty_string(self, provider):
        """Test embedding an empty string."""
        embedding = await provider.embed("")

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    async def test_embed_batch_empty_list(self, provider):
        """Test batch embedding with empty list."""
        embeddings = await provider.embed_batch([])

        assert embeddings == []


class TestEmbeddingService:
    @pytest.fixture
    def mock_provider(self):
        return MockEmbeddingProvider(dimension=384)

    @pytest.fixture
    def service(self, mock_provider):
        return EmbeddingService(provider=mock_provider, dimension=384)

    async def test_create_embedding(self, service, mock_provider):
        """Test creating a single embedding."""
        text = "Test text"
        embedding = await service.create_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert mock_provider.embed_call_count == 1

    async def test_create_embeddings_batch(self, service, mock_provider):
        """Test creating batch embeddings."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await service.create_embeddings_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert mock_provider.embed_batch_call_count == 1

    async def test_create_embedding_failure(self):
        """Test embedding creation failure."""
        failing_provider = MockEmbeddingProvider(should_fail=True)
        service = EmbeddingService(provider=failing_provider, dimension=384)

        with pytest.raises(EmbeddingError, match="Failed to create embedding"):
            await service.create_embedding("Test text")

    async def test_create_embeddings_batch_failure(self):
        """Test batch embedding creation failure."""
        failing_provider = MockEmbeddingProvider(should_fail=True)
        service = EmbeddingService(provider=failing_provider, dimension=384)

        with pytest.raises(EmbeddingError, match="Failed to create batch embeddings"):
            await service.create_embeddings_batch(["Text 1", "Text 2"])

    def test_similarity_identical_vectors(self, service):
        """Test cosine similarity with identical vectors."""
        vec = [1.0, 2.0, 3.0]
        similarity = service.similarity(vec, vec)

        assert abs(similarity - 1.0) < 1e-10

    def test_similarity_orthogonal_vectors(self, service):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = service.similarity(vec1, vec2)

        assert abs(similarity - 0.0) < 1e-10

    def test_similarity_opposite_vectors(self, service):
        """Test cosine similarity with opposite vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = service.similarity(vec1, vec2)

        assert abs(similarity - (-1.0)) < 1e-10

    def test_similarity_zero_vector(self, service):
        """Test cosine similarity with zero vector."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        similarity = service.similarity(vec1, vec2)

        assert similarity == 0.0

    def test_similarity_different_dimensions(self, service):
        """Test cosine similarity with different dimension vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]

        with pytest.raises(
            EmbeddingError, match="Vectors must have the same dimension"
        ):
            service.similarity(vec1, vec2)

    def test_default_provider_creation(self):
        """Test service creation with default provider."""
        service = EmbeddingService(dimension=512)

        assert service.provider is not None
        assert isinstance(service.provider, DefaultEmbeddingProvider)
        assert service.provider.dimension == 512

    async def test_service_with_default_provider(self):
        """Test service functionality with default provider."""
        service = EmbeddingService(dimension=256)

        embedding = await service.create_embedding("Test text")
        assert len(embedding) == 256

        embeddings = await service.create_embeddings_batch(["Text 1", "Text 2"])
        assert len(embeddings) == 2
        assert all(len(emb) == 256 for emb in embeddings)
