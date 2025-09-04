from typing import List, Union, Dict, Any, Optional
import asyncio

from .protocols import EmbeddingProvider
from .exceptions import EmbeddingError
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


class DefaultEmbeddingProvider(EmbeddingProvider):
    """Default embedding provider using deterministic hash-based embeddings."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    async def embed(self, text: str) -> List[float]:
        import hashlib
        import random

        # Log warning for placeholder embedding function
        logger.warning(
            "Using placeholder embedding function (DefaultEmbeddingProvider.embed). "
            "For production use, please provide a real embedding model."
        )

        # Simple deterministic embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        random.seed(seed)

        return [random.uniform(-1, 1) for _ in range(self.dimension)]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Log warning for placeholder embedding function (only once for batch)
        logger.warning(
            "Using placeholder embedding function (DefaultEmbeddingProvider.embed_batch). "
            "For production use, please provide a real embedding model."
        )
        
        # Don't call self.embed to avoid duplicate warnings for each text
        import hashlib
        import random
        
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            random.seed(seed)
            embeddings.append([random.uniform(-1, 1) for _ in range(self.dimension)])
        
        return embeddings


class DefaultSyncEmbeddingProvider:
    """Default synchronous embedding provider using deterministic hash-based embeddings."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        import hashlib
        import random

        # Log warning for placeholder embedding function
        logger.warning(
            "Using placeholder embedding function (DefaultSyncEmbeddingProvider.embed). "
            "For production use, please provide a real embedding model."
        )

        # Simple deterministic embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        random.seed(seed)

        return [random.uniform(-1, 1) for _ in range(self.dimension)]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Log warning for placeholder embedding function (only once for batch)
        logger.warning(
            "Using placeholder embedding function (DefaultSyncEmbeddingProvider.embed_batch). "
            "For production use, please provide a real embedding model."
        )
        
        # Don't call self.embed to avoid duplicate warnings for each text
        import hashlib
        import random
        
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            random.seed(seed)
            embeddings.append([random.uniform(-1, 1) for _ in range(self.dimension)])
        
        return embeddings


class EmbeddingService:
    """Service for managing embeddings with pluggable providers."""

    def __init__(
        self, provider: Optional[EmbeddingProvider] = None, dimension: int = 768
    ) -> None:
        self.provider = provider or DefaultEmbeddingProvider(dimension)

    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        try:
            return await self.provider.embed(text)
        except Exception as e:
            raise EmbeddingError(f"Failed to create embedding: {e}")

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        try:
            return await self.provider.embed_batch(texts)
        except Exception as e:
            raise EmbeddingError(f"Failed to create batch embeddings: {e}")

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise EmbeddingError("Vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class SyncEmbeddingService:
    """Synchronous service for managing embeddings with pluggable providers."""

    def __init__(self, provider: Optional[Any] = None, dimension: int = 768) -> None:
        self.provider = provider or DefaultSyncEmbeddingProvider(dimension)

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        try:
            result = self.provider.embed(text)
            # If the provider returns a coroutine (async method), run it
            if asyncio.iscoroutine(result):
                return asyncio.run(result)
            return result
        except Exception as e:
            raise EmbeddingError(f"Failed to create embedding: {e}")

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        try:
            result = self.provider.embed_batch(texts)
            # If the provider returns a coroutine (async method), run it
            if asyncio.iscoroutine(result):
                return asyncio.run(result)
            return result
        except Exception as e:
            raise EmbeddingError(f"Failed to create batch embeddings: {e}")

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise EmbeddingError("Vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
