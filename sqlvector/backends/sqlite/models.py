"""SQLite-specific models and data structures."""

import json
import struct
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class SQLiteDocument:
    """Document model for SQLite backend."""
    
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQLite insertion."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": json.dumps(self.metadata or {}),
            "hash": self.hash or self._compute_hash()
        }
    
    def _compute_hash(self) -> str:
        """Compute a hash of the content for deduplication."""
        import hashlib
        return hashlib.md5(self.content.encode()).hexdigest()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLiteDocument":
        """Create from dictionary."""
        metadata = None
        if data.get("metadata"):
            metadata = json.loads(data["metadata"]) if isinstance(data["metadata"], str) else data["metadata"]
        
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=metadata,
            hash=data.get("hash")
        )


@dataclass
class SQLiteEmbedding:
    """Embedding model for SQLite backend."""
    
    id: str
    document_id: str
    hash: str
    embedding: List[float]
    model_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQLite insertion."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "hash": self.hash,
            "embedding": self._serialize_embedding(self.embedding),
            "model_name": self.model_name
        }
    
    @staticmethod
    def _serialize_embedding(embedding: List[float]) -> bytes:
        """Serialize embedding as binary data for efficient storage."""
        # Pack as array of 32-bit floats
        return struct.pack(f'{len(embedding)}f', *embedding)
    
    @staticmethod
    def _deserialize_embedding(data: bytes) -> List[float]:
        """Deserialize embedding from binary data."""
        # Unpack array of 32-bit floats
        num_floats = len(data) // 4
        return list(struct.unpack(f'{num_floats}f', data))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLiteEmbedding":
        """Create from dictionary."""
        embedding = data["embedding"]
        if isinstance(embedding, bytes):
            embedding = cls._deserialize_embedding(embedding)
        
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            hash=data["hash"],
            embedding=embedding,
            model_name=data.get("model_name")
        )


# SQLiteQueryResult class removed - queries now return raw dictionaries