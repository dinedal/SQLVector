"""DuckDB-specific models and data structures."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json


@dataclass
class DuckDBDocument:
    """Document model for DuckDB backend."""
    
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DuckDB insertion."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": json.dumps(self.metadata or {}),
        }
    
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DuckDBDocument":
        """Create from dictionary."""
        metadata = None
        if data.get("metadata"):
            metadata = json.loads(data["metadata"]) if isinstance(data["metadata"], str) else data["metadata"]
        
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=metadata
        )


@dataclass
class DuckDBEmbedding:
    """Embedding model for DuckDB backend."""
    
    id: str
    document_id: str
    embedding: List[float]
    model_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DuckDB insertion."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "embedding": self.embedding,
            "model_name": self.model_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DuckDBEmbedding":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            embedding=data["embedding"],
            model_name=data.get("model_name")
        )


# DuckDBQueryResult class removed - queries now return raw dictionaries