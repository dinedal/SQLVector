"""PostgreSQL-specific models and data structures."""

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class PostgresDocument:
    """Document model for PostgreSQL backend."""
    
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PostgreSQL insertion."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": json.dumps(self.metadata or {}),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostgresDocument":
        """Create from dictionary."""
        metadata = None
        if data.get("metadata"):
            if isinstance(data["metadata"], str):
                metadata = json.loads(data["metadata"])
            elif isinstance(data["metadata"], dict):
                metadata = data["metadata"]
            else:
                # Handle PostgreSQL JSONB type which might be returned as dict
                metadata = dict(data["metadata"])
        
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=metadata
        )


@dataclass
class PostgresEmbedding:
    """Embedding model for PostgreSQL backend."""
    
    id: str
    document_id: str
    embedding: List[float]
    model_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PostgreSQL insertion."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "embedding": self.format_vector(self.embedding),
            "model_name": self.model_name
        }
    
    @staticmethod
    def format_vector(embedding: List[float]) -> str:
        """Format embedding as PostgreSQL vector string."""
        # PostgreSQL pgvector expects format: '[1,2,3]'
        return f"[{','.join(map(str, embedding))}]"
    
    @staticmethod
    def parse_vector(vector_data: Any) -> List[float]:
        """Parse vector data from PostgreSQL."""
        if isinstance(vector_data, list):
            return vector_data
        elif isinstance(vector_data, str):
            # Parse string format '[1,2,3]'
            vector_str = vector_data.strip('[]')
            return [float(x) for x in vector_str.split(',')]
        else:
            # Handle pgvector native type or numpy array
            return list(vector_data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostgresEmbedding":
        """Create from dictionary."""
        embedding = cls.parse_vector(data["embedding"])
        
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            embedding=embedding,
            model_name=data.get("model_name")
        )


@dataclass
class PostgresQueryResult:
    """Query result model for PostgreSQL backend."""
    
    id: str
    content: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "content": self.content,
            "similarity": self.similarity,
        }
        
        if self.metadata:
            # Return metadata as JSON string for consistency with other backends
            result["metadata"] = json.dumps(self.metadata)
        
        if self.embedding:
            result["embedding"] = self.embedding
        
        return result
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "PostgresQueryResult":
        """Create from database row."""
        metadata = None
        if row.get("metadata"):
            if isinstance(row["metadata"], str):
                metadata = json.loads(row["metadata"])
            else:
                metadata = dict(row["metadata"])
        
        embedding = None
        if row.get("embedding"):
            embedding = PostgresEmbedding.parse_vector(row["embedding"])
        
        return cls(
            id=row["id"],
            content=row["content"],
            similarity=float(row["similarity"]),
            metadata=metadata,
            embedding=embedding
        )