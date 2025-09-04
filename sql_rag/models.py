from typing import Any, Dict, Optional, List
from sqlalchemy import Column, String, Text, DateTime, Integer, Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
import json

Base = declarative_base()


class Document(Base):
    """SQLAlchemy model for documents."""
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get document metadata as dictionary."""
        if self.doc_metadata:
            return json.loads(self.doc_metadata)
        return {}
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set document metadata from dictionary."""
        self.doc_metadata = json.dumps(metadata)


class Embedding(Base):
    """SQLAlchemy model for embeddings."""
    __tablename__ = 'embeddings'
    
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    vector = Column(Text, nullable=False)  # JSON string for compatibility
    model_name = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    
    def get_vector(self) -> List[float]:
        """Get embedding vector as list of floats."""
        return json.loads(self.vector)
    
    def set_vector(self, vector: List[float]) -> None:
        """Set embedding vector from list of floats."""
        self.vector = json.dumps(vector)