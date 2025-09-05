import pytest
import json
from sqlvector.models import Document, Embedding


class TestDocument:
    def test_document_creation(self):
        """Test document creation with basic fields."""
        doc = Document(
            id="test-doc-1",
            content="Test content"
        )
        
        assert doc.id == "test-doc-1"
        assert doc.content == "Test content"
        assert doc.doc_metadata is None
        assert doc.created_at is None  # Will be set by database
        assert doc.updated_at is None  # Will be set by database
    
    def test_document_metadata_operations(self):
        """Test metadata getter and setter operations."""
        doc = Document(id="test-doc-1", content="Test content")
        
        # Test empty metadata
        assert doc.get_metadata() == {}
        
        # Test setting metadata
        test_metadata = {"key1": "value1", "key2": 42, "key3": True}
        doc.set_metadata(test_metadata)
        
        assert doc.doc_metadata is not None
        assert json.loads(doc.doc_metadata) == test_metadata
        assert doc.get_metadata() == test_metadata
    
    def test_document_metadata_with_existing_json(self):
        """Test document with pre-existing JSON metadata."""
        test_metadata = {"source": "test", "tags": ["tag1", "tag2"]}
        doc = Document(
            id="test-doc-1",
            content="Test content",
            doc_metadata=json.dumps(test_metadata)
        )
        
        assert doc.get_metadata() == test_metadata
    
    def test_document_metadata_none_handling(self):
        """Test metadata handling when metadata is None."""
        doc = Document(id="test-doc-1", content="Test content", doc_metadata=None)
        assert doc.get_metadata() == {}


class TestEmbedding:
    def test_embedding_creation(self):
        """Test embedding creation with basic fields."""
        embedding = Embedding(
            id="test-emb-1",
            document_id="test-doc-1",
            vector="[1.0, 2.0, 3.0]",
            model_name="test-model"
        )
        
        assert embedding.id == "test-emb-1"
        assert embedding.document_id == "test-doc-1"
        assert embedding.vector == "[1.0, 2.0, 3.0]"
        assert embedding.model_name == "test-model"
        assert embedding.created_at is None  # Will be set by database
    
    def test_embedding_vector_operations(self):
        """Test vector getter and setter operations."""
        embedding = Embedding(
            id="test-emb-1",
            document_id="test-doc-1",
            vector="[]"
        )
        
        # Test setting vector
        test_vector = [1.5, -2.3, 0.0, 4.7]
        embedding.set_vector(test_vector)
        
        assert embedding.vector is not None
        assert json.loads(embedding.vector) == test_vector
        assert embedding.get_vector() == test_vector
    
    def test_embedding_vector_with_existing_json(self):
        """Test embedding with pre-existing JSON vector."""
        test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding = Embedding(
            id="test-emb-1",
            document_id="test-doc-1",
            vector=json.dumps(test_vector)
        )
        
        assert embedding.get_vector() == test_vector
    
    def test_embedding_vector_large_dimension(self):
        """Test embedding with large dimension vector."""
        large_vector = [float(i) for i in range(1000)]
        embedding = Embedding(
            id="test-emb-1",
            document_id="test-doc-1",
            vector="[]"
        )
        
        embedding.set_vector(large_vector)
        retrieved_vector = embedding.get_vector()
        
        assert len(retrieved_vector) == 1000
        assert retrieved_vector == large_vector
    
    def test_embedding_vector_empty(self):
        """Test embedding with empty vector."""
        embedding = Embedding(
            id="test-emb-1",
            document_id="test-doc-1",
            vector="[]"
        )
        
        empty_vector = []
        embedding.set_vector(empty_vector)
        
        assert embedding.get_vector() == []