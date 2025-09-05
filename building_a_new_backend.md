# Building a New Backend for SQL RAG

This guide explains how to implement a new database backend for the SQL RAG library. The library's architecture is designed to be extensible, allowing you to add support for any database that can store vectors and perform similarity operations.

## Overview

Each backend in SQL RAG consists of five main components:

1. **Config** - Database configuration and connection management
2. **Models** - Data structures for documents and embeddings
3. **Loader** - Document loading and embedding generation
4. **Querier** - Vector similarity search implementation
5. **RAG** - High-level interface combining loader and querier

## Architecture

### Directory Structure

Create a new directory under `sqlvector/backends/` with your backend name:

```
sqlvector/backends/yourbackend/
├── __init__.py       # Exports main classes
├── config.py         # Configuration class
├── models.py         # Data models
├── loader.py         # Document loader
├── querier.py        # Query interface
└── rag.py           # Main RAG interface
```

### Protocol Requirements

SQL RAG uses Python protocols to define interfaces. Your backend should implement these protocols from `sqlvector/protocols.py`:

- `DatabaseConfigProtocol` - Configuration interface
- `DocumentLoaderProtocol` - Document loading operations
- `DocumentQuerierProtocol` - Query operations
- `RAGSystemProtocol` - Complete RAG system interface

## Step-by-Step Implementation Guide

### Step 1: Create the Config Class

The config class manages database connections and table schemas.

```python
# config.py
from dataclasses import dataclass
from typing import Optional, Any, Union
from contextlib import contextmanager

@dataclass
class YourBackendConfig:
    """Configuration for YourBackend RAG backend."""
    
    # Required attributes
    db_path: str
    documents_table: str = "documents"
    embeddings_table: str = "embeddings"
    embedding_dimension: int = 768
    batch_size: int = 1000
    
    # Optional SQLAlchemy support
    engine: Optional[Any] = None
    use_sqlalchemy: bool = False
    
    # Column name mappings for custom schemas
    documents_id_column: str = "id"
    documents_content_column: str = "content"
    documents_metadata_column: Optional[str] = "metadata"
    embeddings_id_column: str = "id"
    embeddings_document_id_column: str = "document_id"
    embeddings_model_column: Optional[str] = "model_name"
    embeddings_column: str = "embedding"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
    def get_connection(self) -> Any:
        """Get a database connection."""
        # Implement connection logic
        pass
        
    def get_documents_schema(self) -> str:
        """Get CREATE TABLE SQL for documents."""
        metadata_col = (
            f"{self.documents_metadata_column} TEXT,"
            if self.documents_metadata_column
            else ""
        )
        
        return f"""
        CREATE TABLE IF NOT EXISTS {self.documents_table} (
            {self.documents_id_column} TEXT PRIMARY KEY,
            {self.documents_content_column} TEXT NOT NULL,
            {metadata_col}
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
    def get_embeddings_schema(self) -> str:
        """Get CREATE TABLE SQL for embeddings."""
        # Define embedding storage based on your database's vector type
        pass
        
    def setup_database(self, conn: Any) -> None:
        """Set up database schema and functions."""
        # Check if tables exist before creating
        # Create tables if needed
        # Create indexes for performance
        pass
        
    @contextmanager
    def get_connection_context(self):
        """Get a connection with proper context management."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            # Handle connection cleanup
            if not self.use_sqlalchemy:
                conn.close()
```

### Step 2: Define Data Models

Create models for documents and embeddings:

```python
# models.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json

@dataclass
class YourBackendDocument:
    """Document model for YourBackend."""
    
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": json.dumps(self.metadata or {}),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "YourBackendDocument":
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
class YourBackendEmbedding:
    """Embedding model for YourBackend."""
    
    id: str
    document_id: str
    embedding: List[float]
    model_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "embedding": self._serialize_embedding(self.embedding),
            "model_name": self.model_name
        }
    
    def _serialize_embedding(self, embedding: List[float]) -> Any:
        """Serialize embedding for storage."""
        # Implement based on your database's vector storage format
        # Options: binary blob, JSON array, native vector type
        return embedding
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "YourBackendEmbedding":
        """Create from dictionary."""
        embedding = cls._deserialize_embedding(data["embedding"])
        
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            embedding=embedding,
            model_name=data.get("model_name")
        )
    
    @staticmethod
    def _deserialize_embedding(data: Any) -> List[float]:
        """Deserialize embedding from storage."""
        # Implement based on your storage format
        return data
```

### Step 3: Implement the Loader

The loader handles document insertion and embedding generation:

```python
# loader.py
import uuid
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ...embedding import EmbeddingService
from ...exceptions import LoaderError
from ...logger import get_logger
from .config import YourBackendConfig
from .models import YourBackendDocument, YourBackendEmbedding

logger = get_logger(__name__)

class YourBackendLoader:
    """Document loader for YourBackend."""
    
    def __init__(self, config: YourBackendConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
    
    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True
    ) -> str:
        """Load a single document."""
        return self.load_documents_batch([{
            "content": content,
            "metadata": metadata,
            "document_id": document_id
        }], generate_embeddings=generate_embedding)[0]
    
    def load_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True
    ) -> List[str]:
        """Load multiple documents efficiently."""
        try:
            with self.config.get_connection_context() as conn:
                # Prepare documents
                doc_records = []
                for doc_data in documents:
                    doc_id = doc_data.get("document_id") or str(uuid.uuid4())
                    doc = YourBackendDocument(
                        id=doc_id,
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata")
                    )
                    doc_records.append(doc.to_dict())
                
                # Insert documents using batch insert
                self._insert_documents(conn, doc_records)
                
                document_ids = [doc["id"] for doc in doc_records]
                
                if generate_embeddings:
                    self._generate_embeddings_batch(
                        conn, 
                        doc_records, 
                        show_progress=show_progress
                    )
                
                return document_ids
                
        except Exception as e:
            raise LoaderError(f"Failed to load documents batch: {e}")
    
    def _insert_documents(self, conn: Any, doc_records: List[Dict[str, Any]]) -> None:
        """Insert documents into database."""
        # Implement batch insert using your database's syntax
        pass
    
    def _generate_embeddings_batch(
        self,
        conn: Any,
        doc_records: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> None:
        """Generate embeddings in batches."""
        logger.info(f"Generating embeddings for {len(doc_records)} documents")
        
        # Process in batches
        total_docs = len(doc_records)
        batch_size = self.config.batch_size
        
        progress_bar = tqdm(total=total_docs, desc="Generating embeddings") if show_progress else None
        
        try:
            for i in range(0, total_docs, batch_size):
                batch_docs = doc_records[i:i + batch_size]
                
                # Extract content for embedding
                contents = [doc["content"] for doc in batch_docs]
                
                # Generate embeddings
                import asyncio
                embeddings = asyncio.run(
                    self.embedding_service.create_embeddings_batch(contents)
                )
                
                # Prepare embedding records
                embedding_records = []
                for doc, embedding in zip(batch_docs, embeddings):
                    embedding_id = str(uuid.uuid4())
                    emb_record = {
                        "id": embedding_id,
                        "document_id": doc["id"],
                        "embedding": YourBackendEmbedding._serialize_embedding(embedding),
                    }
                    if self.config.embeddings_model_column:
                        emb_record["model_name"] = getattr(
                            self.embedding_service.provider, 'model_name', 'default'
                        )
                    embedding_records.append(emb_record)
                
                # Insert embeddings
                self._insert_embeddings(conn, embedding_records)
                
                if progress_bar:
                    progress_bar.update(len(batch_docs))
                    
        finally:
            if progress_bar:
                progress_bar.close()
    
    def _insert_embeddings(self, conn: Any, embedding_records: List[Dict[str, Any]]) -> None:
        """Insert embeddings into database."""
        # Implement batch insert for embeddings
        pass
    
    def get_document(self, document_id: str) -> Any:
        """Get a single document by ID."""
        with self.config.get_connection_context() as conn:
            # Query document
            pass
    
    def get_documents_batch(self, document_ids: List[str]) -> List[Any]:
        """Get multiple documents by IDs."""
        with self.config.get_connection_context() as conn:
            # Query documents
            pass
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        with self.config.get_connection_context() as conn:
            # Delete document and embeddings
            pass
    
    def create_index(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        **kwargs
    ) -> bool:
        """Create an index on embeddings for accelerated similarity search."""
        # Implement if your database supports vector indexes
        pass
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an existing index."""
        # Implement if your database supports vector indexes
        pass
```

### Step 4: Implement the Querier

The querier handles similarity search operations:

```python
# querier.py
from typing import List, Dict, Any, Optional

from ...embedding import EmbeddingService
from ...exceptions import QueryError
from ...logger import get_logger
from .config import YourBackendConfig
from .models import YourBackendEmbedding

logger = get_logger(__name__)

class YourBackendQuerier:
    """Vector similarity querier for YourBackend."""
    
    def __init__(self, config: YourBackendConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
    
    def setup_similarity_functions(self, conn: Any) -> None:
        """Set up vector similarity functions in the database."""
        # Implement if your database needs custom functions
        # Examples: cosine similarity, euclidean distance, inner product
        pass
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text."""
        try:
            with self.config.get_connection_context() as conn:
                # Generate query embedding
                import asyncio
                query_embedding = asyncio.run(
                    self.embedding_service.create_embedding(query_text)
                )
                
                return self.query_with_precomputed_embedding(
                    query_embedding,
                    top_k,
                    similarity_threshold,
                    similarity_function,
                    **kwargs
                )
                
        except Exception as e:
            raise QueryError(f"Query failed: {e}")
    
    def query_with_precomputed_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector."""
        try:
            with self.config.get_connection_context() as conn:
                # Build and execute similarity query
                results = self._execute_similarity_query(
                    conn,
                    query_embedding,
                    top_k,
                    similarity_threshold,
                    similarity_function
                )
                
                return self._format_results(results)
                
        except Exception as e:
            raise QueryError(f"Query with embedding failed: {e}")
    
    def _execute_similarity_query(
        self,
        conn: Any,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float,
        similarity_function: str
    ) -> List[Any]:
        """Execute the similarity query."""
        # Implement based on your database's vector operations
        # Options:
        # 1. Use native vector similarity functions
        # 2. Use a vector index (if available)
        # 3. Implement similarity calculation in SQL
        pass
    
    def _format_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Format query results."""
        formatted = []
        for row in results:
            formatted.append({
                "document_id": row["document_id"],
                "content": row["content"],
                "similarity": row["similarity"],
                "metadata": json.loads(row.get("metadata", "{}"))
            })
        return formatted
    
    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        results = []
        for query_text in query_texts:
            results.append(
                self.query(
                    query_text,
                    top_k,
                    similarity_threshold,
                    similarity_function,
                    **kwargs
                )
            )
        return results
    
    def query_by_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters and optional similarity search."""
        # Implement metadata filtering with optional vector search
        pass
    
    def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        # Get document's embedding and find similar
        pass
```

### Step 5: Create the Main RAG Interface

Combine loader and querier into a high-level interface:

```python
# rag.py
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ...embedding import EmbeddingService, EmbeddingProvider, DefaultEmbeddingProvider
from .config import YourBackendConfig
from .loader import YourBackendLoader
from .querier import YourBackendQuerier
from .models import YourBackendDocument, YourBackendEmbedding

class YourBackendRAG:
    """High-performance RAG system using YourBackend.
    
    Features:
    - Efficient batch operations
    - Vector similarity search
    - Metadata filtering
    - [Add your backend-specific features]
    """
    
    def __init__(
        self,
        db_path: Union[str, Path] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_dimension: int = 768,
        batch_size: int = 1000,
        documents_table: str = "documents",
        embeddings_table: str = "embeddings",
        engine: Optional[Any] = None,
        use_sqlalchemy: bool = False,
        # Column name mappings for custom schemas
        documents_id_column: str = "id",
        documents_content_column: str = "content",
        documents_metadata_column: Optional[str] = "metadata",
        embeddings_id_column: str = "id",
        embeddings_document_id_column: str = "document_id",
        embeddings_model_column: Optional[str] = "model_name",
        embeddings_column: str = "embedding",
    ):
        """Initialize YourBackend RAG system."""
        # Validate parameters
        if not use_sqlalchemy and db_path is None:
            raise ValueError("db_path is required when not using SQLAlchemy")
        
        self.config = YourBackendConfig(
            db_path=str(db_path) if db_path else ":memory:",
            documents_table=documents_table,
            embeddings_table=embeddings_table,
            embedding_dimension=embedding_dimension,
            batch_size=batch_size,
            engine=engine,
            use_sqlalchemy=use_sqlalchemy,
            documents_id_column=documents_id_column,
            documents_content_column=documents_content_column,
            documents_metadata_column=documents_metadata_column,
            embeddings_id_column=embeddings_id_column,
            embeddings_document_id_column=embeddings_document_id_column,
            embeddings_model_column=embeddings_model_column,
            embeddings_column=embeddings_column,
        )

        self.embedding_service = EmbeddingService(
            provider=embedding_provider or DefaultEmbeddingProvider(embedding_dimension),
            dimension=embedding_dimension,
        )

        self.loader = YourBackendLoader(self.config, self.embedding_service)
        self.querier = YourBackendQuerier(self.config, self.embedding_service)

        # Initialize database schema
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize the database schema."""
        with self.config.get_connection_context() as conn:
            self.config.setup_database(conn)

    # Document Loading Methods
    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True,
    ) -> str:
        """Load a single document."""
        return self.loader.load_document(
            content=content,
            metadata=metadata,
            document_id=document_id,
            generate_embedding=generate_embedding,
        )

    def load_documents(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """Load multiple documents efficiently."""
        return self.loader.load_documents_batch(
            documents=documents,
            generate_embeddings=generate_embeddings,
            show_progress=show_progress,
        )

    # Document Retrieval Methods
    def get_document(self, document_id: str) -> Any:
        """Get a single document by ID."""
        return self.loader.get_document(document_id)

    def get_documents(self, document_ids: List[str]) -> List[Any]:
        """Get multiple documents by IDs."""
        return self.loader.get_documents_batch(document_ids)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        return self.loader.delete_document(document_id)

    # Index Management Methods
    def create_index(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        **kwargs
    ) -> bool:
        """Create an index on embeddings."""
        return self.loader.create_index(
            index_name=index_name,
            similarity_function=similarity_function,
            **kwargs
        )

    def delete_index(self, index_name: str) -> bool:
        """Delete an existing index."""
        return self.loader.delete_index(index_name)

    # Query Methods
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents by similarity to query text."""
        return self.querier.query(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    def query_with_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query using a precomputed embedding vector."""
        return self.querier.query_with_precomputed_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Query multiple texts in batch."""
        return self.querier.query_batch(
            query_texts=query_texts,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    def query_with_filters(
        self,
        filters: Dict[str, Any],
        query_text: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query documents with metadata filters."""
        return self.querier.query_by_filters(
            filters=filters,
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        similarity_function: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        return self.querier.get_similar_documents(
            document_id=document_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            similarity_function=similarity_function,
            **kwargs
        )

    # Analytics and Statistics Methods
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.config.get_connection_context() as conn:
            # Implement statistics gathering
            pass
```

### Step 6: Create the __init__.py File

Export your backend classes:

```python
# __init__.py
"""YourBackend backend for SQL RAG."""

from .config import YourBackendConfig
from .models import YourBackendDocument, YourBackendEmbedding
from .loader import YourBackendLoader
from .querier import YourBackendQuerier
from .rag import YourBackendRAG

__all__ = [
    "YourBackendConfig",
    "YourBackendDocument", 
    "YourBackendEmbedding",
    "YourBackendLoader",
    "YourBackendQuerier", 
    "YourBackendRAG"
]
```

## Key Implementation Considerations

### 1. Connection Management

- Support both native database connections and SQLAlchemy
- Use connection context managers for proper resource cleanup
- Handle in-memory databases specially (maintain connection)
- Consider connection pooling for performance

### 2. Table Schema Flexibility

- Check if tables exist before creating them
- Support custom column names via configuration
- Handle existing tables with different schemas gracefully
- Use "CREATE TABLE IF NOT EXISTS" statements

### 3. Embedding Storage Strategies

Choose based on your database capabilities:

**Binary Storage (Space-efficient):**
```python
# SQLite example
import struct

def _serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding as binary data."""
    return struct.pack(f'{len(embedding)}f', *embedding)

def _deserialize_embedding(data: bytes) -> List[float]:
    """Deserialize embedding from binary data."""
    num_floats = len(data) // 4
    return list(struct.unpack(f'{num_floats}f', data))
```

**Native Vector Type (If available):**
```python
# DuckDB/PostgreSQL pgvector example
def get_embeddings_schema(self) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
        id VARCHAR PRIMARY KEY,
        embedding FLOAT[{self.embedding_dimension}] NOT NULL
    );
    """
```

**JSON Storage (Simple but less efficient):**
```python
import json

def _serialize_embedding(embedding: List[float]) -> str:
    """Serialize embedding as JSON."""
    return json.dumps(embedding)

def _deserialize_embedding(data: str) -> List[float]:
    """Deserialize embedding from JSON."""
    return json.loads(data)
```

### 4. Similarity Functions

Implement at least cosine similarity:

```python
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
```

Additional functions to consider:
- Euclidean distance
- Inner product
- Manhattan distance

### 5. Index Management

If your database supports vector indexes:

```python
def create_index(self, index_name: str, similarity_function: str = "cosine", **kwargs) -> bool:
    """Create an index on embeddings."""
    with self.config.get_connection_context() as conn:
        # Example for databases with HNSW support
        metric_map = {
            "cosine": "cosine",
            "euclidean": "l2",
            "inner_product": "ip"
        }
        
        metric = metric_map.get(similarity_function, "cosine")
        conn.execute(f"""
            CREATE INDEX {index_name} ON {self.embeddings_table}
            USING hnsw ({self.embeddings_column} {metric}_ops)
        """)
        return True
```

### 6. Batch Operations

Optimize for bulk inserts:

```python
def _insert_documents_batch(self, conn, doc_records):
    """Insert multiple documents efficiently."""
    # Use executemany for better performance
    conn.executemany(
        f"INSERT INTO {self.documents_table} VALUES (?, ?, ?, ?)",
        doc_records
    )
    
    # Or use COPY/bulk insert features if available
    # Example for PostgreSQL: COPY FROM
    # Example for DuckDB: INSERT INTO ... SELECT FROM VALUES
```

### 7. Error Handling

Handle database-specific errors gracefully:

```python
try:
    # Database operation
    pass
except YourDatabaseError as e:
    if "table already exists" in str(e):
        logger.warning(f"Table already exists: {e}")
    else:
        raise LoaderError(f"Database error: {e}")
```

## Testing Your Backend

### 1. Create Test Files

Create `tests/test_yourbackend.py`:

```python
import pytest
import tempfile
from pathlib import Path

from sqlvector.backends.yourbackend import YourBackendRAG

@pytest.fixture
def rag_instance():
    """Create a RAG instance for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        rag = YourBackendRAG(
            db_path=f.name,
            embedding_dimension=384
        )
        yield rag

def test_load_document(rag_instance):
    """Test loading a single document."""
    doc_id = rag_instance.load_document(
        content="Test document",
        metadata={"test": True}
    )
    assert doc_id is not None
    
    # Verify document was loaded
    doc = rag_instance.get_document(doc_id)
    assert doc["content"] == "Test document"

def test_query(rag_instance):
    """Test similarity query."""
    # Load test documents
    docs = [
        {"content": "Machine learning algorithms"},
        {"content": "Deep learning neural networks"},
        {"content": "Python programming language"}
    ]
    rag_instance.load_documents(docs)
    
    # Query
    results = rag_instance.query("artificial intelligence", top_k=2)
    assert len(results) <= 2
    assert all("similarity" in r for r in results)

def test_batch_operations(rag_instance):
    """Test batch loading and querying."""
    # Load batch
    docs = [{"content": f"Document {i}"} for i in range(100)]
    doc_ids = rag_instance.load_documents(docs, show_progress=False)
    assert len(doc_ids) == 100
    
    # Query batch
    queries = ["Document 1", "Document 50", "Document 99"]
    results = rag_instance.query_batch(queries, top_k=1)
    assert len(results) == 3
```

### 2. Integration Tests

Test integration with the main SQL RAG interfaces:

```python
def test_protocol_compliance():
    """Verify backend implements required protocols."""
    from sqlvector.protocols import RAGSystemProtocol
    
    rag = YourBackendRAG(db_path=":memory:")
    assert isinstance(rag, RAGSystemProtocol)
```

## Example Usage

Create an example file `examples/example_yourbackend.py`:

```python
"""Example usage of YourBackend RAG backend."""

from sqlvector.backends.yourbackend import YourBackendRAG

def main():
    print("YourBackend RAG Example")
    print("=" * 60)
    
    # Initialize RAG
    rag = YourBackendRAG(
        db_path="example.db",
        embedding_dimension=384,
        batch_size=500
    )
    
    # Load documents
    documents = [
        {"content": "Machine learning transforms AI", "metadata": {"topic": "ML"}},
        {"content": "Python is great for data science", "metadata": {"topic": "Python"}},
        {"content": "Neural networks enable deep learning", "metadata": {"topic": "DL"}},
    ]
    
    doc_ids = rag.load_documents(documents, show_progress=True)
    print(f"Loaded {len(doc_ids)} documents")
    
    # Query
    results = rag.query("artificial intelligence", top_k=2)
    
    print("\nQuery Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['content'][:50]}... (similarity: {result['similarity']:.3f})")
    
    # Query with filters
    filtered_results = rag.query_with_filters(
        filters={"topic": "ML"},
        query_text="learning algorithms"
    )
    
    print(f"\nFiltered results: {len(filtered_results)} documents")
    
    # Create index if supported
    if hasattr(rag, 'create_index'):
        rag.create_index("embedding_idx", similarity_function="cosine")
        print("Created vector index")

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Performance Optimization

- Use batch operations whenever possible
- Implement connection pooling for concurrent access
- Cache frequently accessed data
- Use prepared statements for repeated queries
- Consider async operations for I/O-bound tasks

### 2. Memory Management

- Stream large result sets instead of loading all at once
- Use generators for batch processing
- Clean up resources properly in context managers

### 3. Database-Specific Features

- Leverage native vector types if available
- Use database-specific optimizations (e.g., COPY in PostgreSQL)
- Implement vector indexes for large datasets
- Use database extensions for enhanced functionality

### 4. Error Recovery

- Implement retry logic for transient errors
- Provide meaningful error messages
- Log errors for debugging
- Handle partial failures in batch operations

### 5. Compatibility

- Support both synchronous and asynchronous operations
- Provide SQLAlchemy integration as an option
- Handle different database versions gracefully
- Document minimum requirements

## Registering Your Backend

To make your backend available through the main SQL RAG module:

1. Update `sqlvector/__init__.py`:

```python
# Add to backend imports section
try:
    from .backends.yourbackend import YourBackendRAG, YourBackendConfig
    __yourbackend_available__ = True
except ImportError:
    __yourbackend_available__ = False

# Add to __all__ exports if available
if __yourbackend_available__:
    __all__.extend(["YourBackendRAG", "YourBackendConfig"])
```

2. Add optional dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
yourbackend = ["your-database-driver>=1.0.0"]
```

## Advanced Features

Consider implementing these advanced features:

### 1. Vector Extensions

If your database has vector extensions (like pgvector, sqlite-vss, DuckDB VSS):

```python
def setup_vector_extension(self, conn):
    """Install and configure vector extension."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Configure extension settings
```

### 2. Hybrid Search

Combine vector similarity with full-text search:

```python
def hybrid_search(self, query_text: str, alpha: float = 0.5):
    """Combine vector and keyword search."""
    # Get vector search results
    vector_results = self.query(query_text)
    
    # Get full-text search results
    keyword_results = self.keyword_search(query_text)
    
    # Combine scores with weighting
    combined = self._combine_results(
        vector_results, keyword_results, alpha
    )
    return combined
```

### 3. Incremental Indexing

Support adding documents to existing indexes:

```python
def update_index(self, index_name: str, document_ids: List[str]):
    """Add new documents to existing index."""
    # Implementation depends on index type
    pass
```

### 4. Export/Import Functionality

Allow data portability:

```python
def export_to_parquet(self, output_path: str):
    """Export data to Parquet format."""
    pass

def import_from_csv(self, csv_path: str, has_embeddings: bool = False):
    """Import documents from CSV."""
    pass
```

## Troubleshooting Common Issues

### 1. Connection Issues

```python
# Handle connection timeout
import socket
try:
    conn = self.get_connection()
except socket.timeout:
    logger.error("Database connection timeout")
    raise
```

### 2. Memory Issues with Large Embeddings

```python
# Use chunked processing
def process_large_dataset(self, documents, chunk_size=1000):
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i+chunk_size]
        self.load_documents(chunk)
        # Force garbage collection if needed
        import gc
        gc.collect()
```

### 3. Concurrent Access

```python
# Use locking for SQLite
import threading
_lock = threading.Lock()

def thread_safe_operation(self):
    with _lock:
        # Perform operation
        pass
```

## Conclusion

Building a new backend for SQL RAG involves implementing the five core components (Config, Models, Loader, Querier, RAG) while following the protocol interfaces. Focus on:

1. Robust connection management
2. Efficient batch operations
3. Flexible schema handling
4. Optimized similarity search
5. Proper error handling

Refer to the existing SQLite and DuckDB backends for concrete implementation examples, and adapt the patterns to your specific database's capabilities and syntax.