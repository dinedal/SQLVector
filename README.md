# SQL-RAG

A flexible and efficient Retrieval-Augmented Generation (RAG) library that uses SQL databases as vector stores through SQLAlchemy. Supports multiple backends including DuckDB with HNSW indexing and SQLite with VSS (Vector Similarity Search).

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/dinedal/sql-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/dinedal/sql-rag/actions/workflows/tests.yml)

## Features

- **Multiple Backend Support**: DuckDB with HNSW indexing, SQLite with VSS extension
- **Async and Sync Interfaces**: Both `SQLRAG` (async) and `SyncSQLRAG` interfaces
- **Flexible Embedding Providers**: Bring your own embeddings or use the default provider
- **Batch Operations**: Efficient batch loading and querying
- **Multiple Similarity Functions**: Cosine, Euclidean, Inner Product similarity
- **Metadata Filtering**: Query with metadata filters and complex conditions
- **SQLAlchemy Integration**: Works with any SQLAlchemy-supported database
- **Export Capabilities**: Export to Polars DataFrames (DuckDB) or dictionaries

## Installation

### Basic Installation

```bash
pip install sql-rag
```

### With DuckDB Support

```bash
pip install "sql-rag[duckdb]"
```

### With SQLite Async Support

```bash
# For async SQLite support
pip install sql-rag aiosqlite sqlalchemy
```

### With Custom Embedding Providers

If you want to use custom embedding providers (e.g., with Sentence Transformers), install the required dependencies:

```bash
# For Sentence Transformers based embeddings
pip install sql-rag transformers sentence-transformers torch
```

### With Test Dependencies

```bash
pip install "sql-rag[test]"
```

### Development Installation

```bash
git clone https://github.com/dinedal/sql-rag.git
cd sql-rag
pip install -e ".[duckdb,test]"
```

## Quick Start

### Using Custom Embeddings

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sql_rag import SQLRAG, EmbeddingProvider
from sentence_transformers import SentenceTransformer
from typing import List
import torch
import numpy as np

class CustomEmbeddingProvider(EmbeddingProvider):
    def __init__(self, batch_size=4, max_seq_length=512, use_gpu=False):
        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        
        device = "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                raise RuntimeError(
                    "No suitable GPU found. Please check your PyTorch installation."
                )
        
        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device,
        )
        # We can reduce the max_seq_length from the default for faster encoding
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        
        super().__init__()
    
    async def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=self.batch_size).tolist()
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return self.model.similarity(
            np.array(vec1, dtype=np.float32),
            np.array(vec2, dtype=np.float32),
        ).item()

async def main():
    # Create async engine
    engine = create_async_engine("sqlite+aiosqlite:///example.db")
    
    # Initialize with custom embedding provider
    rag = SQLRAG(
        engine=engine, 
        embedding_provider=CustomEmbeddingProvider(use_gpu=True)
    )
    
    # Create tables
    await rag.create_tables()
    
    # Load documents
    documents = [
        {
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "example", "category": "animals"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "textbook", "category": "technology"}
        }
    ]
    
    document_ids = await rag.load_documents(documents)
    print(f"Loaded {len(document_ids)} documents")
    
    # Query similar documents
    results = await rag.query("artificial intelligence", top_k=5)
    
    for result in results:
        print(f"Content: {result['content']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Metadata: {result.get('metadata', {})}")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Synchronous Interface

```python
from sqlalchemy import create_engine
from sql_rag import SyncSQLRAG

# Create sync engine
engine = create_engine("sqlite:///example.db")

# Initialize RAG
rag = SyncSQLRAG(engine=engine)

# Create tables
rag.create_tables()

# Load and query documents
documents = [{"content": "Your content here", "metadata": {"key": "value"}}]
document_ids = rag.load_documents(documents)

results = rag.query("search term", top_k=5)
```

## Backend Configuration

### DuckDB with HNSW Indexing

```python
from sql_rag.backends.duckdb import DuckDBConfig, DuckDBRAG

config = DuckDBConfig(
    connection_string="duckdb:///rag.duckdb",
    embedding_dim=384,
    use_hnsw=True,  # Enable HNSW indexing
    hnsw_config={
        "max_elements": 100000,
        "ef_construction": 200,
        "ef": 100,
        "M": 16
    }
)

rag = DuckDBRAG(config=config)
```

### SQLite with VSS Extension

```python
from sql_rag.backends.sqlite import SQLiteConfig, SQLiteRAG

config = SQLiteConfig(
    connection_string="sqlite:///rag.db",
    embedding_dim=384,
    use_vss=True,  # Enable VSS extension
    vss_version="v0.1.2"
)

rag = SQLiteRAG(config=config)
```

## Advanced Features

### Metadata Filtering

```python
# Query with metadata filters
results = await rag.query_with_filters(
    filters={"category": "technology", "year": 2025},
    query_text="machine learning",
    top_k=10
)
```

### Batch Operations

```python
# Batch document loading
documents = [{"content": f"Document {i}", "metadata": {"id": i}} for i in range(100)]
document_ids = await rag.load_documents(documents, batch_size=10)

# Batch querying
queries = ["query1", "query2", "query3"]
batch_results = await rag.query_batch(queries, top_k=5)
```

### Export Data

```python
# For DuckDB backend - export to Polars DataFrame
df = rag.export_to_polars(include_embeddings=True)

# For SQLite backend - export to dictionary
data = rag.export_documents(include_embeddings=False)
```

## Architecture

SQL-RAG uses a protocol-based architecture that allows for flexible backend implementations:

- **Protocols**: Define interfaces for backends (`RAGSystemProtocol`, `DocumentLoaderProtocol`, `DocumentQuerierProtocol`)
- **Backends**: Pluggable database backends with specific optimizations
- **Embedding Service**: Handles text-to-vector conversion with pluggable providers
- **Models**: SQLAlchemy models for documents and embeddings

## Performance Considerations

- **DuckDB**: Best for analytical workloads, supports HNSW indexing for fast similarity search
- **SQLite**: Lightweight option with VSS extension for vector similarity
- **Batch Size**: Tune batch sizes based on your hardware and document sizes
- **Embedding Dimensions**: Lower dimensions generally provide faster search at the cost of accuracy

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific backend tests
pytest tests/test_duckdb.py
pytest tests/test_sqlite.py

# Run with coverage
pytest --cov=sql_rag tests/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## Examples

Check out the [examples/](examples/) directory for more detailed examples:

- Basic usage with different backends
- Custom embedding providers
- Advanced querying techniques
- Performance benchmarks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SQL-RAG in your research, please cite:

```bibtex
@software{sql-rag,
  title = {SQL-RAG: SQL-based Retrieval-Augmented Generation},
  year = {2025},
  url = {https://github.com/dinedal/sql-rag}
}
```

## Acknowledgments

- Built on [SQLAlchemy](https://www.sqlalchemy.org/) for database abstraction
- DuckDB backend uses [DuckDB](https://duckdb.org/) with VSS extension
- SQLite backend uses [sqlite-vss](https://github.com/asg017/sqlite-vss) for vector similarity

## Support

- =� [Documentation](https://github.com/dinedal/sql-rag/wiki)
- = [Issue Tracker](https://github.com/dinedal/sql-rag/issues)
- =� [Discussions](https://github.com/dinedal/sql-rag/discussions)