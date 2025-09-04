# SQL-RAG Examples

This directory contains example scripts demonstrating various features and use cases of SQL-RAG.

## Basic Examples

### example.py
Basic usage of SQL-RAG with SQLite backend, demonstrating:
- Document loading and embedding
- Similarity search queries
- Batch operations
- Document updates and retrieval

### example_with_embedding_model.py
Shows how to integrate custom embedding models:
- Using Sentence Transformers for embeddings
- GPU acceleration support
- Custom embedding provider implementation
- Configurable vector dimensions

## Backend-Specific Examples

### DuckDB Examples

#### example_duckdb.py
Comprehensive DuckDB backend demonstration:
- Native DuckDB operations
- Polars DataFrame integration
- CSV and Parquet data loading
- Different similarity functions (cosine, euclidean, inner product)
- Metadata filtering and complex queries
- Export to Polars DataFrame

#### example_duckdb_vss.py
DuckDB with VSS (Vector Similarity Search) extension:
- HNSW index configuration
- Performance comparisons with flat search
- Advanced indexing parameters

#### example_duckdb_with_sqlalchemy.py
DuckDB using SQLAlchemy ORM:
- SQLAlchemy integration
- ORM-based operations
- Connection management

### SQLite Examples

#### example_sqlite_vss.py
SQLite with VSS extension for accelerated similarity search:
- sqlite-vss setup and configuration
- Faiss-based indexing
- IVF index training
- Binary embedding storage

## Synchronous Interface Examples

### example_sync_duckdb.py
Using the synchronous interface with DuckDB:
- SyncSQLRAG usage
- Synchronous document operations
- No async/await required

## Running the Examples

1. Install SQL-RAG with the appropriate backend support:
```bash
# For DuckDB examples
pip install "sql-rag[duckdb]"

# For basic SQLite examples
pip install sql-rag

# For development (all backends)
pip install -e ".[duckdb,test]"
```

2. Install additional dependencies for embedding models:
```bash
# For example_with_embedding_model.py
pip install sentence-transformers torch
```

3. Run an example:
```bash
python examples/example.py
python examples/example_duckdb.py
```

## Performance Considerations

- **DuckDB examples**: Best for analytical workloads and large-scale operations
- **SQLite examples**: Lightweight, good for embedded applications
- **VSS examples**: Demonstrate accelerated vector search capabilities
- **Embedding model example**: Shows how to use state-of-the-art embedding models

## Custom Embedding Models

The `example_with_embedding_model.py` demonstrates how to:
1. Create a custom embedding provider
2. Use GPU acceleration when available
3. Configure embedding dimensions
4. Integrate with popular embedding libraries

## Notes

- Examples create database files in the current directory
- Most examples clean up their database files on start
- Adjust batch sizes based on your hardware capabilities
- GPU support requires appropriate PyTorch installation