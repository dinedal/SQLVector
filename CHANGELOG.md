# Changelog

All notable changes to SQLVector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-09-03

### Added
- Initial release of SQLVector
- Core RAG functionality with SQLAlchemy backend support
- DuckDB backend with HNSW indexing support
- SQLite backend with VSS extension support
- Async and synchronous interfaces (`SQLRAG` and `SyncSQLRAG`)
- Flexible embedding provider system
- Batch operations for document loading and querying
- Multiple similarity functions (cosine, euclidean, inner product)
- Metadata filtering and complex query support
- Export capabilities to Polars DataFrames (DuckDB) and dictionaries
- Comprehensive test suite
- Example scripts demonstrating various features
- Protocol-based architecture for extensibility

### Features
- Document CRUD operations (Create, Read, Update, Delete)
- Vector similarity search with configurable thresholds
- Metadata-based filtering
- Batch processing for improved performance
- Custom embedding provider support
- SQLAlchemy ORM integration
- Context manager support
- Type hints throughout the codebase

### Backend Support
- **DuckDB**: High-performance analytical database with native vector operations
  - HNSW index support for fast similarity search
  - Polars DataFrame integration
  - CSV/Parquet data loading
- **SQLite**: Lightweight embedded database
  - VSS extension for accelerated vector search
  - Faiss-based indexing
  - Binary embedding storage

[Unreleased]: https://github.com/dinedal/sqlvector/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dinedal/sqlvector/releases/tag/v0.1.0