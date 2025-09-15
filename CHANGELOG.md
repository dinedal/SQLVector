# Changelog

All notable changes to SQLVector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-09-15

### Added
- **PostgreSQL backend** with pgvector extension support
- Complete async and synchronous PostgreSQL RAG implementations (`PostgresRAG` class)
- Auto-managed Docker testing infrastructure with PostgreSQL + pgvector containers
- Comprehensive PostgreSQL test suite with integration and unit tests
- PostgreSQL example scripts demonstrating sync and async usage patterns
- Docker Compose configuration for testing environments
- Custom table name support for PostgreSQL backend

### Features
- **pgvector integration**: Native PostgreSQL vector operations with pgvector extension
- **Advanced indexing**: Support for both HNSW and IVFFlat indexing strategies
- **Connection pooling**: Efficient asyncpg-based connection management
- **JSONB metadata filtering**: Advanced querying capabilities with PostgreSQL's JSONB support
- **Batch operations**: Optimized bulk loading and querying for PostgreSQL
- **Zero-setup testing**: Automatic Docker container management for PostgreSQL tests
- **Health checking**: Robust container startup and readiness verification

### Backend Support
- **PostgreSQL**: Enterprise-grade database with native vector operations
  - pgvector extension for efficient vector storage and similarity search
  - HNSW and IVFFlat index support for scalable vector operations
  - AsyncPG driver integration for high-performance async operations
  - SQLAlchemy ORM integration with PostgreSQL-specific optimizations
  - JSONB metadata storage and advanced filtering capabilities
  - Custom table name configuration for flexible deployment scenarios

### Testing Infrastructure
- **Auto-managed PostgreSQL containers**: Tests automatically start and manage PostgreSQL + pgvector Docker containers
- **Container reuse**: Efficient testing with persistent containers across test sessions
- **Docker detection**: Graceful fallback when Docker is not available
- **Custom database support**: Override with existing PostgreSQL instances via environment variables

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

[Unreleased]: https://github.com/dinedal/sqlvector/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dinedal/sqlvector/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dinedal/sqlvector/releases/tag/v0.1.0