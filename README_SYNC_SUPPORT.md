# Synchronous Engine Support for SQL RAG

This document describes the new synchronous engine support added to the sqlvector library, enabling compatibility with sync-only databases like DuckDB.

## Overview

The sqlvector library now supports both asynchronous and synchronous SQLAlchemy engines:

- **Async Support** (existing): For databases with async drivers like PostgreSQL with asyncpg, SQLite with aiosqlite
- **Sync Support** (new): For databases with sync-only drivers like DuckDB, regular SQLite, MySQL, etc.

## New Components

### Core Classes

1. **SyncRAGConfig** - Configuration for synchronous RAG systems
2. **SyncSQLRAG** - Main synchronous interface (mirrors SQLRAG API)
3. **SyncEmbeddingService** - Synchronous embedding operations
4. **SyncLoaderInterface** - Synchronous document loading
5. **SyncQueryInterface** - Synchronous document querying
6. **DefaultSyncEmbeddingProvider** - Synchronous embedding provider

### Usage Examples

#### Synchronous Usage (NEW)
```python
from sqlalchemy import create_engine
from sqlvector import SyncSQLRAG

# Create sync engine for DuckDB
engine = create_engine("duckdb:///mydb.db")
rag = SyncSQLRAG(engine=engine)

# Create tables
rag.create_tables()

# Load documents (no async/await needed!)
doc_id = rag.load_document("Hello world", {"source": "test"})

# Query documents
results = rag.query("Hello", top_k=5)
```

#### Asynchronous Usage (EXISTING - unchanged)
```python
from sqlalchemy.ext.asyncio import create_async_engine
from sqlvector import SQLRAG

# Create async engine
engine = create_async_engine("sqlite+aiosqlite:///mydb.db")
rag = SQLRAG(engine=engine)

# All operations require async/await
await rag.create_tables()
doc_id = await rag.load_document("Hello world", {"source": "test"})
results = await rag.query("Hello", top_k=5)
```

## Supported Databases

### Synchronous Engines
- **DuckDB**: `create_engine("duckdb:///path/to/db.db")`
- **SQLite**: `create_engine("sqlite:///path/to/db.db")`
- **PostgreSQL**: `create_engine("postgresql://user:pass@host/db")`
- **MySQL**: `create_engine("mysql://user:pass@host/db")`

### Asynchronous Engines
- **PostgreSQL**: `create_async_engine("postgresql+asyncpg://user:pass@host/db")`
- **SQLite**: `create_async_engine("sqlite+aiosqlite:///path/to/db.db")`

## API Compatibility

The `SyncSQLRAG` class provides the same API as `SQLRAG` but without async/await:

| Async Method | Sync Method | Description |
|--------------|-------------|-------------|
| `await rag.create_tables()` | `rag.create_tables()` | Create database tables |
| `await rag.load_document(...)` | `rag.load_document(...)` | Load single document |
| `await rag.load_documents(...)` | `rag.load_documents(...)` | Load multiple documents |
| `await rag.query(...)` | `rag.query(...)` | Query by similarity |
| `await rag.query_batch(...)` | `rag.query_batch(...)` | Batch queries |
| `await rag.get_document(...)` | `rag.get_document(...)` | Get document by ID |
| `await rag.delete_document(...)` | `rag.delete_document(...)` | Delete document |
| `await rag.update_document(...)` | `rag.update_document(...)` | Update document |

## Migration Guide

### For DuckDB Users
Previously, DuckDB users had to use the specialized `DuckDBRAG` backend:

```python
# Old way (still works)
from sqlvector.backends.duckdb import DuckDBRAG
rag = DuckDBRAG(db_path="mydb.db")

# New way (unified interface)
from sqlalchemy import create_engine
from sqlvector import SyncSQLRAG
engine = create_engine("duckdb:///mydb.db")
rag = SyncSQLRAG(engine=engine)
```

### For Sync-Only Databases
Any database that only has synchronous SQLAlchemy drivers can now use the main interface:

```python
from sqlalchemy import create_engine
from sqlvector import SyncSQLRAG

# Works with any sync SQLAlchemy-supported database
engine = create_engine("your-database-url")
rag = SyncSQLRAG(engine=engine)
```

## Implementation Details

### Architecture
- **Dual Implementation**: Maintains separate sync and async code paths
- **Shared Models**: Uses the same SQLAlchemy models for both sync and async
- **Parallel APIs**: Sync classes mirror async classes exactly
- **No Mixing**: Sync and async components don't mix (prevents async/sync conflicts)

### Performance
- **Zero Overhead**: Sync implementation has no async overhead
- **Native Performance**: Uses native sync SQLAlchemy sessions
- **Memory Efficient**: No event loop or async context overhead

## Backward Compatibility

- ✅ All existing async code continues to work unchanged
- ✅ Existing `SQLRAG` class unchanged
- ✅ All backend implementations (`DuckDBRAG`, `SQLiteRAG`) still work
- ✅ No breaking changes to any existing APIs

## Testing

Run the example to verify sync support:

```bash
python example_sync_duckdb.py
```

This should demonstrate successful:
- Document loading
- Similarity queries  
- Batch queries
- Document updates
- Document deletion
