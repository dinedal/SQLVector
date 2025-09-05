"""Example demonstrating DuckDB backend with SQLAlchemy sync Engine support."""

import sys
import os

# Add parent directory to path to import sqlvector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlvector.backends.duckdb import DuckDBRAG
from sqlvector.embedding import DefaultEmbeddingProvider


def cleanup_duckdb_files(db_path: str) -> None:
    """Remove existing DuckDB database and WAL files to ensure clean start."""
    if db_path == ":memory:":
        return
    
    # Remove the main database file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Remove the WAL file if it exists
    wal_path = f"{db_path}.wal"
    if os.path.exists(wal_path):
        os.remove(wal_path)
        print(f"Removed existing WAL file: {wal_path}")


def main():
    print("=== DuckDB RAG with SQLAlchemy sync Engine Example ===\n")

    # Method 1: Create your own sync engine
    print("1. Using user-provided sync SQLAlchemy engine:")
    cleanup_duckdb_files("example_sqlalchemy.db")
    engine = create_engine("duckdb:///example_sqlalchemy.db")

    rag1 = DuckDBRAG(
        db_path="example_sqlalchemy.db",
        engine=engine,
        use_sqlalchemy=True,
        embedding_dimension=384,
    )

    # Load some documents
    documents = [
        {
            "content": "DuckDB is a fast analytical database",
            "metadata": {"category": "database", "type": "analytical"},
        },
        {
            "content": "SQLAlchemy provides database abstraction for Python",
            "metadata": {"category": "python", "type": "library"},
        },
        {
            "content": "Vector databases enable semantic search capabilities",
            "metadata": {"category": "database", "type": "vector"},
        },
    ]

    doc_ids = rag1.load_documents(documents, show_progress=False)
    print(f"Loaded {len(doc_ids)} documents using SQLAlchemy engine")

    # Query the documents
    results = rag1.query("database technology", top_k=2)
    print(f"Query results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Similarity: {result['similarity']:.3f}")
        print(f"     Content: {result['content']}")
        print(f"     Category: {result.get('metadata', {}).get('category', 'N/A')}")
        print()

    print("-" * 60)

    # Method 2: Auto-create engine
    print("2. Using auto-created SQLAlchemy engine:")
    cleanup_duckdb_files("example_auto_engine.db")
    rag2 = DuckDBRAG(
        db_path="example_auto_engine.db", use_sqlalchemy=True, embedding_dimension=384
    )

    # Load a document
    doc_id = rag2.load_document(
        content="Auto-created engines work seamlessly with DuckDB",
        metadata={"source": "example", "auto_created": True},
    )
    print(f"Loaded document with auto-created engine: {doc_id}")

    # Verify it works
    doc = rag2.get_document(doc_id)
    if doc:
        print(f"Retrieved document content: {doc.content}")
        print(f"Document metadata: {doc.metadata}")
    else:
        print("Error: Could not retrieve document")

    print("-" * 60)

    # Method 3: In-memory database with SQLAlchemy
    print("3. Using in-memory database with SQLAlchemy:")
    rag3 = DuckDBRAG(db_path=":memory:", use_sqlalchemy=True, embedding_dimension=384)

    # Load documents
    memory_docs = [
        {"content": "In-memory databases are fast for temporary data"},
        {"content": "SQLAlchemy works with both file and memory databases"},
    ]

    memory_doc_ids = rag3.load_documents(memory_docs, show_progress=False)
    print(f"Loaded {len(memory_doc_ids)} documents in memory database")

    # Query
    memory_results = rag3.query("memory database", top_k=1)
    print(f"Memory database query result: {memory_results[0]['content']}")

    print("-" * 60)

    # Method 4: Custom engine with connection pooling
    print("4. Using custom engine with StaticPool:")
    from sqlalchemy.pool import StaticPool

    cleanup_duckdb_files("example_custom.db")
    custom_engine = create_engine(
        "duckdb:///example_custom.db",
        poolclass=StaticPool,
        pool_pre_ping=True,
        echo=False,  # Set to True to see SQL queries
    )

    rag4 = DuckDBRAG(
        db_path="example_custom.db",
        engine=custom_engine,
        use_sqlalchemy=True,
        embedding_dimension=384,
    )

    # Load and query
    custom_doc_id = rag4.load_document(
        "Custom engine configurations provide more control",
        metadata={"engine_type": "custom", "pooling": "static"},
    )

    custom_results = rag4.query("custom configuration", top_k=1)
    print(f"Custom engine result: {custom_results[0]['content']}")

    print("-" * 60)

    # Method 5: Sharing engine across multiple RAG instances
    print("5. Sharing engine across multiple RAG instances:")
    cleanup_duckdb_files("shared_engine.db")
    shared_engine = create_engine("duckdb:///shared_engine.db")

    # Create two RAG instances with different table names
    rag5a = DuckDBRAG(
        db_path="shared_engine.db",
        engine=shared_engine,
        use_sqlalchemy=True,
        embedding_dimension=384,
        documents_table="docs_a",
        embeddings_table="embeddings_a",
    )

    rag5b = DuckDBRAG(
        db_path="shared_engine.db",
        engine=shared_engine,
        use_sqlalchemy=True,
        embedding_dimension=384,
        documents_table="docs_b",
        embeddings_table="embeddings_b",
    )

    # Load different documents in each
    doc_a = rag5a.load_document("Document in instance A")
    doc_b = rag5b.load_document("Document in instance B")

    print(f"Instance A documents: {len(rag5a.get_documents([doc_a]))}")
    print(f"Instance B documents: {len(rag5b.get_documents([doc_b]))}")

    # Each instance only sees its own documents
    print(f"A can see B's document: {rag5a.get_document(doc_b) is not None}")
    print(f"B can see A's document: {rag5b.get_document(doc_a) is not None}")

    print("-" * 60)

    # Compare with native DuckDB connections
    print("6. Comparison with native DuckDB connections:")

    # Native approach
    cleanup_duckdb_files("example_native.db")
    rag_native = DuckDBRAG(
        db_path="example_native.db",
        use_sqlalchemy=False,  # Use native DuckDB connections
        embedding_dimension=384,
    )

    native_doc_id = rag_native.load_document("Native DuckDB connection example")
    native_results = rag_native.query("native connection", top_k=1)
    print(f"Native connection result: {native_results[0]['content']}")

    # SQLAlchemy approach
    cleanup_duckdb_files("example_sqlalchemy_comparison.db")
    rag_sqlalchemy = DuckDBRAG(
        db_path="example_sqlalchemy_comparison.db",
        use_sqlalchemy=True,
        embedding_dimension=384,
    )

    sqlalchemy_doc_id = rag_sqlalchemy.load_document("SQLAlchemy connection example")
    sqlalchemy_results = rag_sqlalchemy.query("sqlalchemy connection", top_k=1)
    print(f"SQLAlchemy connection result: {sqlalchemy_results[0]['content']}")

    print("\n=== Benefits of SQLAlchemy Integration ===")
    print("• Unified database interface across different backends")
    print("• Connection pooling and management")
    print("• Compatibility with SQLAlchemy ecosystem tools")
    print("• Easier integration with existing SQLAlchemy applications")
    print("• Consistent API regardless of underlying database")

    print("\n=== When to Use Each Approach ===")
    print("Native DuckDB:")
    print("  + Direct access to DuckDB-specific features")
    print("  + Potentially lower overhead")
    print("  + Simpler for DuckDB-only applications")

    print("\nSQLAlchemy Integration:")
    print("  + Better for multi-database applications")
    print("  + Connection pooling and management")
    print("  + Easier testing and mocking")
    print("  + Integration with existing SQLAlchemy code")


if __name__ == "__main__":
    main()
    
    # Clean up all created database files
    cleanup_duckdb_files("example_sqlalchemy.db")
    cleanup_duckdb_files("example_auto_engine.db")
    cleanup_duckdb_files("example_custom.db")
    cleanup_duckdb_files("shared_engine.db")
    cleanup_duckdb_files("example_native.db")
    cleanup_duckdb_files("example_sqlalchemy_comparison.db")
