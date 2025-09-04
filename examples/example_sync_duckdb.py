"""Example usage of the synchronous SQL RAG with DuckDB."""

import sys
import os

# Add parent directory to path to import sql_rag
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sql_rag import SyncSQLRAG


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
    # Clean up any existing database files
    cleanup_duckdb_files("example_sync.db")
    
    # Create a sync DuckDB engine
    engine = create_engine("duckdb:///example_sync.db")

    # Create sync RAG instance
    rag = SyncSQLRAG(engine=engine, embedding_dimension=384)

    # Create tables
    rag.create_tables()

    print("=== Loading Documents (Sync) ===")

    # Load documents
    documents = [
        {
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "example", "category": "animals"},
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "textbook", "category": "technology"},
        },
        {
            "content": "DuckDB is a fast analytical database.",
            "metadata": {"source": "documentation", "category": "database"},
        },
    ]

    document_ids = rag.load_documents(documents)
    print(f"Loaded documents: {document_ids}")

    print("\n=== Querying Documents (Sync) ===")

    # Query similar documents
    results = rag.query("artificial intelligence", top_k=2)

    print(f"\nQuery results for 'artificial intelligence':")
    for result in results:
        print(f"  ID: {result['id']}")
        print(f"  Content: {result['content']}")
        print(f"  Similarity: {result['similarity']:.3f}")
        print(f"  Metadata: {result.get('metadata', {})}")
        print()

    # Batch query
    batch_results = rag.query_batch(["fox", "database"], top_k=1)

    print("Batch query results:")
    for i, query_results in enumerate(batch_results):
        query = ["fox", "database"][i]
        print(f"  Query '{query}':")
        for result in query_results:
            print(f"    {result['content']} (similarity: {result['similarity']:.3f})")

    # Get document
    doc = rag.get_document(document_ids[0])
    if doc:
        print(f"\nRetrieved document: {doc.content}")
        print(f"Metadata: {doc.get_metadata()}")

    # Update document
    success = rag.update_document(
        document_ids[0],
        content="The quick brown fox jumps over the sleeping dog.",
        metadata={"source": "example", "category": "animals", "updated": True},
    )
    print(f"\nDocument updated: {success}")

    # Query again to see the change
    results = rag.query("sleeping dog", top_k=1)
    if results:
        print(f"Updated content found: {results[0]['content']}")

    print("\n=== Cleanup ===")
    deleted = rag.delete_document(document_ids[0])
    print(f"Document deleted: {deleted}")

    # Final query to confirm deletion effect
    final_results = rag.query("fox", top_k=5)
    print(f"Documents remaining after deletion: {len(final_results)}")


if __name__ == "__main__":
    print("Synchronous SQL RAG with DuckDB Example")
    print("=" * 50)

    main()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    
    # Clean up database files
    cleanup_duckdb_files("example_sync.db")
