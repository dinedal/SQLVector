"""Example demonstrating custom table names with SQLite backend."""

import sys
import os
import sqlite3
from pathlib import Path

# Add parent directory to path to import sqlvector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlvector.backends.sqlite import SQLiteRAG, SQLiteConfig
from sqlvector.embedding import DefaultEmbeddingProvider


def inspect_database_schema(db_path: str, config: SQLiteConfig):
    """Helper function to inspect the database schema."""
    print("\n" + "="*60)
    print("DATABASE SCHEMA INSPECTION")
    print("="*60)
    
    with sqlite3.connect(db_path) as conn:
        # List all tables
        cursor = conn.execute("""
            SELECT name, type FROM sqlite_master 
            WHERE type IN ('table', 'index')
            ORDER BY type, name
        """)
        
        print("\nTables and Indexes:")
        current_type = None
        for name, obj_type in cursor.fetchall():
            if obj_type != current_type:
                current_type = obj_type
                print(f"\n{obj_type.upper()}S:")
            print(f"  - {name}")
        
        # Show schema for custom tables
        print(f"\nSchema for '{config.documents_table}':")
        cursor = conn.execute(f"PRAGMA table_info({config.documents_table})")
        for row in cursor.fetchall():
            cid, name, dtype, notnull, default, pk = row
            print(f"  {name:20} {dtype:15} {'PRIMARY KEY' if pk else ''}")
        
        print(f"\nSchema for '{config.embeddings_table}':")
        cursor = conn.execute(f"PRAGMA table_info({config.embeddings_table})")
        for row in cursor.fetchall():
            cid, name, dtype, notnull, default, pk = row
            print(f"  {name:20} {dtype:15} {'PRIMARY KEY' if pk else ''}")


def main():
    print("SQLite Custom Table Names Example")
    print("="*60)
    
    # Create configuration with custom table names
    config = SQLiteConfig(
        db_path="custom_tables_example.db",
        documents_table="my_knowledge_base",    # Custom document table name
        embeddings_table="my_vector_store",     # Custom embeddings table name
        vss_table="my_vss_index",              # Custom VSS virtual table name
        embedding_dimension=384,
        enable_vss_extension=True,              # Try to enable VSS if available
        
        # You can also customize column names
        documents_id_column="doc_id",
        documents_content_column="text",
        documents_metadata_column="properties",
        embeddings_id_column="vec_id",
        embeddings_document_id_column="doc_reference",
        embeddings_column="vector"
    )
    
    print("\nConfiguration:")
    print(f"  Database: {config.db_path}")
    print(f"  Documents Table: {config.documents_table}")
    print(f"  Embeddings Table: {config.embeddings_table}")
    print(f"  VSS Table: {config.vss_table}")
    print(f"  Custom Columns: doc_id, text, properties, vec_id, doc_reference, vector")
    
    # Create embedding service
    embedding_service = DefaultEmbeddingProvider(dimension=384)
    
    # Create RAG instance with custom configuration
    rag = SQLiteRAG(
        db_path=config.db_path,
        embedding_provider=embedding_service,
        documents_table=config.documents_table,
        embeddings_table=config.embeddings_table,
        vss_table=config.vss_table,
        embedding_dimension=config.embedding_dimension,
        enable_vss_extension=config.enable_vss_extension,
        # Custom column names
        documents_id_column=config.documents_id_column,
        documents_content_column=config.documents_content_column,
        documents_metadata_column=config.documents_metadata_column,
        embeddings_id_column=config.embeddings_id_column,
        embeddings_document_id_column=config.embeddings_document_id_column,
        embeddings_column=config.embeddings_column
    )
    
    print("\nInitializing database with custom tables...")
    
    # Sample documents to load
    documents = [
        {
            "content": "Python is a high-level programming language known for its simplicity.",
            "metadata": {"category": "programming", "language": "Python", "difficulty": "beginner"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"category": "AI", "topic": "ML", "level": "intermediate"}
        },
        {
            "content": "SQLite is a lightweight, serverless database engine that stores data in a single file.",
            "metadata": {"category": "database", "type": "embedded", "usage": "local"}
        },
        {
            "content": "Vector databases are optimized for storing and searching high-dimensional vectors efficiently.",
            "metadata": {"category": "database", "type": "vector", "technology": "modern"}
        },
        {
            "content": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
            "metadata": {"category": "AI", "technique": "RAG", "application": "NLP"}
        }
    ]
    
    print(f"\nLoading {len(documents)} documents into custom tables...")
    doc_ids = rag.load_documents(documents)
    print(f"Successfully loaded documents with IDs: {doc_ids[:3]}...")
    
    # Inspect the database to show custom tables were created
    inspect_database_schema(config.db_path, config)
    
    # Verify data is in custom tables
    with sqlite3.connect(config.db_path) as conn:
        # Load VSS extension if enabled to allow querying VSS tables
        if config.enable_vss_extension:
            try:
                conn.enable_load_extension(True)
                import sqlite_vss
                sqlite_vss.load(conn)
                conn.enable_load_extension(False)
            except (ImportError, sqlite3.OperationalError):
                # VSS extension not available
                pass
        
        doc_count = conn.execute(f"SELECT COUNT(*) FROM {config.documents_table}").fetchone()[0]
        emb_count = conn.execute(f"SELECT COUNT(*) FROM {config.embeddings_table}").fetchone()[0]
        
        print(f"\nData Statistics:")
        print(f"  Documents in '{config.documents_table}': {doc_count}")
        print(f"  Embeddings in '{config.embeddings_table}': {emb_count}")
        
        if config.enable_vss_extension:
            # Check if VSS table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (config.vss_table,)
            )
            if cursor.fetchone():
                try:
                    vss_count = conn.execute(f"SELECT COUNT(*) FROM {config.vss_table}").fetchone()[0]
                    print(f"  VSS entries in '{config.vss_table}': {vss_count}")
                except sqlite3.OperationalError:
                    # VSS extension couldn't be loaded
                    print(f"  VSS table '{config.vss_table}' exists but extension not available")
            else:
                print(f"  VSS extension not available (table '{config.vss_table}' not created)")
    
    # Perform similarity search
    print("\n" + "="*60)
    print("SIMILARITY SEARCH")
    print("="*60)
    
    query = "database technology"
    print(f"\nSearching for: '{query}'")
    results = rag.query(query, top_k=3)
    
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        # Use custom column names when accessing results
        doc_id = result.get('doc_id') or result.get('id')  # Handle both custom and default
        content = result.get('text') or result.get('content')  # Handle both custom and default
        
        print(f"\n{i}. Document ID: {doc_id}")
        print(f"   Content: {content[:100] if content else 'N/A'}...")
        print(f"   Similarity: {result['similarity']:.4f}")
        if result.get('properties') or result.get('metadata'):
            metadata = result.get('properties') or result.get('metadata')
            print(f"   Metadata: {metadata}")
    
    # Demonstrate querying with metadata filter using custom columns
    print("\n" + "="*60)
    print("ADVANCED QUERY WITH CUSTOM COLUMNS")
    print("="*60)
    
    with sqlite3.connect(config.db_path) as conn:
        # Query using custom column names
        cursor = conn.execute(f"""
            SELECT 
                d.{config.documents_id_column},
                d.{config.documents_content_column},
                d.{config.documents_metadata_column}
            FROM {config.documents_table} d
            WHERE d.{config.documents_metadata_column} LIKE '%"category": "AI"%'
        """)
        
        print("\nDocuments with category='AI' (using custom columns):")
        for row in cursor.fetchall():
            doc_id, content, metadata = row
            print(f"  - {doc_id}: {content[:60]}...")
    
    # Demonstrate multiple RAG instances with different tables in same database
    print("\n" + "="*60)
    print("MULTIPLE TABLE SETS IN SAME DATABASE")
    print("="*60)
    
    # Create another RAG instance with different table names
    config2 = SQLiteConfig(
        db_path="custom_tables_example.db",  # Same database
        documents_table="secondary_docs",     # Different tables
        embeddings_table="secondary_vectors",
        embedding_dimension=384
    )
    
    rag2 = SQLiteRAG(
        db_path=config2.db_path,
        embedding_provider=embedding_service,
        documents_table=config2.documents_table,
        embeddings_table=config2.embeddings_table,
        embedding_dimension=config2.embedding_dimension
    )
    
    # Load different data
    secondary_documents = [
        {"content": "This is data in the secondary table set"},
        {"content": "Completely separate from the primary tables"}
    ]
    
    rag2.load_documents(secondary_documents)
    
    print("\nCreated second set of tables in same database:")
    print(f"  Primary tables: {config.documents_table}, {config.embeddings_table}")
    print(f"  Secondary tables: {config2.documents_table}, {config2.embeddings_table}")
    
    # Show all tables now exist
    with sqlite3.connect(config.db_path) as conn:
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        all_tables = [row[0] for row in cursor.fetchall()]
        print(f"\nAll tables in database: {', '.join(all_tables)}")
    
    # Clean up
    print("\n" + "="*60)
    print("Cleaning up...")
    
    # Close any shared connections
    config.close_shared_connection()
    config2.close_shared_connection()
    
    # Remove the example database
    db_path = Path("custom_tables_example.db")
    if db_path.exists():
        db_path.unlink()
        print(f"Removed example database: {db_path}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()