"""Example demonstrating custom table names with DuckDB backend."""

import sys
import os
from pathlib import Path
import duckdb
import polars as pl

# Add parent directory to path to import sqlvector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlvector.backends.duckdb import DuckDBRAG, DuckDBConfig
from sqlvector.embedding import DefaultEmbeddingProvider


def inspect_database_schema(db_path: str, config: DuckDBConfig):
    """Helper function to inspect the DuckDB database schema."""
    print("\n" + "="*60)
    print("DATABASE SCHEMA INSPECTION")
    print("="*60)
    
    conn = duckdb.connect(db_path)
    
    # List all tables
    tables = conn.execute("""
        SELECT table_name, table_type 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()
    
    print("\nTables in database:")
    for table_name, table_type in tables:
        print(f"  - {table_name} ({table_type})")
    
    # Show schema for custom tables
    print(f"\nSchema for '{config.documents_table}':")
    columns = conn.execute(f"""
        SELECT column_name, data_type, is_nullable 
        FROM information_schema.columns 
        WHERE table_name = '{config.documents_table}'
        ORDER BY ordinal_position
    """).fetchall()
    
    for col_name, data_type, nullable in columns:
        print(f"  {col_name:25} {data_type:20} {'NULL' if nullable == 'YES' else 'NOT NULL'}")
    
    print(f"\nSchema for '{config.embeddings_table}':")
    columns = conn.execute(f"""
        SELECT column_name, data_type, is_nullable 
        FROM information_schema.columns 
        WHERE table_name = '{config.embeddings_table}'
        ORDER BY ordinal_position
    """).fetchall()
    
    for col_name, data_type, nullable in columns:
        print(f"  {col_name:25} {data_type:20} {'NULL' if nullable == 'YES' else 'NOT NULL'}")
    
    conn.close()


def main():
    print("DuckDB Custom Table Names Example")
    print("="*60)
    
    # Create configuration with custom table names
    config = DuckDBConfig(
        db_path="custom_tables_example.duckdb",
        documents_table="knowledge_articles",    # Custom document table name
        embeddings_table="article_vectors",      # Custom embeddings table name
        embedding_dimension=384,
        batch_size=500,
        enable_vector_similarity=True,
        enable_vss_extension=True,              # Try to enable VSS (HNSW) if available
        
        # Custom column names
        documents_id_column="article_id",
        documents_content_column="article_text",
        documents_metadata_column="properties",
        embeddings_id_column="vector_id",
        embeddings_document_id_column="article_ref",
        embeddings_column="embedding_vector",
        embeddings_model_column="model_used"
    )
    
    print("\nConfiguration:")
    print(f"  Database: {config.db_path}")
    print(f"  Documents Table: {config.documents_table}")
    print(f"  Embeddings Table: {config.embeddings_table}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Vector Similarity: {config.enable_vector_similarity}")
    print(f"  VSS Extension: {config.enable_vss_extension}")
    print(f"\nCustom Columns:")
    print(f"  Document ID: {config.documents_id_column}")
    print(f"  Content: {config.documents_content_column}")
    print(f"  Metadata: {config.documents_metadata_column}")
    print(f"  Embedding ID: {config.embeddings_id_column}")
    print(f"  Document Reference: {config.embeddings_document_id_column}")
    print(f"  Vector: {config.embeddings_column}")
    
    # Create embedding service
    embedding_service = DefaultEmbeddingProvider(dimension=384)
    
    # Create RAG instance with custom configuration
    rag = DuckDBRAG(
        db_path=config.db_path,
        embedding_provider=embedding_service,
        documents_table=config.documents_table,
        embeddings_table=config.embeddings_table,
        embedding_dimension=config.embedding_dimension,
        batch_size=config.batch_size,
        enable_vss_extension=config.enable_vss_extension,
        vss_enable_persistence=config.vss_enable_persistence,
        # Custom column names
        documents_id_column=config.documents_id_column,
        documents_content_column=config.documents_content_column,
        documents_metadata_column=config.documents_metadata_column,
        embeddings_id_column=config.embeddings_id_column,
        embeddings_document_id_column=config.embeddings_document_id_column,
        embeddings_column=config.embeddings_column,
        embeddings_model_column=config.embeddings_model_column
    )
    
    print("\nInitializing database with custom tables...")
    
    # Sample documents to load
    documents = [
        {
            "content": "DuckDB is an in-process SQL OLAP database management system designed for analytical workloads.",
            "metadata": {"category": "database", "type": "OLAP", "language": "C++"}
        },
        {
            "content": "Polars is a lightning-fast DataFrame library implemented in Rust with Python bindings.",
            "metadata": {"category": "data-processing", "language": "Rust/Python", "type": "DataFrame"}
        },
        {
            "content": "Vector similarity search enables finding similar items in high-dimensional spaces efficiently.",
            "metadata": {"category": "ML", "technique": "similarity", "application": "search"}
        },
        {
            "content": "HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search.",
            "metadata": {"category": "algorithm", "type": "ANN", "complexity": "logarithmic"}
        },
        {
            "content": "Column-oriented storage is optimized for analytical queries that process large amounts of data.",
            "metadata": {"category": "database", "storage": "columnar", "use-case": "analytics"}
        },
        {
            "content": "Apache Arrow provides a standardized columnar memory format for data interchange.",
            "metadata": {"category": "data-format", "type": "columnar", "ecosystem": "Apache"}
        }
    ]
    
    print(f"\nLoading {len(documents)} documents into custom tables...")
    doc_ids = rag.load_documents(documents)
    print(f"Successfully loaded documents with IDs: {doc_ids[:3]}...")
    
    # Inspect the database to show custom tables were created
    inspect_database_schema(config.db_path, config)
    
    # Verify data is in custom tables
    conn = config.get_connection()
    
    doc_count = conn.execute(f"SELECT COUNT(*) FROM {config.documents_table}").fetchone()[0]
    emb_count = conn.execute(f"SELECT COUNT(*) FROM {config.embeddings_table}").fetchone()[0]
    
    print(f"\nData Statistics:")
    print(f"  Documents in '{config.documents_table}': {doc_count}")
    print(f"  Embeddings in '{config.embeddings_table}': {emb_count}")
    
    # Show sample data with custom columns
    print(f"\nSample data from '{config.documents_table}':")
    sample = conn.execute(f"""
        SELECT 
            {config.documents_id_column},
            SUBSTRING({config.documents_content_column}, 1, 50) as content_preview,
            {config.documents_metadata_column}
        FROM {config.documents_table}
        LIMIT 3
    """).fetchall()
    
    for article_id, preview, metadata in sample:
        print(f"  {article_id}: {preview}...")
    
    conn.close()
    
    # Perform similarity search
    print("\n" + "="*60)
    print("SIMILARITY SEARCH")
    print("="*60)
    
    query = "columnar database storage"
    print(f"\nSearching for: '{query}'")
    results = rag.query(query, top_k=3)
    
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        # Use custom column names when accessing results
        doc_id = result.get('article_id') or result.get('id')  # Handle both custom and default
        content = result.get('article_text') or result.get('content')  # Handle both custom and default
        
        print(f"\n{i}. Document ID: {doc_id}")
        print(f"   Content: {content[:100] if content else 'N/A'}...")
        print(f"   Similarity: {result['similarity']:.4f}")
        if result.get('properties') or result.get('metadata'):
            metadata = result.get('properties') or result.get('metadata')
            print(f"   Metadata: {metadata}")
    
    # Export to Polars DataFrame
    print("\n" + "="*60)
    print("EXPORT TO POLARS DATAFRAME")
    print("="*60)
    
    df = rag.export_to_polars()
    print(f"\nExported DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns}")
    
    # Show first few rows
    print("\nFirst 3 rows of exported data:")
    print(df.head(3))
    
    # Demonstrate batch operations with custom tables
    print("\n" + "="*60)
    print("BATCH OPERATIONS")
    print("="*60)
    
    batch_queries = [
        "database analytics",
        "vector search algorithms",
        "data processing frameworks"
    ]
    
    print(f"\nPerforming batch query with {len(batch_queries)} queries...")
    batch_results = rag.query_batch(batch_queries, top_k=2)
    
    for query, results in zip(batch_queries, batch_results):
        print(f"\nQuery: '{query}'")
        for result in results:
            # Use custom column names when accessing results
            content = result.get('article_text') or result.get('content')
            if content:
                print(f"  - {content[:60]}... (sim: {result['similarity']:.3f})")
    
    # Create HNSW index if VSS is enabled
    if config.enable_vss_extension:
        print("\n" + "="*60)
        print("VSS/HNSW INDEX")
        print("="*60)
        
        conn = config.get_connection()
        try:
            # Try to create HNSW index on custom embeddings table
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS hnsw_{config.embeddings_table} 
                ON {config.embeddings_table} 
                USING HNSW ({config.embeddings_column})
            """)
            print(f"\nCreated HNSW index on '{config.embeddings_table}.{config.embeddings_column}'")
        except Exception as e:
            print(f"\nVSS extension not available: {e}")
        conn.close()
    
    # Demonstrate multiple RAG instances with different tables
    print("\n" + "="*60)
    print("MULTIPLE TABLE SETS IN SAME DATABASE")
    print("="*60)
    
    # Create another RAG instance with different table names
    config2 = DuckDBConfig(
        db_path="custom_tables_example.duckdb",  # Same database
        documents_table="research_papers",        # Different tables
        embeddings_table="paper_embeddings",
        embedding_dimension=384
    )
    
    rag2 = DuckDBRAG(
        db_path=config2.db_path,
        embedding_provider=embedding_service,
        documents_table=config2.documents_table,
        embeddings_table=config2.embeddings_table,
        embedding_dimension=config2.embedding_dimension
    )
    
    # Load different data
    research_documents = [
        {"content": "Research paper on neural networks and deep learning fundamentals"},
        {"content": "Study on distributed computing and parallel processing systems"}
    ]
    
    rag2.load_documents(research_documents)
    
    print("\nCreated second set of tables in same database:")
    print(f"  First set: {config.documents_table}, {config.embeddings_table}")
    print(f"  Second set: {config2.documents_table}, {config2.embeddings_table}")
    
    # Show all tables now exist
    conn = duckdb.connect(config.db_path)
    tables = conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()
    
    print(f"\nAll tables in database:")
    for (table_name,) in tables:
        print(f"  - {table_name}")
    
    conn.close()
    
    # Demonstrate loading from CSV with custom tables
    print("\n" + "="*60)
    print("LOADING FROM CSV")
    print("="*60)
    
    # Create a sample CSV file
    csv_path = "sample_data.csv"
    sample_df = pl.DataFrame({
        "content": [
            "CSV data: Machine learning model training techniques",
            "CSV data: Data preprocessing and feature engineering",
            "CSV data: Model evaluation metrics and validation"
        ],
        "metadata": [
            '{"source": "csv", "topic": "ML"}',
            '{"source": "csv", "topic": "data-prep"}',
            '{"source": "csv", "topic": "evaluation"}'
        ]
    })
    sample_df.write_csv(csv_path)
    
    print(f"Created sample CSV with {len(sample_df)} rows")
    
    # Load from CSV into custom tables
    csv_doc_ids = rag.load_from_csv(csv_path)
    print(f"Loaded {len(csv_doc_ids)} documents from CSV into '{config.documents_table}'")
    
    # Verify the data was loaded
    conn = config.get_connection()
    csv_docs = conn.execute(f"""
        SELECT {config.documents_content_column} 
        FROM {config.documents_table}
        WHERE {config.documents_content_column} LIKE 'CSV data:%'
    """).fetchall()
    print(f"Found {len(csv_docs)} CSV-loaded documents in custom table")
    conn.close()
    
    # Clean up
    print("\n" + "="*60)
    print("Cleaning up...")
    
    # Remove the example files
    db_path = Path("custom_tables_example.duckdb")
    if db_path.exists():
        db_path.unlink()
        print(f"Removed example database: {db_path}")
    
    csv_file = Path(csv_path)
    if csv_file.exists():
        csv_file.unlink()
        print(f"Removed sample CSV: {csv_file}")
    
    # DuckDB WAL file cleanup
    wal_file = Path("custom_tables_example.duckdb.wal")
    if wal_file.exists():
        wal_file.unlink()
        print(f"Removed WAL file: {wal_file}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()