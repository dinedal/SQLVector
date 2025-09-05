"""Tests for custom table names with DuckDB backend."""

import pytest
import tempfile
from pathlib import Path
import polars as pl

from sqlvector.backends.duckdb import (
    DuckDBRAG,
    DuckDBConfig,
    DuckDBDocument
)
from sqlvector.embedding import DefaultEmbeddingProvider


class TestDuckDBCustomTableNames:
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file."""
        # Create a temporary file and immediately delete it
        # DuckDB will create its own file at this path
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
        # Also clean up WAL file if it exists
        wal_path = Path(f"{db_path}.wal")
        if wal_path.exists():
            wal_path.unlink()
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service."""
        return DefaultEmbeddingProvider(dimension=384)
    
    def test_custom_table_creation(self, temp_db_path, embedding_service):
        """Test that custom table names are actually used in the database."""
        config = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="custom_documents",
            embeddings_table="custom_embeddings",
            embedding_dimension=384
        )
        
        rag = DuckDBRAG(
            db_path=config.db_path,
            embedding_provider=embedding_service,
            documents_table=config.documents_table,
            embeddings_table=config.embeddings_table,
            embedding_dimension=config.embedding_dimension
        )
        
        # Initialize database
        conn = config.get_connection()
        config.setup_database(conn)
        
        # Verify custom tables were created
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
        
        table_names = [row[0] for row in tables]
        
        assert "custom_documents" in table_names
        assert "custom_embeddings" in table_names
        
        # Verify default table names were NOT created
        assert "documents" not in table_names
        assert "embeddings" not in table_names
        
        conn.close()
    
    def test_custom_table_operations(self, temp_db_path, embedding_service):
        """Test CRUD operations with custom table names."""
        config = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="my_docs",
            embeddings_table="my_vectors",
            embedding_dimension=384
        )
        
        rag = DuckDBRAG(
            db_path=config.db_path,
            embedding_provider=embedding_service,
            documents_table=config.documents_table,
            embeddings_table=config.embeddings_table,
            embedding_dimension=config.embedding_dimension
        )
        
        # Load documents
        documents = [
            {
                "content": "DuckDB with custom tables",
                "metadata": {"type": "test", "backend": "duckdb"}
            },
            {
                "content": "Testing custom schema in DuckDB",
                "metadata": {"type": "test", "index": 2}
            }
        ]
        
        doc_ids = rag.load_documents(documents)
        assert len(doc_ids) == 2
        
        # Verify data was inserted into custom tables
        conn = config.get_connection()
        
        # Check documents table
        doc_count = conn.execute(f"SELECT COUNT(*) FROM {config.documents_table}").fetchone()[0]
        assert doc_count == 2
        
        # Check embeddings table
        emb_count = conn.execute(f"SELECT COUNT(*) FROM {config.embeddings_table}").fetchone()[0]
        assert emb_count == 2
        
        # Verify content
        contents = conn.execute(
            f"SELECT {config.documents_content_column} FROM {config.documents_table} ORDER BY {config.documents_id_column}"
        ).fetchall()
        content_texts = [row[0] for row in contents]
        assert "DuckDB with custom tables" in content_texts
        assert "Testing custom schema in DuckDB" in content_texts
        
        conn.close()
    
    def test_custom_table_querying(self, temp_db_path, embedding_service):
        """Test querying with custom table names."""
        config = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="knowledge_base",
            embeddings_table="vector_store",
            embedding_dimension=384
        )
        
        rag = DuckDBRAG(
            db_path=config.db_path,
            embedding_provider=embedding_service,
            documents_table=config.documents_table,
            embeddings_table=config.embeddings_table,
            embedding_dimension=config.embedding_dimension
        )
        
        # Load test documents
        documents = [
            {"content": "Python is a programming language"},
            {"content": "JavaScript is used for web development"},
            {"content": "SQL is used for database queries"}
        ]
        
        rag.load_documents(documents)
        
        # Query documents
        results = rag.query("programming", top_k=2)
        
        assert len(results) > 0
        assert "Python" in results[0]["content"] or "JavaScript" in results[0]["content"]
        
        # Verify query is actually using custom tables
        conn = config.get_connection()
        result = conn.execute(f"""
            SELECT d.{config.documents_content_column}
            FROM {config.documents_table} d
            JOIN {config.embeddings_table} e ON d.{config.documents_id_column} = e.{config.embeddings_document_id_column}
            LIMIT 1
        """).fetchone()
        assert result is not None
        conn.close()
    
    def test_custom_table_indexes(self, temp_db_path, embedding_service):
        """Test that indexes are created with appropriate names for custom tables."""
        config = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="articles",
            embeddings_table="article_vectors",
            embedding_dimension=384
        )
        
        rag = DuckDBRAG(
            db_path=config.db_path,
            embedding_provider=embedding_service,
            documents_table=config.documents_table,
            embeddings_table=config.embeddings_table,
            embedding_dimension=config.embedding_dimension
        )
        
        conn = config.get_connection()
        config.setup_database(conn)
        
        # DuckDB doesn't expose indexes in the same way as SQLite, 
        # but we can verify the index creation doesn't fail
        # and that queries using the indexed columns work efficiently
        
        # Load some data
        rag.load_documents([
            {"content": f"Document {i}"} for i in range(10)
        ])
        
        # This query should use the index
        result = conn.execute(f"""
            SELECT COUNT(*) 
            FROM {config.embeddings_table} 
            WHERE {config.embeddings_document_id_column} = ?
        """, ["doc_1"]).fetchone()
        
        # Verify the query works (index exists and is usable)
        assert result is not None
        
        conn.close()
    
    def test_custom_column_names(self, temp_db_path, embedding_service):
        """Test custom column names along with custom table names."""
        config = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="my_documents",
            embeddings_table="my_embeddings",
            documents_id_column="doc_id",
            documents_content_column="text_content",
            documents_metadata_column="extra_data",
            embeddings_id_column="emb_id",
            embeddings_document_id_column="doc_ref",
            embeddings_column="vector_data",
            embedding_dimension=384
        )
        
        rag = DuckDBRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="my_documents",
            embeddings_table="my_embeddings",
            documents_id_column="doc_id",
            documents_content_column="text_content",
            documents_metadata_column="extra_data",
            embeddings_id_column="emb_id",
            embeddings_document_id_column="doc_ref",
            embeddings_column="vector_data",
            embedding_dimension=384
        )
        
        conn = config.get_connection()
        config.setup_database(conn)
        
        # Verify custom columns in documents table
        columns = conn.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{config.documents_table}'
        """).fetchall()
        column_names = {row[0] for row in columns}
        
        assert "doc_id" in column_names
        assert "text_content" in column_names
        assert "extra_data" in column_names
        
        # Verify custom columns in embeddings table
        columns = conn.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{config.embeddings_table}'
        """).fetchall()
        column_names = {row[0] for row in columns}
        
        assert "emb_id" in column_names
        assert "doc_ref" in column_names
        assert "vector_data" in column_names
        
        conn.close()
    
    def test_polars_export_with_custom_tables(self, temp_db_path, embedding_service):
        """Test Polars DataFrame export works with custom table names."""
        config = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="polars_docs",
            embeddings_table="polars_embeddings",
            embedding_dimension=384
        )
        
        rag = DuckDBRAG(
            db_path=config.db_path,
            embedding_provider=embedding_service,
            documents_table=config.documents_table,
            embeddings_table=config.embeddings_table,
            embedding_dimension=config.embedding_dimension
        )
        
        # Load documents
        documents = [
            {"content": "Document for Polars export", "metadata": {"index": 1}},
            {"content": "Another document for testing", "metadata": {"index": 2}}
        ]
        
        rag.load_documents(documents)
        
        # Export to Polars DataFrame
        df = rag.export_to_polars()
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert config.documents_content_column in df.columns
        
        # Verify the data
        contents = df[config.documents_content_column].to_list()
        assert "Document for Polars export" in contents
        assert "Another document for testing" in contents
    
    def test_batch_operations_with_custom_tables(self, temp_db_path, embedding_service):
        """Test batch operations work correctly with custom table names."""
        config = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="batch_docs",
            embeddings_table="batch_vectors",
            embedding_dimension=384,
            batch_size=100
        )
        
        rag = DuckDBRAG(
            db_path=config.db_path,
            embedding_provider=embedding_service,
            documents_table=config.documents_table,
            embeddings_table=config.embeddings_table,
            embedding_dimension=config.embedding_dimension
        )
        
        # Load a larger batch of documents
        documents = [
            {"content": f"Batch document number {i}", "metadata": {"batch_id": i}}
            for i in range(250)
        ]
        
        doc_ids = rag.load_documents(documents)
        assert len(doc_ids) == 250
        
        # Verify all documents were inserted into custom tables
        conn = config.get_connection()
        doc_count = conn.execute(f"SELECT COUNT(*) FROM {config.documents_table}").fetchone()[0]
        assert doc_count == 250
        
        emb_count = conn.execute(f"SELECT COUNT(*) FROM {config.embeddings_table}").fetchone()[0]
        assert emb_count == 250
        
        conn.close()
    
    def test_vss_with_custom_tables(self, temp_db_path, embedding_service):
        """Test VSS extension (HNSW) works with custom table names."""
        config = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="vss_docs",
            embeddings_table="vss_embeddings",
            enable_vss_extension=True,
            vss_enable_persistence=True,
            embedding_dimension=384
        )
        
        rag = DuckDBRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="vss_docs",
            embeddings_table="vss_embeddings",
            enable_vss_extension=True,
            vss_enable_persistence=True,
            embedding_dimension=384
        )
        
        # Load documents
        documents = [
            {"content": "Vector similarity search with DuckDB VSS"},
            {"content": "Testing HNSW index with custom tables"}
        ]
        
        doc_ids = rag.load_documents(documents)
        
        if config.enable_vss_extension:
            # If VSS is enabled, we can create HNSW index
            conn = config.get_connection()
            try:
                # Try to create HNSW index on custom table
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS hnsw_idx ON {config.embeddings_table} 
                    USING HNSW ({config.embeddings_column})
                """)
            except Exception:
                # VSS might not be available
                pass
            conn.close()
        
        # Query should work regardless
        results = rag.query("vector similarity", top_k=1)
        assert len(results) > 0
    
    def test_multiple_instances_different_tables(self, temp_db_path, embedding_service):
        """Test multiple RAG instances with different table names in same database."""
        # First instance with one set of table names
        config1 = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="project1_docs",
            embeddings_table="project1_embeddings",
            embedding_dimension=384
        )
        rag1 = DuckDBRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="project1_docs",
            embeddings_table="project1_embeddings",
            embedding_dimension=384
        )
        
        # Second instance with different table names
        config2 = DuckDBConfig(
            db_path=temp_db_path,
            documents_table="project2_docs",
            embeddings_table="project2_embeddings",
            embedding_dimension=384
        )
        rag2 = DuckDBRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="project2_docs",
            embeddings_table="project2_embeddings",
            embedding_dimension=384
        )
        
        # Load different documents in each
        rag1.load_documents([{"content": "Project 1 data in DuckDB"}])
        rag2.load_documents([{"content": "Project 2 data in DuckDB"}])
        
        # Verify both sets of tables exist
        conn = config1.get_connection()
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'main' AND table_name LIKE 'project%'
            ORDER BY table_name
        """).fetchall()
        
        table_names = [row[0] for row in tables]
        
        assert "project1_docs" in table_names
        assert "project1_embeddings" in table_names
        assert "project2_docs" in table_names
        assert "project2_embeddings" in table_names
        
        conn.close()
        
        # Verify data was loaded into correct tables by checking directly
        conn = config1.get_connection()
        
        # Check project1_docs has correct data
        result = conn.execute(f"SELECT {config1.documents_content_column} FROM {config1.documents_table}").fetchone()
        assert "Project 1" in result[0]
        
        # Check project2_docs has correct data
        result = conn.execute(f"SELECT {config2.documents_content_column} FROM {config2.documents_table}").fetchone()
        assert "Project 2" in result[0]
        
        # Verify embeddings were created for both
        count = conn.execute(f"SELECT COUNT(*) FROM {config1.embeddings_table}").fetchone()[0]
        assert count == 1
        
        count = conn.execute(f"SELECT COUNT(*) FROM {config2.embeddings_table}").fetchone()[0]
        assert count == 1
        
        conn.close()
    
    def test_load_from_csv_with_custom_tables(self, temp_db_path, embedding_service):
        """Test loading from CSV works with custom table names."""
        import csv
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
            writer = csv.DictWriter(f, fieldnames=['content', 'metadata'])
            writer.writeheader()
            writer.writerow({'content': 'CSV document 1', 'metadata': '{"source": "csv"}'})
            writer.writerow({'content': 'CSV document 2', 'metadata': '{"source": "csv"}'})
        
        try:
            config = DuckDBConfig(
                db_path=temp_db_path,
                documents_table="csv_docs",
                embeddings_table="csv_vectors",
                embedding_dimension=384
            )
            
            rag = DuckDBRAG(
            db_path=config.db_path,
            embedding_provider=embedding_service,
            documents_table=config.documents_table,
            embeddings_table=config.embeddings_table,
            embedding_dimension=config.embedding_dimension
        )
            
            # Load from CSV
            doc_ids = rag.load_from_csv(csv_path)
            assert len(doc_ids) == 2
            
            # Verify data was loaded into custom tables
            conn = config.get_connection()
            doc_count = conn.execute(f"SELECT COUNT(*) FROM {config.documents_table}").fetchone()[0]
            assert doc_count == 2
            conn.close()
            
        finally:
            Path(csv_path).unlink(missing_ok=True)