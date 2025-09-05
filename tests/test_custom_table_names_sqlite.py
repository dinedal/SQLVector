"""Tests for custom table names with SQLite backend."""

import pytest
import tempfile
from pathlib import Path
import sqlite3

from sqlvector.backends.sqlite import (
    SQLiteRAG,
    SQLiteConfig,
    SQLiteDocument
)
from sqlvector.embedding import DefaultEmbeddingProvider


class TestSQLiteCustomTableNames:
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service."""
        return DefaultEmbeddingProvider(dimension=384)
    
    def test_custom_table_creation(self, temp_db_path, embedding_service):
        """Test that custom table names are actually used in the database."""
        config = SQLiteConfig(
            db_path=temp_db_path,
            documents_table="my_custom_docs",
            embeddings_table="my_custom_embeddings",
            vss_table="my_custom_vss",
            embedding_dimension=384
        )
        
        rag = SQLiteRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="my_custom_docs",
            embeddings_table="my_custom_embeddings",
            vss_table="my_custom_vss",
            embedding_dimension=384
        )
        
        # Initialize database
        with config.get_connection() as conn:
            config.setup_database(conn)
            
            # Verify custom tables were created
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "my_custom_docs" in tables
            assert "my_custom_embeddings" in tables
            # Note: VSS virtual table might not appear if extension isn't loaded
            
            # Verify default table names were NOT created
            assert "documents" not in tables
            assert "embeddings" not in tables
    
    def test_custom_table_operations(self, temp_db_path, embedding_service):
        """Test CRUD operations with custom table names."""
        config = SQLiteConfig(
            db_path=temp_db_path,
            documents_table="custom_documents",
            embeddings_table="custom_vectors",
            embedding_dimension=384
        )
        
        rag = SQLiteRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="custom_documents",
            embeddings_table="custom_vectors",
            embedding_dimension=384
        )
        
        # Load documents
        documents = [
            {
                "content": "First document with custom tables",
                "metadata": {"type": "test", "custom_table": True}
            },
            {
                "content": "Second document in custom schema",
                "metadata": {"type": "test", "index": 2}
            }
        ]
        
        doc_ids = rag.load_documents(documents)
        assert len(doc_ids) == 2
        
        # Verify data was inserted into custom tables
        with config.get_connection() as conn:
            # Check documents table
            cursor = conn.execute(f"SELECT COUNT(*) FROM {config.documents_table}")
            doc_count = cursor.fetchone()[0]
            assert doc_count == 2
            
            # Check embeddings table
            cursor = conn.execute(f"SELECT COUNT(*) FROM {config.embeddings_table}")
            emb_count = cursor.fetchone()[0]
            assert emb_count == 2
            
            # Verify content
            cursor = conn.execute(
                f"SELECT {config.documents_content_column} FROM {config.documents_table} ORDER BY {config.documents_id_column}"
            )
            contents = [row[0] for row in cursor.fetchall()]
            assert "First document with custom tables" in contents
            assert "Second document in custom schema" in contents
    
    def test_custom_table_querying(self, temp_db_path, embedding_service):
        """Test querying with custom table names."""
        config = SQLiteConfig(
            db_path=temp_db_path,
            documents_table="knowledge_base",
            embeddings_table="vector_store",
            embedding_dimension=384
        )
        
        rag = SQLiteRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="knowledge_base",
            embeddings_table="vector_store",
            embedding_dimension=384
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
        with config.get_connection() as conn:
            # This should work with custom table names
            cursor = conn.execute(f"""
                SELECT d.{config.documents_content_column}
                FROM {config.documents_table} d
                JOIN {config.embeddings_table} e ON d.{config.documents_id_column} = e.{config.embeddings_document_id_column}
                LIMIT 1
            """)
            result = cursor.fetchone()
            assert result is not None
    
    def test_custom_table_indexes(self, temp_db_path, embedding_service):
        """Test that indexes are created with appropriate names for custom tables."""
        config = SQLiteConfig(
            db_path=temp_db_path,
            documents_table="articles",
            embeddings_table="article_vectors",
            embedding_dimension=384
        )
        
        rag = SQLiteRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="articles",
            embeddings_table="article_vectors",
            embedding_dimension=384
        )
        
        with config.get_connection() as conn:
            config.setup_database(conn)
            
            # Check indexes
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name LIKE 'idx_%'
                ORDER BY name
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Indexes should use custom table names
            assert f"idx_articles_hash" in indexes
            assert f"idx_article_vectors_hash" in indexes
            assert f"idx_article_vectors_{config.embeddings_document_id_column}" in indexes
    
    def test_custom_column_names(self, temp_db_path, embedding_service):
        """Test custom column names along with custom table names."""
        config = SQLiteConfig(
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
        
        rag = SQLiteRAG(
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
        
        with config.get_connection() as conn:
            config.setup_database(conn)
            
            # Verify custom columns in documents table
            cursor = conn.execute(f"PRAGMA table_info({config.documents_table})")
            columns = {row[1] for row in cursor.fetchall()}
            
            assert "doc_id" in columns
            assert "text_content" in columns
            assert "extra_data" in columns
            
            # Verify custom columns in embeddings table
            cursor = conn.execute(f"PRAGMA table_info({config.embeddings_table})")
            columns = {row[1] for row in cursor.fetchall()}
            
            assert "emb_id" in columns
            assert "doc_ref" in columns
            assert "vector_data" in columns
    
    def test_vss_with_custom_tables(self, temp_db_path, embedding_service):
        """Test VSS extension works with custom table names."""
        config = SQLiteConfig(
            db_path=temp_db_path,
            documents_table="vss_docs",
            embeddings_table="vss_embeddings",
            vss_table="custom_vss_index",
            enable_vss_extension=True,
            embedding_dimension=384
        )
        
        rag = SQLiteRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="vss_docs",
            embeddings_table="vss_embeddings",
            vss_table="custom_vss_index",
            enable_vss_extension=True,
            embedding_dimension=384
        )
        
        # Load documents
        documents = [
            {"content": "Vector similarity search test"},
            {"content": "Testing custom VSS table names"}
        ]
        
        # This will use VSS if available
        doc_ids = rag.load_documents(documents)
        
        if config.enable_vss_extension:
            # VSS was loaded successfully
            with config.get_connection() as conn:
                # Check if VSS virtual table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (config.vss_table,))
                result = cursor.fetchone()
                if result:
                    assert result[0] == "custom_vss_index"
        
        # Query should still work regardless of VSS
        results = rag.query("vector search", top_k=1)
        assert len(results) > 0
    
    def test_multiple_instances_different_tables(self, temp_db_path, embedding_service):
        """Test multiple RAG instances with different table names in same database."""
        # First instance with one set of table names
        config1 = SQLiteConfig(
            db_path=temp_db_path,
            documents_table="project1_docs",
            embeddings_table="project1_embeddings",
            embedding_dimension=384
        )
        rag1 = SQLiteRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="project1_docs",
            embeddings_table="project1_embeddings",
            embedding_dimension=384
        )
        
        # Second instance with different table names
        config2 = SQLiteConfig(
            db_path=temp_db_path,
            documents_table="project2_docs",
            embeddings_table="project2_embeddings",
            embedding_dimension=384
        )
        rag2 = SQLiteRAG(
            db_path=temp_db_path,
            embedding_provider=embedding_service,
            documents_table="project2_docs",
            embeddings_table="project2_embeddings",
            embedding_dimension=384
        )
        
        # Load different documents in each
        rag1.load_documents([{"content": "Project 1 data from first RAG instance"}])
        rag2.load_documents([{"content": "Project 2 data from second RAG instance"}])
        
        # Verify both sets of tables exist
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'project%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "project1_docs" in tables
            assert "project1_embeddings" in tables
            assert "project2_docs" in tables
            assert "project2_embeddings" in tables
        
        # Verify data was loaded into correct tables by checking directly
        with sqlite3.connect(temp_db_path) as conn:
            # Check project1_docs has correct data
            cursor = conn.execute(f"SELECT {config1.documents_content_column} FROM {config1.documents_table}")
            project1_content = cursor.fetchone()[0]
            assert "Project 1" in project1_content
            
            # Check project2_docs has correct data  
            cursor = conn.execute(f"SELECT {config2.documents_content_column} FROM {config2.documents_table}")
            project2_content = cursor.fetchone()[0]
            assert "Project 2" in project2_content
            
            # Verify embeddings were created for both
            cursor = conn.execute(f"SELECT COUNT(*) FROM {config1.embeddings_table}")
            assert cursor.fetchone()[0] == 1
            
            cursor = conn.execute(f"SELECT COUNT(*) FROM {config2.embeddings_table}")
            assert cursor.fetchone()[0] == 1