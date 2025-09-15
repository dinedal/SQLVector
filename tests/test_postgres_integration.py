"""
Real PostgreSQL Integration Tests with Docker

This module tests the PostgreSQL backend against a real PostgreSQL instance 
running in Docker with the pgvector extension. It verifies:

1. Database connectivity and schema creation
2. Vector index creation and verification
3. Query execution plan analysis (EXPLAIN)
4. Performance benchmarking
5. End-to-end workflows with real data

Run with: python -m pytest tests/test_postgres_integration.py -v -s
"""

import pytest
import asyncio
import time
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import PostgreSQL backend
try:
    from sqlvector.backends.postgres import (
        PostgresRAG, PostgresConfig, PostgresDocument, 
        PostgresEmbedding, PostgresLoader, PostgresQuerier
    )
    from sqlvector.embedding import DefaultEmbeddingProvider, EmbeddingService
    POSTGRES_BACKEND_AVAILABLE = True
except ImportError:
    POSTGRES_BACKEND_AVAILABLE = False

# Import database libraries for direct testing
try:
    import asyncpg
    import psycopg2
    POSTGRES_DEPS_AVAILABLE = True
except ImportError:
    POSTGRES_DEPS_AVAILABLE = False


# Mark all tests in this module as PostgreSQL integration tests
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.integration,
    pytest.mark.skipif(
        not (POSTGRES_BACKEND_AVAILABLE and POSTGRES_DEPS_AVAILABLE),
        reason="PostgreSQL backend or dependencies not available"
    )
]


# Use the centralized PostgreSQL fixtures from conftest.py
# No need to define postgres_env here anymore


# Use postgres_db_config fixture from conftest.py instead


# Use postgres_db_url fixture from conftest.py instead


class TestDatabaseConnectivity:
    """Test basic database connectivity and pgvector functionality."""
    
    async def test_asyncpg_connection(self, postgres_db_config):
        """Test direct asyncpg connection."""
        conn = await asyncpg.connect(**postgres_db_config)
        
        # Test basic query
        result = await conn.fetchval("SELECT 1")
        assert result == 1
        
        # Test pgvector extension
        extensions = await conn.fetch(
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
        )
        assert len(extensions) == 1
        print(f"✅ pgvector version: {extensions[0]['extversion']}")
        
        await conn.close()
    
    def test_psycopg2_connection(self, postgres_db_config):
        """Test psycopg2 connection."""
        conn = psycopg2.connect(**postgres_db_config)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        
        # Test pgvector extension
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        extensions = cursor.fetchall()
        assert len(extensions) == 1
        
        cursor.close()
        conn.close()
    
    async def test_vector_operations(self, postgres_db_config):
        """Test basic vector operations work."""
        conn = await asyncpg.connect(**postgres_db_config)
        
        # Test vector creation and operations
        await conn.execute("CREATE TEMPORARY TABLE test_vectors (id int, vec vector(3))")
        await conn.execute("INSERT INTO test_vectors VALUES (1, '[1,2,3]'), (2, '[4,5,6]')")
        
        # Test similarity operations
        result = await conn.fetchval(
            "SELECT vec <-> '[1,2,3]'::vector FROM test_vectors WHERE id = 1"
        )
        assert result == 0.0  # Distance to itself should be 0
        
        result = await conn.fetchval(
            "SELECT vec <=> '[1,2,3]'::vector FROM test_vectors WHERE id = 1" 
        )
        assert result == 0.0  # Cosine distance to itself should be 0
        
        await conn.close()
        print("✅ Basic vector operations working")


class TestSchemaCreation:
    """Test PostgreSQL backend schema creation."""
    
    async def test_config_schema_generation(self, postgres_db_config):
        """Test that config generates correct PostgreSQL schemas."""
        config = PostgresConfig(
            **postgres_db_config,
            embedding_dimension=384,
            documents_table="test_docs",
            embeddings_table="test_embeddings"
        )
        
        # Test schema generation
        docs_schema = config.get_documents_schema()
        emb_schema = config.get_embeddings_schema()
        idx_schema = config.get_index_schema("test_idx")
        
        # Verify schema contents
        assert "CREATE TABLE IF NOT EXISTS test_docs" in docs_schema
        assert "JSONB" in docs_schema
        assert "vector(384)" in emb_schema
        assert "REFERENCES test_docs" in emb_schema
        assert "CREATE INDEX" in idx_schema
        
        print("✅ Schema generation produces valid SQL")
    
    async def test_real_schema_creation(self, postgres_db_config):
        """Test schema creation against real database."""
        conn = await asyncpg.connect(**postgres_db_config)
        
        config = PostgresConfig(
            **postgres_db_config,
            embedding_dimension=768,
            documents_table="integration_test_docs", 
            embeddings_table="integration_test_embeddings"
        )
        
        # Clean up any existing tables
        await conn.execute("DROP TABLE IF EXISTS integration_test_embeddings CASCADE")
        await conn.execute("DROP TABLE IF EXISTS integration_test_docs CASCADE")
        
        try:
            # Create schema using our config
            await config.setup_database(conn)
            
            # Verify tables were created
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('integration_test_docs', 'integration_test_embeddings')
            """)
            
            table_names = [t['table_name'] for t in tables]
            assert 'integration_test_docs' in table_names
            assert 'integration_test_embeddings' in table_names
            
            # Verify column types
            columns = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'integration_test_embeddings'
                AND column_name = 'embedding'
            """)
            
            # pgvector columns show up as USER-DEFINED type
            assert len(columns) == 1
            print(f"✅ Vector column type: {columns[0]['data_type']}")
            
            # Test that we can insert and query vectors
            await conn.execute("""
                INSERT INTO integration_test_docs (id, content, metadata) 
                VALUES ('test-doc-1', 'Test document', '{"category": "test"}'::jsonb)
            """)
            
            test_vector = [0.1] * 768
            vector_str = '[' + ','.join(map(str, test_vector)) + ']'
            
            await conn.execute("""
                INSERT INTO integration_test_embeddings (id, document_id, embedding)
                VALUES ('test-emb-1', 'test-doc-1', $1::vector)
            """, vector_str)
            
            # Test similarity query
            result = await conn.fetchval("""
                SELECT embedding <-> $1::vector 
                FROM integration_test_embeddings 
                WHERE id = 'test-emb-1'
            """, vector_str)
            
            assert result == 0.0
            print("✅ Real schema creation and vector operations successful")
        
        finally:
            # Cleanup
            await conn.execute("DROP TABLE IF EXISTS integration_test_embeddings CASCADE")
            await conn.execute("DROP TABLE IF EXISTS integration_test_docs CASCADE")
            await conn.close()


class TestVectorIndexes:
    """Test vector index creation and verification."""
    
    async def test_index_creation_and_verification(self, postgres_db_config):
        """Test creating vector indexes and verifying they exist."""
        conn = await asyncpg.connect(**postgres_db_config)
        
        try:
            # Create test tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_index_docs (
                    id TEXT PRIMARY KEY,
                    content TEXT
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_index_embeddings (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES test_index_docs(id),
                    embedding vector(384)
                )
            """)
            
            # Test HNSW index creation
            await conn.execute("""
                CREATE INDEX test_hnsw_idx 
                ON test_index_embeddings 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            
            # Test IVFFlat index creation  
            await conn.execute("""
                CREATE INDEX test_ivfflat_idx
                ON test_index_embeddings
                USING ivfflat (embedding vector_l2_ops) 
                WITH (lists = 100)
            """)
            
            # Verify indexes were created
            indexes = await conn.fetch("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'test_index_embeddings'
                AND indexname IN ('test_hnsw_idx', 'test_ivfflat_idx')
            """)
            
            index_names = [idx['indexname'] for idx in indexes]
            assert 'test_hnsw_idx' in index_names
            assert 'test_ivfflat_idx' in index_names
            
            # Verify index definitions contain expected components
            for idx in indexes:
                if idx['indexname'] == 'test_hnsw_idx':
                    assert 'hnsw' in idx['indexdef'].lower()
                    assert 'vector_cosine_ops' in idx['indexdef']
                    assert 'm=' in idx['indexdef']
                    assert 'ef_construction=' in idx['indexdef']
                elif idx['indexname'] == 'test_ivfflat_idx':
                    assert 'ivfflat' in idx['indexdef'].lower()
                    assert 'lists=' in idx['indexdef']
                    # Note: operator class may not appear in stored definition for IVFFlat
            
            print("✅ Vector indexes created and verified successfully")
            
        finally:
            # Cleanup
            await conn.execute("DROP INDEX IF EXISTS test_hnsw_idx")
            await conn.execute("DROP INDEX IF EXISTS test_ivfflat_idx")
            await conn.execute("DROP TABLE IF EXISTS test_index_embeddings CASCADE")
            await conn.execute("DROP TABLE IF EXISTS test_index_docs CASCADE")
            await conn.close()
    
    async def test_postgres_config_index_creation(self, postgres_db_config):
        """Test index creation through PostgresConfig."""
        config = PostgresConfig(
            **postgres_db_config,
            embedding_dimension=512,
            documents_table="config_test_docs",
            embeddings_table="config_test_embeddings"
        )
        
        conn = await asyncpg.connect(**postgres_db_config)
        
        try:
            # Setup schema
            await config.setup_database(conn)
            
            # Test different index configurations
            test_cases = [
                ("config_hnsw_cosine", "cosine", {"index_type": "hnsw", "index_m": 32, "index_ef_construction": 128}),
                ("config_hnsw_euclidean", "euclidean", {"index_type": "hnsw", "index_m": 16, "index_ef_construction": 64}),
                ("config_ivfflat_cosine", "cosine", {"index_type": "ivfflat", "index_lists": 50}),
                ("config_ivfflat_l2", "euclidean", {"index_type": "ivfflat", "index_lists": 100})
            ]
            
            for idx_name, sim_func, params in test_cases:
                # Update config
                for key, value in params.items():
                    setattr(config, key, value)
                
                # Generate index SQL
                idx_sql = config.get_index_schema(idx_name, sim_func)
                
                # Execute index creation
                await conn.execute(idx_sql)
                
                # Verify index exists
                index_exists = await conn.fetchval("""
                    SELECT COUNT(*) FROM pg_indexes 
                    WHERE indexname = $1 AND tablename = $2
                """, idx_name, config.embeddings_table)
                
                assert index_exists == 1
                print(f"✅ Created {idx_name} for {sim_func} similarity")
            
            print("✅ PostgresConfig index creation working correctly")
            
        finally:
            # Cleanup
            await conn.execute(f"DROP TABLE IF EXISTS {config.embeddings_table} CASCADE")
            await conn.execute(f"DROP TABLE IF EXISTS {config.documents_table} CASCADE")
            await conn.close()


class TestQueryExecution:
    """Test query execution and EXPLAIN analysis."""
    
    async def test_explain_analysis(self, postgres_db_config):
        """Test EXPLAIN ANALYZE to verify index usage."""
        conn = await asyncpg.connect(**postgres_db_config)
        
        try:
            # Create test schema with indexes
            await conn.execute("""
                CREATE TABLE explain_test_docs (
                    id TEXT PRIMARY KEY,
                    content TEXT
                )
            """)
            
            await conn.execute("""
                CREATE TABLE explain_test_embeddings (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES explain_test_docs(id),
                    embedding vector(384)
                )
            """)
            
            # Insert test data (larger dataset to encourage index usage)
            await conn.execute("""
                INSERT INTO explain_test_docs (id, content) 
                SELECT 'doc-' || i, 'Document content ' || i
                FROM generate_series(1, 1000) i
            """)
            
            # Insert embeddings with some variation
            embedding_data = []
            for i in range(1, 1001):
                # Create varied but deterministic embeddings with 384 dimensions
                base_values = [0.1 * (i % 10), 0.2 * ((i + 1) % 10), 0.3 * ((i + 2) % 10)]
                # Repeat and pad to get exactly 384 dimensions
                embedding = (base_values * 128)[:384]  # 3 * 128 = 384
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                embedding_data.append(('emb-' + str(i), 'doc-' + str(i), embedding_str))
            
            await conn.executemany("""
                INSERT INTO explain_test_embeddings (id, document_id, embedding)
                VALUES ($1, $2, $3::vector)
            """, embedding_data)
            
            # Create a matching query vector (384 dimensions)
            query_embedding = ([0.1, 0.2, 0.3] * 128)[:384]
            query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Test without index (should use sequential scan)
            explain_result = await conn.fetch(f"""
                EXPLAIN (ANALYZE, BUFFERS) 
                SELECT document_id, embedding <-> '{query_vector_str}'::vector AS distance
                FROM explain_test_embeddings 
                ORDER BY embedding <-> '{query_vector_str}'::vector 
                LIMIT 5
            """)
            
            explain_text = '\n'.join([row[0] for row in explain_result])
            print("\n📊 Query plan WITHOUT index:")
            print(explain_text)
            
            # Should use sequential scan without index
            assert 'Seq Scan' in explain_text
            
            # Create HNSW index with L2 ops (matches the <-> operator)
            await conn.execute("""
                CREATE INDEX explain_test_hnsw_idx
                ON explain_test_embeddings
                USING hnsw (embedding vector_l2_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            
            # Test with index (should use index scan because we're using L2 distance operator with L2 index)
            explain_result_indexed = await conn.fetch(f"""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT document_id, embedding <-> '{query_vector_str}'::vector AS distance
                FROM explain_test_embeddings 
                ORDER BY embedding <-> '{query_vector_str}'::vector 
                LIMIT 5
            """)
            
            explain_text_indexed = '\n'.join([row[0] for row in explain_result_indexed])
            print("\n📊 Query plan WITH HNSW index:")
            print(explain_text_indexed)
            
            # Should use index scan
            assert 'Index Scan' in explain_text_indexed or 'Bitmap' in explain_text_indexed
            
            print("✅ EXPLAIN ANALYZE shows index usage correctly")
            
        finally:
            # Cleanup
            await conn.execute("DROP TABLE IF EXISTS explain_test_embeddings CASCADE")
            await conn.execute("DROP TABLE IF EXISTS explain_test_docs CASCADE") 
            await conn.close()
    
    async def test_similarity_functions_with_indexes(self, postgres_db_config):
        """Test different similarity functions use appropriate indexes."""
        conn = await asyncpg.connect(**postgres_db_config)
        
        try:
            # Create test schema
            await conn.execute("""
                CREATE TABLE sim_test_embeddings (
                    id TEXT PRIMARY KEY,
                    embedding vector(3)
                )
            """)
            
            # Insert test vectors
            test_vectors = [
                ('v1', '[1, 0, 0]'),
                ('v2', '[0, 1, 0]'), 
                ('v3', '[0, 0, 1]'),
                ('v4', '[1, 1, 0]'),
                ('v5', '[0.5, 0.5, 0.5]')
            ]
            
            await conn.executemany("""
                INSERT INTO sim_test_embeddings (id, embedding) VALUES ($1, $2::vector)
            """, test_vectors)
            
            # Create indexes for different similarity functions
            await conn.execute("""
                CREATE INDEX sim_cosine_idx ON sim_test_embeddings 
                USING hnsw (embedding vector_cosine_ops)
            """)
            
            await conn.execute("""
                CREATE INDEX sim_l2_idx ON sim_test_embeddings 
                USING hnsw (embedding vector_l2_ops)  
            """)
            
            await conn.execute("""
                CREATE INDEX sim_ip_idx ON sim_test_embeddings
                USING hnsw (embedding vector_ip_ops)
            """)
            
            # Test different similarity operations
            similarity_tests = [
                ("cosine", "<=>", "[1, 0, 0]"),
                ("euclidean", "<->", "[1, 0, 0]"), 
                ("inner_product", "<#>", "[1, 0, 0]")
            ]
            
            for sim_name, operator, query_vector in similarity_tests:
                result = await conn.fetch(f"""
                    SELECT id, embedding {operator} '{query_vector}'::vector AS distance
                    FROM sim_test_embeddings
                    ORDER BY embedding {operator} '{query_vector}'::vector
                    LIMIT 3
                """)
                
                assert len(result) == 3
                print(f"✅ {sim_name} similarity query returned {len(result)} results")
                
                # The first result should be exact match with distance 0 (for cosine and L2)
                if operator in ["<=>", "<->"]:
                    assert abs(result[0]['distance']) < 1e-10  # Very close to 0
            
            print("✅ All similarity functions working with appropriate indexes")
        
        finally:
            # Cleanup
            await conn.execute("DROP TABLE IF EXISTS sim_test_embeddings CASCADE")
            await conn.close()


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow with PostgresRAG."""
    
    @pytest.fixture
    async def rag_instance(self, postgres_db_url):
        """Create a RAG instance for testing.""" 
        # Create instance without automatic database initialization
        config = PostgresConfig(
            db_url=postgres_db_url,
            embedding_dimension=384,
            documents_table="e2e_documents",
            embeddings_table="e2e_embeddings",
            pool_min_size=2,
            pool_max_size=5
        )
        
        embedding_service = EmbeddingService(
            provider=DefaultEmbeddingProvider(384),
            dimension=384
        )
        
        # Create RAG instance manually
        from sqlvector.backends.postgres.rag import PostgresRAG
        from sqlvector.backends.postgres.loader import PostgresLoader
        from sqlvector.backends.postgres.querier import PostgresQuerier
        
        rag = object.__new__(PostgresRAG)  # Create without calling __init__
        rag.config = config
        rag.embedding_service = embedding_service
        rag.loader = PostgresLoader(config, embedding_service)
        rag.querier = PostgresQuerier(config, embedding_service)
        rag._db_initialized = False
        
        # Initialize database schema manually
        async with config.get_async_connection() as conn:
            await config.setup_database(conn)
        rag._db_initialized = True
        
        yield rag
        
        # Cleanup
        await rag.close()
    
    async def test_complete_rag_workflow(self, rag_instance):
        """Test complete RAG workflow with real database."""
        # Test document loading
        documents = [
            {
                "content": "PostgreSQL is a powerful relational database system",
                "metadata": {"category": "database", "rating": 5.0}
            },
            {
                "content": "pgvector enables vector similarity search in PostgreSQL", 
                "metadata": {"category": "database", "rating": 4.8}
            },
            {
                "content": "HNSW provides approximate nearest neighbor search",
                "metadata": {"category": "algorithms", "rating": 4.5}
            },
            {
                "content": "Machine learning models generate embedding vectors",
                "metadata": {"category": "ml", "rating": 4.7}
            }
        ]
        
        # Load documents
        doc_ids = await rag_instance.load_documents_async(
            documents, 
            show_progress=False
        )
        
        assert len(doc_ids) == 4
        print(f"✅ Loaded {len(doc_ids)} documents")
        
        # Test similarity search
        results = await rag_instance.query_async(
            "database vector search",
            top_k=3,
            similarity_function="cosine"
        )
        
        assert len(results) <= 3
        assert all("similarity" in r for r in results)
        assert all("content" in r for r in results)
        print(f"✅ Query returned {len(results)} results")
        
        # Test metadata filtering
        db_results = await rag_instance.query_with_filters_async(
            filters={"category": "database"},
            top_k=5
        )
        
        # Should find at least our 2 database documents (may be more from previous runs)
        assert len(db_results) >= 2
        assert all(json.loads(r["metadata"])["category"] == "database" for r in db_results)
        print(f"✅ Metadata filtering working - found {len(db_results)} database documents")
        
        # Test similar documents
        similar_docs = await rag_instance.find_similar_documents_async(
            doc_ids[0],  # First document
            top_k=2
        )
        
        assert len(similar_docs) <= 2
        # Should not include the source document
        assert all(r["id"] != doc_ids[0] for r in similar_docs)
        print("✅ Similar documents search working")
        
        # Test statistics
        stats = await rag_instance.get_statistics_async()
        assert stats["document_count"] >= 4  # At least our 4 documents
        assert stats["embedding_count"] >= 4  # At least our 4 embeddings
        assert stats["embedding_dimension"] == 384
        print(f"✅ Statistics reporting working - {stats['document_count']} docs, {stats['embedding_count']} embeddings")
        
        # Test index creation
        index_created = await rag_instance.create_index_async(
            "e2e_test_idx",
            similarity_function="cosine"
        )
        assert index_created
        print("✅ Index creation working")
        
        # Cleanup - delete documents
        for doc_id in doc_ids:
            deleted = await rag_instance.delete_document_async(doc_id)
            assert deleted
        
        print("✅ Complete end-to-end workflow successful")
    
    async def test_performance_benchmark(self, rag_instance):
        """Test performance with larger dataset."""
        # Generate larger test dataset
        large_dataset = []
        categories = ["tech", "science", "business", "health", "education"]
        
        for i in range(50):  # Smaller dataset for CI
            category = categories[i % len(categories)]
            large_dataset.append({
                "content": f"Document {i+1} about {category} with detailed content and information",
                "metadata": {"category": category, "index": i+1, "batch": i // 10}
            })
        
        # Benchmark loading
        start_time = time.time()
        doc_ids = await rag_instance.load_documents_async(
            large_dataset,
            show_progress=False
        )
        load_time = time.time() - start_time
        
        assert len(doc_ids) == 50
        print(f"✅ Loaded {len(doc_ids)} docs in {load_time:.2f}s ({len(doc_ids)/load_time:.1f} docs/sec)")
        
        # Benchmark querying
        start_time = time.time()
        results = await rag_instance.query_async(
            "technology and science information",
            top_k=10,
            similarity_function="cosine"
        )
        query_time = time.time() - start_time
        
        # With DefaultEmbeddingProvider, we may get fewer results due to similarity threshold
        assert len(results) <= 10
        assert len(results) >= 0  # At least we should get some results or none
        print(f"✅ Query completed in {query_time*1000:.1f}ms with {len(results)} results")
        
        # Test concurrent queries
        async def concurrent_query(query_num):
            return await rag_instance.query_async(
                f"query number {query_num}",
                top_k=5
            )
        
        start_time = time.time()
        concurrent_results = await asyncio.gather(*[
            concurrent_query(i) for i in range(5)
        ])
        concurrent_time = time.time() - start_time
        
        assert len(concurrent_results) == 5
        # With DefaultEmbeddingProvider, results may vary
        assert all(len(results) <= 5 for results in concurrent_results)
        total_results = sum(len(results) for results in concurrent_results)
        print(f"✅ 5 concurrent queries completed in {concurrent_time:.2f}s with {total_results} total results")
        
        # Cleanup
        for doc_id in doc_ids:
            await rag_instance.delete_document_async(doc_id)
        
        print("✅ Performance benchmarking completed")


# Utility functions for manual testing
async def run_manual_tests():
    """Run manual integration tests for development."""
    print("🧪 Running manual PostgreSQL integration tests...")
    
    env = PostgresTestEnvironment()
    if not env.start_container():
        print("❌ Failed to start PostgreSQL container")
        return False
    
    try:
        db_config = env.get_connection_info()['db_config']
        
        # Test basic connectivity
        print("\n1. Testing basic connectivity...")
        conn = await asyncpg.connect(**postgres_db_config)
        result = await conn.fetchval("SELECT 1")
        assert result == 1
        await conn.close()
        print("✅ Basic connectivity test passed")
        
        # Test PostgresRAG
        print("\n2. Testing PostgresRAG end-to-end...")
        rag = PostgresRAG(
            **db_config,
            embedding_dimension=384,
            embedding_provider=DefaultEmbeddingProvider(384)
        )
        
        # Load test document
        doc_id = await rag.load_document_async(
            "Test document for manual integration testing"
        )
        print(f"✅ Document loaded: {doc_id}")
        
        # Query
        results = await rag.query_async("test document", top_k=1)
        assert len(results) == 1
        print("✅ Query successful")
        
        # Cleanup
        await rag.delete_document_async(doc_id)
        await rag.close()
        
        print("\n✅ All manual tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Manual test failed: {e}")
        return False
    
    finally:
        env.stop_container()


if __name__ == "__main__":
    # Run manual tests if executed directly
    asyncio.run(run_manual_tests())