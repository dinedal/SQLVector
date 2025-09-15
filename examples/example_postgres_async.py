"""Advanced async PostgreSQL RAG example with connection pooling and concurrent operations."""

import asyncio
import time
import sys
import os

# Add parent directory to path to import sqlvector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlvector.backends.postgres import PostgresRAG
from sqlvector.embedding import DefaultEmbeddingProvider


async def main():
    """Main async example demonstrating advanced PostgreSQL features."""
    print("Advanced PostgreSQL RAG Async Example")
    print("=" * 60)
    
    # Connection configuration
    DB_URL = "postgresql://postgres:password@localhost:5432/sqlvector_async_demo"
    
    await advanced_async_operations(DB_URL)
    await concurrent_operations_example(DB_URL)
    await connection_pooling_example(DB_URL)


async def advanced_async_operations(db_url):
    """Demonstrate advanced async operations."""
    print("\n=== Advanced Async Operations ===")
    
    async with PostgresRAG(
        db_url=db_url,
        embedding_dimension=512,
        batch_size=500,
        # Optimized connection pool
        pool_min_size=5,
        pool_max_size=20,
        # Advanced pgvector indexing
        index_type="hnsw",
        index_m=32,  # Higher M for better recall
        index_ef_construction=128,  # Higher ef_construction for better index quality
        embedding_provider=DefaultEmbeddingProvider(512)
    ) as rag:
        
        print("Connected with optimized pool settings")
        
        # === Large Batch Loading ===
        print("\n--- Large Batch Loading ---")
        
        # Generate a large dataset
        large_dataset = []
        topics = ["AI", "database", "web", "mobile", "security", "cloud", "ML", "data"]
        
        for i in range(100):
            topic = topics[i % len(topics)]
            large_dataset.append({
                "content": f"Advanced {topic} technology article #{i+1} discussing modern approaches and implementations",
                "metadata": {
                    "topic": topic,
                    "article_id": i+1,
                    "complexity": "advanced",
                    "word_count": 1000 + (i % 500)
                }
            })
        
        # Load with progress tracking
        start_time = time.time()
        doc_ids = await rag.load_documents_async(
            large_dataset,
            generate_embeddings=True,
            show_progress=True
        )
        load_time = time.time() - start_time
        print(f"Loaded {len(doc_ids)} documents in {load_time:.2f} seconds")
        print(f"Rate: {len(doc_ids)/load_time:.1f} docs/second")
        
        # === Advanced Indexing ===
        print("\n--- Advanced Vector Indexing ---")
        
        # Create multiple indexes for different similarity functions
        index_configs = [
            ("hnsw_cosine_advanced", "cosine", {"index_type": "hnsw", "index_m": 32, "index_ef_construction": 128}),
            ("hnsw_euclidean", "euclidean", {"index_type": "hnsw", "index_m": 16, "index_ef_construction": 64}),
            ("ivfflat_inner_product", "inner_product", {"index_type": "ivfflat", "index_lists": 100})
        ]
        
        for idx_name, sim_func, config in index_configs:
            # Update config
            for key, value in config.items():
                setattr(rag.config, key, value)
            
            created = await rag.create_index_async(idx_name, similarity_function=sim_func)
            print(f"Created {idx_name}: {created}")
        
        # === High-Performance Querying ===
        print("\n--- High-Performance Querying ---")
        
        # Benchmark different similarity functions
        query_text = "machine learning artificial intelligence data processing"
        
        for sim_func in ["cosine", "euclidean", "inner_product"]:
            start_time = time.time()
            results = await rag.query_async(
                query_text,
                top_k=10,
                similarity_function=sim_func
            )
            query_time = time.time() - start_time
            
            print(f"{sim_func.title()} similarity: {len(results)} results in {query_time*1000:.1f}ms")
            if results:
                print(f"  Best match: {results[0]['content'][:50]}... (score: {results[0]['similarity']:.4f})")
        
        # === Complex Metadata Filtering ===
        print("\n--- Complex Metadata Filtering ---")
        
        # Filter by multiple criteria
        complex_filters = [
            {"topic": "AI", "complexity": "advanced"},
            {"topic": "database"},
            {"word_count": 1200}  # Exact match
        ]
        
        for i, filters in enumerate(complex_filters, 1):
            results = await rag.query_with_filters_async(
                filters=filters,
                query_text="advanced technology solutions",
                top_k=5
            )
            print(f"Filter set {i} ({filters}): {len(results)} results")
            
            for result in results[:2]:  # Show top 2
                meta = result['metadata']
                print(f"  - Topic: {meta['topic']}, Words: {meta['word_count']}, ID: {meta['article_id']}")
        
        # === Statistical Analysis ===
        print("\n--- Database Statistics ---")
        stats = await rag.get_statistics_async()
        
        print(f"Total documents: {stats['document_count']:,}")
        print(f"Total embeddings: {stats['embedding_count']:,}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Active indexes: {len(stats['indexes'])}")
        
        # Calculate storage statistics
        vector_size_mb = (stats['embedding_count'] * stats['embedding_dimension'] * 4) / (1024 * 1024)
        print(f"Estimated vector storage: {vector_size_mb:.1f} MB")
        
        # Cleanup for next example
        print("\n--- Partial Cleanup ---")
        cleanup_count = len(doc_ids) // 2
        for doc_id in doc_ids[:cleanup_count]:
            await rag.delete_document_async(doc_id)
        print(f"Deleted {cleanup_count} documents for next example")
        
        return doc_ids[cleanup_count:]  # Return remaining IDs


async def concurrent_operations_example(db_url):
    """Demonstrate concurrent operations and connection pooling."""
    print("\n=== Concurrent Operations Example ===")
    
    async with PostgresRAG(
        db_url=db_url,
        embedding_dimension=256,  # Smaller for faster processing
        pool_min_size=10,  # Larger pool for concurrency
        pool_max_size=30,
        embedding_provider=DefaultEmbeddingProvider(256)
    ) as rag:
        
        print("Testing concurrent operations with connection pooling")
        
        # === Concurrent Document Loading ===
        print("\n--- Concurrent Document Loading ---")
        
        async def load_batch(batch_id, size=10):
            """Load a batch of documents concurrently."""
            documents = [
                {
                    "content": f"Concurrent batch {batch_id} document {i} about distributed systems",
                    "metadata": {"batch_id": batch_id, "doc_index": i, "category": "distributed"}
                }
                for i in range(size)
            ]
            
            start_time = time.time()
            doc_ids = await rag.load_documents_async(documents, show_progress=False)
            duration = time.time() - start_time
            
            print(f"Batch {batch_id}: {len(doc_ids)} docs in {duration:.2f}s")
            return doc_ids
        
        # Run multiple batches concurrently
        tasks = [load_batch(i, size=20) for i in range(5)]
        batch_results = await asyncio.gather(*tasks)
        
        total_docs = sum(len(batch) for batch in batch_results)
        print(f"Loaded {total_docs} documents across {len(batch_results)} concurrent batches")
        
        # === Concurrent Querying ===
        print("\n--- Concurrent Querying ---")
        
        async def concurrent_query(query_id, query_text):
            """Execute a query concurrently."""
            start_time = time.time()
            results = await rag.query_async(
                query_text,
                top_k=5,
                similarity_function="cosine"
            )
            duration = time.time() - start_time
            
            print(f"Query {query_id}: {len(results)} results in {duration*1000:.0f}ms")
            return results
        
        # Run multiple queries simultaneously
        queries = [
            "distributed systems architecture",
            "concurrent processing algorithms", 
            "database performance optimization",
            "system scalability patterns",
            "parallel computing frameworks"
        ]
        
        start_time = time.time()
        query_tasks = [
            concurrent_query(i, query) 
            for i, query in enumerate(queries, 1)
        ]
        query_results = await asyncio.gather(*query_tasks)
        total_time = time.time() - start_time
        
        total_results = sum(len(results) for results in query_results)
        print(f"Executed {len(queries)} concurrent queries in {total_time:.2f}s")
        print(f"Total results: {total_results}")
        print(f"Average query time: {total_time/len(queries)*1000:.0f}ms")
        
        # === Concurrent Mixed Operations ===
        print("\n--- Mixed Concurrent Operations ---")
        
        async def mixed_operations():
            """Mix of different operations."""
            operations = []
            
            # Add some queries
            for i in range(3):
                operations.append(
                    rag.query_async(f"system operation {i}", top_k=3)
                )
            
            # Add some document retrievals
            all_doc_ids = [doc_id for batch in batch_results for doc_id in batch]
            for doc_id in all_doc_ids[:5]:
                operations.append(
                    rag.get_document_async(doc_id)
                )
            
            # Add some similarity searches
            if all_doc_ids:
                for doc_id in all_doc_ids[:2]:
                    operations.append(
                        rag.find_similar_documents_async(doc_id, top_k=3)
                    )
            
            start_time = time.time()
            results = await asyncio.gather(*operations)
            duration = time.time() - start_time
            
            print(f"Executed {len(operations)} mixed operations in {duration:.2f}s")
            return results
        
        await mixed_operations()


async def connection_pooling_example(db_url):
    """Demonstrate connection pool monitoring and management."""
    print("\n=== Connection Pool Management ===")
    
    # Create RAG with detailed pool configuration
    rag = PostgresRAG(
        db_url=db_url,
        embedding_dimension=128,
        pool_min_size=3,
        pool_max_size=15,
        embedding_provider=DefaultEmbeddingProvider(128)
    )
    
    try:
        print("Monitoring connection pool behavior")
        
        # === Pool Stress Test ===
        async def pool_stress_test():
            """Stress test the connection pool."""
            
            async def db_operation(op_id):
                """Single database operation."""
                await asyncio.sleep(0.1)  # Simulate work
                
                # Perform a lightweight query
                results = await rag.query_async(f"test operation {op_id}", top_k=1)
                return len(results)
            
            # Run many operations simultaneously
            num_operations = 50
            print(f"Running {num_operations} simultaneous operations...")
            
            start_time = time.time()
            tasks = [db_operation(i) for i in range(num_operations)]
            results = await asyncio.gather(*tasks)
            duration = time.time() - start_time
            
            print(f"Completed {len(results)} operations in {duration:.2f}s")
            print(f"Average operation time: {duration/len(results)*1000:.1f}ms")
            print(f"Operations per second: {len(results)/duration:.1f}")
            
        await pool_stress_test()
        
        # === Connection Lifecycle ===
        print("\n--- Connection Lifecycle Test ---")
        
        # Test rapid connection acquisition/release
        async def rapid_connections():
            for i in range(10):
                async with rag.config.get_async_connection() as conn:
                    # Verify connection is working
                    if hasattr(conn, 'fetchval'):
                        result = await conn.fetchval("SELECT 1")
                        assert result == 1
                        print(f"Connection {i+1}: OK")
                await asyncio.sleep(0.05)  # Brief pause
        
        await rapid_connections()
        
        print("Connection pool stress test completed successfully")
        
    finally:
        # Cleanup
        await rag.close()
        print("Connection pool closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"\nExample failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is running: sudo service postgresql start")
        print("2. Create the database: createdb sqlvector_async_demo")
        print("3. Install pgvector: CREATE EXTENSION vector;")
        print("4. Install dependencies: pip install sqlvector[postgres]")
        print("5. Check connection settings in the script")