"""Example usage of the PostgreSQL RAG backend with pgvector."""

import asyncio
import sys
import os

# Add parent directory to path to import sqlvector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlvector.backends.postgres import PostgresRAG
from sqlvector.embedding import DefaultEmbeddingProvider


def main():
    """Main example function demonstrating PostgreSQL RAG usage."""
    print("PostgreSQL RAG Backend Example")
    print("=" * 60)
    
    # Database connection - adjust these settings for your PostgreSQL instance
    # You can also use a full connection URL instead of individual parameters
    DB_CONFIG = {
        # Option 1: Individual parameters
        "host": "localhost",
        "port": 5432,
        "user": "postgres", 
        "password": "password",
        "database": "sqlvector_demo",
        
        # Option 2: Connection URL (comment out individual params above if using this)
        # "db_url": "postgresql://postgres:password@localhost:5432/sqlvector_demo",
    }
    
    # Run the async example
    asyncio.run(async_example(DB_CONFIG))
    
    print("\n=== Sync Example ===")
    sync_example(DB_CONFIG)


async def async_example(db_config):
    """Asynchronous example demonstrating PostgreSQL RAG features."""
    print("\n=== Async PostgreSQL RAG Example ===")
    
    try:
        # Initialize PostgreSQL RAG with pgvector support
        async with PostgresRAG(
            **db_config,
            embedding_dimension=384,
            batch_size=1000,
            # pgvector index configuration
            index_type="hnsw",  # or "ivfflat"
            index_m=16,         # HNSW parameter
            index_ef_construction=64,  # HNSW parameter
            # Connection pooling
            pool_min_size=2,
            pool_max_size=10,
            embedding_provider=DefaultEmbeddingProvider(384)
        ) as rag:
            
            print("Connected to PostgreSQL with pgvector support")
            
            # === Document Loading ===
            print("\n=== Loading Documents ===")
            
            # Load individual document
            doc_id1 = await rag.load_document_async(
                content="PostgreSQL is a powerful open-source relational database",
                metadata={"category": "database", "type": "relational", "rating": 5.0}
            )
            print(f"Loaded document: {doc_id1}")
            
            # Load batch of documents
            documents = [
                {
                    "content": "pgvector enables vector similarity search in PostgreSQL",
                    "metadata": {"category": "database", "type": "extension", "rating": 4.8}
                },
                {
                    "content": "HNSW indexing provides fast approximate nearest neighbor search",
                    "metadata": {"category": "algorithms", "type": "indexing", "rating": 4.5}
                },
                {
                    "content": "Machine learning embeddings capture semantic meaning",
                    "metadata": {"category": "AI", "type": "embeddings", "rating": 4.9}
                },
                {
                    "content": "Vector databases are essential for RAG applications",
                    "metadata": {"category": "AI", "type": "infrastructure", "rating": 4.7}
                },
                {
                    "content": "Cosine similarity measures angle between vectors",
                    "metadata": {"category": "mathematics", "type": "metric", "rating": 4.3}
                }
            ]
            
            doc_ids = await rag.load_documents_async(
                documents, 
                generate_embeddings=True,
                show_progress=True
            )
            print(f"Loaded {len(doc_ids)} documents in batch")
            
            # === Vector Index Management ===
            print("\n=== Index Management ===")
            
            # Create HNSW index for cosine similarity
            index_created = await rag.create_index_async(
                "hnsw_cosine_idx",
                similarity_function="cosine"
            )
            print(f"HNSW index created: {index_created}")
            
            # Create IVFFlat index for euclidean distance  
            rag.config.index_type = "ivfflat"
            rag.config.index_lists = 50
            ivf_index_created = await rag.create_index_async(
                "ivf_euclidean_idx", 
                similarity_function="euclidean"
            )
            print(f"IVFFlat index created: {ivf_index_created}")
            
            # === Similarity Search ===
            print("\n=== Similarity Search ===")
            
            # Query with different similarity functions
            similarity_functions = ["cosine", "euclidean", "inner_product"]
            
            for sim_func in similarity_functions:
                print(f"\n--- {sim_func.title()} Similarity ---")
                results = await rag.query_async(
                    "vector search database technology",
                    top_k=3,
                    similarity_threshold=0.1,
                    similarity_function=sim_func
                )
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['content'][:60]}...")
                    print(f"   Similarity: {result['similarity']:.4f}")
                    print(f"   Category: {result['metadata'].get('category', 'N/A')}")
            
            # === Metadata Filtering ===
            print("\n=== Metadata Filtering ===")
            
            # Filter by category
            db_results = await rag.query_with_filters_async(
                filters={"category": "database"},
                query_text="powerful search capabilities",
                top_k=2
            )
            print(f"\nDatabase category results: {len(db_results)}")
            for result in db_results:
                print(f"- {result['content'][:50]}... (rating: {result['metadata'].get('rating')})")
            
            # Filter by rating (exact match)
            high_rated = await rag.query_with_filters_async(
                filters={"rating": 4.9},
                top_k=5
            )
            print(f"\nHigh-rated results: {len(high_rated)}")
            
            # === Batch Operations ===
            print("\n=== Batch Operations ===")
            
            # Batch query
            queries = [
                "database indexing performance",
                "artificial intelligence embeddings", 
                "vector similarity algorithms"
            ]
            
            batch_results = await rag.query_batch_async(
                queries,
                top_k=2,
                similarity_function="cosine"
            )
            
            for i, query_results in enumerate(batch_results):
                print(f"\nQuery {i+1}: '{queries[i]}'")
                for j, result in enumerate(query_results, 1):
                    print(f"  {j}. {result['content'][:40]}... ({result['similarity']:.3f})")
            
            # === Document Operations ===
            print("\n=== Document Operations ===")
            
            # Retrieve specific document
            doc = await rag.get_document_async(doc_ids[0])
            print(f"Retrieved document: {doc['content'][:50]}...")
            
            # Get multiple documents
            multi_docs = await rag.get_documents_async(doc_ids[:2])
            print(f"Retrieved {len(multi_docs)} documents")
            
            # Find similar documents
            similar_docs = await rag.find_similar_documents_async(
                doc_ids[0],
                top_k=2,
                similarity_function="cosine"
            )
            print(f"Found {len(similar_docs)} similar documents")
            for doc in similar_docs:
                print(f"- {doc['content'][:40]}... (similarity: {doc['similarity']:.3f})")
            
            # === Statistics ===
            print("\n=== Database Statistics ===")
            stats = await rag.get_statistics_async()
            print(f"Documents: {stats['document_count']}")
            print(f"Embeddings: {stats['embedding_count']}")
            print(f"Dimension: {stats['embedding_dimension']}")
            print(f"Indexes: {len(stats['indexes'])}")
            
            for idx in stats['indexes']:
                print(f"- {idx['indexname']}: {idx['indexdef'][:50]}...")
            
            # === Cleanup ===
            print("\n=== Cleanup ===")
            
            # Delete some documents
            for doc_id in doc_ids[-2:]:
                deleted = await rag.delete_document_async(doc_id)
                print(f"Deleted document {doc_id}: {deleted}")
            
            # Delete indexes
            await rag.delete_index_async("hnsw_cosine_idx")
            await rag.delete_index_async("ivf_euclidean_idx")
            print("Deleted test indexes")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. PostgreSQL is running and accessible")
        print("2. Database exists and user has proper permissions")
        print("3. pgvector extension is installed: CREATE EXTENSION vector;")
        print("4. Required Python packages are installed:")
        print("   pip install sqlvector[postgres]")


def sync_example(db_config):
    """Synchronous example for comparison."""
    try:
        # Initialize with sync mode
        with PostgresRAG(
            **db_config,
            embedding_dimension=384,
            embedding_provider=DefaultEmbeddingProvider(384)
        ) as rag:
            
            print("Connected to PostgreSQL (sync mode)")
            
            # Load document synchronously
            doc_id = rag.load_document(
                content="Sync example: PostgreSQL with pgvector is awesome!",
                metadata={"mode": "sync", "example": True}
            )
            print(f"Loaded document (sync): {doc_id}")
            
            # Query synchronously
            results = rag.query(
                "PostgreSQL database example",
                top_k=2,
                similarity_function="cosine"
            )
            print(f"Found {len(results)} results (sync)")
            
            for result in results:
                print(f"- {result['content'][:50]}...")
            
            # Cleanup
            rag.delete_document(doc_id)
            print("Deleted sync example document")
            
    except Exception as e:
        print(f"Sync example error: {e}")


if __name__ == "__main__":
    main()