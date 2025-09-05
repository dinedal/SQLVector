"""Example usage of DuckDB RAG backend with Vector Similarity Search (VSS) extension."""

import time
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path to import sqlvector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlvector.backends.duckdb import DuckDBRAG
from sqlvector.embedding import DefaultEmbeddingProvider


def main():
    print("DuckDB RAG with Vector Similarity Search Extension")
    print("=" * 60)
    
    # Create a temporary database file path
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        db_path = f.name
    
    try:
        # Initialize DuckDB RAG with VSS extension enabled
        print("=== Initializing DuckDB RAG with VSS Extension ===")
        rag = DuckDBRAG(
            db_path=db_path, 
            embedding_dimension=384,
            enable_vss_extension=True,  # Enable Vector Similarity Search
            vss_enable_persistence=True  # Enable experimental persistence
        )
        
        print(f"VSS Extension enabled: {rag.config.enable_vss_extension}")
        print(f"VSS Persistence enabled: {rag.config.vss_enable_persistence}")
        
        # Load a larger dataset for demonstrating index benefits
        print("\n=== Loading Sample Documents ===")
        documents = [
            {"content": "Machine learning algorithms are transforming artificial intelligence", "metadata": {"category": "AI", "complexity": "high"}},
            {"content": "Deep neural networks enable pattern recognition in large datasets", "metadata": {"category": "AI", "complexity": "high"}},
            {"content": "Natural language processing helps computers understand human text", "metadata": {"category": "AI", "complexity": "medium"}},
            {"content": "Computer vision allows machines to interpret visual information", "metadata": {"category": "AI", "complexity": "medium"}},
            {"content": "Reinforcement learning trains agents through reward systems", "metadata": {"category": "AI", "complexity": "high"}},
            {"content": "Python is a popular programming language for data science", "metadata": {"category": "programming", "complexity": "low"}},
            {"content": "JavaScript enables interactive web applications", "metadata": {"category": "programming", "complexity": "medium"}},
            {"content": "SQL databases store and query structured data efficiently", "metadata": {"category": "database", "complexity": "medium"}},
            {"content": "NoSQL databases handle unstructured data at scale", "metadata": {"category": "database", "complexity": "medium"}},
            {"content": "Vector databases enable semantic search capabilities", "metadata": {"category": "database", "complexity": "high"}},
            {"content": "Cloud computing provides scalable infrastructure services", "metadata": {"category": "infrastructure", "complexity": "medium"}},
            {"content": "Microservices architecture improves application scalability", "metadata": {"category": "infrastructure", "complexity": "high"}},
            {"content": "DevOps practices streamline software development and deployment", "metadata": {"category": "infrastructure", "complexity": "medium"}},
            {"content": "Quantum computing promises exponential speedups for specific problems", "metadata": {"category": "quantum", "complexity": "high"}},
            {"content": "Blockchain technology enables decentralized digital transactions", "metadata": {"category": "blockchain", "complexity": "high"}},
        ]
        
        # Load documents
        start_time = time.time()
        document_ids = rag.load_documents(documents, show_progress=True)
        load_time = time.time() - start_time
        print(f"Loaded {len(document_ids)} documents in {load_time:.2f} seconds")
        
        # Create HNSW indexes for different similarity functions
        print("\n=== Creating HNSW Indexes ===")
        
        # Create cosine similarity index
        print("Creating cosine similarity index...")
        start_time = time.time()
        rag.create_index(
            index_name="cosine_idx",
            similarity_function="cosine",
            ef_construction=128,
            ef_search=64,
            M=16
        )
        cosine_index_time = time.time() - start_time
        print(f"Cosine index created in {cosine_index_time:.3f} seconds")
        
        # Create inner product index
        print("Creating inner product index...")
        start_time = time.time()
        rag.create_index(
            index_name="ip_idx",
            similarity_function="inner_product",
            ef_construction=128,
            ef_search=64,
            M=16
        )
        ip_index_time = time.time() - start_time
        print(f"Inner product index created in {ip_index_time:.3f} seconds")
        
        # Demonstrate query performance with and without index
        print("\n=== Query Performance Comparison ===")
        query = "machine learning artificial intelligence"
        
        # Query without HNSW optimization
        print("\\nQuerying without HNSW optimization...")
        start_time = time.time()
        results_standard = rag.query(
            query_text=query,
            top_k=5,
            similarity_function="cosine",
            use_hnsw_optimization=False
        )
        standard_time = time.time() - start_time
        print(f"Standard query completed in {standard_time:.4f} seconds")
        print(f"Found {len(results_standard)} results")
        
        # Query with HNSW optimization
        print("\\nQuerying with HNSW optimization...")
        start_time = time.time()
        results_hnsw = rag.query(
            query_text=query,
            top_k=5,
            similarity_function="cosine",
            use_hnsw_optimization=True
        )
        hnsw_time = time.time() - start_time
        print(f"HNSW-optimized query completed in {hnsw_time:.4f} seconds")
        print(f"Found {len(results_hnsw)} results")
        
        if standard_time > 0 and hnsw_time > 0:
            speedup = standard_time / hnsw_time
            print(f"Speedup factor: {speedup:.2f}x")
        
        # Show top results
        print("\\nTop results from HNSW-optimized query:")
        for i, result in enumerate(results_hnsw[:3], 1):
            print(f"{i}. [{result['similarity']:.3f}] {result['content'][:60]}...")
        
        # Demonstrate different similarity functions with indexes
        print("\\n=== Testing Different Similarity Functions ===")
        
        similarity_functions = ["cosine", "inner_product"]
        for sim_func in similarity_functions:
            results = rag.query(
                query_text=query,
                top_k=3,
                similarity_function=sim_func,
                use_hnsw_optimization=True
            )
            print(f"\\n{sim_func.title()} similarity (with HNSW):")
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['similarity']:.3f}] {result['content'][:50]}...")
        
        # Demonstrate index management
        print("\\n=== Index Management ===")
        
        # Get database statistics
        stats = rag.get_statistics()
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        
        # Compact index (useful after many deletions)
        print("\\nCompacting cosine index...")
        rag.compact_index("cosine_idx")
        print("Index compacted successfully")
        
        # Delete an index
        print("\\nDeleting inner product index...")
        rag.delete_index("ip_idx")
        print("Index deleted successfully")
        
        # Filtered query with HNSW
        print("\\n=== Filtered Queries with HNSW ===")
        filtered_results = rag.query_with_filters(
            filters={"category": "AI"},
            query_text=query,
            top_k=3,
            similarity_function="cosine"
        )
        print(f"\\nFiltered query (category=AI) found {len(filtered_results)} results:")
        for i, result in enumerate(filtered_results, 1):
            print(f"  {i}. [{result['similarity']:.3f}] {result['content'][:50]}...")
        
        print("\\n=== Performance Notes ===")
        print("• HNSW indexes provide approximate nearest neighbor search")
        print("• Index creation time scales with dataset size and parameters")
        print("• Query performance improves significantly with large datasets")
        print("• Index persistence requires experimental flag for file databases")
        print("• Multiple indexes can be created for different similarity functions")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\\nNote: This example requires the DuckDB VSS extension.")
        print("Install it with: pip install duckdb[vss] (if available)")
        print("Or ensure DuckDB version supports the vss extension.")
    
    finally:
        # Clean up
        try:
            Path(db_path).unlink()
        except:
            pass


if __name__ == "__main__":
    main()