"""Example usage of SQLite RAG backend with Vector Similarity Search (VSS) extension."""

import time
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path to import sqlvector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlvector.backends.sqlite import SQLiteRAG
from sqlvector.embedding import DefaultEmbeddingProvider


def main():
    print("SQLite RAG with Vector Similarity Search Extension")
    print("=" * 60)
    
    # Create a temporary database file path
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        db_path = f.name
    
    try:
        # Initialize SQLite RAG with VSS extension enabled
        print("=== Initializing SQLite RAG with VSS Extension ===")
        rag = SQLiteRAG(
            db_path=db_path,
            embedding_dimension=384,
            enable_vss_extension=True,  # Enable Vector Similarity Search
            vss_factory_string="Flat"   # Start with Flat index (can be changed)
        )
        
        print(f"VSS Extension enabled: {rag.config.enable_vss_extension}")
        print(f"VSS Factory string: {rag.config.vss_factory_string}")
        
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
        
        # Demonstrate different Faiss factory strings for VSS indexes
        print("\n=== Creating VSS Indexes with Different Configurations ===")
        
        # Create Flat index (exact search, good for small datasets)
        print("Creating Flat index (exact search)...")
        start_time = time.time()
        rag.create_index(
            index_name="flat_idx",
            factory_string="Flat"
        )
        flat_index_time = time.time() - start_time
        print(f"Flat index created in {flat_index_time:.3f} seconds")
        
        # Demonstrate query performance with Flat index
        print("\n=== Query Performance with Flat Index ===")
        query = "machine learning artificial intelligence"
        
        # Query without VSS optimization
        print("\nQuerying without VSS optimization...")
        start_time = time.time()
        results_standard = rag.query(
            query_text=query,
            top_k=5,
            similarity_function="cosine",
            use_vss_optimization=False
        )
        standard_time = time.time() - start_time
        print(f"Standard query completed in {standard_time:.4f} seconds")
        print(f"Found {len(results_standard)} results")
        
        # Query with VSS optimization
        print("\nQuerying with VSS optimization (Flat index)...")
        start_time = time.time()
        results_vss = rag.query(
            query_text=query,
            top_k=5,
            similarity_function="cosine",
            use_vss_optimization=True
        )
        vss_time = time.time() - start_time
        print(f"VSS-optimized query completed in {vss_time:.4f} seconds")
        print(f"Found {len(results_vss)} results")
        
        if standard_time > 0 and vss_time > 0:
            speedup = standard_time / vss_time
            print(f"Speedup factor: {speedup:.2f}x")
        
        # Show top results
        print("\nTop results from VSS-optimized query:")
        for i, result in enumerate(results_vss[:3], 1):
            print(f"{i}. [{result['similarity']:.3f}] {result['content'][:60]}...")
        
        # Try IVF index for larger datasets (approximate search)
        print("\n=== Creating IVF Index (Approximate Search) ===")
        print("Creating IVF index for scalable approximate search...")
        start_time = time.time()
        rag.create_index(
            index_name="ivf_idx",
            factory_string="IVF16,Flat"  # 16 clusters with Flat quantizer
        )
        ivf_index_time = time.time() - start_time
        print(f"IVF index created in {ivf_index_time:.3f} seconds")
        
        # Train IVF index (required for IVF factory strings)
        print("\nTraining IVF index with existing embeddings...")
        start_time = time.time()
        training_success = rag.train_index(training_data_limit=None)
        train_time = time.time() - start_time
        print(f"Index training {'successful' if training_success else 'failed'} in {train_time:.3f} seconds")
        
        # Query with IVF index
        print("\nQuerying with VSS optimization (IVF index)...")
        start_time = time.time()
        results_ivf = rag.query(
            query_text=query,
            top_k=5,
            similarity_function="cosine",
            use_vss_optimization=True
        )
        ivf_time = time.time() - start_time
        print(f"IVF-optimized query completed in {ivf_time:.4f} seconds")
        print(f"Found {len(results_ivf)} results")
        
        # Demonstrate different similarity functions
        print("\n=== Testing Different Similarity Functions ===")
        
        similarity_functions = ["cosine", "inner_product", "euclidean"]
        for sim_func in similarity_functions:
            results = rag.query(
                query_text=query,
                top_k=3,
                similarity_function=sim_func,
                use_vss_optimization=True
            )
            print(f"\n{sim_func.title()} similarity (with VSS):")
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['similarity']:.3f}] {result['content'][:50]}...")
        
        # Demonstrate index management
        print("\n=== Index Management ===")
        
        # Get database statistics
        stats = rag.get_statistics()
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"VSS enabled: {stats['vss_enabled']}")
        print(f"Current VSS factory: {stats['vss_factory_string']}")
        
        # Delete and recreate index with different configuration
        print("\nDeleting current index...")
        rag.delete_index("ivf_idx")
        print("Index deleted, reverting to default Flat factory")
        
        # Create a more complex IVF index
        print("\nCreating complex IVF index with IDMap...")
        rag.create_index(
            index_name="complex_idx",
            factory_string="IVF32,Flat,IDMap2"  # More clusters with ID mapping
        )
        print("Complex index created successfully")
        
        # Filtered query with VSS
        print("\n=== Filtered Queries with VSS ===")
        filtered_results = rag.query_with_filters(
            filters={"category": "AI"},
            query_text=query,
            top_k=3,
            similarity_function="cosine",
            use_vss_optimization=True
        )
        print(f"\nFiltered query (category=AI) found {len(filtered_results)} results:")
        for i, result in enumerate(filtered_results, 1):
            print(f"  {i}. [{result['similarity']:.3f}] {result['content'][:50]}...")
        
        # Export data for analysis
        print("\n=== Export Capabilities ===")
        export_data = rag.export_to_dict(include_embeddings=False)
        print(f"Exported {len(export_data)} documents to dictionary")
        if export_data:
            print(f"Sample export keys: {list(export_data[0].keys())}")
        
        # Query with precomputed embedding
        print("\n=== Advanced Query Features ===")
        if results_vss:
            # Use the embedding from a previous result
            # Need to deserialize the embedding from blob format
            from sqlvector.backends.sqlite.models import SQLiteEmbedding
            embedding_blob = results_vss[0].get('embedding')
            if embedding_blob:
                query_embedding = SQLiteEmbedding._deserialize_embedding(embedding_blob)
            else:
                query_embedding = None
            
        if query_embedding:
                precomputed_results = rag.query_with_embedding(
                    query_embedding=query_embedding,
                    top_k=3,
                    use_vss_optimization=True
                )
                print(f"Query with precomputed embedding: {len(precomputed_results)} results")
        
        # Demonstrate batch operations
        print("\n=== Batch Query Operations ===")
        queries = [
            "database systems",
            "python programming",
            "artificial intelligence"
        ]
        batch_results = rag.query_batch(queries, top_k=2, use_vss_optimization=True)
        print("Batch query results with VSS:")
        for i, (query_text, results) in enumerate(zip(queries, batch_results)):
            print(f"  '{query_text}': {len(results)} results")
            if results:
                print(f"    Top match: [{results[0]['similarity']:.3f}] {results[0]['content'][:40]}...")
        
        print("\n=== Performance Notes ===")
        print("• sqlite-vss uses Faiss for accelerated similarity search")
        print("• Flat index provides exact search (best for <10K documents)")
        print("• IVF indexes provide approximate search (scalable to millions)")
        print("• Factory strings control index type and configuration")
        print("• Training is required for IVF indexes to build clusters")
        print("• VSS optimization typically provides 2-10x speedup")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This example requires the sqlite-vss extension.")
        print("Install it with: pip install sqlite-vss")
        print("Or ensure the VSS extension is available in your environment.")
    
    finally:
        # Clean up
        try:
            Path(db_path).unlink()
        except:
            pass


if __name__ == "__main__":
    main()