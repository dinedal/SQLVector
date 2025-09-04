"""Example usage of SQLite RAG backend without VSS extension.

This example demonstrates the core SQLite backend functionality using:
- Binary embedding storage for space efficiency
- Custom SQLite functions for vector similarity
- Batch operations and efficient data loading
- Metadata filtering and complex queries
"""

import time
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path to import sql_rag
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sql_rag.backends.sqlite import SQLiteRAG
from sql_rag.embedding import DefaultEmbeddingProvider


def main():
    print("SQLite RAG Backend Example (Without VSS Extension)")
    print("=" * 60)
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        db_path = f.name
    
    try:
        # Initialize SQLite RAG without VSS extension
        print("=== Initializing SQLite RAG ===")
        rag = SQLiteRAG(
            db_path=db_path,
            embedding_dimension=384,
            enable_vss_extension=False,  # Explicitly disable VSS extension
            batch_size=500  # Smaller batch size for SQLite
        )
        
        print(f"Database path: {db_path}")
        print(f"Embedding dimension: {rag.config.embedding_dimension}")
        print(f"VSS Extension enabled: {rag.config.enable_vss_extension}")
        print(f"Batch size: {rag.config.batch_size}")
        
        # Load sample documents
        print("\n=== Loading Sample Documents ===")
        
        # Load individual document
        doc_id1 = rag.load_document(
            content="The quick brown fox jumps over the lazy dog",
            metadata={"category": "example", "type": "pangram"}
        )
        print(f"Loaded document ID: {doc_id1}")
        
        # Load batch of documents with metadata
        documents = [
            {"content": "Machine learning algorithms transform artificial intelligence", "metadata": {"category": "AI", "topic": "ML"}},
            {"content": "Deep neural networks enable complex pattern recognition", "metadata": {"category": "AI", "topic": "DL"}},
            {"content": "Natural language processing helps computers understand text", "metadata": {"category": "AI", "topic": "NLP"}},
            {"content": "Python is a versatile programming language for data science", "metadata": {"category": "programming", "language": "Python"}},
            {"content": "JavaScript enables interactive web applications", "metadata": {"category": "programming", "language": "JavaScript"}},
            {"content": "SQL databases store structured data efficiently", "metadata": {"category": "database", "type": "relational"}},
            {"content": "NoSQL databases handle unstructured data at scale", "metadata": {"category": "database", "type": "non-relational"}},
            {"content": "Cloud computing provides scalable infrastructure", "metadata": {"category": "infrastructure", "service": "cloud"}},
            {"content": "Containerization simplifies application deployment", "metadata": {"category": "infrastructure", "tool": "docker"}},
            {"content": "Version control systems track code changes over time", "metadata": {"category": "development", "tool": "git"}},
        ]
        
        # Load documents with progress tracking
        start_time = time.time()
        doc_ids = rag.load_documents(documents, show_progress=True)
        load_time = time.time() - start_time
        print(f"Loaded {len(doc_ids)} documents in {load_time:.2f} seconds")
        print(f"Average time per document: {load_time/len(doc_ids):.3f} seconds")
        
        # Basic queries
        print("\n=== Basic Similarity Search ===")
        
        query = "machine learning and artificial intelligence"
        start_time = time.time()
        results = rag.query(query_text=query, top_k=5)
        query_time = time.time() - start_time
        
        print(f"Query: '{query}'")
        print(f"Query completed in {query_time:.3f} seconds")
        print(f"Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['similarity']:.3f}] {result['content'][:60]}...")
            if result.get('metadata'):
                print(f"     Metadata: {result['metadata']}")
        
        # Test different similarity functions
        print("\n=== Different Similarity Functions ===")
        
        similarity_functions = ["cosine", "inner_product", "euclidean"]
        test_query = "programming language"
        
        for sim_func in similarity_functions:
            results = rag.query(
                query_text=test_query,
                top_k=3,
                similarity_function=sim_func
            )
            print(f"\n{sim_func.title()} similarity:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['similarity']:.3f}] {result['content'][:50]}...")
        
        # Metadata filtering
        print("\n=== Queries with Metadata Filters ===")
        
        # Filter by category
        ai_results = rag.query_with_filters(
            filters={"category": "AI"},
            query_text="neural networks",
            top_k=5
        )
        print(f"AI category results: {len(ai_results)}")
        for result in ai_results[:3]:
            print(f"  - {result['content'][:60]}...")
            metadata = result.get('metadata', {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            print(f"    Topic: {metadata.get('topic', 'N/A')}")
        
        # Filter by programming language
        python_results = rag.query_with_filters(
            filters={"language": "Python"},
            top_k=5
        )
        print(f"\nPython language results: {len(python_results)}")
        for result in python_results:
            print(f"  - {result['content']}")
        
        # Multiple filters
        db_relational = rag.query_with_filters(
            filters={"category": "database", "type": "relational"},
            top_k=5
        )
        print(f"\nRelational database results: {len(db_relational)}")
        for result in db_relational:
            print(f"  - {result['content']}")
        
        # Find similar documents
        print("\n=== Find Similar Documents ===")
        
        if doc_ids:
            target_doc = rag.get_document(doc_ids[0])
            if target_doc:
                print(f"Finding documents similar to: '{target_doc.content[:60]}...'")
                
                similar_docs = rag.find_similar_documents(
                    document_id=doc_ids[0],
                    top_k=3
                )
                print(f"Found {len(similar_docs)} similar documents:")
                for doc in similar_docs:
                    print(f"  [{doc['similarity']:.3f}] {doc['content'][:60]}...")
        
        # Batch querying
        print("\n=== Batch Query Operations ===")
        
        queries = [
            "database systems",
            "python programming",
            "artificial intelligence",
            "cloud infrastructure"
        ]
        
        start_time = time.time()
        batch_results = rag.query_batch(queries, top_k=2)
        batch_time = time.time() - start_time
        
        print(f"Batch query completed in {batch_time:.3f} seconds")
        print("Results per query:")
        for query_text, results in zip(queries, batch_results):
            print(f"  '{query_text}': {len(results)} results")
            if results:
                print(f"    Top match: [{results[0]['similarity']:.3f}] {results[0]['content'][:40]}...")
        
        # Advanced features
        print("\n=== Advanced Features ===")
        
        # Query with precomputed embedding
        if results and results[0].get('embedding'):
            query_embedding = results[0]['embedding']
            precomputed_results = rag.query_with_embedding(
                query_embedding=query_embedding,
                top_k=3
            )
            print(f"Query with precomputed embedding: {len(precomputed_results)} results")
            for i, result in enumerate(precomputed_results[:2], 1):
                print(f"  {i}. {result['content'][:60]}...")
        
        # Export to dictionary
        print("\n=== Export Capabilities ===")
        
        export_data = rag.export_to_dict(include_embeddings=False)
        print(f"Exported {len(export_data)} documents to dictionary")
        if export_data:
            print(f"Export format keys: {list(export_data[0].keys())}")
            print(f"Sample document: {export_data[0]['content'][:50]}...")
        
        # Export with embeddings for a subset
        subset_ids = doc_ids[:2] if doc_ids else []
        if subset_ids:
            export_with_emb = rag.export_to_dict(
                include_embeddings=True,
                document_ids=subset_ids
            )
            print(f"Exported {len(export_with_emb)} documents with embeddings")
            if export_with_emb and 'embedding' in export_with_emb[0]:
                print(f"Embedding dimension: {len(export_with_emb[0]['embedding'])}")
        
        # Database statistics
        print("\n=== Database Statistics ===")
        
        stats = rag.get_statistics()
        print("Database information:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Document management
        print("\n=== Document Management ===")
        
        total_before = stats['total_documents']
        print(f"Total documents before deletion: {total_before}")
        
        # Delete a document
        if doc_ids:
            deleted = rag.delete_document(doc_ids[0])
            print(f"Deleted document: {deleted}")
            
            stats_after = rag.get_statistics()
            total_after = stats_after['total_documents']
            print(f"Total documents after deletion: {total_after}")
        
        # Performance summary
        print("\n=== Performance Summary ===")
        print(f"• Document loading: {load_time/len(doc_ids):.3f} sec/doc")
        print(f"• Single query time: {query_time:.3f} seconds")
        print(f"• Batch query time: {batch_time/len(queries):.3f} sec/query")
        print("• Binary embedding storage reduces database size")
        print("• Custom SQLite functions provide good performance for small-medium datasets")
        print("• No external dependencies required (pure SQLite)")
        
        # Demonstrate context manager usage
        print("\n=== Alternative Usage with Context Manager ===")
        
        # Create another temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path2 = f.name
        
        try:
            # Use context manager for automatic cleanup
            with SQLiteRAG(db_path=db_path2, embedding_dimension=384) as rag2:
                # Add a few documents
                rag2.load_document("Context managers ensure proper resource cleanup")
                rag2.load_document("SQLite is lightweight and serverless")
                rag2.load_document("Binary storage optimizes space usage")
                
                # Query
                ctx_results = rag2.query("resource cleanup", top_k=2)
                print(f"Context manager query: {len(ctx_results)} results")
                for result in ctx_results:
                    print(f"  - {result['content']}")
                
                final_stats = rag2.get_statistics()
                print(f"Context manager database: {final_stats['total_documents']} documents")
        
        finally:
            # Clean up second database
            Path(db_path2).unlink(missing_ok=True)
        
        print("\n=== Notes on SQLite Backend ===")
        print("• Ideal for small to medium datasets (up to ~100K documents)")
        print("• No external dependencies or extensions required")
        print("• Binary embedding storage saves ~50% space vs text")
        print("• Custom similarity functions work well for most use cases")
        print("• Consider VSS extension for larger datasets or faster queries")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary database
        try:
            Path(db_path).unlink()
        except:
            pass


def demo_persistence():
    """Demonstrate persistent database usage."""
    print("\n=== Persistent Database Demo ===")
    print("=" * 60)
    
    db_path = "example_sqlite_rag.db"
    
    try:
        # First session: Create and populate database
        print("\nSession 1: Creating and populating database")
        rag1 = SQLiteRAG(db_path=db_path, embedding_dimension=384)
        
        # Load initial documents
        initial_docs = [
            {"content": "Data persistence allows information to survive program termination"},
            {"content": "SQLite databases are self-contained and portable"},
            {"content": "Embedding vectors can be stored efficiently in binary format"},
        ]
        
        doc_ids = rag1.load_documents(initial_docs, show_progress=False)
        print(f"Loaded {len(doc_ids)} documents")
        
        stats1 = rag1.get_statistics()
        print(f"Database size: {stats1['total_documents']} documents")
        
        # Close first session
        del rag1
        
        # Second session: Read and add more data
        print("\nSession 2: Reading existing data and adding more")
        rag2 = SQLiteRAG(db_path=db_path, embedding_dimension=384)
        
        # Verify existing data
        stats2 = rag2.get_statistics()
        print(f"Existing documents: {stats2['total_documents']}")
        
        # Query existing data
        results = rag2.query("data persistence", top_k=2)
        print(f"Query found {len(results)} results from existing data")
        
        # Add new documents
        new_docs = [
            {"content": "Persistent storage maintains state across application restarts"},
            {"content": "Database transactions ensure data consistency"},
        ]
        
        new_ids = rag2.load_documents(new_docs, show_progress=False)
        print(f"Added {len(new_ids)} new documents")
        
        final_stats = rag2.get_statistics()
        print(f"Final database size: {final_stats['total_documents']} documents")
        
        # Query combined dataset
        all_results = rag2.query("database storage", top_k=3)
        print(f"\nQuery across all data: {len(all_results)} results")
        for result in all_results:
            print(f"  - {result['content'][:60]}...")
        
    finally:
        # Clean up persistent database
        Path(db_path).unlink(missing_ok=True)
        print("\nPersistent database demo cleaned up")


if __name__ == "__main__":
    print("SQLite RAG Backend Examples")
    print("=" * 60)
    
    # Run main example
    main()
    
    # Run persistence demo
    demo_persistence()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")