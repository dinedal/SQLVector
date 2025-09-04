"""Example usage of the DuckDB RAG backend."""

import polars as pl
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path to import sql_rag
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sql_rag.backends.duckdb import DuckDBRAG
from sql_rag.embedding import DefaultEmbeddingProvider


def main():
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        db_path = f.name

    try:
        # Option 1: Use with context manager
        with DuckDBRAG(
            db_path=db_path, embedding_dimension=384, batch_size=1000
        ) as rag:

            print("=== Loading Documents ===")

            # Load individual documents
            doc_id1 = rag.load_document(
                content="The quick brown fox jumps over the lazy dog",
                metadata={"category": "animals", "source": "example"},
            )
            print(f"Loaded document: {doc_id1}")

            # Load batch of documents
            documents = [
                {
                    "content": "Machine learning is transforming how we process data",
                    "metadata": {"category": "technology", "topic": "AI"},
                },
                {
                    "content": "Python is a versatile programming language",
                    "metadata": {"category": "technology", "topic": "programming"},
                },
                {
                    "content": "The sun rises in the east and sets in the west",
                    "metadata": {"category": "science", "topic": "astronomy"},
                },
            ]

            doc_ids = rag.load_documents(documents, show_progress=True)
            print(f"Loaded {len(doc_ids)} documents in batch")

            # Load from Polars DataFrame
            df = pl.DataFrame(
                {
                    "content": [
                        "DuckDB is a fast analytical database",
                        "Polars provides fast DataFrames for Python",
                        "Vector similarity search enables semantic retrieval",
                    ],
                    "category": ["database", "data-processing", "AI"],
                    "difficulty": ["intermediate", "beginner", "advanced"],
                    "rating": [9.0, 8.5, 9.2],
                }
            )

            polars_doc_ids = rag.load_from_polars(
                df=df,
                metadata_columns=["category", "difficulty", "rating"],
                show_progress=True,
            )
            print(f"Loaded {len(polars_doc_ids)} documents from Polars DataFrame")

            print("\n=== Querying Documents ===")

            # Basic similarity search
            results = rag.query("machine learning AI", top_k=3)
            print(f"\nQuery 'machine learning AI' returned {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['similarity']:.3f}] {result['content']}")
                print(f"   Metadata: {result.get('metadata', {})}")

            # Query with different similarity functions
            print(f"\nTesting different similarity functions:")

            for sim_func in ["cosine", "inner_product", "euclidean"]:
                results = rag.query(
                    "programming language",
                    top_k=2,
                    similarity_function=sim_func,
                    similarity_threshold=(
                        -100
                        if sim_func == "inner_product"
                        else 1000 if sim_func == "euclidean" else -1.0
                    ),
                )
                print(f"{sim_func}: {len(results)} results")
                if results:
                    print(
                        f"  Best match: [{results[0]['similarity']:.3f}] {results[0]['content'][:50]}..."
                    )

            # Query with metadata filters
            tech_results = rag.query_with_filters(
                filters={"category": "technology"}, query_text="programming", top_k=5
            )
            print(
                f"\nFiltered query (category=technology): {len(tech_results)} results"
            )
            for result in tech_results:
                print(f"  - {result['content']}")

            # Find similar documents
            if doc_ids:
                similar_docs = rag.find_similar_documents(
                    document_id=doc_ids[0], top_k=3, similarity_threshold=-1.0
                )
                doc_obj = rag.get_document(doc_ids[0])
                if doc_obj:
                    print(
                        f"\nDocuments similar to '{doc_obj.content[:30]}...': {len(similar_docs)}"
                    )
                for doc in similar_docs:
                    print(f"  [{doc['similarity']:.3f}] {doc['content'][:50]}...")

            # Batch querying
            queries = [
                "database systems",
                "python programming",
                "artificial intelligence",
            ]
            batch_results = rag.query_batch(queries, top_k=2)
            print(f"\nBatch query results:")
            for i, (query, results) in enumerate(zip(queries, batch_results)):
                print(f"  {query}: {len(results)} results")

            print("\n=== Statistics and Export ===")

            # Get database statistics
            stats = rag.get_statistics()
            print(f"Database statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Export to Polars DataFrame
            export_df = rag.export_to_polars(include_embeddings=False)
            print(f"\nExported {len(export_df)} documents to Polars DataFrame")
            print(f"Columns: {export_df.columns}")

            # Export with embeddings (smaller sample)
            sample_ids = polars_doc_ids[:2] if polars_doc_ids else []
            if sample_ids:
                export_with_emb = rag.export_to_polars(
                    include_embeddings=True, document_ids=sample_ids
                )
                print(f"Exported {len(export_with_emb)} documents with embeddings")
                print(f"Embedding dimension: {len(export_with_emb['embedding'][0])}")

            print("\n=== Advanced Features ===")

            # Query with precomputed embedding
            if results and results[0].get('embedding'):
                # Use the embedding from a previous result
                query_embedding = results[0].get('embedding')
                precomputed_results = rag.query_with_embedding_duckdb(
                    query_embedding=query_embedding, top_k=3, similarity_threshold=-1.0
                )
                print(
                    f"Query with precomputed embedding: {len(precomputed_results)} results"
                )

            # Demonstrate different operations
            print(
                f"\nTotal documents before deletion: {rag.get_statistics()['total_documents']}"
            )

            # Delete a document
            if doc_ids:
                deleted = rag.delete_document(doc_ids[0])
                print(f"Deleted document: {deleted}")
                print(
                    f"Total documents after deletion: {rag.get_statistics()['total_documents']}"
                )

        # Option 2: Use without context manager
        print(f"\n=== Alternative Usage (Without Context Manager) ===")

        rag2 = DuckDBRAG(db_path=db_path, embedding_dimension=384)

        # Add a few more documents
        rag2.load_document("Context managers are useful in Python")
        rag2.load_document("DuckDB supports both SQL and Python APIs")

        # Query the enhanced database
        enhanced_results = rag2.query("Python API", top_k=3)
        print(f"Enhanced database query: {len(enhanced_results)} results")
        for result in enhanced_results:
            print(f"  - {result['content']}")

        final_stats = rag2.get_statistics()
        print(f"Final database size: {final_stats['total_documents']} documents")

    finally:
        # Clean up temporary file
        Path(db_path).unlink(missing_ok=True)


def demo_csv_loading():
    """Demonstrate loading from CSV files."""
    print("\n=== CSV Loading Demo ===")

    # Create a sample CSV file
    csv_data = """content,category,priority,author
"Introduction to machine learning concepts",AI,high,Alice
"Python data structures and algorithms",Programming,medium,Bob
"Database design principles and practices",Database,high,Charlie
"Web development with modern frameworks",Web,medium,Alice
"Statistical analysis and visualization",Statistics,low,Bob"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_data)
        csv_path = f.name

    try:
        with DuckDBRAG(db_path=":memory:", embedding_dimension=384) as rag:
            # Load from CSV
            doc_ids = rag.load_from_csv(
                csv_path=csv_path,
                metadata_columns=["category", "priority", "author"],
                show_progress=True,
            )

            print(f"Loaded {len(doc_ids)} documents from CSV")

            # Query by author
            alice_docs = rag.query_with_filters({"author": "Alice"})
            print(f"Documents by Alice: {len(alice_docs)}")

            # Query high priority items
            high_priority = rag.query_with_filters({"priority": "high"})
            print(f"High priority documents: {len(high_priority)}")

            # Combined filter and similarity search
            ai_high_priority = rag.query_with_filters(
                filters={"category": "AI", "priority": "high"},
                query_text="machine learning",
            )
            print(f"AI + high priority + ML similarity: {len(ai_high_priority)}")

    finally:
        Path(csv_path).unlink(missing_ok=True)


if __name__ == "__main__":
    print("DuckDB RAG Backend Example")
    print("=" * 40)

    main()
    demo_csv_loading()

    print("\n" + "=" * 40)
    print("Example completed successfully!")
