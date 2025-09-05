"""Example usage of the SQL RAG library."""

import asyncio
import sys
import os
from typing import List

# Add parent directory to path to import sqlvector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlvector import SQLRAG, EmbeddingProvider


class CustomEmbeddingProvider(EmbeddingProvider):
    """Custom embedding provider example."""
    
    async def embed(self, text: str) -> List[float]:
        # Implement your custom embedding logic here
        # This is just a placeholder
        import hashlib
        import random
        
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        random.seed(seed)
        
        return [random.uniform(-1, 1) for _ in range(384)]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]


async def main():
    # Create async engine (SQLite example)
    engine = create_async_engine("sqlite+aiosqlite:///example.db")
    
    # Option 1: Use default embedding provider
    rag = SQLRAG(engine=engine)
    
    # Option 2: Use custom embedding provider
    # custom_provider = CustomEmbeddingProvider()
    # rag = SQLRAG(engine=engine, embedding_provider=custom_provider)
    
    # Create tables
    await rag.create_tables()
    
    # Load documents
    documents = [
        {
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "example", "category": "animals"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "textbook", "category": "technology"}
        }
    ]
    
    document_ids = await rag.load_documents(documents)
    print(f"Loaded documents: {document_ids}")
    
    # Query similar documents
    results = await rag.query("artificial intelligence", top_k=2)
    
    print(f"\nQuery results for 'artificial intelligence':")
    for result in results:
        print(f"  ID: {result['id']}")
        print(f"  Content: {result['content']}")
        print(f"  Similarity: {result['similarity']:.3f}")
        print(f"  Metadata: {result.get('metadata', {})}")
        print()
    
    # Batch query
    batch_results = await rag.query_batch(
        ["fox", "machine learning"], 
        top_k=1
    )
    
    print("Batch query results:")
    for i, query_results in enumerate(batch_results):
        query = ["fox", "machine learning"][i]
        print(f"  Query '{query}':")
        for result in query_results:
            print(f"    {result['content']} (similarity: {result['similarity']:.3f})")
    
    # Update document
    success = await rag.update_document(
        document_ids[0],
        content="The quick brown fox jumps over the sleeping dog.",
        metadata={"source": "example", "category": "animals", "updated": True}
    )
    print(f"\nDocument updated: {success}")
    
    # Get document
    doc = await rag.get_document(document_ids[0])
    if doc:
        print(f"Retrieved document: {doc.content}")
        print(f"Metadata: {doc.get_metadata()}")
    
    # Clean up
    await engine.dispose()
    
    # Remove database file
    import pathlib
    db_path = pathlib.Path("example.db")
    if db_path.exists():
        db_path.unlink()
    # Remove SQLite auxiliary files if they exist
    for suffix in ["-shm", "-wal"]:
        aux_file = pathlib.Path(f"example.db{suffix}")
        if aux_file.exists():
            aux_file.unlink()


if __name__ == "__main__":
    asyncio.run(main())