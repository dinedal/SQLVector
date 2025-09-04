"""Example usage of the SQL RAG library."""

import asyncio
import sys
import os
from typing import List

# Add parent directory to path to import sql_rag
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine
from sql_rag import SQLRAG, EmbeddingProvider
from sentence_transformers import SentenceTransformer
import torch
import numpy as np


class CustomEmbeddingProvider(EmbeddingProvider):

    def __init__(self, batch_size=4, max_seq_length=512, use_gpu=False) -> None:
        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

        device = "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                raise RuntimeError(
                    "No suitable GPU found. Please check your PyTorch installation."
                )

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device,
        )
        # We can reduce the max_seq_length from the default for faster encoding
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

        super().__init__()

    async def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=self.batch_size).tolist()

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return self.model.similarity(
            np.array(vec1, dtype=np.float32),
            np.array(vec2, dtype=np.float32),
        ).item()


async def main():
    # remove the database file if it exists
    import os
    import json

    if os.path.exists("example.db"):
        os.remove("example.db")
    # Create async engine (SQLite example)
    engine = create_async_engine("sqlite+aiosqlite:///example.db")

    # Option 1: Use default embedding provider
    # rag = SQLRAG(engine=engine)

    # Option 2: Use custom embedding provider
    # custom_provider = CustomEmbeddingProvider()
    rag = SQLRAG(
        engine=engine, embedding_provider=CustomEmbeddingProvider(use_gpu=True)
    )

    # Create tables
    await rag.create_tables()

    # Load documents
    documents = [
        {
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "example", "category": "animals"},
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "textbook", "category": "technology"},
        },
    ]

    document_ids = await rag.load_documents(documents)
    print(f"Loaded documents: {document_ids}")

    # Query similar documents
    results = await rag.query("artificial intelligence", top_k=5)

    print(f"\nQuery results for 'artificial intelligence':")
    for result in results:
        print(f"  ID: {result['id']}")
        print(f"  Content: {result['content']}")
        print(f"  Similarity: {result['similarity']:.3f}")
        metadata = json.loads(result.get("doc_metadata", "{}"))
        print(f"  Metadata: {metadata}")
        print()

    # Batch query
    batch_results = await rag.query_batch(["fox", "machine learning"], top_k=1)

    print("Batch query results:")
    for i, query_results in enumerate(batch_results):
        query = ["fox", "machine learning"][i]
        print(f"  Query '{query}':")
        for result in query_results:
            print(
                f"    {result['content']} (similarity: {result['similarity']:.3f})\n   Metadata: {json.loads(result.get('doc_metadata', '{}'))}"
            )

    # Update document
    success = await rag.update_document(
        document_ids[0],
        content="The quick brown fox jumps over the sleeping dog.",
        metadata={"source": "example", "category": "animals", "updated": True},
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
    if os.path.exists("example.db"):
        os.remove("example.db")
    # Remove SQLite auxiliary files if they exist
    for suffix in ["-shm", "-wal"]:
        aux_file = f"example.db{suffix}"
        if os.path.exists(aux_file):
            os.remove(aux_file)


if __name__ == "__main__":
    asyncio.run(main())
