"""SQLite-specific loader implementation with efficient batch operations."""

import uuid
import sqlite3
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from ...embedding import EmbeddingService
from ...exceptions import LoaderError
from ...logger import get_logger
from .config import SQLiteConfig
from .models import SQLiteDocument, SQLiteEmbedding

# Get logger for this module
logger = get_logger(__name__)


class SQLiteLoader:
    """High-performance loader for SQLite backend with batch operations and VSS support."""
    
    def __init__(self, config: SQLiteConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
    
    def load_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        generate_embedding: bool = True
    ) -> str:
        """Load a single document."""
        return self.load_documents_batch([{
            "content": content,
            "metadata": metadata,
            "document_id": document_id
        }], generate_embeddings=generate_embedding)[0]
    
    def load_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        show_progress: bool = True
    ) -> List[str]:
        """Load multiple documents efficiently using batch operations."""
        try:
            with self.config.get_connection_context() as conn:
                # Prepare documents
                doc_records = []
                for doc_data in documents:
                    doc_id = doc_data.get("document_id") or str(uuid.uuid4())
                    doc = SQLiteDocument(
                        id=doc_id,
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata")
                    )
                    doc_records.append(doc.to_dict())
                
                # Insert documents using batch insert
                metadata_col = f"{self.config.documents_metadata_column}, " if self.config.documents_metadata_column else ""
                metadata_val = ":metadata, " if self.config.documents_metadata_column else ""
                
                conn.executemany(f"""
                    INSERT OR REPLACE INTO {self.config.documents_table} 
                    ({self.config.documents_id_column}, {self.config.documents_content_column}, {metadata_col}hash)
                    VALUES (:id, :content, {metadata_val}:hash)
                """, doc_records)
                
                document_ids = [doc["id"] for doc in doc_records]
                
                if generate_embeddings:
                    self._generate_embeddings_batch(
                        conn, 
                        doc_records, 
                        show_progress=show_progress
                    )
                
                return document_ids
                
        except Exception as e:
            raise LoaderError(f"Failed to load documents batch: {e}")
    
    def _generate_embeddings_batch(
        self,
        conn: sqlite3.Connection,
        doc_records: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> None:
        """Generate embeddings in batches for loaded documents."""
        logger.info(f"Generating embeddings for {len(doc_records)} documents")
        # Get documents that don't have embeddings yet
        doc_hashes = [doc["hash"] for doc in doc_records]
        
        if not doc_hashes:
            return
            
        placeholders = ",".join(["?" for _ in doc_hashes])
        
        cursor = conn.execute(f"""
            SELECT hash FROM {self.config.embeddings_table}
            WHERE hash IN ({placeholders})
        """, doc_hashes)
        existing_hashes = {row[0] for row in cursor.fetchall()}
        
        # Filter to only unembedded documents
        unembedded_docs = [
            doc for doc in doc_records 
            if doc["hash"] not in existing_hashes
        ]
        
        if not unembedded_docs:
            return
        
        # Process in batches
        total_docs = len(unembedded_docs)
        batch_size = self.config.batch_size
        
        progress_bar = tqdm(total=total_docs, desc="Generating embeddings") if show_progress else None
        
        try:
            for i in range(0, total_docs, batch_size):
                batch_docs = unembedded_docs[i:i + batch_size]
                
                if not batch_docs:
                    break
                
                # Extract content for embedding
                contents = [doc["content"] for doc in batch_docs]
                
                # Generate embeddings using the async service (run synchronously here)
                import asyncio
                embeddings = asyncio.run(
                    self.embedding_service.create_embeddings_batch(contents)
                )
                
                # Prepare embedding records
                embedding_records = []
                
                for doc, embedding in zip(batch_docs, embeddings):
                    embedding_id = str(uuid.uuid4())
                    
                    # Regular embeddings table record
                    emb_record = {
                        "id": embedding_id,
                        "document_id": doc["id"],
                        "hash": doc["hash"],
                        "embedding": SQLiteEmbedding._serialize_embedding(embedding),
                    }
                    if self.config.embeddings_model_column:
                        emb_record["model_name"] = getattr(self.embedding_service.provider, 'model_name', 'default')
                    embedding_records.append(emb_record)
                
                # Insert into regular embeddings table
                model_col = f"{self.config.embeddings_model_column}, " if self.config.embeddings_model_column else ""
                model_val = ", :model_name" if self.config.embeddings_model_column else ""
                
                conn.executemany(f"""
                    INSERT OR REPLACE INTO {self.config.embeddings_table}
                    ({self.config.embeddings_id_column}, {self.config.embeddings_document_id_column}, hash, {self.config.embeddings_column}{', ' + self.config.embeddings_model_column if self.config.embeddings_model_column else ''})
                    VALUES (:id, :document_id, :hash, :embedding{model_val})
                """, embedding_records)
                
                conn.commit()
                
                # Insert into VSS table if enabled
                # VSS expects JSON format for embeddings, not blobs
                if self.config.enable_vss_extension:
                    logger.info("Inserting embeddings into VSS index")
                    try:
                        import json
                        # Get the rowids and embeddings we just inserted
                        # We need to insert with matching rowids for efficient lookups
                        doc_ids = [doc["id"] for doc in batch_docs]
                        placeholders = ",".join(["?" for _ in doc_ids])
                        
                        cursor = conn.execute(f"""
                            SELECT rowid, {self.config.embeddings_column}
                            FROM {self.config.embeddings_table}
                            WHERE {self.config.embeddings_document_id_column} IN ({placeholders})
                        """, doc_ids)
                        
                        vss_records = []
                        for rowid, embedding_blob in cursor.fetchall():
                            # Convert blob back to vector, then to JSON for VSS
                            embedding_vector = SQLiteEmbedding._deserialize_embedding(embedding_blob)
                            embedding_json = json.dumps(embedding_vector)
                            vss_records.append((rowid, embedding_json))
                        
                        # Insert into VSS table with explicit rowids and JSON embeddings
                        conn.executemany(f"""
                            INSERT OR REPLACE INTO {self.config.vss_table} (rowid, embedding)
                            VALUES (?, ?)
                        """, vss_records)
                        
                        conn.commit()
                        
                    except sqlite3.OperationalError as e:
                        # Silently skip VSS operations if extension not available
                        if "no such table" not in str(e).lower():
                            print(f"Warning: Could not insert into VSS table: {e}")
                
                if progress_bar:
                    progress_bar.update(len(batch_docs))
        
        finally:
            if progress_bar:
                progress_bar.close()
    
    def get_document(self, document_id: str) -> Optional[SQLiteDocument]:
        """Get a single document by ID."""
        try:
            with self.config.get_connection_context() as conn:
                metadata_col = f"{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                
                cursor = conn.execute(f"""
                    SELECT {self.config.documents_id_column}, {self.config.documents_content_column}, {metadata_col}, hash 
                    FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} = ?
                """, [document_id])
                
                row = cursor.fetchone()
                if row:
                    return SQLiteDocument.from_dict(dict(row))
                return None
                
        except Exception as e:
            raise LoaderError(f"Failed to get document: {e}")
    
    def get_documents_batch(self, document_ids: List[str]) -> List[SQLiteDocument]:
        """Get multiple documents by IDs."""
        try:
            with self.config.get_connection_context() as conn:
                # Use prepared statement for batch query
                placeholders = ",".join(["?" for _ in document_ids])
                metadata_col = f"{self.config.documents_metadata_column}" if self.config.documents_metadata_column else "NULL as metadata"
                
                cursor = conn.execute(f"""
                    SELECT {self.config.documents_id_column}, {self.config.documents_content_column}, {metadata_col}, hash
                    FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} IN ({placeholders})
                """, document_ids)
                
                return [
                    SQLiteDocument.from_dict(dict(row))
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            raise LoaderError(f"Failed to get documents batch: {e}")
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its embeddings."""
        try:
            with self.config.get_connection_context() as conn:
                # Check if document exists first
                cursor = conn.execute(f"""
                    SELECT COUNT(*) FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} = ?
                """, [document_id])
                
                if cursor.fetchone()[0] == 0:
                    return False
                
                # Delete from VSS table first if enabled
                if self.config.enable_vss_extension:
                    try:
                        # Get embedding rowid from embeddings table
                        cursor = conn.execute(f"""
                            SELECT rowid FROM {self.config.embeddings_table}
                            WHERE {self.config.embeddings_document_id_column} = ?
                        """, [document_id])
                        
                        for row in cursor.fetchall():
                            emb_rowid = row[0]
                            
                            # Delete from VSS table using the same rowid
                            conn.execute(f"""
                                DELETE FROM {self.config.vss_table}
                                WHERE rowid = ?
                            """, [emb_rowid])
                            
                    except sqlite3.OperationalError:
                        pass  # VSS table might not exist or be accessible
                
                # Delete embeddings
                conn.execute(f"""
                    DELETE FROM {self.config.embeddings_table}
                    WHERE {self.config.embeddings_document_id_column} = ?
                """, [document_id])
                
                # Delete document
                conn.execute(f"""
                    DELETE FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} = ?
                """, [document_id])
                
                # Verify deletion
                cursor = conn.execute(f"""
                    SELECT COUNT(*) FROM {self.config.documents_table}
                    WHERE {self.config.documents_id_column} = ?
                """, [document_id])
                
                conn.commit()
                return cursor.fetchone()[0] == 0
                
        except Exception as e:
            raise LoaderError(f"Failed to delete document: {e}")
    
    def create_index(
        self,
        index_name: str,
        similarity_function: str = "cosine",
        factory_string: Optional[str] = None
    ) -> bool:
        """Create a VSS index on embeddings for accelerated similarity search.
        
        Args:
            index_name: Name for the VSS index (currently only one index supported)
            similarity_function: Similarity function (only used for compatibility)
            factory_string: Custom Faiss factory string (e.g., "IVF4096,Flat,IDMap2")
            
        Returns:
            True if index was created successfully
            
        Raises:
            LoaderError: If VSS extension is not enabled or index creation fails
            
        Note:
            - Requires enable_vss_extension=True in SQLiteConfig
            - sqlite-vss uses Faiss factory strings for index configuration
            - For large datasets, consider training with IVF factory strings
        """
        if not self.config.enable_vss_extension:
            raise LoaderError("VSS extension is not enabled. Set enable_vss_extension=True in SQLiteConfig.")
        
        try:
            with self.config.get_connection_context() as conn:
                # Store the factory string for future use
                if factory_string:
                    self.config.vss_factory_string = factory_string
                
                # Drop and recreate the VSS table
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {self.config.vss_table}")
                    conn.execute(self.config.get_vss_schema())
                    
                    # Repopulate VSS table with existing embeddings
                    import json
                    cursor = conn.execute(f"""
                        SELECT rowid, {self.config.embeddings_column}
                        FROM {self.config.embeddings_table}
                    """)
                    
                    vss_records = []
                    for rowid, embedding_blob in cursor.fetchall():
                        # Convert blob to JSON for VSS
                        embedding_vector = SQLiteEmbedding._deserialize_embedding(embedding_blob)
                        embedding_json = json.dumps(embedding_vector)
                        vss_records.append((rowid, embedding_json))
                    
                    if vss_records:
                        conn.executemany(f"""
                            INSERT INTO {self.config.vss_table} (rowid, embedding)
                            VALUES (?, ?)
                        """, vss_records)
                    
                    conn.commit()
                    
                except Exception as e:
                    raise LoaderError(f"Failed to recreate VSS index: {e}")
                
                return True
                
        except Exception as e:
            raise LoaderError(f"Failed to create VSS index '{index_name}': {e}")
    
    def delete_index(self, index_name: str) -> bool:
        """Delete VSS index (recreates table with default factory).
        
        Args:
            index_name: Name of the index to delete (currently only one index supported)
            
        Returns:
            True if index was deleted successfully
        """
        if not self.config.enable_vss_extension:
            raise LoaderError("VSS extension is not enabled.")
        
        try:
            with self.config.get_connection_context() as conn:
                # Reset to default factory string
                self.config.vss_factory_string = "Flat"
                
                # Recreate VSS table with default factory
                conn.execute(f"DROP TABLE IF EXISTS {self.config.vss_table}")
                conn.execute(self.config.get_vss_schema())
                
                # Repopulate with existing embeddings
                import json
                cursor = conn.execute(f"""
                    SELECT rowid, {self.config.embeddings_column}
                    FROM {self.config.embeddings_table}
                """)
                
                vss_records = []
                for rowid, embedding_blob in cursor.fetchall():
                    # Convert blob to JSON for VSS
                    embedding_vector = SQLiteEmbedding._deserialize_embedding(embedding_blob)
                    embedding_json = json.dumps(embedding_vector)
                    vss_records.append((rowid, embedding_json))
                
                if vss_records:
                    conn.executemany(f"""
                        INSERT INTO {self.config.vss_table} (rowid, embedding)
                        VALUES (?, json(?))
                    """, vss_records)
                
                conn.commit()
                
                return True
                
        except Exception as e:
            raise LoaderError(f"Failed to delete index '{index_name}': {e}")
    
    def train_index(self, training_data_limit: Optional[int] = None) -> bool:
        """Train VSS index with existing embeddings (for IVF factory strings).
        
        Args:
            training_data_limit: Limit on number of training vectors (None for all)
            
        Returns:
            True if training was successful
        """
        if not self.config.enable_vss_extension:
            raise LoaderError("VSS extension is not enabled.")
        
        # Only needed for IVF factory strings
        if "IVF" not in self.config.vss_factory_string:
            return True
        
        try:
            with self.config.get_connection_context() as conn:
                # For IVF indexes, sqlite-vss handles training internally
                # We just need to make sure the table has data
                # The training happens automatically when we query
                
                # Verify we have data in VSS table
                cursor = conn.execute(f"""
                    SELECT COUNT(*) FROM {self.config.vss_table}
                """)
                count = cursor.fetchone()[0]
                
                if count == 0:
                    # If no data, populate from embeddings table
                    import json
                    cursor = conn.execute(f"""
                        SELECT rowid, {self.config.embeddings_column}
                        FROM {self.config.embeddings_table}
                        {f'LIMIT {training_data_limit}' if training_data_limit else ''}
                    """)
                    
                    vss_records = []
                    for rowid, embedding_blob in cursor.fetchall():
                        # Convert blob to JSON for VSS
                        embedding_vector = SQLiteEmbedding._deserialize_embedding(embedding_blob)
                        embedding_json = json.dumps(embedding_vector)
                        vss_records.append((rowid, embedding_json))
                    
                    if vss_records:
                        conn.executemany(f"""
                            INSERT INTO {self.config.vss_table} (rowid, embedding)
                            VALUES (?, ?)
                        """, vss_records)
                        
                        conn.commit()
                
                return True
                
        except Exception as e:
            raise LoaderError(f"Failed to train index: {e}")
    
    def _compute_hash(self, content: str) -> str:
        """Compute a hash of the content."""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()