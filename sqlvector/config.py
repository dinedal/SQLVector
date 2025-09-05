from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy import Engine, Table, Column, String, Text, DateTime, MetaData
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.sql import func
import json

from .exceptions import ConfigurationError
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class RAGConfig:
    """Configuration for SQL RAG system.

    Args:
        engine: SQLAlchemy async engine for database connection
        documents_table: Name of the documents table
        embeddings_table: Name of the embeddings table
        embedding_dimension: Dimension of embedding vectors
        embeddings_column: Name of the column storing embedding vectors
        documents_id_column: Name of the documents ID column
        documents_content_column: Name of the documents content column
        documents_metadata_column: Name of the documents metadata column
        embeddings_id_column: Name of the embeddings ID column
        embeddings_document_id_column: Name of the embeddings document_id column
        embeddings_model_column: Name of the embeddings model column
    """

    engine: AsyncEngine
    documents_table: str
    embeddings_table: str
    embedding_dimension: int = 768
    embeddings_column: str = "vector"
    # Document table columns
    documents_id_column: str = "id"
    documents_content_column: str = "content"
    documents_metadata_column: Optional[str] = "doc_metadata"
    # Embedding table columns
    embeddings_id_column: str = "id"
    embeddings_document_id_column: str = "document_id"
    embeddings_model_column: Optional[str] = "model_name"

    def __post_init__(self) -> None:
        if not self.documents_table:
            raise ConfigurationError("documents_table is required")
        if not self.embeddings_table:
            raise ConfigurationError("embeddings_table is required")
        if self.embedding_dimension <= 0:
            raise ConfigurationError("embedding_dimension must be positive")
        
        logger.debug(f"Initializing RAGConfig with tables: {self.documents_table}, {self.embeddings_table}")

        # Create dynamic table objects based on configuration
        self.metadata = MetaData()

        # Create documents table object
        documents_columns = [
            Column(self.documents_id_column, String, primary_key=True),
            Column(self.documents_content_column, Text, nullable=False),
        ]

        # Add metadata column only if specified
        if self.documents_metadata_column:
            documents_columns.append(Column(self.documents_metadata_column, Text))

        documents_columns.extend(
            [
                Column("created_at", DateTime, server_default=func.now()),
                Column(
                    "updated_at",
                    DateTime,
                    server_default=func.now(),
                    onupdate=func.now(),
                ),
            ]
        )

        self.documents_table_obj = Table(
            self.documents_table, self.metadata, *documents_columns
        )

        # Create embeddings table object
        embeddings_columns = []

        # Add ID column as primary key
        embeddings_columns.append(
            Column(self.embeddings_id_column, String, primary_key=True)
        )

        # Add document_id column only if it's different from id column
        if self.embeddings_document_id_column != self.embeddings_id_column:
            embeddings_columns.append(
                Column(
                    self.embeddings_document_id_column,
                    String,
                    nullable=False,
                    index=True,
                )
            )

        # Add embedding vector column
        embeddings_columns.append(Column(self.embeddings_column, Text, nullable=False))

        # Add model column if specified and not already added
        if self.embeddings_model_column and self.embeddings_model_column not in [
            self.embeddings_id_column,
            self.embeddings_document_id_column,
            self.embeddings_column,
        ]:
            embeddings_columns.append(Column(self.embeddings_model_column, String))

        # Add timestamp column
        embeddings_columns.append(
            Column("created_at", DateTime, server_default=func.now())
        )

        self.embeddings_table_obj = Table(
            self.embeddings_table, self.metadata, *embeddings_columns
        )

        # Create dynamic ORM models based on the table structure
        self._create_dynamic_models()

    def _create_dynamic_models(self) -> None:
        """Create dynamic ORM models that match the table structure."""
        # Create a separate base class for our dynamic models to avoid table conflicts
        dynamic_metadata = MetaData()
        Base = declarative_base(metadata=dynamic_metadata)

        # Create dynamic Document model
        document_attrs = {
            "__tablename__": self.documents_table,
            self.documents_id_column: Column(String, primary_key=True),
            self.documents_content_column: Column(Text, nullable=False),
            "created_at": Column(DateTime, server_default=func.now()),
            "updated_at": Column(
                DateTime, server_default=func.now(), onupdate=func.now()
            ),
        }

        # Add metadata column only if it exists
        if self.documents_metadata_column:
            document_attrs[self.documents_metadata_column] = Column(Text)

            # Add metadata methods with closure to capture column name
            metadata_column_name = self.documents_metadata_column

            def get_metadata(self) -> Dict[str, Any]:
                metadata_value = getattr(self, metadata_column_name, None)
                if metadata_value:
                    return json.loads(metadata_value)
                return {}

            def set_metadata(self, metadata: Dict[str, Any]) -> None:
                setattr(self, metadata_column_name, json.dumps(metadata))

            document_attrs["get_metadata"] = get_metadata
            document_attrs["set_metadata"] = set_metadata
        else:
            # Add empty metadata methods when no metadata column
            def get_metadata(self) -> Dict[str, Any]:
                return {}

            def set_metadata(self, metadata: Dict[str, Any]) -> None:
                pass  # No-op when no metadata column

            document_attrs["get_metadata"] = get_metadata
            document_attrs["set_metadata"] = set_metadata

        self.DocumentModel = type("DynamicDocument", (Base,), document_attrs)

        # Create dynamic Embedding model
        embedding_attrs = {
            "__tablename__": self.embeddings_table,
            self.embeddings_id_column: Column(String, primary_key=True),
            self.embeddings_column: Column(Text, nullable=False),
            "created_at": Column(DateTime, server_default=func.now()),
        }

        # Add document_id column only if it's different from id column
        if self.embeddings_document_id_column != self.embeddings_id_column:
            embedding_attrs[self.embeddings_document_id_column] = Column(
                String, nullable=False, index=True
            )

        # Add model column if specified
        if self.embeddings_model_column and self.embeddings_model_column not in [
            self.embeddings_id_column,
            self.embeddings_document_id_column,
            self.embeddings_column,
        ]:
            embedding_attrs[self.embeddings_model_column] = Column(String)

        # Add vector methods with closure to capture column name
        embeddings_column_name = self.embeddings_column

        def get_vector(self):
            return json.loads(getattr(self, embeddings_column_name))

        def set_vector(self, vector):
            setattr(self, embeddings_column_name, json.dumps(vector))

        embedding_attrs["get_vector"] = get_vector
        embedding_attrs["set_vector"] = set_vector

        self.EmbeddingModel = type("DynamicEmbedding", (Base,), embedding_attrs)

    def get_session(self) -> AsyncSession:
        """Get an async database session."""
        return AsyncSession(self.engine)


@dataclass
class SyncRAGConfig:
    """Configuration for synchronous SQL RAG system.

    Args:
        engine: SQLAlchemy sync engine for database connection
        documents_table: Name of the documents table
        embeddings_table: Name of the embeddings table
        embedding_dimension: Dimension of embedding vectors
        embeddings_column: Name of the column storing embedding vectors
        documents_id_column: Name of the documents ID column
        documents_content_column: Name of the documents content column
        documents_metadata_column: Name of the documents metadata column
        embeddings_id_column: Name of the embeddings ID column
        embeddings_document_id_column: Name of the embeddings document_id column
        embeddings_model_column: Name of the embeddings model column
    """

    engine: Engine
    documents_table: str
    embeddings_table: str
    embedding_dimension: int = 768
    embeddings_column: str = "vector"
    # Document table columns
    documents_id_column: str = "id"
    documents_content_column: str = "content"
    documents_metadata_column: Optional[str] = "doc_metadata"
    # Embedding table columns
    embeddings_id_column: str = "id"
    embeddings_document_id_column: str = "document_id"
    embeddings_model_column: Optional[str] = "model_name"

    def __post_init__(self) -> None:
        if not self.documents_table:
            raise ConfigurationError("documents_table is required")
        if not self.embeddings_table:
            raise ConfigurationError("embeddings_table is required")
        if self.embedding_dimension <= 0:
            raise ConfigurationError("embedding_dimension must be positive")
        
        logger.debug(f"Initializing SyncRAGConfig with tables: {self.documents_table}, {self.embeddings_table}")

        # Create dynamic table objects based on configuration
        self.metadata = MetaData()

        # Create documents table object
        documents_columns = [
            Column(self.documents_id_column, String, primary_key=True),
            Column(self.documents_content_column, Text, nullable=False),
        ]

        # Add metadata column only if specified
        if self.documents_metadata_column:
            documents_columns.append(Column(self.documents_metadata_column, Text))

        documents_columns.extend(
            [
                Column("created_at", DateTime, server_default=func.now()),
                Column(
                    "updated_at",
                    DateTime,
                    server_default=func.now(),
                    onupdate=func.now(),
                ),
            ]
        )

        self.documents_table_obj = Table(
            self.documents_table, self.metadata, *documents_columns
        )

        # Create embeddings table object
        embeddings_columns = []

        # Add ID column as primary key
        embeddings_columns.append(
            Column(self.embeddings_id_column, String, primary_key=True)
        )

        # Add document_id column only if it's different from id column
        if self.embeddings_document_id_column != self.embeddings_id_column:
            embeddings_columns.append(
                Column(
                    self.embeddings_document_id_column,
                    String,
                    nullable=False,
                    index=True,
                )
            )

        # Add embedding vector column
        embeddings_columns.append(Column(self.embeddings_column, Text, nullable=False))

        # Add model column if specified and not already added
        if self.embeddings_model_column and self.embeddings_model_column not in [
            self.embeddings_id_column,
            self.embeddings_document_id_column,
            self.embeddings_column,
        ]:
            embeddings_columns.append(Column(self.embeddings_model_column, String))

        # Add timestamp column
        embeddings_columns.append(
            Column("created_at", DateTime, server_default=func.now())
        )

        self.embeddings_table_obj = Table(
            self.embeddings_table, self.metadata, *embeddings_columns
        )

        # Create dynamic ORM models based on the table structure
        self._create_dynamic_models()

    def _create_dynamic_models(self) -> None:
        """Create dynamic ORM models that match the table structure."""
        # Create a separate base class for our dynamic models to avoid table conflicts
        dynamic_metadata = MetaData()
        Base = declarative_base(metadata=dynamic_metadata)

        # Create dynamic Document model
        document_attrs = {
            "__tablename__": self.documents_table,
            self.documents_id_column: Column(String, primary_key=True),
            self.documents_content_column: Column(Text, nullable=False),
            "created_at": Column(DateTime, server_default=func.now()),
            "updated_at": Column(
                DateTime, server_default=func.now(), onupdate=func.now()
            ),
        }

        # Add metadata column only if it exists
        if self.documents_metadata_column:
            document_attrs[self.documents_metadata_column] = Column(Text)

            # Add metadata methods with closure to capture column name
            metadata_column_name = self.documents_metadata_column

            def get_metadata(self) -> Dict[str, Any]:
                metadata_value = getattr(self, metadata_column_name, None)
                if metadata_value:
                    return json.loads(metadata_value)
                return {}

            def set_metadata(self, metadata: Dict[str, Any]) -> None:
                setattr(self, metadata_column_name, json.dumps(metadata))

            document_attrs["get_metadata"] = get_metadata
            document_attrs["set_metadata"] = set_metadata
        else:
            # Add empty metadata methods when no metadata column
            def get_metadata(self) -> Dict[str, Any]:
                return {}

            def set_metadata(self, metadata: Dict[str, Any]) -> None:
                pass  # No-op when no metadata column

            document_attrs["get_metadata"] = get_metadata
            document_attrs["set_metadata"] = set_metadata

        self.DocumentModel = type("DynamicDocument", (Base,), document_attrs)

        # Create dynamic Embedding model
        embedding_attrs = {
            "__tablename__": self.embeddings_table,
            self.embeddings_id_column: Column(String, primary_key=True),
            self.embeddings_column: Column(Text, nullable=False),
            "created_at": Column(DateTime, server_default=func.now()),
        }

        # Add document_id column only if it's different from id column
        if self.embeddings_document_id_column != self.embeddings_id_column:
            embedding_attrs[self.embeddings_document_id_column] = Column(
                String, nullable=False, index=True
            )

        # Add model column if specified
        if self.embeddings_model_column and self.embeddings_model_column not in [
            self.embeddings_id_column,
            self.embeddings_document_id_column,
            self.embeddings_column,
        ]:
            embedding_attrs[self.embeddings_model_column] = Column(String)

        # Add vector methods with closure to capture column name
        embeddings_column_name = self.embeddings_column

        def get_vector(self):
            return json.loads(getattr(self, embeddings_column_name))

        def set_vector(self, vector):
            setattr(self, embeddings_column_name, json.dumps(vector))

        embedding_attrs["get_vector"] = get_vector
        embedding_attrs["set_vector"] = set_vector

        self.EmbeddingModel = type("DynamicEmbedding", (Base,), embedding_attrs)

    def get_session(self) -> Session:
        """Get a sync database session."""
        return Session(self.engine)
