"""DuckDB configuration."""

from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
import duckdb

try:
    from sqlalchemy import Engine, create_engine
    from sqlalchemy.pool import StaticPool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@dataclass
class DuckDBConfig:
    """Configuration for DuckDB RAG backend."""

    db_path: str
    documents_table: str = "documents"
    embeddings_table: str = "embeddings"
    embedding_dimension: int = 768
    batch_size: int = 1000
    enable_vector_similarity: bool = True
    enable_vss_extension: bool = False
    vss_enable_persistence: bool = True
    engine: Optional[Engine] = None  # SQLAlchemy engine (optional)
    use_sqlalchemy: bool = (
        False  # Whether to use SQLAlchemy instead of native connections
    )
    # Column name mappings for custom schemas
    documents_id_column: str = "id"
    documents_content_column: str = "content"
    documents_metadata_column: Optional[str] = "metadata"
    embeddings_id_column: str = "id"
    embeddings_document_id_column: str = "document_id"
    embeddings_model_column: Optional[str] = "model_name"
    embeddings_column: str = "embedding"

    def __post_init__(self):
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Validate SQLAlchemy usage
        if self.use_sqlalchemy:
            if not SQLALCHEMY_AVAILABLE:
                raise ValueError(
                    "SQLAlchemy is not available. Install with: pip install sqlalchemy"
                )
            if self.engine is None:
                # Create a default SQLAlchemy engine with StaticPool
                self.engine = self._create_default_engine()

    def _create_default_engine(self):
        """Create a default SQLAlchemy engine with StaticPool for DuckDB."""
        if not SQLALCHEMY_AVAILABLE:
            raise ValueError("SQLAlchemy is not available")

        # DuckDB SQLAlchemy support with StaticPool
        if self.db_path == ":memory:":
            # In-memory databases need StaticPool to maintain connection
            engine = create_engine(
                "duckdb:///:memory:", poolclass=StaticPool, connect_args={}, echo=False
            )
        else:
            # File databases can use StaticPool for consistency
            engine = create_engine(
                f"duckdb:///{self.db_path}",
                poolclass=StaticPool,
                connect_args={},
                echo=False,
            )

        return engine

    def get_connection(self) -> Any:
        """Get a DuckDB connection (native duckdb or SQLAlchemy proxied)."""
        if self.use_sqlalchemy:
            if self.engine is None:
                raise ValueError(
                    "SQLAlchemy engine not configured. Set use_sqlalchemy=True and provide an engine."
                )
            # Get raw connection from SQLAlchemy engine
            raw_conn = self.engine.raw_connection()
            # SQLAlchemy returns a PoolProxiedConnection which wraps the DuckDB connection
            return raw_conn
        else:
            # Use native duckdb connections
            return duckdb.connect(self.db_path)

    def get_sqlalchemy_engine(self):
        """Get the SQLAlchemy engine if available."""
        if not self.use_sqlalchemy or self.engine is None:
            raise ValueError(
                "SQLAlchemy engine not configured. Set use_sqlalchemy=True"
            )
        return self.engine

    def get_documents_schema(self) -> str:
        """Get the CREATE TABLE SQL for documents."""
        metadata_col = (
            f"{self.documents_metadata_column} TEXT,"
            if self.documents_metadata_column
            else ""
        )

        return f"""
        CREATE TABLE IF NOT EXISTS {self.documents_table} (
            {self.documents_id_column} VARCHAR PRIMARY KEY,
            {self.documents_content_column} TEXT NOT NULL,
            {metadata_col}
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

    def get_embeddings_schema(self) -> str:
        """Get the CREATE TABLE SQL for embeddings."""
        model_col = (
            f"{self.embeddings_model_column} VARCHAR,"
            if self.embeddings_model_column
            else ""
        )

        return f"""
        CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
            {self.embeddings_id_column} VARCHAR PRIMARY KEY,
            {self.embeddings_document_id_column} VARCHAR NOT NULL,
            {self.embeddings_column} FLOAT[{self.embedding_dimension}] NOT NULL,
            {model_col}
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

    def setup_database(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Set up the database schema and functions."""
        # Install and load VSS extension if enabled
        if self.enable_vss_extension:
            try:
                conn.execute("INSTALL vss")
                conn.execute("LOAD vss")

                # Enable experimental persistence if using file-based database
                if self.vss_enable_persistence and self.db_path != ":memory:":
                    conn.execute("SET hnsw_enable_experimental_persistence = true")

            except Exception as e:
                # VSS extension might not be available, continue without it
                print(f"Warning: Could not load VSS extension: {e}")
                self.enable_vss_extension = False

        # Check if tables already exist before trying to create them
        documents_exists = self._table_exists(conn, self.documents_table)
        embeddings_exists = self._table_exists(conn, self.embeddings_table)

        # Only create tables if they don't exist
        if not documents_exists:
            try:
                conn.execute(self.get_documents_schema())
            except Exception as e:
                print(
                    f"Warning: Could not create documents table {self.documents_table}: {e}"
                )

        if not embeddings_exists:
            try:
                conn.execute(self.get_embeddings_schema())
            except Exception as e:
                print(
                    f"Warning: Could not create embeddings table {self.embeddings_table}: {e}"
                )

        # Create indexes for performance - with error handling for existing tables with different schemas
        try:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.embeddings_table}_{self.embeddings_document_id_column} ON {self.embeddings_table}({self.embeddings_document_id_column});"
            )
        except Exception as e:
            # Column might not exist in existing table
            print(
                f"Warning: Could not create {self.embeddings_document_id_column} index on {self.embeddings_table}: {e}"
            )

    def _table_exists(self, conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            result = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            return result is not None
        except Exception:
            # If we can't check, assume it doesn't exist and let the CREATE TABLE IF NOT EXISTS handle it
            return False
