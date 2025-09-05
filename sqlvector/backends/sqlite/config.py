"""SQLite configuration."""

import sqlite3
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from contextlib import contextmanager
from pathlib import Path

try:
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import create_engine

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@dataclass
class SQLiteConfig:
    """Configuration for SQLite RAG backend."""

    db_path: str
    documents_table: str = "documents"
    embeddings_table: str = "embeddings"
    vss_table: str = "vss_embeddings"
    embedding_dimension: int = 768
    batch_size: int = 1000
    enable_vss_extension: bool = False
    vss_factory_string: str = "Flat"
    engine: Optional[Union[AsyncEngine, Any]] = None  # SQLAlchemy engine (optional)
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
    _shared_connection: Optional[sqlite3.Connection] = None

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
        """Create a default SQLAlchemy engine with StaticPool for SQLite."""
        if not SQLALCHEMY_AVAILABLE:
            raise ValueError("SQLAlchemy is not available")

        # Create engine with StaticPool for SQLite
        if self.db_path == ":memory:":
            # In-memory databases need StaticPool to maintain connection
            engine = create_engine(
                "sqlite:///:memory:",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            # File databases can use StaticPool for consistency
            engine = create_engine(
                f"sqlite:///{self.db_path}",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False,
            )

        return engine

    def _load_vss_extension(self, conn: sqlite3.Connection) -> bool:
        """Load VSS extension into a connection if enabled.
        
        Returns:
            True if VSS was successfully loaded or not needed, False otherwise.
        """
        if not self.enable_vss_extension:
            return True
        
        try:
            # Enable extension loading
            conn.enable_load_extension(True)
            
            # Try to load sqlite-vss extension
            try:
                import sqlite_vss
                sqlite_vss.load(conn)
                # Disable extension loading after successful load
                conn.enable_load_extension(False)
                return True
            except ImportError:
                # sqlite_vss package not available
                self.enable_vss_extension = False
                return False
                
        except sqlite3.OperationalError:
            # Extension loading not supported
            self.enable_vss_extension = False
            return False
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection (native sqlite3)."""
        if self.use_sqlalchemy:
            # Get raw connection from SQLAlchemy engine
            raw_conn = self.engine.raw_connection()
            # Load VSS extension if enabled
            self._load_vss_extension(raw_conn)
            # SQLAlchemy returns a DBAPI connection which for SQLite is sqlite3.Connection
            return raw_conn
        else:
            # Use native sqlite3 connections
            # For in-memory databases, reuse the same connection to preserve data
            if self.db_path == ":memory:":
                if self._shared_connection is None:
                    self._shared_connection = sqlite3.connect(
                        self.db_path,
                        check_same_thread=False,  # Allow use from different threads
                    )
                    self._shared_connection.row_factory = sqlite3.Row
                    self._shared_connection.execute("PRAGMA foreign_keys = ON")
                    # Load VSS extension for the shared connection
                    self._load_vss_extension(self._shared_connection)
                return self._shared_connection
            else:
                # For file databases, create new connections each time
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                # Load VSS extension for each new connection
                self._load_vss_extension(conn)
                return conn

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
            {self.documents_id_column} TEXT PRIMARY KEY,
            {self.documents_content_column} TEXT NOT NULL,
            {metadata_col}
            hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

    def get_embeddings_schema(self) -> str:
        """Get the CREATE TABLE SQL for embeddings."""
        model_col = (
            f"{self.embeddings_model_column} TEXT,"
            if self.embeddings_model_column
            else ""
        )
        
        return f"""
        CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
            {self.embeddings_id_column} TEXT PRIMARY KEY,
            {self.embeddings_document_id_column} TEXT NOT NULL,
            hash TEXT NOT NULL,
            {self.embeddings_column} BLOB NOT NULL,
            {model_col}
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

    def get_vss_schema(self) -> str:
        """Get the CREATE VIRTUAL TABLE SQL for VSS embeddings."""
        # Note: sqlite-vss doesn't support factory parameter in CREATE TABLE
        # The factory string is used during index operations, not table creation
        return f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self.vss_table} USING vss0(
            embedding({self.embedding_dimension})
        )
        """

    def setup_database(self, conn: sqlite3.Connection) -> None:
        """Set up the database schema and functions.
        
        Note: VSS extension loading is now handled in get_connection() for each connection.
        """

        # Create tables
        conn.execute(self.get_documents_schema())
        conn.execute(self.get_embeddings_schema())

        # Create VSS virtual table if extension is available
        if self.enable_vss_extension:
            try:
                conn.execute(self.get_vss_schema())
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not create VSS virtual table: {e}")
                self.enable_vss_extension = False

        # No need for mapping table - we'll use direct rowid relationships

        # Create indexes for performance
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.documents_table}_hash ON {self.documents_table}(hash)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.embeddings_table}_hash ON {self.embeddings_table}(hash)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.embeddings_table}_{self.embeddings_document_id_column} ON {self.embeddings_table}({self.embeddings_document_id_column})"
        )

        conn.commit()

    @contextmanager
    def get_connection_context(self):
        """Get a connection with proper context management."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            # Handle connection cleanup based on connection type
            if self.use_sqlalchemy:
                # SQLAlchemy engine manages connection lifecycle
                conn.close()
            else:
                # Only close if it's not a shared in-memory connection
                if self.db_path != ":memory:":
                    conn.close()

    def close_shared_connection(self):
        """Close shared connection (for in-memory databases)."""
        if self._shared_connection is not None:
            self._shared_connection.close()
            self._shared_connection = None
