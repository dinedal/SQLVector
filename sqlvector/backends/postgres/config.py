"""PostgreSQL configuration with pgvector support."""

import asyncpg
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

try:
    from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool, QueuePool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL RAG backend with pgvector support."""

    # Database connection parameters
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    db_url: Optional[str] = None  # Alternative: full connection URL
    
    # Table configuration
    documents_table: str = "documents"
    embeddings_table: str = "embeddings"
    embedding_dimension: int = 768
    batch_size: int = 1000
    
    # pgvector index configuration
    index_type: str = "ivfflat"  # "ivfflat" or "hnsw"
    index_lists: int = 100  # For IVFFlat
    index_m: int = 16  # For HNSW
    index_ef_construction: int = 64  # For HNSW
    
    # SQLAlchemy support
    engine: Optional[Union[AsyncEngine, Any]] = None  # SQLAlchemy engine (optional)
    use_sqlalchemy: bool = False  # Whether to use SQLAlchemy instead of asyncpg
    use_async_sqlalchemy: bool = True  # If using SQLAlchemy, use async version
    
    # Column name mappings for custom schemas
    documents_id_column: str = "id"
    documents_content_column: str = "content"
    documents_metadata_column: Optional[str] = "metadata"
    embeddings_id_column: str = "id"
    embeddings_document_id_column: str = "document_id"
    embeddings_model_column: Optional[str] = "model_name"
    embeddings_column: str = "embedding"
    
    # Connection pool settings (for asyncpg)
    pool_min_size: int = 2
    pool_max_size: int = 10
    
    # Private attributes
    _pool: Optional[asyncpg.Pool] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Build connection URL if not provided
        if not self.db_url and self.host:
            password_part = f":{self.password}@" if self.password else "@"
            user_part = f"{self.user}{password_part}" if self.user else ""
            port_part = f":{self.port}" if self.port else ""
            self.db_url = f"postgresql://{user_part}{self.host}{port_part}/{self.database}"
        
        # Validate SQLAlchemy usage
        if self.use_sqlalchemy:
            if not SQLALCHEMY_AVAILABLE:
                raise ValueError(
                    "SQLAlchemy is not available. Install with: pip install sqlalchemy[asyncio]"
                )
            if self.engine is None:
                # Create a default SQLAlchemy engine
                self.engine = self._create_default_engine()
    
    def _create_default_engine(self):
        """Create a default SQLAlchemy engine."""
        if not SQLALCHEMY_AVAILABLE:
            raise ValueError("SQLAlchemy is not available")
        
        if not self.db_url:
            raise ValueError("Database URL is required for SQLAlchemy engine")
        
        if self.use_async_sqlalchemy:
            # Create async engine
            async_url = self.db_url.replace("postgresql://", "postgresql+asyncpg://")
            engine = create_async_engine(
                async_url,
                pool_size=self.pool_max_size,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
            )
        else:
            # Create sync engine
            sync_url = self.db_url.replace("postgresql://", "postgresql+psycopg2://")
            engine = create_engine(
                sync_url,
                pool_size=self.pool_max_size,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
            )
        
        return engine
    
    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create asyncpg connection pool."""
        if self._pool is None:
            if not self.db_url:
                raise ValueError("Database URL is required")
            
            self._pool = await asyncpg.create_pool(
                self.db_url,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=60,
            )
        return self._pool
    
    async def close_pool(self):
        """Close the asyncpg connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get an async database connection."""
        if self.use_sqlalchemy:
            if not self.use_async_sqlalchemy:
                raise ValueError("Async connection requested but use_async_sqlalchemy is False")
            
            # Get connection from SQLAlchemy async engine
            async with self.engine.connect() as conn:
                yield conn
        else:
            # Get connection from asyncpg pool
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                yield conn
    
    @contextmanager
    def get_sync_connection(self):
        """Get a sync database connection (SQLAlchemy only)."""
        if not self.use_sqlalchemy:
            raise ValueError("Sync connections require SQLAlchemy. Set use_sqlalchemy=True")
        
        if self.use_async_sqlalchemy:
            raise ValueError("Sync connection requested but use_async_sqlalchemy is True")
        
        # Get connection from SQLAlchemy sync engine
        with self.engine.connect() as conn:
            yield conn
    
    @contextmanager 
    def get_connection_context(self):
        """Get a connection context manager (for compatibility with sync operations)."""
        if self.use_sqlalchemy and not self.use_async_sqlalchemy:
            with self.get_sync_connection() as conn:
                yield conn
        else:
            # For async connections, this would need to be handled differently
            # This is primarily for SQLite compatibility
            raise ValueError(
                "Sync context manager not available for async connections. "
                "Use get_async_connection() for async operations or set "
                "use_sqlalchemy=True and use_async_sqlalchemy=False for sync operations."
            )
    
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
            f"{self.documents_metadata_column} JSONB,"
            if self.documents_metadata_column
            else ""
        )
        
        return f"""
        CREATE TABLE IF NOT EXISTS {self.documents_table} (
            {self.documents_id_column} TEXT PRIMARY KEY,
            {self.documents_content_column} TEXT NOT NULL,
            {metadata_col}
            created_at TIMESTAMP DEFAULT NOW()
        )
        """
    
    def get_embeddings_schema(self) -> str:
        """Get the CREATE TABLE SQL for embeddings with pgvector support."""
        model_col = (
            f"{self.embeddings_model_column} TEXT,"
            if self.embeddings_model_column
            else ""
        )
        
        return f"""
        CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
            {self.embeddings_id_column} TEXT PRIMARY KEY,
            {self.embeddings_document_id_column} TEXT NOT NULL REFERENCES {self.documents_table}({self.documents_id_column}) ON DELETE CASCADE,
            {self.embeddings_column} vector({self.embedding_dimension}) NOT NULL,
            {model_col}
            created_at TIMESTAMP DEFAULT NOW()
        )
        """
    
    def get_index_schema(self, index_name: str, similarity_function: str = "cosine") -> str:
        """Get the CREATE INDEX SQL for vector similarity search."""
        # Map similarity functions to pgvector operators
        ops_map = {
            "cosine": "vector_cosine_ops",
            "euclidean": "vector_l2_ops", 
            "inner_product": "vector_ip_ops"
        }
        
        ops = ops_map.get(similarity_function, "vector_cosine_ops")
        
        if self.index_type == "hnsw":
            return f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {self.embeddings_table} 
            USING hnsw ({self.embeddings_column} {ops})
            WITH (m = {self.index_m}, ef_construction = {self.index_ef_construction})
            """
        else:  # ivfflat
            return f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {self.embeddings_table}
            USING ivfflat ({self.embeddings_column} {ops})
            WITH (lists = {self.index_lists})
            """
    
    async def setup_database(self, conn: Any) -> None:
        """Set up the database schema and extensions."""
        # Check if we're using asyncpg or SQLAlchemy
        is_asyncpg = isinstance(conn, asyncpg.Connection)
        
        if is_asyncpg:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create tables
            await conn.execute(self.get_documents_schema())
            await conn.execute(self.get_embeddings_schema())
            
            # Create default index
            await conn.execute(
                self.get_index_schema(f"{self.embeddings_table}_idx", "cosine")
            )
        else:
            # SQLAlchemy connection
            from sqlalchemy import text
            
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Create tables
            conn.execute(text(self.get_documents_schema()))
            conn.execute(text(self.get_embeddings_schema()))
            
            # Create default index
            conn.execute(
                text(self.get_index_schema(f"{self.embeddings_table}_idx", "cosine"))
            )
            
            conn.commit()