"""PostgreSQL configuration with pgvector support."""

import asyncpg
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

# Context variable to track sync-to-async operations
_sync_context: ContextVar[bool] = ContextVar('sync_context', default=False)

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class Psycopg2AsyncPGWrapper:
    """Wrapper to make psycopg2 connections behave like asyncpg connections."""
    
    def __init__(self, psycopg2_conn):
        self._conn = psycopg2_conn
        
    def __getattr__(self, name):
        # Delegate any other attributes to the underlying connection
        return getattr(self._conn, name)
        
    async def execute(self, query, *args):
        """Execute a query and return the cursor (like asyncpg)."""
        cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Handle SQLAlchemy text() objects
        if hasattr(query, 'text'):
            query = str(query)
        
        # Adapt query parameters
        if args:
            # Convert $1, $2 style to %s style
            import re
            adapted_query = re.sub(r'\$\d+', '%s', query)
            cursor.execute(adapted_query, args)
        else:
            cursor.execute(query)
        
        return cursor
        
    async def executemany(self, query, param_list):
        """Execute a query with multiple parameter sets."""
        cursor = self._conn.cursor()
        
        # Handle SQLAlchemy text() objects
        if hasattr(query, 'text'):
            query = str(query)
        
        # Convert $1, $2 style to %s style  
        import re
        adapted_query = re.sub(r'\$\d+', '%s', query)
        cursor.executemany(adapted_query, param_list)
        
        return cursor
        
    async def fetch(self, query, *args, **kwargs):
        """Execute a query and return all results as a list of dicts."""
        cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Handle SQLAlchemy text() objects
        if hasattr(query, 'text'):
            query = str(query)
        
        # Handle both positional and named parameters
        if kwargs:
            # Named parameters - convert :param to %(param)s format for psycopg2
            # Use smart parsing to avoid PostgreSQL cast operators like ::vector
            import re
            
            adapted_query = query
            # Sort kwargs keys by length (longest first) to avoid partial matches
            sorted_params = sorted(kwargs.keys(), key=len, reverse=True)
            
            for param_name in sorted_params:
                # Match :param when followed by ::, whitespace, end of string, or newline
                pattern = f":{param_name}(?=::|\\s|$|\\n)"
                if re.search(pattern, adapted_query):
                    adapted_query = re.sub(pattern, f"%({param_name})s", adapted_query)
            
            cursor.execute(adapted_query, kwargs)
        elif args:
            # Positional parameters - convert $1, $2, etc. to %s for psycopg2
            import re
            adapted_query = re.sub(r'\$\d+', '%s', query)
            cursor.execute(adapted_query, args)
        else:
            cursor.execute(query)
        
        results = cursor.fetchall()
        cursor.close()
        return results
        
    async def fetchrow(self, query, *args, **kwargs):
        """Execute a query and return the first result as a dict."""
        cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Handle SQLAlchemy text() objects
        if hasattr(query, 'text'):
            query = str(query)
        
        # Handle both positional and named parameters
        if kwargs:
            # Named parameters - convert :param to %(param)s format for psycopg2
            # Use smart parsing to avoid PostgreSQL cast operators like ::vector
            import re
            
            adapted_query = query
            # Sort kwargs keys by length (longest first) to avoid partial matches
            sorted_params = sorted(kwargs.keys(), key=len, reverse=True)
            
            for param_name in sorted_params:
                # Match :param when followed by ::, whitespace, end of string, or newline
                pattern = f":{param_name}(?=::|\\s|$|\\n)"
                if re.search(pattern, adapted_query):
                    adapted_query = re.sub(pattern, f"%({param_name})s", adapted_query)
            
            cursor.execute(adapted_query, kwargs)
        elif args:
            # Positional parameters - convert $1, $2, etc. to %s for psycopg2
            import re
            adapted_query = re.sub(r'\$\d+', '%s', query)
            cursor.execute(adapted_query, args)
        else:
            cursor.execute(query)
        
        result = cursor.fetchone()
        cursor.close()
        return result
        
    async def fetchval(self, query, *args, **kwargs):
        """Execute a query and return the first column of the first row."""
        cursor = self._conn.cursor()
        
        # Handle SQLAlchemy text() objects
        if hasattr(query, 'text'):
            query = str(query)
        
        # Handle both positional and named parameters
        if kwargs:
            # Named parameters - convert :param to %(param)s format for psycopg2
            # Use smart parsing to avoid PostgreSQL cast operators like ::vector
            import re
            
            adapted_query = query
            # Sort kwargs keys by length (longest first) to avoid partial matches
            sorted_params = sorted(kwargs.keys(), key=len, reverse=True)
            
            for param_name in sorted_params:
                # Match :param when followed by ::, whitespace, end of string, or newline
                pattern = f":{param_name}(?=::|\\s|$|\\n)"
                if re.search(pattern, adapted_query):
                    adapted_query = re.sub(pattern, f"%({param_name})s", adapted_query)
            
            cursor.execute(adapted_query, kwargs)
        elif args:
            # Positional parameters - convert $1, $2, etc. to %s for psycopg2
            import re
            adapted_query = re.sub(r'\$\d+', '%s', query)
            cursor.execute(adapted_query, args)
        else:
            cursor.execute(query)
        
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else None


class AsyncPGToWrapperAdapter:
    """Adapter to make asyncpg connections work like our psycopg2 wrapper."""
    
    def __init__(self, asyncpg_conn):
        self._conn = asyncpg_conn
    
    async def fetch(self, query, *args, **kwargs):
        """Execute a query and return all results as a list of dicts."""
        # Handle both positional and named parameters
        if kwargs:
            # Named parameters - convert :param to $1, $2 format for asyncpg
            # Simple approach: replace specific known patterns
            import re
            
            # Replace our known named parameters in order
            asyncpg_query = query
            param_counter = 1
            param_mapping = {}
            
            # Sort kwargs keys by length (longest first) to avoid partial matches
            sorted_params = sorted(kwargs.keys(), key=len, reverse=True)
            
            for param_name in sorted_params:
                # Match :param when followed by ::, whitespace, end of string, or newline
                pattern = f":{param_name}(?=::|\\s|$|\\n)"  
                if re.search(pattern, asyncpg_query):
                    param_mapping[param_name] = param_counter
                    asyncpg_query = re.sub(pattern, f"${param_counter}", asyncpg_query)
                    param_counter += 1
            # Create ordered parameter list based on mapping
            ordered_params = [None] * (param_counter - 1)
            for param_name, index in param_mapping.items():
                ordered_params[index - 1] = kwargs[param_name]
            
            rows = await self._conn.fetch(asyncpg_query, *ordered_params)
        elif args:
            # Positional parameters - convert $1, $2, etc. to asyncpg format (already correct)
            rows = await self._conn.fetch(query, *args)
        else:
            rows = await self._conn.fetch(query)
        
        return [dict(row) for row in rows]
    
    async def fetchrow(self, query, *args, **kwargs):
        """Execute a query and return the first result as a dict."""
        # Handle both positional and named parameters
        if kwargs:
            # Named parameters - convert :param to $1, $2 format for asyncpg
            # Simple approach: replace specific known patterns
            import re
            
            # Replace our known named parameters in order
            asyncpg_query = query
            param_counter = 1
            param_mapping = {}
            
            # Sort kwargs keys by length (longest first) to avoid partial matches
            sorted_params = sorted(kwargs.keys(), key=len, reverse=True)
            
            for param_name in sorted_params:
                # Match :param when followed by ::, whitespace, end of string, or newline
                pattern = f":{param_name}(?=::|\\s|$|\\n)"
                if re.search(pattern, asyncpg_query):
                    param_mapping[param_name] = param_counter
                    asyncpg_query = re.sub(pattern, f"${param_counter}", asyncpg_query)
                    param_counter += 1
            # Create ordered parameter list based on mapping
            ordered_params = [None] * (param_counter - 1)
            for param_name, index in param_mapping.items():
                ordered_params[index - 1] = kwargs[param_name]
            
            row = await self._conn.fetchrow(asyncpg_query, *ordered_params)
        elif args:
            # Positional parameters
            row = await self._conn.fetchrow(query, *args)
        else:
            row = await self._conn.fetchrow(query)
        
        return dict(row) if row else None
    
    async def fetchval(self, query, *args, **kwargs):
        """Execute a query and return the first column of the first row."""
        # Handle both positional and named parameters
        if kwargs:
            # Named parameters - convert :param to $1, $2 format for asyncpg
            # Simple approach: replace specific known patterns
            import re
            
            # Replace our known named parameters in order
            asyncpg_query = query
            param_counter = 1
            param_mapping = {}
            
            # Sort kwargs keys by length (longest first) to avoid partial matches
            sorted_params = sorted(kwargs.keys(), key=len, reverse=True)
            
            for param_name in sorted_params:
                # Match :param when followed by ::, whitespace, end of string, or newline
                pattern = f":{param_name}(?=::|\\s|$|\\n)"
                if re.search(pattern, asyncpg_query):
                    param_mapping[param_name] = param_counter
                    asyncpg_query = re.sub(pattern, f"${param_counter}", asyncpg_query)
                    param_counter += 1
            # Create ordered parameter list based on mapping
            ordered_params = [None] * (param_counter - 1)
            for param_name, index in param_mapping.items():
                ordered_params[index - 1] = kwargs[param_name]
            
            result = await self._conn.fetchval(asyncpg_query, *ordered_params)
        elif args:
            # Positional parameters
            result = await self._conn.fetchval(query, *args)
        else:
            result = await self._conn.fetchval(query)
        
        return result
    
    async def execute(self, query, *args, **kwargs):
        """Execute a query without returning results."""
        # Handle SQLAlchemy text() objects
        if hasattr(query, 'text'):
            query = str(query)
        
        # Handle both positional and named parameters
        if kwargs:
            # Named parameters - convert :param to $1, $2 format for asyncpg
            # Simple approach: replace specific known patterns
            import re
            
            # Replace our known named parameters in order
            asyncpg_query = query
            param_counter = 1
            param_mapping = {}
            
            # Sort kwargs keys by length (longest first) to avoid partial matches
            sorted_params = sorted(kwargs.keys(), key=len, reverse=True)
            
            for param_name in sorted_params:
                # Match :param when followed by ::, whitespace, end of string, or newline
                pattern = f":{param_name}(?=::|\\s|$|\\n)"  
                if re.search(pattern, asyncpg_query):
                    param_mapping[param_name] = param_counter
                    asyncpg_query = re.sub(pattern, f"${param_counter}", asyncpg_query)
                    param_counter += 1
            # Create ordered parameter list based on mapping
            ordered_params = [None] * (param_counter - 1)
            for param_name, index in param_mapping.items():
                ordered_params[index - 1] = kwargs[param_name]
            
            result = await self._conn.execute(asyncpg_query, *ordered_params)
        elif args:
            # Positional parameters
            result = await self._conn.execute(query, *args)
        else:
            result = await self._conn.execute(query)
        
        return result
    
    async def executemany(self, query, param_list):
        """Execute a query multiple times with different parameters."""
        # Handle SQLAlchemy text() objects
        if hasattr(query, 'text'):
            query = str(query)
        
        # Use asyncpg's executemany method
        await self._conn.executemany(query, param_list)


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
    
    # Connection management  
    max_total_connections: Optional[int] = None  # Global connection limit
    connection_timeout: float = 30.0  # Connection acquisition timeout
    
    # Private attributes
    _pool: Optional[asyncpg.Pool] = None
    _pool_lock: Optional[threading.Lock] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Initialize connection pool lock
        self._pool_lock = threading.Lock()
        
        # Set default max_total_connections based on pool settings
        if self.max_total_connections is None:
            self.max_total_connections = self.pool_max_size
        
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
        """Get or create a connection pool for async operations only."""
        # Use lock to ensure thread-safe pool creation
        with self._pool_lock:
            if self._pool is not None and not self._pool._closed:
                return self._pool
            
            if not self.db_url:
                raise ValueError("Database URL is required")
            
            # Create pool for async operations only
            self._pool = await self._create_pool_with_retry()
            
            return self._pool
    
    
    async def _create_pool_with_retry(self) -> asyncpg.Pool:
        """Create connection pool with retry logic for connection limit errors."""
        import asyncio
        import random
        import logging
        
        logger = logging.getLogger(__name__)
        
        max_retries = 3
        base_delay = 0.5
        max_delay = 5.0
        
        # Use conservative pool settings to minimize connection usage
        min_size = max(1, self.pool_min_size // 2)  
        max_size = self.max_total_connections or self.pool_max_size
        
        for attempt in range(max_retries + 1):
            try:
                pool = await asyncpg.create_pool(
                    self.db_url,
                    min_size=min_size,
                    max_size=max_size,
                    command_timeout=60,
                    server_settings={
                        'application_name': 'sqlvector_postgres_rag',
                    },
                    init=self._init_connection,
                )
                
                logger.info(f"Created connection pool with {min_size}-{max_size} connections")
                return pool
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a connection limit error
                if ("too many clients" in error_msg or 
                    "connection limit exceeded" in error_msg or
                    "maximum number of connections" in error_msg):
                    
                    if attempt < max_retries:
                        # Reduce connection count and retry
                        max_size = max(1, max_size // 2)
                        min_size = min(min_size, max_size)
                        
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = delay * 0.1 * random.random()
                        total_delay = delay + jitter
                        
                        logger.warning(
                            f"Connection pool creation failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"reducing to {min_size}-{max_size} connections and retrying in {total_delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(total_delay)
                        continue
                    else:
                        logger.error(f"Failed to create connection pool after {max_retries} retries: {e}")
                        raise
                else:
                    # Non-retry-able error
                    logger.error(f"Connection pool creation error: {e}")
                    raise
        
        raise RuntimeError("Unreachable code")
    
    
    async def _init_connection(self, conn):
        """Initialize connection settings."""
        # Set proper encoding and timezone
        await conn.execute("SET CLIENT_ENCODING TO 'UTF8'")
        await conn.execute("SET TIMEZONE TO 'UTC'")
    
    async def close_pool(self):
        """Close the single global connection pool."""
        pool_to_close = None
        
        with self._pool_lock:
            if self._pool and not self._pool._closed:
                pool_to_close = self._pool
                self._pool = None
        
        if pool_to_close:
            try:
                await pool_to_close.close()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error closing connection pool: {e}")
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get a database connection - async pool for real async, psycopg2 for sync operations."""
        if self.use_sqlalchemy:
            if not self.use_async_sqlalchemy:
                raise ValueError("Async connection requested but use_async_sqlalchemy is False")
            
            # Get connection from SQLAlchemy async engine
            async with self.engine.connect() as conn:
                yield conn
        else:
            # Check if we're in a sync-to-async bridge thread
            if self._is_in_sync_thread():
                # Use psycopg2 for sync operations to avoid event loop conflicts
                with self.get_sync_connection() as conn:
                    yield conn
            else:
                # Use asyncpg pool for real async operations but wrap it
                async with self._get_connection_with_retry() as raw_conn:
                    # Convert asyncpg connection to use our wrapper interface
                    wrapper = AsyncPGToWrapperAdapter(raw_conn)
                    yield wrapper
    
    @contextmanager
    def get_sync_connection(self):
        """Get a synchronous database connection using psycopg2."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 is required for sync operations but is not installed")
        
        if not self.db_url:
            raise ValueError("Database URL is required")
        
        # Use the URL directly - psycopg2 supports postgresql:// URLs
        conn = None
        try:
            # Create synchronous connection using DSN
            conn = psycopg2.connect(self.db_url)
            
            # Configure connection
            conn.autocommit = True
            
            # Initialize connection (pgvector extension, etc.)
            with conn.cursor() as cursor:
                self._init_sync_connection(cursor)
            
            # Wrap psycopg2 connection to provide AsyncPG-like interface
            wrapped_conn = self._wrap_psycopg2_connection(conn)
            yield wrapped_conn
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Sync connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _parse_db_url_for_psycopg2(self, db_url: str) -> Dict[str, Any]:
        """Parse database URL for psycopg2 connection parameters."""
        # psycopg2 can actually use the full URL directly
        # But let's also support manual parsing for compatibility
        try:
            import urllib.parse as urlparse
            
            parsed = urlparse.urlparse(db_url)
            
            params = {}
            
            if parsed.hostname:
                params['host'] = parsed.hostname
            if parsed.port:
                params['port'] = parsed.port
            if parsed.path and parsed.path.lstrip('/'):
                params['database'] = parsed.path.lstrip('/')
            if parsed.username:
                params['user'] = parsed.username
            if parsed.password:
                params['password'] = parsed.password
            
            return params
            
        except Exception:
            # Fallback: just use the URL directly (psycopg2 supports this)
            return {'dsn': db_url}
    
    def _init_sync_connection(self, cursor):
        """Initialize synchronous connection with required settings."""
        try:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Set encoding and timezone
            cursor.execute("SET CLIENT_ENCODING TO 'UTF8'")
            cursor.execute("SET TIMEZONE TO 'UTC'")
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Sync connection initialization: {e}")
    
    def _is_in_sync_thread(self) -> bool:
        """Check if we're in a sync-to-async bridge context."""
        # Check context variable first (works for asyncio.run case)
        try:
            if _sync_context.get():
                return True
        except LookupError:
            pass
        
        # Fallback: check thread name (works for isolated thread case)
        thread_name = threading.current_thread().name
        return "sqlvector_async" in thread_name
    
    def _wrap_psycopg2_connection(self, conn):
        """Wrap a psycopg2 connection to provide AsyncPG-like interface."""
        return Psycopg2AsyncPGWrapper(conn)
    
    def adapt_sql_for_driver(self, query: str, params: tuple, is_psycopg2: bool = False):
        """Adapt SQL query and parameters for different drivers."""
        if is_psycopg2:
            # Convert asyncpg-style ($1, $2) to psycopg2-style (%s) parameters
            import re
            
            # Replace $1, $2, etc. with %s
            adapted_query = re.sub(r'\$(\d+)', '%s', query)
            
            return adapted_query, params
        else:
            # Return as-is for asyncpg
            return query, params
    
    def execute_query_with_driver(self, conn, query: str, params: tuple = ()):
        """Execute query with the appropriate driver method."""
        # Detect driver type
        if hasattr(conn, 'fetch'):
            # AsyncPG connection
            return conn.fetch(query, *params)
        elif hasattr(conn, 'cursor'):
            # Psycopg2 connection
            adapted_query, adapted_params = self.adapt_sql_for_driver(query, params, is_psycopg2=True)
            
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(adapted_query, adapted_params)
            return cursor.fetchall()
        else:
            # SQLAlchemy connection
            from sqlalchemy import text
            # For SQLAlchemy, adapt parameter style
            adapted_query = query.replace('$1', ':param1').replace('$2', ':param2').replace('$3', ':param3').replace('$4', ':param4')
            param_dict = {f'param{i+1}': param for i, param in enumerate(params)}
            result = conn.execute(text(adapted_query), param_dict)
            return result.fetchall()
    
    
    @asynccontextmanager
    async def _get_connection_with_retry(self):
        """Get connection from pool with exponential backoff retry for connection limits."""
        import asyncio
        import logging
        import random
        
        logger = logging.getLogger(__name__)
        
        max_retries = 3
        base_delay = 0.1  # Start with 100ms delay
        max_delay = 2.0   # Maximum delay of 2 seconds
        
        for attempt in range(max_retries + 1):
            try:
                pool = await self._get_pool()
                conn = None
                try:
                    # Use configured timeout
                    conn = await pool.acquire(timeout=self.connection_timeout)
                    
                    try:
                        yield conn
                    finally:
                        # Connection cleanup in finally block
                        if conn:
                            try:
                                await self._safe_release_connection(pool, conn)
                            except Exception as release_error:
                                logger.warning(f"Error during pool connection cleanup: {release_error}")
                    
                    # If we reach here, connection was successful
                    return
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check if it's a connection limit error
                    if ("too many clients" in error_msg or 
                        "connection limit exceeded" in error_msg or
                        "maximum number of connections" in error_msg):
                        
                        if attempt < max_retries:
                            # Calculate delay with jitter
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            jitter = delay * 0.1 * random.random()  # Add up to 10% jitter
                            total_delay = delay + jitter
                            
                            logger.warning(
                                f"Connection limit reached (attempt {attempt + 1}/{max_retries + 1}), "
                                f"retrying in {total_delay:.2f}s: {e}"
                            )
                            await asyncio.sleep(total_delay)
                            continue
                        else:
                            logger.error(f"Failed to acquire connection after {max_retries} retries: {e}")
                            raise
                    else:
                        # Non-retry-able error, log and re-raise immediately
                        logger.error(f"Connection error: {e}")
                        raise
                        
            except Exception as e:
                # If this is the last attempt or a non-retryable error, re-raise
                if attempt >= max_retries:
                    raise
                
                # For other exceptions, only retry if it might be connection-related
                error_msg = str(e).lower()
                if not any(keyword in error_msg for keyword in [
                    "connection", "timeout", "network", "pool"
                ]):
                    # Non-connection error, don't retry
                    raise
    
    async def _safe_release_connection(self, pool: asyncpg.Pool, conn: asyncpg.Connection):
        """Safely release a connection back to the pool."""
        import asyncio
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Check if we have a valid event loop before attempting release
            try:
                loop = asyncio.get_running_loop()
                if loop.is_closed():
                    logger.warning("Event loop is closed, cannot release connection properly")
                    return
            except RuntimeError:
                logger.warning("No running event loop for connection release")
                return
            
            # Ensure connection is properly released
            await pool.release(conn, timeout=10)
            
        except RuntimeError as e:
            # Handle event loop errors specifically
            if "event loop is closed" in str(e).lower() or "no running event loop" in str(e).lower():
                logger.debug("Event loop closed during connection release, attempting synchronous cleanup")
                # Try to forcefully close the connection synchronously
                try:
                    if not conn.is_closed() and hasattr(conn, '_transport') and conn._transport:
                        conn._transport.close()
                except Exception as sync_close_error:
                    logger.debug(f"Synchronous connection close also failed: {sync_close_error}")
            else:
                logger.warning(f"Runtime error releasing connection: {e}")
                # Try to forcefully close the connection if release fails
        except Exception as e:
            logger.warning(f"Error releasing connection: {e}")
            
            # Try to forcefully close the connection if release fails
            try:
                if not conn.is_closed():
                    conn.close()
            except Exception:
                pass  # Ignore errors during forced close
    
    
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
        # Check if we're using asyncpg/our adapter or SQLAlchemy
        is_asyncpg_style = isinstance(conn, asyncpg.Connection) or isinstance(conn, AsyncPGToWrapperAdapter) or hasattr(conn, '_conn')
        
        if is_asyncpg_style:
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