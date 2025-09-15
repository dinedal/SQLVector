"""Centralized event loop management utilities for PostgreSQL backend.

This module provides robust async-to-sync bridging functionality that properly
handles event loop lifecycle, connection cleanup, and resource management.
"""

import asyncio
import threading
import queue
import weakref
import warnings
from contextvars import copy_context
from typing import Any, Callable, Coroutine, TypeVar, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from ...logger import get_logger

# Import the sync context variable from config
try:
    from .config import _sync_context
except ImportError:
    # Fallback if circular import
    from contextvars import ContextVar
    _sync_context: ContextVar[bool] = ContextVar('sync_context', default=False)

logger = get_logger(__name__)

T = TypeVar('T')

# Global thread pool for event loop operations
_thread_pool: Optional[ThreadPoolExecutor] = None
_thread_pool_lock = threading.Lock()

def _get_thread_pool() -> ThreadPoolExecutor:
    """Get or create the global thread pool for event loop operations."""
    global _thread_pool
    if _thread_pool is None:
        with _thread_pool_lock:
            if _thread_pool is None:
                _thread_pool = ThreadPoolExecutor(
                    max_workers=4, 
                    thread_name_prefix="sqlvector_async"
                )
    return _thread_pool

def _cleanup_thread_pool():
    """Cleanup the global thread pool."""
    global _thread_pool
    if _thread_pool is not None:
        with _thread_pool_lock:
            if _thread_pool is not None:
                _thread_pool.shutdown(wait=True)
                _thread_pool = None

# Register cleanup on module exit
import atexit
atexit.register(_cleanup_thread_pool)


class EventLoopManager:
    """Manages event loops for async-to-sync operations."""
    
    def __init__(self):
        self._loops = weakref.WeakSet()
        self._lock = threading.Lock()
    
    def run_async_in_sync(
        self, 
        coro_or_factory: Union[Coroutine[Any, Any, T], Callable[[], Coroutine[Any, Any, T]]], 
        timeout: float = 300.0
    ) -> T:
        """Run an async coroutine in a sync context with proper cleanup.
        
        Args:
            coro_or_factory: The coroutine to run, or a factory function that creates the coroutine
            timeout: Maximum time to wait for completion (seconds)
            
        Returns:
            The result of the coroutine
            
        Raises:
            TimeoutError: If the operation times out
            RuntimeError: If no event loop is available and asyncio.run fails
            Exception: Any exception raised by the coroutine
        """
        # Helper function to get a fresh coroutine
        def get_coro():
            if callable(coro_or_factory):
                return coro_or_factory()
            else:
                return coro_or_factory
        
        try:
            # First, try to see if we're already in an async context
            loop = asyncio.get_running_loop()
            
            # We're in an async context, need to run in a separate thread
            warnings.warn(
                "Running sync PostgreSQL methods from within an async context. "
                "Consider using the async methods (e.g., load_documents_async) "
                "for better performance and to avoid event loop complications.",
                UserWarning,
                stacklevel=3
            )
            
            return self._run_in_thread_with_new_loop(get_coro, timeout)
            
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No running event loop, safe to use asyncio.run
                try:
                    # Set sync context before running the coroutine
                    async def run_with_context():
                        _sync_context.set(True)
                        return await get_coro()
                    
                    return asyncio.run(run_with_context())
                except Exception as run_error:
                    logger.error(f"Failed to run coroutine with asyncio.run: {run_error}")
                    # Fallback to thread-based approach with fresh coroutine
                    return self._run_in_thread_with_new_loop(get_coro, timeout)
            else:
                raise e
    
    def _run_in_thread_with_new_loop(
        self, 
        coro_factory: Callable[[], Coroutine[Any, Any, T]], 
        timeout: float
    ) -> T:
        """Run coroutine in a separate thread with a new event loop."""
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def run_with_isolated_loop():
            """Run the coroutine in a completely isolated event loop."""
            new_loop = None
            try:
                # Create a new event loop
                new_loop = asyncio.new_event_loop()
                
                # Track the loop for cleanup
                with self._lock:
                    self._loops.add(new_loop)
                
                # Set as the current event loop for this thread only
                asyncio.set_event_loop(new_loop)
                
                # Create a fresh coroutine and run it with sync context
                async def run_with_context():
                    _sync_context.set(True)
                    return await coro_factory()
                
                result = new_loop.run_until_complete(run_with_context())
                result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Error in isolated event loop: {e}", exc_info=True)
                exception_queue.put(e)
            finally:
                # Comprehensive cleanup
                if new_loop and not new_loop.is_closed():
                    try:
                        self._cleanup_loop_safely(new_loop)
                    except Exception as cleanup_error:
                        logger.warning(f"Error during loop cleanup: {cleanup_error}")
                
                # Clear the event loop for this thread
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass  # Ignore errors when clearing event loop
        
        # Use thread pool for better resource management
        thread_pool = _get_thread_pool()
        future = thread_pool.submit(run_with_isolated_loop)
        
        try:
            # Wait for completion with timeout
            future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"Thread execution failed: {e}")
            raise
        
        # Check for exceptions from the thread
        if not exception_queue.empty():
            raise exception_queue.get()
        
        # Return result
        if not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("No result returned from async operation")
    
    def _cleanup_loop_safely(self, loop: asyncio.AbstractEventLoop):
        """Safely cleanup an event loop and its resources."""
        try:
            # Cancel all pending tasks
            pending_tasks = asyncio.all_tasks(loop)
            if pending_tasks:
                logger.debug(f"Cancelling {len(pending_tasks)} pending tasks")
                for task in pending_tasks:
                    task.cancel()
                
                # Wait for cancellation to complete with timeout
                try:
                    loop.run_until_complete(
                        asyncio.wait_for(
                            asyncio.gather(*pending_tasks, return_exceptions=True),
                            timeout=5.0
                        )
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Task cancellation took too long or failed: {e}")
            
            # Close the loop
            if not loop.is_closed():
                loop.close()
                logger.debug("Event loop closed successfully")
                
        except Exception as e:
            logger.warning(f"Error during safe loop cleanup: {e}")


# Global instance
_event_loop_manager = EventLoopManager()

def run_async_in_sync(
    coro_or_factory: Union[Coroutine[Any, Any, T], Callable[[], Coroutine[Any, Any, T]]], 
    timeout: float = 300.0
) -> T:
    """Run an async coroutine from a sync context.
    
    This is the main function that should be used throughout the PostgreSQL backend
    to bridge async operations in sync methods.
    
    Args:
        coro_or_factory: The coroutine to run, or a factory function that creates the coroutine
        timeout: Maximum time to wait for completion (seconds)
        
    Returns:
        The result of the coroutine
        
    Raises:
        TimeoutError: If the operation times out
        Exception: Any exception raised by the coroutine
    """
    return _event_loop_manager.run_async_in_sync(coro_or_factory, timeout)


def is_event_loop_running() -> bool:
    """Check if there's a running event loop in the current thread."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def safe_close_connection_pool(close_coro: Coroutine[Any, Any, None]) -> None:
    """Safely close a connection pool, handling event loop state."""
    try:
        if is_event_loop_running():
            # We're in an async context, schedule the cleanup
            loop = asyncio.get_running_loop()
            try:
                # Try to run the cleanup in the current loop
                task = loop.create_task(close_coro)
                # Don't wait for completion to avoid blocking
                task.add_done_callback(lambda t: None)  # Prevent warnings about unawaited task
            except Exception as e:
                logger.warning(f"Could not schedule connection pool cleanup: {e}")
        else:
            # We're in a sync context, run the cleanup
            try:
                asyncio.run(close_coro)
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error during connection pool cleanup: {e}")