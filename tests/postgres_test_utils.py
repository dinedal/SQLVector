"""PostgreSQL Test Utilities

Centralized utilities for managing PostgreSQL test containers and fixtures.
This module provides consistent PostgreSQL test infrastructure across all test files.
"""

import asyncio
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pytest

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check for PostgreSQL dependencies
try:
    import asyncpg
    import psycopg2
    POSTGRES_DEPS_AVAILABLE = True
except ImportError:
    POSTGRES_DEPS_AVAILABLE = False

# Check for PostgreSQL backend
try:
    from sqlvector.backends.postgres import PostgresRAG, PostgresConfig
    from sqlvector.embedding import DefaultEmbeddingProvider
    POSTGRES_BACKEND_AVAILABLE = True
except ImportError:
    POSTGRES_BACKEND_AVAILABLE = False


class PostgreSQLTestManager:
    """Manages PostgreSQL test environment with automatic container lifecycle."""
    
    def __init__(self):
        self.container_name = "sqlvector-postgres-test"
        self.port = 5433
        self.db_config = {
            'host': 'localhost',
            'port': self.port,
            'user': 'testuser',
            'password': 'testpass',
            'database': 'sqlvector_test'
        }
        self.db_url = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        self._container_started = False
        self._startup_attempted = False
        
    def get_db_url(self) -> str:
        """Get the database URL for connections."""
        # Allow override via environment variable
        return os.environ.get("TEST_POSTGRES_URL", self.db_url)
    
    def get_db_config(self) -> Dict[str, Any]:
        """Get database configuration dict."""
        # If using custom URL, parse it
        custom_url = os.environ.get("TEST_POSTGRES_URL")
        if custom_url and custom_url != self.db_url:
            # For custom URLs, return minimal config (tests will use URL directly)
            return {"db_url": custom_url}
        return self.db_config
    
    def is_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ['docker', 'info'], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def is_container_running(self) -> bool:
        """Check if the PostgreSQL test container is running."""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return self.container_name in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def start_container(self, force_rebuild: bool = False) -> bool:
        """Start the PostgreSQL container if needed."""
        if self._startup_attempted and self._container_started:
            return True
            
        self._startup_attempted = True
        
        # Check if we should use existing database URL
        custom_url = os.environ.get("TEST_POSTGRES_URL")
        if custom_url and custom_url != self.db_url:
            print(f"📌 Using custom PostgreSQL URL: {custom_url}")
            self._container_started = True
            return True
        
        print("🐳 Starting PostgreSQL test container...")
        
        if not self.is_docker_available():
            print("❌ Docker is not available. PostgreSQL tests will be skipped.")
            return False
        
        # Check if container is already running
        if self.is_container_running() and not force_rebuild:
            print("✅ PostgreSQL test container is already running")
            if self.wait_for_ready():
                self._container_started = True
                return True
            else:
                print("⚠️ Container running but not ready, attempting restart...")
                self.stop_container()
        
        try:
            # Start container using docker-compose
            compose_file = project_root / "docker-compose.test.yml"
            if not compose_file.exists():
                print(f"❌ Docker compose file not found: {compose_file}")
                return False
            
            cmd = ['docker-compose', '-f', str(compose_file), 'up', '-d']
            if force_rebuild:
                cmd.append('--force-recreate')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"❌ Failed to start container: {result.stderr}")
                return False
            
            print("⏳ Waiting for PostgreSQL to be ready...")
            if self.wait_for_ready():
                print("✅ PostgreSQL is ready for connections")
                self._container_started = True
                return True
            else:
                print("❌ PostgreSQL failed to become ready")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout starting PostgreSQL container")
            return False
        except Exception as e:
            print(f"❌ Error starting container: {e}")
            return False
    
    def stop_container(self) -> bool:
        """Stop the PostgreSQL container."""
        # Don't stop if using custom URL
        custom_url = os.environ.get("TEST_POSTGRES_URL")
        if custom_url and custom_url != self.db_url:
            return True
            
        try:
            compose_file = project_root / "docker-compose.test.yml"
            subprocess.run(
                ['docker-compose', '-f', str(compose_file), 'down'],
                capture_output=True,
                timeout=30
            )
            self._container_started = False
            return True
        except Exception:
            return False
    
    def wait_for_ready(self, max_attempts: int = 30) -> bool:
        """Wait for PostgreSQL to be ready to accept connections."""
        if not POSTGRES_DEPS_AVAILABLE:
            print("⚠️ PostgreSQL dependencies not available, waiting 30 seconds...")
            time.sleep(30)
            return True
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Try to connect using psycopg2
                conn = psycopg2.connect(**self.db_config, connect_timeout=5)
                conn.close()
                return True
            except psycopg2.OperationalError:
                if attempt < max_attempts:
                    time.sleep(2)
                else:
                    return False
            except Exception:
                return False
        
        return False
    
    def is_available(self) -> bool:
        """Check if PostgreSQL testing is available."""
        return POSTGRES_DEPS_AVAILABLE and POSTGRES_BACKEND_AVAILABLE
    
    def get_skip_reason(self) -> Optional[str]:
        """Get reason why PostgreSQL tests should be skipped, if any."""
        if not POSTGRES_BACKEND_AVAILABLE:
            return "PostgreSQL backend not available"
        if not POSTGRES_DEPS_AVAILABLE:
            return "PostgreSQL dependencies not installed"
        if not self._container_started and not self.start_container():
            return "PostgreSQL container could not be started"
        return None


# Global instance
_postgres_manager = PostgreSQLTestManager()


def get_postgres_manager() -> PostgreSQLTestManager:
    """Get the global PostgreSQL test manager instance."""
    return _postgres_manager


def pytest_configure(config):
    """Configure pytest with PostgreSQL markers."""
    config.addinivalue_line(
        "markers", "postgres: mark test as requiring PostgreSQL database"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark PostgreSQL tests and handle container startup."""
    postgres_tests = []
    
    # Find PostgreSQL tests
    for item in items:
        if (
            "postgres" in item.nodeid.lower() or
            "postgresql" in item.nodeid.lower() or
            any(fixture in item.fixturenames for fixture in ["postgres_env", "postgres_rag", "postgres_db_config"])
        ):
            item.add_marker(pytest.mark.postgres)
            postgres_tests.append(item)
    
    # If we have PostgreSQL tests and container isn't started, try to start it
    if postgres_tests:
        manager = get_postgres_manager()
        if not manager.is_available():
            # Mark all PostgreSQL tests as skipped
            skip_reason = manager.get_skip_reason()
            for test in postgres_tests:
                test.add_marker(pytest.mark.skip(reason=skip_reason))
        elif not manager._startup_attempted:
            # Try to start container during collection phase
            print("\n🔍 PostgreSQL tests detected, preparing container...")
            force_rebuild = config.getoption("--postgres-rebuild", False) if hasattr(config, 'getoption') else False
            if not manager.start_container(force_rebuild=force_rebuild):
                skip_reason = "Failed to start PostgreSQL container"
                for test in postgres_tests:
                    test.add_marker(pytest.mark.skip(reason=skip_reason))


def pytest_addoption(parser):
    """Add PostgreSQL-specific command line options."""
    parser.addoption(
        "--postgres-rebuild", 
        action="store_true", 
        default=False,
        help="Force rebuild of PostgreSQL test container"
    )
    parser.addoption(
        "--skip-postgres", 
        action="store_true", 
        default=False,
        help="Skip all PostgreSQL tests"
    )


def pytest_runtest_setup(item):
    """Skip PostgreSQL tests if requested."""
    if item.config.getoption("--skip-postgres"):
        if "postgres" in str(item.fspath).lower():
            pytest.skip("PostgreSQL tests skipped via --skip-postgres")


def pytest_sessionfinish(session, exitstatus):
    """Clean up PostgreSQL container after test session."""
    manager = get_postgres_manager()
    if manager._container_started and not os.environ.get("TEST_POSTGRES_URL"):
        print("\n🧹 Cleaning up PostgreSQL test container...")
        manager.stop_container()