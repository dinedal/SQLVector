#!/usr/bin/env python3
"""
PostgreSQL Test Environment Setup Script

This script manages the Docker PostgreSQL instance for integration testing.
It handles container lifecycle, health checks, and provides connection utilities.
"""

import asyncio
import subprocess
import time
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import asyncpg
    import psycopg2
    POSTGRES_DEPS_AVAILABLE = True
except ImportError:
    POSTGRES_DEPS_AVAILABLE = False


class PostgresTestEnvironment:
    """Manages PostgreSQL test environment with Docker."""
    
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
    
    def is_docker_running(self) -> bool:
        """Check if Docker daemon is running."""
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
    
    def start_container(self, force_rebuild=False, wait_for_ready=True):
        """Start the PostgreSQL test container."""
        print("🐳 Starting PostgreSQL test container...")
        
        if not self.is_docker_running():
            print("❌ Docker daemon is not running. Please start Docker first.")
            return False
        
        # Stop existing container if running
        if self.is_container_running():
            if force_rebuild:
                print("🔄 Stopping existing container for rebuild...")
                self.stop_container()
            else:
                print("✅ PostgreSQL test container is already running")
                if wait_for_ready:
                    return self.wait_for_ready()
                return True
        
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
            
            print("✅ Container started successfully")
            
            if wait_for_ready:
                print("⏳ Waiting for PostgreSQL to be ready...")
                if self.wait_for_ready():
                    print("✅ PostgreSQL is ready for connections")
                    return True
                else:
                    print("❌ PostgreSQL failed to become ready")
                    return False
            else:
                print("📌 Container started in background (not waiting for ready)")
                return True
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout starting PostgreSQL container")
            return False
        except Exception as e:
            print(f"❌ Error starting container: {e}")
            return False
    
    def stop_container(self):
        """Stop the PostgreSQL test container."""
        print("🛑 Stopping PostgreSQL test container...")
        try:
            compose_file = project_root / "docker-compose.test.yml"
            subprocess.run(
                ['docker-compose', '-f', str(compose_file), 'down'],
                capture_output=True,
                timeout=30
            )
            print("✅ PostgreSQL test container stopped")
            return True
        except Exception as e:
            print(f"❌ Error stopping container: {e}")
            return False
    
    def wait_for_ready(self, max_attempts=30) -> bool:
        """Wait for PostgreSQL to be ready to accept connections."""
        if not POSTGRES_DEPS_AVAILABLE:
            print("⚠️ PostgreSQL dependencies not available for health check")
            print("   Waiting 30 seconds and assuming ready...")
            time.sleep(30)
            return True
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Try to connect using psycopg2 (sync)
                conn = psycopg2.connect(**self.db_config, connect_timeout=5)
                conn.close()
                print(f"✅ PostgreSQL ready after {attempt} attempts")
                return True
            except psycopg2.OperationalError:
                if attempt < max_attempts:
                    print(f"⏳ Attempt {attempt}/{max_attempts}: PostgreSQL not ready yet...")
                    time.sleep(2)
                else:
                    print("❌ PostgreSQL failed to become ready")
                    return False
            except Exception as e:
                print(f"❌ Unexpected error checking PostgreSQL readiness: {e}")
                return False
        
        return False
    
    async def test_async_connection(self) -> bool:
        """Test async connection to PostgreSQL."""
        if not POSTGRES_DEPS_AVAILABLE:
            print("⚠️ Cannot test async connection - dependencies not available")
            return False
        
        try:
            conn = await asyncpg.connect(**self.db_config)
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            assert result == 1
            
            # Test pgvector is available
            extensions = await conn.fetch("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            assert len(extensions) > 0, "pgvector extension not found"
            
            # Test vector operations
            vector_test = await conn.fetchval("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector")
            assert vector_test is not None
            
            await conn.close()
            print("✅ Async connection test passed")
            return True
            
        except Exception as e:
            print(f"❌ Async connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get connection information for tests."""
        return {
            'db_config': self.db_config,
            'db_url': self.db_url,
            'container_name': self.container_name,
            'port': self.port
        }
    
    def show_logs(self, tail=50):
        """Show PostgreSQL container logs."""
        try:
            result = subprocess.run(
                ['docker', 'logs', '--tail', str(tail), self.container_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout:
                print("📋 PostgreSQL container logs:")
                print(result.stdout)
            if result.stderr:
                print("⚠️ PostgreSQL container errors:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Error getting logs: {e}")


async def main():
    """Main function for interactive usage."""
    env = PostgresTestEnvironment()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start':
            force_rebuild = '--rebuild' in sys.argv
            no_wait = '--no-wait' in sys.argv
            
            wait_for_ready = not no_wait
            
            if env.start_container(force_rebuild=force_rebuild, wait_for_ready=wait_for_ready):
                if wait_for_ready and POSTGRES_DEPS_AVAILABLE:
                    await env.test_async_connection()
                
                print("\n📊 Connection Info:")
                info = env.get_connection_info()
                print(f"  Database URL: {info['db_url']}")
                print(f"  Host: {info['db_config']['host']}:{info['db_config']['port']}")
                print(f"  Database: {info['db_config']['database']}")
                print(f"  User: {info['db_config']['user']}")
                
                if no_wait:
                    print("\n⚠️  Container started in background. Use 'status' to check readiness.")
            else:
                sys.exit(1)
        
        elif command == 'stop':
            env.stop_container()
        
        elif command == 'status':
            if env.is_container_running():
                print("✅ PostgreSQL test container is running")
                if POSTGRES_DEPS_AVAILABLE:
                    await env.test_async_connection()
            else:
                print("❌ PostgreSQL test container is not running")
        
        elif command == 'logs':
            env.show_logs()
        
        elif command == 'info':
            info = env.get_connection_info()
            print("📊 PostgreSQL Test Environment Info:")
            print(f"  Database URL: {info['db_url']}")
            print(f"  Container: {info['container_name']}")
            print(f"  Port: {info['port']}")
            print(f"  Dependencies available: {POSTGRES_DEPS_AVAILABLE}")
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: start, stop, status, logs, info")
    else:
        print("PostgreSQL Test Environment Manager")
        print("Usage: python postgres_test_setup.py <command>")
        print("Commands:")
        print("  start [--rebuild] [--no-wait]  - Start PostgreSQL container")
        print("    --rebuild       - Force recreate container")
        print("    --no-wait       - Don't wait for PostgreSQL to be ready")
        print("  stop                           - Stop PostgreSQL container") 
        print("  status                         - Check container status")
        print("  logs                           - Show container logs")
        print("  info                           - Show connection info")


if __name__ == "__main__":
    asyncio.run(main())