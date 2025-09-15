-- Initialize pgvector extension and set up test environment
-- This script runs automatically when the PostgreSQL container starts

\echo 'Setting up pgvector extension and test environment...'

-- Create the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Create additional databases for testing different scenarios
CREATE DATABASE sqlvector_integration_test OWNER testuser;
CREATE DATABASE sqlvector_performance_test OWNER testuser;

-- Connect to integration test database and set up extension there too
\c sqlvector_integration_test;
CREATE EXTENSION IF NOT EXISTS vector;

-- Connect to performance test database and set up extension there too  
\c sqlvector_performance_test;
CREATE EXTENSION IF NOT EXISTS vector;

-- Go back to main test database
\c sqlvector_test;

-- Create some test data types for validation
DO $$
BEGIN
    -- Test that we can create vector columns of different dimensions
    CREATE TABLE IF NOT EXISTS vector_test_table (
        id SERIAL PRIMARY KEY,
        embedding_384 vector(384),
        embedding_768 vector(768),
        embedding_1536 vector(1536)
    );
    
    -- Insert a test vector to verify functionality
    INSERT INTO vector_test_table (embedding_384) 
    VALUES (array_fill(0.1::float, ARRAY[384])::vector);
    
    -- Test basic vector operations
    SELECT embedding_384 <-> '[0.1, 0.2, 0.3]'::vector FROM vector_test_table LIMIT 1;
    
    RAISE NOTICE 'pgvector extension successfully initialized and tested';
    
EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION 'Failed to initialize pgvector: %', SQLERRM;
END $$;

-- Set up logging for query analysis
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_duration = 'on';
ALTER SYSTEM SET log_min_duration_statement = 0;

-- Reload configuration
SELECT pg_reload_conf();

\echo 'PostgreSQL with pgvector initialization completed successfully!'