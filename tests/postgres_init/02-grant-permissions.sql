-- Grant necessary permissions for testing
-- This ensures the test user can create indexes, tables, etc.

\echo 'Granting permissions for testing...'

-- Grant all privileges on all databases to test user
GRANT ALL PRIVILEGES ON DATABASE sqlvector_test TO testuser;
GRANT ALL PRIVILEGES ON DATABASE sqlvector_integration_test TO testuser;
GRANT ALL PRIVILEGES ON DATABASE sqlvector_performance_test TO testuser;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO testuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO testuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO testuser;

-- Allow creating extensions (needed for some tests)
ALTER USER testuser CREATEDB;

-- Connect to each database and grant permissions there too
\c sqlvector_integration_test;
GRANT ALL ON SCHEMA public TO testuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO testuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO testuser;

\c sqlvector_performance_test;
GRANT ALL ON SCHEMA public TO testuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO testuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO testuser;

\c sqlvector_test;

\echo 'Permissions granted successfully!'