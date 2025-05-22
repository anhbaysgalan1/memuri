#!/bin/bash
# Database initialization script for CI/CD

set -e

echo "Setting up PostgreSQL database..."

# Set environment variables
export PGUSER=memuri
export PGPASSWORD=memuri
export PGHOST=localhost
export PGDATABASE=memuri

# Check if PostgreSQL is running
echo "Checking PostgreSQL connection..."
pg_isready -h $PGHOST -U $PGUSER || (echo "PostgreSQL is not ready" && exit 1)

# Create pgvector extension
echo "Creating pgvector extension..."
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Display PostgreSQL version and extensions
echo "PostgreSQL version:"
psql -c "SELECT version();"

echo "Installed extensions:"
psql -c "\dx"

echo "Database users:"
psql -c "\du"

# Fix permissions if needed
echo "Ensuring proper permissions..."
psql -c "ALTER USER memuri WITH SUPERUSER;"

# Create test database tables if needed
echo "Setting up test tables..."
psql -c "CREATE TABLE IF NOT EXISTS test_vectors (id SERIAL PRIMARY KEY, data vector(1536));"

echo "Database initialization complete!" 