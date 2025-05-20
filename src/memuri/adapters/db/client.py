"""
PostgreSQL client for database operations.
"""
from typing import Any, Dict, List, Optional, Union, Callable
import asyncio
import logging

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)


class PostgresClient:
    """PostgreSQL client for database operations."""

    def __init__(
        self,
        connection_string: str,
        max_connections: int = 10,
        min_connections: int = 2,
    ):
        """Initialize PostgreSQL client.
        
        Args:
            connection_string: PostgreSQL connection string
            max_connections: Maximum number of connections in the pool
            min_connections: Minimum number of connections in the pool
        """
        self.connection_string = connection_string
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self, setup_schema: bool = True) -> None:
        """Initialize connection pool and create schema if needed.
        
        Args:
            setup_schema: Whether to create tables and indexes
        """
        logger.info("Initializing PostgreSQL connection pool")
        
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=self.min_connections,
            max_size=self.max_connections,
            setup=self._setup_connection,
        )
        
        if setup_schema:
            await self._create_schema()

    async def _setup_connection(self, conn: asyncpg.Connection) -> None:
        """Set up connection with vector extension."""
        await register_vector(conn)

    async def _create_schema(self) -> None:
        """Create database schema if needed."""
        if not self.pool:
            await self.initialize(setup_schema=False)
            
        create_tables_query = """
        -- Memory entries table
        CREATE TABLE IF NOT EXISTS memory_entries (
            id SERIAL PRIMARY KEY,
            memory_id TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            category TEXT,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- User feedback table for classifier feedback
        CREATE TABLE IF NOT EXISTS user_feedback (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            source TEXT DEFAULT 'user',
            used_for_training BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS memory_entries_category_idx ON memory_entries(category);
        CREATE INDEX IF NOT EXISTS memory_entries_created_at_idx ON memory_entries(created_at);
        CREATE INDEX IF NOT EXISTS user_feedback_category_idx ON user_feedback(category);
        CREATE INDEX IF NOT EXISTS user_feedback_used_for_training_idx ON user_feedback(used_for_training);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_tables_query)
            logger.info("Database schema created or already exists")

    async def close(self) -> None:
        """Close database connections."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    async def execute(
        self, query: str, *args: Any, timeout: Optional[float] = None
    ) -> str:
        """Execute a query and return the status string.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query status string
        """
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetch(
        self, query: str, *args: Any, timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query and return all rows.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            List of rows as dictionaries
        """
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args, timeout=timeout)
            return [dict(row) for row in rows]

    async def fetchrow(
        self, query: str, *args: Any, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a query and return the first row.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            First row as dictionary or None
        """
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args, timeout=timeout)
            return dict(row) if row else None

    async def fetch_val(
        self, query: str, *args: Any, column: int = 0, timeout: Optional[float] = None
    ) -> Any:
        """Execute a query and return a single value.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            column: Column index to return
            timeout: Query timeout in seconds
            
        Returns:
            Single value or None
        """
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def transaction(self) -> asyncpg.Transaction:
        """Get a transaction object that can be used as a context manager.
        
        Returns:
            Transaction object
        """
        if not self.pool:
            await self.initialize()
            
        conn = await self.pool.acquire()
        transaction = conn.transaction()
        
        try:
            await transaction.start()
            return transaction
        except Exception:
            await conn.close()
            raise 