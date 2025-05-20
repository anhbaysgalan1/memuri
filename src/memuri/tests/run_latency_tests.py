#!/usr/bin/env python
"""Command-line script to run latency tests for Memuri."""

import argparse
import asyncio
import json
import os
import sys
import datetime
from pathlib import Path

# Add parent directory to path so we can import test_latency
src_path = str(Path(__file__).parent.parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from memuri import Memuri, MemoryCategory, MemuriSettings
from memuri.factory import EmbedderFactory, VectorStoreFactory
from memuri.core.config import EmbeddingSettings, DatabaseSettings, VectorStoreSettings
from memuri.core.logging import configure_logging, get_logger
from memuri.core.config import LoggingSettings
from memuri.tests.test_latency import LatencyBenchmark

logger = get_logger(__name__)


async def setup_memuri_client(api_key=None, db_url=None):
    """Set up a Memuri client for testing.
    
    Args:
        api_key: OpenAI API key (optional)
        db_url: PostgreSQL URL (optional)
        
    Returns:
        Memuri: Configured Memuri client
    """
    # OpenAI API key configuration
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set. Please provide an API key via --api-key or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Set environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Create embedding service
    embedding_settings = EmbeddingSettings(
        provider="openai",
        model_name="text-embedding-3-small",
        api_key=openai_api_key,
    )
    embedding_service = EmbedderFactory.create(provider="openai", settings=embedding_settings)
    
    # Database configuration
    postgres_url = db_url or os.environ.get(
        "MEMURI_DATABASE__POSTGRES_URL", 
        "postgresql://memuri:memuri@localhost:5432/memuri"
    )
    
    # Create vector store settings
    vector_store_settings = VectorStoreSettings(
        provider="pgvector",
        dimension=1536,
        connection_string=postgres_url,
    )
    
    # Create database settings
    database_settings = DatabaseSettings(postgres_url=postgres_url)
    
    # Create overall settings
    settings = MemuriSettings()
    settings.database = database_settings
    settings.vector_store = vector_store_settings
    settings.embedding = embedding_settings
    
    # Initialize vector store
    vector_store = VectorStoreFactory.create(
        provider="pgvector",
        settings=vector_store_settings
    )
    
    # Initialize memuri with explicit services
    memuri = Memuri(
        settings=settings,
        embedding_service=embedding_service,
        memory_service=vector_store,
    )
    
    logger.info("Memuri client initialized for testing")
    return memuri


async def run_latency_tests(
    test_type: str = "all",
    iterations: int = 10,
    api_key: str = None,
    db_url: str = None,
    save_results: bool = True,
):
    """Run latency tests.
    
    Args:
        test_type: Type of test to run ("add", "search", "all")
        iterations: Number of iterations for each test
        api_key: OpenAI API key (optional)
        db_url: PostgreSQL URL (optional)
        save_results: Whether to save results to a file
    """
    # Set up Memuri client
    client = await setup_memuri_client(api_key, db_url)
    
    try:
        logger.info(f"Running latency tests: {test_type}, iterations: {iterations}")
        
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "test_type": test_type,
            "iterations": iterations,
            "database_url": db_url or os.environ.get("MEMURI_DATABASE__POSTGRES_URL", "default"),
            "embedding_model": os.environ.get("MEMURI_EMBEDDING__MODEL_NAME", "text-embedding-3-small"),
            "tests": {}
        }
        
        if test_type in ["add", "all"]:
            # Create benchmark
            benchmark = LatencyBenchmark("Add Memory")
            
            # Perform benchmarking
            for i in range(iterations):
                content = f"Test memory {i}: This is a test memory with content that needs to be processed, embedded, and stored."
                await benchmark.measure_operation(
                    client.add_memory,
                    content=content,
                    category="FACT",
                    metadata={"test_id": i, "importance": "medium"}
                )
            
            # Print results
            benchmark.print_stats()
            
            # Store results
            results["tests"]["add_memory"] = benchmark.get_stats()
        
        if test_type in ["search", "all"]:
            # Create benchmark
            benchmark = LatencyBenchmark("Search Memory")
            
            # Sample queries
            queries = [
                "test memory",
                "content processed",
                "embedded and stored",
                "memory with content",
                "test",
                "processing information",
                "storage requirements",
                "embedding process",
                "memory content",
                "database operations"
            ]
            
            # Perform benchmarking
            for i in range(iterations):
                query = queries[i % len(queries)]
                await benchmark.measure_operation(
                    client.search_memory,
                    query=query,
                    top_k=5,
                    rerank=True
                )
            
            # Print results
            benchmark.print_stats()
            
            # Store results
            results["tests"]["search_memory"] = benchmark.get_stats()
            
        # Save results to file if requested
        if save_results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path(f"test_results_{timestamp}.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
            
    finally:
        # Clean up
        await client.close()
        logger.info("Tests completed")


def main():
    """Main entry point for the script."""
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description="Run latency tests for Memuri")
    parser.add_argument(
        "--test-type",
        choices=["add", "search", "all"],
        default="all",
        help="Type of test to run",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for each test",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (optional, can use OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--db-url",
        help="PostgreSQL URL (optional, default: postgresql://memuri:memuri@localhost:5432/memuri)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to a file",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging with proper LoggingSettings object
    log_settings = LoggingSettings(level=args.log_level)
    configure_logging(log_settings)
    
    # Run tests
    asyncio.run(
        run_latency_tests(
            test_type=args.test_type,
            iterations=args.iterations,
            api_key=args.api_key,
            db_url=args.db_url,
            save_results=not args.no_save,
        )
    )


if __name__ == "__main__":
    main() 