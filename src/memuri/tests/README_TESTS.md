# Memuri Tests

This directory contains tests for the Memuri system, including unit tests, integration tests, and performance benchmarks.

## Latency Tests

The latency tests are designed to measure the performance of memory operations in the Memuri system. These tests can be used to:

1. Establish performance baselines
2. Detect performance regressions
3. Optimize memory operations
4. Validate performance on different hardware or configurations

### Running Latency Tests

You can run the latency tests using the provided command-line script:

```bash
# Run all latency tests with default settings
./run_latency_tests.py

# Run only memory addition tests
./run_latency_tests.py --test-type add

# Run only memory search tests
./run_latency_tests.py --test-type search

# Run tests with a custom number of iterations
./run_latency_tests.py --iterations 50

# Run tests with a specific OpenAI API key
./run_latency_tests.py --api-key YOUR_API_KEY

# Run tests with a custom database URL
./run_latency_tests.py --db-url postgresql://user:password@hostname:port/database

# Enable debug logging
./run_latency_tests.py --log-level DEBUG

# Don't save results to a file
./run_latency_tests.py --no-save
```

Alternatively, you can run the tests using pytest directly:

```bash
# Run all latency tests
pytest -xvs test_latency.py

# Run a specific test
pytest -xvs test_latency.py::test_add_memory_latency
```

### Test Results

The latency tests save results to JSON files with timestamps in the current directory. These files contain:

- Test configuration details (timestamp, iterations, database URL, etc.)
- Performance metrics for each test
- Raw latency data

Example results file:

```json
{
  "timestamp": "2023-05-21T12:34:56.789123",
  "test_type": "all",
  "iterations": 10,
  "database_url": "postgresql://memuri:memuri@localhost:5432/memuri",
  "embedding_model": "text-embedding-3-small",
  "tests": {
    "add_memory": {
      "min": 123.45,
      "max": 456.78,
      "mean": 234.56,
      "median": 222.33,
      "count": 10,
      "stdev": 45.67
    },
    "search_memory": {
      "min": 78.90,
      "max": 234.56,
      "mean": 123.45,
      "median": 111.22,
      "count": 10,
      "stdev": 23.45
    }
  }
}
```

### Interpreting Results

The latency tests provide statistics for each operation, including:

- Minimum latency
- Maximum latency
- Mean (average) latency
- Median latency
- Standard deviation (for statistical analysis)

A good performance baseline depends on your specific requirements, but generally:

- Add memory operations: <500ms is excellent, <1000ms is good
- Search memory operations: <200ms is excellent, <500ms is good

### Test Configuration

The latency tests use the following configuration by default:

- Provider: OpenAI
- Embedding model: text-embedding-3-small
- Vector store: pgvector (PostgreSQL with pgvector extension)
- Database URL: postgresql://memuri:memuri@localhost:5432/memuri

You can customize these settings by modifying the environment variables or using the command-line arguments.

## Other Tests

In addition to the latency tests, this directory also contains:

- **Unit tests**: Tests for individual components
- **Integration tests**: Tests for component interactions
- **Configuration fixtures**: Test fixtures and utilities

# Memuri Performance Testing

This directory contains tools for performance testing and benchmarking the Memuri memory infrastructure.

## Latency Benchmark Tool

The `test_latency.py` module provides a `LatencyBenchmark` class for measuring the latency of asynchronous operations in Memuri. This tool is useful for evaluating the performance of memory operations in different environments and configurations.

### Features

- Detailed measurement of operation latency in milliseconds
- Statistical analysis (mean, median, min, max, 95th percentile, standard deviation)
- Capture of operation results for verification
- Readable output formatting
- Async-compatible for modern Python codebases

## Command-Line Interface

The `run_latency_tests.py` script provides a convenient command-line interface for running latency tests on Memuri operations.

### Usage

```bash
./run_latency_tests.py [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--test-type` | Type of test to run: "add", "search", or "all" | "all" |
| `--iterations` | Number of iterations for each test | 10 |
| `--api-key` | OpenAI API key (can use OPENAI_API_KEY env var instead) | sk-your-openai-key |
| `--model` | Embedding model name | text-embedding-3-small |
| `--operation` | Operation to test (add, search, both) | both |
| `--duration` | Duration in seconds to run the test | 30 |
| `--concurrency` | Number of concurrent operations | 10 |
| `--db-url` | PostgreSQL URL (can use MEMURI_DATABASE_POSTGRES_URL env var instead) | postgresql://memuri:memuri@localhost:5432/memuri |
| `--log-level` | Logging level: DEBUG, INFO, WARNING, ERROR | INFO |
| `--no-save` | Don't save results to a file | False |

### Examples

```bash
# Run all tests with default settings
./run_latency_tests.py

# Run only add memory tests with 20 iterations
./run_latency_tests.py --test-type add --iterations 20

# Run only search tests with custom database URL
./run_latency_tests.py --test-type search --db-url postgresql://user:pass@localhost:5432/mydb

# Set log level to DEBUG for more detailed output
./run_latency_tests.py --log-level DEBUG

# Don't save results to a file
./run_latency_tests.py --no-save
```

### Output

The tool produces console output with detailed statistics:

```
=== Add Memory Benchmark Results ===
Total operations: 10
Mean latency: 78.45ms
Median latency: 76.32ms
Min latency: 68.91ms
Max latency: 102.58ms
95th percentile: 94.37ms
Standard deviation: 10.21ms
==================================
```

It also saves results to a JSON file with timestamp in the current directory (unless `--no-save` is specified).

## Programmatic Usage

You can use the `LatencyBenchmark` class in your own code:

```python
import asyncio
from memuri import Memuri
from memuri.tests.test_latency import LatencyBenchmark

async def measure_performance():
    client = Memuri()
    benchmark = LatencyBenchmark("Custom Test")
    
    # Measure add_memory operation
    for i in range(5):
        await benchmark.measure_operation(
            client.add_memory,
            content=f"Test memory {i}",
            category="FACT"
        )
    
    # Print stats
    benchmark.print_stats()
    
    # Get stats as dictionary
    stats = benchmark.get_stats()
    print(f"95th percentile latency: {stats['p95']:.2f}ms")

asyncio.run(measure_performance())
```

## Prerequisites

Before running the tests, ensure:

1. You have a running PostgreSQL instance with pgvector extension
2. You have a valid OpenAI API key (or other embedding provider configured)
3. Required Python packages are installed via Poetry or pip

## Environment Variables

These tests respect the following environment variables:

- `OPENAI_API_KEY`: API key for OpenAI embeddings
- `MEMURI_DATABASE_POSTGRES_URL`: PostgreSQL connection string
- `MEMURI_EMBEDDING_MODEL_NAME`: Embedding model to use

## Troubleshooting

If you encounter issues:

1. Verify that Postgres is running and accessible
2. Check your API key is valid and has sufficient quota
3. Increase log level to DEBUG for more detailed output
4. Ensure numpy is properly installed for statistical calculations

### Required Environment Variables

For most tests, you'll need these environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `MEMURI_DATABASE_POSTGRES_URL`: PostgreSQL connection string
- `MEMURI_EMBEDDING_MODEL_NAME`: Embedding model to use 