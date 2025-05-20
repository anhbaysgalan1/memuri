"""
Test module for the LatencyBenchmark functionality.
"""
import asyncio
import pytest
from typing import Dict, Any

from memuri.tests.test_latency import LatencyBenchmark


@pytest.mark.asyncio
async def test_latency_benchmark_initialization():
    """Test that LatencyBenchmark initializes correctly."""
    benchmark = LatencyBenchmark("Test Benchmark")
    assert benchmark.name == "Test Benchmark"
    assert benchmark.measurements == []
    assert benchmark.operation_results == []


@pytest.mark.asyncio
async def test_latency_benchmark_measure_operation():
    """Test that LatencyBenchmark can measure operations."""
    benchmark = LatencyBenchmark("Test Measurement")
    
    # Create a mock async operation
    async def mock_operation(delay: float = 0.1, return_value: Any = "result") -> Any:
        await asyncio.sleep(delay)
        return return_value
    
    # Measure the operation
    result = await benchmark.measure_operation(mock_operation, delay=0.1, return_value="test_result")
    
    # Check results
    assert result == "test_result"
    assert len(benchmark.measurements) == 1
    assert benchmark.measurements[0] > 0  # Time should be positive
    assert len(benchmark.operation_results) == 1
    assert benchmark.operation_results[0] == "test_result"


@pytest.mark.asyncio
async def test_latency_benchmark_stats():
    """Test that LatencyBenchmark calculates statistics correctly."""
    benchmark = LatencyBenchmark("Stats Test")
    
    # Create a mock async operation with varying delays
    async def mock_operation(delay: float = 0.1) -> str:
        await asyncio.sleep(delay)
        return "result"
    
    # Perform multiple operations with different delays
    for i in range(5):
        delay = 0.01 * (i + 1)  # Increasing delays
        await benchmark.measure_operation(mock_operation, delay=delay)
    
    # Get statistics
    stats = benchmark.get_stats()
    
    # Basic validations
    assert stats["count"] == 5
    assert stats["min"] > 0
    assert stats["max"] > stats["min"]
    assert stats["mean"] > 0
    assert stats["median"] > 0
    assert stats["p95"] > 0
    assert stats["std"] >= 0  # Standard deviation can be 0 in some cases


@pytest.mark.asyncio
async def test_latency_benchmark_clear():
    """Test that LatencyBenchmark can clear measurements."""
    benchmark = LatencyBenchmark("Clear Test")
    
    # Create a mock async operation
    async def mock_operation() -> str:
        await asyncio.sleep(0.01)
        return "result"
    
    # Perform operations
    await benchmark.measure_operation(mock_operation)
    await benchmark.measure_operation(mock_operation)
    
    # Verify we have measurements
    assert len(benchmark.measurements) == 2
    assert len(benchmark.operation_results) == 2
    
    # Clear and verify
    benchmark.clear()
    assert len(benchmark.measurements) == 0
    assert len(benchmark.operation_results) == 0 