"""Latency benchmark tools for Memuri.

This module provides tools for measuring and analyzing the performance of
various Memuri operations.
"""

import time
import statistics
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Awaitable
import numpy as np

# Add the src directory to the Python path for imports to work in tests
src_path = str(Path(__file__).parent.parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from memuri.core.logging import get_logger

logger = get_logger(__name__)


class LatencyBenchmark:
    """Benchmark tool for measuring latency of asynchronous operations."""

    def __init__(self, name: str = "Benchmark"):
        """Initialize a new benchmark.
        
        Args:
            name: Name of the benchmark for reporting
        """
        self.name = name
        self.measurements: List[float] = []
        self.operation_results: List[Any] = []
        self.logger = logger

    async def measure_operation(self, operation: Callable[..., Awaitable[Any]], **kwargs) -> Any:
        """Measure the latency of an async operation.
        
        Args:
            operation: Asynchronous function to measure
            **kwargs: Arguments to pass to the operation
            
        Returns:
            Any: The result of the operation
        
        Example:
            ```python
            benchmark = LatencyBenchmark("Search")
            await benchmark.measure_operation(
                client.search_memory, 
                query="test query", 
                top_k=5
            )
            ```
        """
        try:
            # Record start time
            start_time = time.time()
            
            # Execute operation
            result = await operation(**kwargs)
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Store results
            self.measurements.append(latency)
            self.operation_results.append(result)
            
            self.logger.debug(f"{self.name}: Operation completed in {latency:.2f}ms")
            
            return result
        except Exception as e:
            self.logger.error(f"{self.name}: Error in operation: {str(e)}")
            raise

    def print_stats(self):
        """Print statistics about the benchmark measurements."""
        if not self.measurements:
            self.logger.warning(f"{self.name}: No measurements recorded")
            return
        
        # Calculate statistics
        stats = self.get_stats()
        
        # Print results
        print(f"\n=== {self.name} Benchmark Results ===")
        print(f"Total operations: {stats['count']}")
        print(f"Mean latency: {stats['mean']:.2f}ms")
        print(f"Median latency: {stats['median']:.2f}ms")
        print(f"Min latency: {stats['min']:.2f}ms")
        print(f"Max latency: {stats['max']:.2f}ms")
        print(f"95th percentile: {stats['p95']:.2f}ms")
        print(f"Standard deviation: {stats['std']:.2f}ms")
        print(f"==================================\n")

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about the benchmark measurements.
        
        Returns:
            Dict[str, float]: Dictionary with statistical metrics
        """
        if not self.measurements:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "p95": 0,
                "std": 0,
            }
        
        # Calculate statistics
        return {
            "count": len(self.measurements),
            "mean": statistics.mean(self.measurements),
            "median": statistics.median(self.measurements),
            "min": min(self.measurements),
            "max": max(self.measurements),
            "p95": np.percentile(self.measurements, 95),
            "std": statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0,
        }

    def clear(self):
        """Clear all measurements and results."""
        self.measurements = []
        self.operation_results = []
        self.logger.debug(f"{self.name}: Benchmark data cleared") 