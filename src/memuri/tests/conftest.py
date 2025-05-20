"""
Pytest configuration for Memuri tests.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, AsyncGenerator

import pytest
import pytest_asyncio

# Add the src directory to the Python path so imports work correctly
src_path = str(Path(__file__).parent.parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Test environment settings - override with environment variables or defaults
TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL", "postgresql://memuri:memuri@localhost:5432/memuri"
)
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379")

# Note: We don't redefine the event_loop fixture here anymore
# pytest-asyncio will provide its own event_loop fixture 