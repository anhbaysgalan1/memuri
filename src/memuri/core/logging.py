"""Structured logging module for memuri SDK."""

import json
import logging
import sys
from typing import Dict, Optional, Union

from memuri.core.config import LoggingSettings


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            str: JSON-formatted log string
        """
        log_data: Dict[str, Union[str, int, float, bool, None]] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if available
        if hasattr(record, "props"):
            log_data.update(record.props)
        
        return json.dumps(log_data)


def configure_logging(settings: Optional[LoggingSettings] = None) -> logging.Logger:
    """Configure and return a logger for the memuri SDK.
    
    Args:
        settings: Logging settings to use
        
    Returns:
        logging.Logger: Configured logger
    """
    # Use default settings if none provided
    if settings is None:
        settings = LoggingSettings()
    
    # Create logger
    logger = logging.getLogger("memuri")
    logger.setLevel(settings.level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter based on settings
    if settings.json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(settings.format))
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    This function returns a child logger of the main memuri logger.
    If the main logger has not been configured, it will be configured
    with default settings.
    
    Args:
        name: Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(f"memuri.{name}")
    
    # If no handlers are configured, configure the root logger
    if not logger.handlers and not logger.parent.handlers:
        configure_logging()
    
    return logger 