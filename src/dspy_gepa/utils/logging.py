"""Logging utilities for dspy-gepa.

Provides standardized logging configuration and utilities
for the dspy-gepa framework.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """Configure logging for the dspy-gepa framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log format
    """
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        stream=sys.stdout,
        force=True  # Override existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Ensure logger has handlers (in case setup_logging wasn't called)
    if not logger.handlers:
        # Add a simple console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def create_module_logger(module_name: str) -> logging.Logger:
    """Create a logger for a specific module.
    
    Convenience function that creates a logger with a standardized name
    format for dspy-gepa modules.
    
    Args:
        module_name: Name of the module (without dspy_gepa prefix)
        
    Returns:
        Logger instance with name "dspy_gepa.{module_name}"
    """
    full_name = f"dspy_gepa.{module_name}"
    return get_logger(full_name)


def set_logger_level(logger: logging.Logger, level: str) -> None:
    """Set the logging level for a specific logger.
    
    Args:
        logger: Logger instance to configure
        level: Logging level as string
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)


def log_exception(logger: logging.Logger, message: str, exception: Exception) -> None:
    """Log an exception with context.
    
    Args:
        logger: Logger instance
        message: Context message
        exception: Exception to log
    """
    logger.error(f"{message}: {type(exception).__name__}: {exception}", exc_info=True)


def configure_debug_logging() -> None:
    """Configure verbose debug logging for development.
    
    This function sets up detailed logging suitable for
    development and troubleshooting.
    """
    setup_logging(
        level="DEBUG",
        format_string="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
    )


def configure_production_logging() -> None:
    """Configure production logging with minimal output.
    
    This function sets up logging suitable for production
    environments with focus on warnings and errors.
    """
    setup_logging(
        level="WARNING",
        include_timestamp=True
    )


# Default logger for the utils module
_logger = get_logger(__name__)
_logger.debug("Logging utilities module initialized")