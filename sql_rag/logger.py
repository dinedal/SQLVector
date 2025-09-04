"""Centralized logging configuration for SQL RAG."""

import logging
import os
import sys
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure logging for the SQL RAG library.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If not provided, uses SQL_RAG_LOG_LEVEL env var or defaults to INFO.
    """
    # Get log level from parameter, environment, or default
    if level is None:
        level = os.environ.get("SQL_RAG_LOG_LEVEL", "INFO")

    # Convert to uppercase and validate
    level = level.upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        level = "INFO"

    # Configure the root logger for the sql_rag package
    logger = logging.getLogger("sql_rag")

    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    # Set the log level
    logger.setLevel(getattr(logging, level))

    # Optionally log the configuration
    if level == "DEBUG":
        logger.debug(f"Logging configured with level: {level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: The name of the module (typically __name__)

    Returns:
        A configured logger instance
    """
    # Ensure name is under the sql_rag namespace
    if not name.startswith("sql_rag"):
        name = f"sql_rag.{name}"

    return logging.getLogger(name)