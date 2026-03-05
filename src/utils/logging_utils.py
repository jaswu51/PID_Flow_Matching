"""
Logging utilities for cmu-10799-diffusion.

Provides unified logging setup for training, sampling, and evaluation.
"""

import os
import sys
import logging
from typing import Optional


def setup_logger(
    log_dir: str,
    name: str = 'main',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up logger that writes to both console and file.

    Args:
        log_dir: Directory to save log files
        name: Logger name
        log_file: Log filename (default: {name}.log)
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')

    # File handler - detailed logs
    if log_file is None:
        log_file = f'{name}.log'

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler - simple logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    return logger


def log_section(logger: logging.Logger, title: str, width: int = 80):
    """Log a section header."""
    logger.info("")
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)
    logger.info("")
