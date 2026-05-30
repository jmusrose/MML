#!/usr/bin/env python3
"""
Logger with dual output to console and file.
"""

import logging
import time
from datetime import timedelta


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath, args):
    """Create a logger with dual output to console and file.

    Args:
        filepath: Path to log file
        args: Configuration namespace (reserved for future use)

    Returns:
        logging.Logger with file and console handlers
    """
    log_formatter = LogFormatter()

    # File handler
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # Logger
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Reset time helper
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger
