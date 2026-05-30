#!/usr/bin/env python3
"""Unit tests for the logger module."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.logger import create_logger


class TestLogger:
    """Tests for create_logger."""

    def test_logger_creates_file(self, mock_args):
        log_dir = tempfile.mkdtemp()
        log_path = os.path.join(log_dir, "test.log")
        logger = create_logger(log_path, mock_args)
        logger.info("Test message")
        assert os.path.exists(log_path)

    def test_logger_writes_message(self, mock_args):
        log_dir = tempfile.mkdtemp()
        log_path = os.path.join(log_dir, "test.log")
        logger = create_logger(log_path, mock_args)
        logger.info("Hello World")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        with open(log_path, "r") as f:
            content = f.read()
        assert "Hello World" in content

    def test_logger_format_contains_level(self, mock_args):
        log_dir = tempfile.mkdtemp()
        log_path = os.path.join(log_dir, "test.log")
        logger = create_logger(log_path, mock_args)
        logger.info("Format test")

        for handler in logger.handlers:
            handler.flush()

        with open(log_path, "r") as f:
            content = f.read()
        assert "INFO" in content

    def test_logger_has_handlers(self, mock_args):
        log_dir = tempfile.mkdtemp()
        log_path = os.path.join(log_dir, "test.log")
        logger = create_logger(log_path, mock_args)
        # Should have file handler and console handler
        assert len(logger.handlers) == 2

    def test_logger_reset_time(self, mock_args):
        log_dir = tempfile.mkdtemp()
        log_path = os.path.join(log_dir, "test.log")
        logger = create_logger(log_path, mock_args)
        # Should not raise
        logger.reset_time()
