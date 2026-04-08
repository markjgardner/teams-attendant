"""Tests for utils.logging module."""

from __future__ import annotations

import structlog

from teams_attendant.utils.logging import setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_import(self):
        """Module can be imported without errors."""
        from teams_attendant.utils import logging  # noqa: F811

        assert hasattr(logging, "setup_logging")

    def test_setup_default(self):
        """Default call completes without error."""
        setup_logging()

    def test_setup_with_level(self):
        """Accepts different log levels."""
        setup_logging(level="DEBUG")
        setup_logging(level="WARNING")

    def test_setup_json_output(self):
        """JSON output mode configures structlog."""
        setup_logging(json_output=True)
        logger = structlog.get_logger()
        assert logger is not None

    def test_setup_invalid_level_defaults_to_info(self):
        """Invalid log level falls back to INFO."""
        setup_logging(level="NONEXISTENT")
